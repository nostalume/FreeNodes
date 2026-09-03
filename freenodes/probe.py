from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx
import yaml
from pydantic import TypeAdapter, ValidationError

from freenodes.capability import (
    CapabilityPolicy,
    CapabilityRunReceipt,
    CapabilityTarget,
    DelayObservation,
    NodeCapabilityDecision,
    ProbeCandidate,
    ProbeDiagnostic,
    ProbeFailureCode,
    ProbePlan,
    TargetControlWindow,
    TargetObservation,
    ValidatedProbeBatch,
    classify_capability,
)
from freenodes.mihomo import DelayPayload, MihomoValidator, loopback_port
from freenodes.nodes import ProbeableNode

JsonRequest = Callable[[str, float], Awaitable[dict[str, Any]]]
_JSON_OBJECT = TypeAdapter(dict[str, Any])


class ProbeSessionError(RuntimeError):
    def __init__(self, code: ProbeFailureCode, detail: str = ""):
        self.diagnostic = ProbeDiagnostic(code=code, detail=detail[:200])
        super().__init__(code)


class MihomoDelayProbe:
    def __init__(
        self,
        *,
        request_json: JsonRequest | None = None,
        timeout_ms: int = 4000,
        concurrency: int = 64,
        deadline: float = 300.0,
        max_candidates: int = 4000,
    ):
        if min(timeout_ms, concurrency, deadline, max_candidates) <= 0:
            raise ValueError("probe limits must be positive")
        self.request_override = request_json
        self.timeout_ms = timeout_ms
        self.concurrency = concurrency
        self.deadline = deadline
        self.max_candidates = max_candidates

    async def observe(
        self,
        controller: str,
        candidate: ProbeCandidate,
        endpoint: str,
        request_json: JsonRequest,
        expected_status: int,
    ) -> DelayObservation:
        query = urlencode(
            {
                "url": endpoint,
                "timeout": str(self.timeout_ms),
                "expected": str(expected_status),
            }
        )
        name = quote(candidate.proxy_name, safe="")
        url = f"{controller.rstrip('/')}/proxies/{name}/delay?{query}"
        try:
            value = await request_json(url, self.timeout_ms / 1000 + 1)
            delay = DelayPayload.model_validate(value)
            return DelayObservation(
                endpoint=endpoint,
                status="success",
                delay_ms=delay.delay,
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            return DelayObservation.failed(endpoint, "timeout", "request_timeout")
        except httpx.HTTPStatusError as error:
            return DelayObservation.failed(
                endpoint,
                "api_error",
                "controller_http",
                f"HTTP {error.response.status_code}",
            )
        except ValidationError:
            return DelayObservation.failed(endpoint, "api_error", "controller_payload")
        except Exception as error:
            return DelayObservation.failed(
                endpoint, "api_error", "controller_error", type(error).__name__
            )

    async def probe_wave(
        self,
        controller: str,
        candidates: Sequence[ProbeCandidate],
        endpoint: str,
        request_json: JsonRequest,
        deadline: float,
        expected_status: int,
    ) -> dict[str, DelayObservation]:
        if len(candidates) > self.max_candidates:
            raise ValueError("probe candidate budget exceeded")
        pending_candidates = deque(candidates)
        observations: dict[str, DelayObservation] = {}

        async def worker() -> None:
            while pending_candidates:
                candidate = pending_candidates.popleft()
                observations[candidate.fingerprint] = await self.observe(
                    controller,
                    candidate,
                    endpoint,
                    request_json,
                    expected_status,
                )

        tasks = tuple(
            asyncio.create_task(worker())
            for _ in range(min(self.concurrency, len(candidates)))
        )
        if tasks:
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                )
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        for candidate in candidates:
            observations.setdefault(
                candidate.fingerprint,
                DelayObservation.failed(endpoint, "cancelled", "run_deadline"),
            )
        return observations


class MihomoProbeSession:
    BLOCK_SIZE = 128
    WORKERS = 64

    def __init__(
        self,
        executable: Path,
        *,
        delay_probe: MihomoDelayProbe | None = None,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        validate_config: Callable[[Path, Path], None] | None = None,
        startup_timeout: float = 20.0,
        max_validation_invocations: int = 64,
    ):
        if min(startup_timeout, max_validation_invocations) <= 0:
            raise ValueError("probe limits must be positive")
        self.executable = executable.resolve()
        self.delay_probe = delay_probe or MihomoDelayProbe()
        self.process_factory = process_factory
        self.startup_timeout = startup_timeout
        self.validate_config = (
            validate_config
            or MihomoValidator(self.executable, timeout=startup_timeout).validate_config
        )
        self.max_validation_invocations = max_validation_invocations

    async def probe_capabilities(
        self,
        plan: ProbePlan,
        targets: Sequence[CapabilityTarget],
        policy: CapabilityPolicy,
    ) -> CapabilityRunReceipt:
        admitted_targets = CapabilityTarget.admit_registry(targets, quorum=2)
        nodes = plan.nodes
        if not nodes:
            return CapabilityRunReceipt(status="complete")
        if len(nodes) > self.delay_probe.max_candidates:
            raise ValueError("probe candidate budget exceeded")
        names = {node.display_name for node in nodes}
        control_names = {self._control_name(target) for target in admitted_targets}
        if len(names) != len(nodes) or names & control_names:
            raise ValueError("Mihomo probe names must be unique and non-reserved")
        with tempfile.TemporaryDirectory(prefix="freenodes-mihomo-capability-") as raw:
            root = Path(raw).resolve()
            validated = await asyncio.to_thread(
                self._validate_candidates,
                root,
                loopback_port(),
                nodes,
            )
            rejected = {item.fingerprint: item for item in validated.failures}
            if not validated.complete:
                return CapabilityRunReceipt(
                    status="inconclusive",
                    diagnostic=ProbeDiagnostic(code="validation_budget"),
                )
            if not validated.nodes:
                return CapabilityRunReceipt(
                    status="complete", decisions=tuple(rejected.values())
                )
            try:
                async with self._controller(
                    root, validated.nodes, admitted_targets
                ) as connection:
                    result = await self._probe_blocks(
                        *connection,
                        validated.nodes,
                        admitted_targets,
                        policy,
                    )
            except ProbeSessionError as error:
                result = CapabilityRunReceipt(
                    status="inconclusive",
                    diagnostic=error.diagnostic,
                )
            measured = {item.fingerprint: item for item in result.decisions}
            measured.update(rejected)
            return CapabilityRunReceipt(
                status=result.status,
                decisions=tuple(
                    measured[node.fingerprint]
                    for node in nodes
                    if node.fingerprint in measured
                ),
                accepted_fingerprints=result.accepted_fingerprints,
                controls=result.controls,
                deadline_reached=result.deadline_reached,
                diagnostic=result.diagnostic,
            )

    async def _probe_blocks(
        self,
        controller: str,
        request_json: JsonRequest,
        nodes: tuple[ProbeableNode, ...],
        targets: tuple[CapabilityTarget, ...],
        policy: CapabilityPolicy,
    ) -> CapabilityRunReceipt:
        deadline = asyncio.get_running_loop().time() + self.delay_probe.deadline
        decisions: list[NodeCapabilityDecision] = []
        controls: list[TargetControlWindow] = []

        async def observe(proxy_name: str, target: CapabilityTarget):
            value = await self.delay_probe.observe(
                controller,
                ProbeCandidate(fingerprint=proxy_name, proxy_name=proxy_name),
                str(target.url),
                request_json,
                target.expected_status,
            )
            return TargetObservation.from_delay(target.id, value)

        async def observe_controls() -> tuple[TargetObservation, ...]:
            return tuple(
                [
                    await observe(self._control_name(target), target)
                    for target in targets
                ]
            )

        async def close_windows(
            opening: Sequence[TargetObservation],
        ) -> tuple[TargetControlWindow, ...]:
            closing = await observe_controls()
            return tuple(
                TargetControlWindow(
                    target_id=target.id,
                    opening=(
                        "success" if opening[index].status == "success" else "failure"
                    ),
                    closing=(
                        "success" if closing[index].status == "success" else "failure"
                    ),
                )
                for index, target in enumerate(targets)
            )

        for offset in range(0, len(nodes), self.BLOCK_SIZE):
            block = nodes[offset : offset + self.BLOCK_SIZE]
            opening = await observe_controls()
            if sum(item.status == "success" for item in opening) < 2:
                controls.extend(await close_windows(opening))
                return CapabilityRunReceipt(
                    status="inconclusive",
                    controls=tuple(controls),
                    diagnostic=ProbeDiagnostic(code="control_unavailable"),
                )
            candidates = tuple(
                ProbeCandidate(
                    fingerprint=node.fingerprint,
                    proxy_name=node.display_name,
                )
                for node in block
            )
            measured: dict[str, list[TargetObservation]] = defaultdict(list)
            unresolved = candidates
            for target in targets:
                wave = await self.delay_probe.probe_wave(
                    controller,
                    unresolved,
                    str(target.url),
                    request_json,
                    deadline,
                    target.expected_status,
                )
                for candidate in unresolved:
                    measured[candidate.fingerprint].append(
                        TargetObservation.from_delay(
                            target.id, wave[candidate.fingerprint]
                        )
                    )
                unresolved = tuple(
                    candidate
                    for candidate in unresolved
                    if sum(
                        item.status == "success"
                        for item in measured[candidate.fingerprint]
                    )
                    < 2
                    and sum(
                        item.status == "request_error"
                        for item in measured[candidate.fingerprint]
                    )
                    < 2
                )
                if not unresolved:
                    break
            windows = await close_windows(opening)
            controls.extend(windows)
            decisions.extend(
                classify_capability(
                    node.fingerprint,
                    measured.get(node.fingerprint, ()),
                    windows,
                    quorum=2,
                )
                for node in block
            )
            deadline_reached = any(
                item.status == "cancelled"
                for values in measured.values()
                for item in values
            )
            if (
                sum(item.opening == item.closing == "success" for item in windows) < 2
                or deadline_reached
            ):
                return CapabilityRunReceipt(
                    status="inconclusive",
                    decisions=tuple(decisions),
                    controls=tuple(controls),
                    deadline_reached=deadline_reached,
                    diagnostic=ProbeDiagnostic(
                        code=(
                            "run_deadline"
                            if deadline_reached
                            else "control_unavailable"
                        )
                    ),
                )
            if (
                sum(item.status == "capable" for item in decisions)
                >= policy.max_published
            ):
                break
        capable = tuple(
            item.fingerprint for item in decisions if item.status == "capable"
        )
        return CapabilityRunReceipt(
            status="complete",
            decisions=tuple(decisions),
            accepted_fingerprints=capable[: policy.max_published],
            controls=tuple(controls),
        )

    @asynccontextmanager
    async def _controller(
        self,
        root: Path,
        nodes: tuple[ProbeableNode, ...],
        targets: tuple[CapabilityTarget, ...],
    ) -> AsyncIterator[tuple[str, JsonRequest]]:
        port = loopback_port()
        config_path = root / "capability.yaml"
        self._write_config(
            config_path,
            port,
            nodes,
            tuple(self._control_name(target) for target in targets),
        )
        process = self._start_process(config_path, root / "capability-home")
        if process is None:
            raise ProbeSessionError("process_start")
        try:
            controller = f"http://127.0.0.1:{port}"
            limits = httpx.Limits(
                max_connections=self.WORKERS,
                max_keepalive_connections=self.WORKERS,
            )
            async with httpx.AsyncClient(
                timeout=self.delay_probe.timeout_ms / 1000 + 1,
                trust_env=False,
                verify=False,
                limits=limits,
            ) as client:
                request_json = self.delay_probe.request_override or self._request(
                    client
                )
                await self._ready(process, controller, request_json)
                yield controller, request_json
        finally:
            await asyncio.to_thread(self._terminate, process)

    @staticmethod
    def _request(client: httpx.AsyncClient) -> JsonRequest:
        async def request(url: str, timeout: float) -> dict[str, Any]:
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            return _JSON_OBJECT.validate_python(response.json())

        return request

    async def _ready(
        self,
        process: subprocess.Popen[str],
        controller: str,
        request_json: JsonRequest,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if process.poll() is not None:
                raise ProbeSessionError("controller_start", "process exited")
            try:
                if (await request_json(f"{controller}/version", 1.0)).get("version"):
                    return
            except Exception as error:
                last_error = error
            await asyncio.sleep(0.05)
        detail = type(last_error).__name__ if last_error else "timeout"
        raise ProbeSessionError("controller_start", detail)

    def _validate_candidates(
        self,
        root: Path,
        controller_port: int,
        nodes: tuple[ProbeableNode, ...],
    ) -> ValidatedProbeBatch:
        config_path = root / "check.yaml"
        check_home = root / "check-home"
        check_home.mkdir()
        pending = [nodes]
        accepted: list[ProbeableNode] = []
        failures: list[NodeCapabilityDecision] = []
        invocations = 0
        while pending:
            if invocations >= self.max_validation_invocations:
                return ValidatedProbeBatch(
                    nodes=tuple(accepted), failures=tuple(failures), complete=False
                )
            batch = pending.pop()
            invocations += 1
            self._write_config(config_path, controller_port, batch)
            try:
                self.validate_config(config_path, check_home)
            except Exception:
                if len(batch) == 1:
                    failures.append(
                        NodeCapabilityDecision(
                            fingerprint=batch[0].fingerprint,
                            status="failed",
                            reason="config_rejected",
                        )
                    )
                else:
                    midpoint = len(batch) // 2
                    pending.extend((batch[midpoint:], batch[:midpoint]))
            else:
                accepted.extend(batch)
        position = {node.fingerprint: index for index, node in enumerate(nodes)}
        accepted.sort(key=lambda node: position[node.fingerprint])
        failures.sort(key=lambda item: position[item.fingerprint])
        return ValidatedProbeBatch(nodes=tuple(accepted), failures=tuple(failures))

    @staticmethod
    def _control_name(target: CapabilityTarget) -> str:
        return f"CONTROL / {target.id}"

    @staticmethod
    def _write_config(
        config_path: Path,
        controller_port: int,
        nodes: Sequence[ProbeableNode],
        controls: Sequence[str] = (),
    ) -> None:
        config = {
            "allow-lan": False,
            "ipv6": False,
            "log-level": "silent",
            "external-controller": f"127.0.0.1:{controller_port}",
            "proxies": [
                node.proxy.mihomo_payload() | {"name": node.display_name}
                for node in nodes
            ]
            + [{"name": name, "type": "direct"} for name in controls],
        }
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _start_process(
        self,
        config_path: Path,
        home_dir: Path,
    ) -> subprocess.Popen[str] | None:
        home_dir.mkdir()
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            return self.process_factory(
                [str(self.executable), "-d", str(home_dir), "-f", str(config_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=creation_flags,
            )
        except Exception:
            return None

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
