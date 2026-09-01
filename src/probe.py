from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode

import httpx
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from src.config import FrozenModel
from src.mihomo import MihomoValidationError, MihomoValidator, loopback_port
from src.nodes import ProbeableNode
from src.quality import (
    DelayObservation,
    ProbeCandidate,
    ProbeDiagnostic,
    ProbeEvidence,
    ProbeFailureCode,
    ProbePlan,
    ProbeRunFailure,
    ProbeRunPhase,
    ProbeRunResult,
    ProbeRunSuccess,
    QualityPolicy,
    TransferObservation,
    TransferStatus,
    TransferTargetEvidence,
    ValidatedProbeBatch,
)


class DelayPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    delay: int = Field(ge=0, strict=True)


_JSON_OBJECT = TypeAdapter(dict[str, Any])

JsonRequest = Callable[[str, float], Awaitable[dict[str, Any]]]
ChunkSource = Callable[[str, "TransferTarget", int], AsyncGenerator[bytes, None]]
SelectorUpdate = Callable[[str, str], Awaitable[None]]


class TransferTarget(FrozenModel):
    name: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    url: HttpUrl

    @model_validator(mode="after")
    def require_public_https(self) -> "TransferTarget":
        if self.url.scheme != "https" and self.url.host not in {
            "127.0.0.1",
            "localhost",
        }:
            raise ValueError("public transfer target must use HTTPS")
        return self


DEFAULT_TRANSFER_TARGETS = (
    TransferTarget(
        name="cloudflare",
        authority="cloudflare",
        url=HttpUrl("https://speed.cloudflare.com/__down?bytes=1048576"),
    ),
    TransferTarget(
        name="hetzner-hel1",
        authority="hetzner",
        url=HttpUrl("https://hel1-speed.hetzner.com/100MB.bin"),
    ),
)


class TransferWindow(FrozenModel):
    target: str
    opening: TransferObservation
    attempts: tuple[TransferObservation, ...] = Field(default=(), strict=False)
    closing: TransferObservation | None = None

    @property
    def valid(self) -> bool:
        return self.opening.status == "success" and (
            self.closing is not None and self.closing.status == "success"
        )


class ProbeRunAbort(Exception):
    def __init__(self, result: ProbeRunFailure):
        self.result = result
        super().__init__(result.diagnostic.code)


class MihomoDelayProbe:
    COARSE_ENDPOINT = "https://www.gstatic.com/generate_204"
    CONFIRM_ENDPOINT = "https://cp.cloudflare.com/generate_204"

    def __init__(
        self,
        *,
        request_json: JsonRequest | None = None,
        timeout_ms: int = 2500,
        concurrency: int = 64,
        deadline: float = 300.0,
        max_candidates: int = 4000,
    ):
        if timeout_ms <= 0 or concurrency <= 0 or deadline <= 0 or max_candidates <= 0:
            raise ValueError("probe limits must be positive")
        self._request_override = request_json
        self.timeout_ms = timeout_ms
        self.concurrency = concurrency
        self.deadline = deadline
        self.max_candidates = max_candidates

    async def request_json(self, url: str, timeout: float) -> dict[str, Any]:
        if self._request_override is not None:
            return await self._request_override(url, timeout)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            return _JSON_OBJECT.validate_python(response.json())

    async def probe_controller(
        self,
        controller: str,
        candidates: Sequence[ProbeCandidate],
        request_json: JsonRequest | None = None,
    ) -> tuple[ProbeEvidence, ...]:
        if len(candidates) > self.max_candidates:
            raise ValueError("probe candidate budget exceeded")
        if not candidates:
            return ()
        if request_json is not None:
            return await self._probe_controller_with(
                controller, candidates, request_json
            )
        if self._request_override is not None:
            return await self._probe_controller_with(
                controller, candidates, self._request_override
            )
        limits = httpx.Limits(
            max_connections=self.concurrency,
            max_keepalive_connections=self.concurrency,
        )
        async with httpx.AsyncClient(
            timeout=self.timeout_ms / 1000 + 1,
            trust_env=False,
            limits=limits,
        ) as client:

            async def request_json(url: str, timeout: float) -> dict[str, Any]:
                response = await client.get(url)
                response.raise_for_status()
                return _JSON_OBJECT.validate_python(response.json())

            return await self._probe_controller_with(
                controller, candidates, request_json
            )

    async def _probe_controller_with(
        self,
        controller: str,
        candidates: Sequence[ProbeCandidate],
        request_json: JsonRequest,
    ) -> tuple[ProbeEvidence, ...]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.deadline
        coarse = await self._probe_wave(
            controller,
            candidates,
            self.COARSE_ENDPOINT,
            request_json,
            deadline,
        )
        confirm_candidates = [
            candidate
            for candidate in candidates
            if coarse[candidate.fingerprint].status == "success"
        ]
        confirm = await self._probe_wave(
            controller,
            confirm_candidates,
            self.CONFIRM_ENDPOINT,
            request_json,
            deadline,
        )
        return tuple(
            ProbeEvidence(
                fingerprint=candidate.fingerprint,
                proxy_name=candidate.proxy_name,
                coarse=coarse[candidate.fingerprint],
                confirm=confirm.get(candidate.fingerprint),
            )
            for candidate in candidates
        )

    async def _probe_wave(
        self,
        controller: str,
        candidates: Sequence[ProbeCandidate],
        endpoint: str,
        request_json: JsonRequest,
        deadline: float,
    ) -> dict[str, DelayObservation]:
        if not candidates:
            return {}
        pending_candidates = deque(candidates)
        observations: dict[str, DelayObservation] = {}

        async def worker() -> None:
            while pending_candidates:
                candidate = pending_candidates.popleft()
                observations[candidate.fingerprint] = await self._probe_one(
                    controller, candidate, endpoint, request_json
                )

        tasks = tuple(
            asyncio.create_task(worker())
            for _ in range(min(self.concurrency, len(candidates)))
        )
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            task.result()
        for candidate in candidates:
            if candidate.fingerprint not in observations:
                observations[candidate.fingerprint] = DelayObservation.failed(
                    endpoint, "cancelled", "run_deadline"
                )
        return observations

    async def _probe_one(
        self,
        controller: str,
        candidate: ProbeCandidate,
        endpoint: str,
        request_json: JsonRequest,
    ) -> DelayObservation:
        query = urlencode(
            {"url": endpoint, "timeout": str(self.timeout_ms), "expected": "204"}
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
        except (asyncio.TimeoutError, httpx.TimeoutException):
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


class MihomoProbeSession:
    SELECTOR_NAME = "QUALITY-TEST"
    TRANSFER_BLOCK_SIZE = 8

    def __init__(
        self,
        executable: Path,
        *,
        delay_probe: MihomoDelayProbe | None = None,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        validate_config: Callable[[Path, Path], None] | None = None,
        selector_update: SelectorUpdate | None = None,
        chunk_source: ChunkSource | None = None,
        transfer_targets: tuple[
            TransferTarget, TransferTarget
        ] = DEFAULT_TRANSFER_TARGETS,
        chunk_size: int = 64 * 1024,
        transfer_timeout: float = 10.0,
        startup_timeout: float = 20.0,
        max_validation_invocations: int = 64,
    ):
        if (
            min(
                max_validation_invocations,
                chunk_size,
                transfer_timeout,
                startup_timeout,
            )
            <= 0
        ):
            raise ValueError("probe limits must be positive")
        if len({target.authority for target in transfer_targets}) != 2:
            raise ValueError("transfer targets must have distinct authorities")
        self.executable = executable.resolve()
        self.delay_probe = delay_probe or MihomoDelayProbe()
        self.process_factory = process_factory
        self.startup_timeout = startup_timeout
        self.validate_config = (
            validate_config
            or MihomoValidator(self.executable, timeout=startup_timeout).validate_config
        )
        self.selector_update = selector_update
        self.chunk_source = chunk_source
        self.transfer_targets = transfer_targets
        self.chunk_size = chunk_size
        self.transfer_timeout = transfer_timeout
        self.max_validation_invocations = max_validation_invocations

    async def probe(
        self,
        plan: ProbePlan,
        policy: QualityPolicy,
    ) -> ProbeRunResult:
        nodes = plan.nodes
        candidates = tuple(
            ProbeCandidate(fingerprint=node.fingerprint, proxy_name=node.display_name)
            for node in nodes
        )
        if not candidates:
            return ProbeRunSuccess()
        if len(candidates) > self.delay_probe.max_candidates:
            raise ValueError("probe candidate budget exceeded")
        names = {item.proxy_name for item in candidates}
        if len(names) != len(candidates) or names & {"DIRECT", self.SELECTOR_NAME}:
            raise ValueError("Mihomo probe names must be unique and non-reserved")

        with tempfile.TemporaryDirectory(prefix="freenodes-mihomo-probe-") as temporary:
            return await self._probe_root(
                Path(temporary).resolve(),
                nodes,
                candidates,
                policy,
            )

    async def _probe_root(
        self,
        root: Path,
        nodes: tuple[ProbeableNode, ...],
        candidates: tuple[ProbeCandidate, ...],
        policy: QualityPolicy,
    ) -> ProbeRunResult:
        validated = await asyncio.to_thread(
            self._validate_candidates,
            root,
            loopback_port(),
            nodes,
        )
        if not validated.complete:
            return self._run_failure("validation", "validation_budget")
        if not validated.nodes:
            by_id = {item.fingerprint: item for item in validated.failures}
            return ProbeRunSuccess(
                evidence=tuple(by_id[item.fingerprint] for item in candidates)
            )

        try:
            delays = await self._probe_delays(root, validated.nodes)
            delay_by_id = {item.fingerprint: item for item in delays}
            finalists = tuple(
                node
                for node in validated.nodes
                if self._delay_qualifies(
                    delay_by_id[node.fingerprint],
                    policy.max_delay_ms,
                )
            )
            transfers, transfer_targets = await self._probe_transfers(
                root,
                validated.nodes,
                finalists,
                policy.transfer_bytes,
            )
        except ProbeRunAbort as abort:
            return abort.result

        evidence_by_id = {
            item.fingerprint: item for item in (*validated.failures, *delays)
        }
        for transfer in transfers:
            evidence_by_id[transfer.fingerprint] = evidence_by_id[
                transfer.fingerprint
            ].model_copy(update={"transfer": transfer})
        return ProbeRunSuccess(
            evidence=tuple(
                evidence_by_id[candidate.fingerprint] for candidate in candidates
            ),
            transfer_targets=transfer_targets,
        )

    async def _probe_delays(
        self,
        root: Path,
        nodes: tuple[ProbeableNode, ...],
    ) -> tuple[ProbeEvidence, ...]:
        controller_port = loopback_port()
        config_path = root / "delay.yaml"
        self._write_config(config_path, controller_port, nodes)
        process = self._start_process(config_path, root / "delay-home")
        if process is None:
            raise ProbeRunAbort(self._run_failure("process", "process_start", "delay"))
        try:
            controller = f"http://127.0.0.1:{controller_port}"
            limits = httpx.Limits(
                max_connections=self.delay_probe.concurrency,
                max_keepalive_connections=self.delay_probe.concurrency,
            )
            async with httpx.AsyncClient(
                timeout=self.delay_probe.timeout_ms / 1000 + 1,
                trust_env=False,
                verify=False,
                limits=limits,
            ) as controller_client:
                request_json = self._controller_request(controller_client)
                await self._ready(process, controller, request_json, "delay")
                return await self.delay_probe.probe_controller(
                    controller,
                    tuple(
                        ProbeCandidate(
                            fingerprint=node.fingerprint,
                            proxy_name=node.display_name,
                        )
                        for node in nodes
                    ),
                    request_json=request_json,
                )
        finally:
            await asyncio.to_thread(self._terminate, process)

    async def _probe_transfers(
        self,
        root: Path,
        nodes: tuple[ProbeableNode, ...],
        finalists: tuple[ProbeableNode, ...],
        byte_count: int,
    ) -> tuple[tuple[TransferObservation, ...], tuple[TransferTargetEvidence, ...]]:
        if not finalists:
            return (), ()
        controller_port = loopback_port()
        listener_port = loopback_port()
        while listener_port == controller_port:
            listener_port = loopback_port()
        config_path = root / "transfer.yaml"
        self._write_config(
            config_path,
            controller_port,
            nodes,
            listener_port=listener_port,
        )
        try:
            await asyncio.to_thread(
                self.validate_config,
                config_path,
                root / "transfer-check-home",
            )
        except Exception as error:
            raise ProbeRunAbort(
                self._run_failure("validation", "config_rejected", type(error).__name__)
            ) from error
        process = self._start_process(config_path, root / "transfer-home")
        if process is None:
            raise ProbeRunAbort(
                self._run_failure("process", "process_start", "transfer")
            )
        controller = f"http://127.0.0.1:{controller_port}"
        try:
            async with httpx.AsyncClient(
                timeout=self.transfer_timeout,
                trust_env=False,
                verify=False,
            ) as controller_client:
                request_json = self._controller_request(controller_client)
                await self._ready(process, controller, request_json, "transfer")
                return await self._measure_finalists(
                    controller,
                    controller_client,
                    listener_port,
                    finalists,
                    byte_count,
                )
        finally:
            await asyncio.to_thread(self._terminate, process)

    async def _ready(
        self,
        process: subprocess.Popen[str],
        controller: str,
        request_json: JsonRequest,
        activity: Literal["delay", "transfer"],
    ) -> None:
        try:
            await self._await_ready(process, controller, request_json)
        except Exception as error:
            raise ProbeRunAbort(
                self._run_failure(
                    "readiness",
                    "controller_start",
                    f"{activity}/{type(error).__name__}",
                )
            ) from error

    def _controller_request(self, client: httpx.AsyncClient) -> JsonRequest:
        if self.delay_probe._request_override is not None:
            return self.delay_probe._request_override

        async def request(url: str, timeout: float) -> dict[str, Any]:
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            return _JSON_OBJECT.validate_python(response.json())

        return request

    async def _measure_finalists(
        self,
        controller: str,
        controller_client: httpx.AsyncClient,
        listener_port: int,
        nodes: tuple[ProbeableNode, ...],
        byte_count: int,
    ) -> tuple[tuple[TransferObservation, ...], tuple[TransferTargetEvidence, ...]]:
        outcomes: list[TransferObservation] = []
        windows: dict[str, list[TransferWindow]] = defaultdict(list)
        for offset in range(0, len(nodes), self.TRANSFER_BLOCK_SIZE):
            block = nodes[offset : offset + self.TRANSFER_BLOCK_SIZE]
            primary_index = (offset // self.TRANSFER_BLOCK_SIZE) % 2
            primary = self.transfer_targets[primary_index]
            alternate = self.transfer_targets[1 - primary_index]
            first = await self._measure_window(
                controller,
                controller_client,
                listener_port,
                block,
                primary,
                byte_count,
            )
            windows[primary.name].append(first)
            first_by_id = {item.fingerprint: item for item in first.attempts}
            unresolved = tuple(
                node
                for node in block
                if first_by_id.get(node.fingerprint) is None
                or first_by_id[node.fingerprint].status != "success"
            )
            second = (
                await self._measure_window(
                    controller,
                    controller_client,
                    listener_port,
                    unresolved,
                    alternate,
                    byte_count,
                )
                if unresolved
                else None
            )
            if second is not None:
                windows[alternate.name].append(second)
            second_by_id = (
                {item.fingerprint: item for item in second.attempts} if second else {}
            )
            passed = False
            for node in block:
                attempts = tuple(
                    item
                    for item in (
                        first_by_id.get(node.fingerprint),
                        second_by_id.get(node.fingerprint),
                    )
                    if item is not None
                )
                success = next(
                    (item for item in attempts if item.status == "success"), None
                )
                if success is not None:
                    outcomes.append(success)
                    passed = True
                elif first.valid and second is not None and second.valid:
                    outcomes.append(attempts[-1])
                else:
                    outcomes.append(
                        self._transfer_failure(
                            node.fingerprint,
                            f"{primary.name}+{alternate.name}",
                            "inconclusive",
                            "control_transfer",
                        )
                    )
            if (
                not first.valid
                and second is not None
                and not second.valid
                and not passed
            ):
                failure = self._control_failure("pre_control", second.opening)
                raise ProbeRunAbort(
                    failure.model_copy(
                        update={"transfer_targets": self._summarize_targets(windows)}
                    )
                )
        return tuple(outcomes), self._summarize_targets(windows)

    def _summarize_targets(
        self, windows: dict[str, list[TransferWindow]]
    ) -> tuple[TransferTargetEvidence, ...]:
        return tuple(
            TransferTargetEvidence(
                name=target.name,
                authority=target.authority,
                controls_attempted=sum(
                    1 + (window.closing is not None) for window in windows[target.name]
                ),
                controls_passed=sum(
                    (window.opening.status == "success")
                    + (
                        window.closing is not None
                        and window.closing.status == "success"
                    )
                    for window in windows[target.name]
                ),
                candidate_attempts=sum(
                    len(window.attempts) for window in windows[target.name]
                ),
            )
            for target in self.transfer_targets
        )

    async def _measure_window(
        self,
        controller: str,
        controller_client: httpx.AsyncClient,
        listener_port: int,
        nodes: tuple[ProbeableNode, ...],
        target: TransferTarget,
        byte_count: int,
    ) -> TransferWindow:
        await self._select(controller, controller_client, "DIRECT")
        async with self._transfer_transport(listener_port) as transfer_client:
            opening = await self._measure(
                "DIRECT", "DIRECT", transfer_client, target, byte_count
            )
        if opening.status != "success":
            return TransferWindow(target=target.name, opening=opening)
        attempts: list[TransferObservation] = []
        async with self._transfer_transport(listener_port) as transfer_client:
            for node in nodes:
                await self._select(controller, controller_client, node.display_name)
                attempts.append(
                    await self._measure(
                        node.display_name,
                        node.fingerprint,
                        transfer_client,
                        target,
                        byte_count,
                    )
                )
        await self._select(controller, controller_client, "DIRECT")
        async with self._transfer_transport(listener_port) as transfer_client:
            closing = await self._measure(
                "DIRECT", "DIRECT", transfer_client, target, byte_count
            )
        return TransferWindow(
            target=target.name,
            opening=opening,
            attempts=tuple(attempts),
            closing=closing,
        )

    def _transfer_client(self, listener_port: int) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            proxy=f"http://127.0.0.1:{listener_port}",
            timeout=self.transfer_timeout,
            follow_redirects=True,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            headers={"Accept-Encoding": "identity", "Connection": "close"},
        )

    @asynccontextmanager
    async def _transfer_transport(
        self, listener_port: int
    ) -> AsyncIterator[httpx.AsyncClient | None]:
        if self.chunk_source is not None:
            yield None
            return
        async with self._transfer_client(listener_port) as client:
            yield client

    async def _select(
        self,
        controller: str,
        client: httpx.AsyncClient,
        proxy_name: str,
    ) -> None:
        try:
            if self.selector_update is not None:
                await self.selector_update(controller, proxy_name)
            else:
                selector = quote(self.SELECTOR_NAME, safe="")
                response = await client.put(
                    f"{controller.rstrip('/')}/proxies/{selector}",
                    json={"name": proxy_name},
                    timeout=self.transfer_timeout,
                )
                response.raise_for_status()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ProbeRunAbort(
                self._run_failure("selector", "selector_update", type(error).__name__)
            ) from error

    async def _measure(
        self,
        proxy_name: str,
        fingerprint: str,
        client: httpx.AsyncClient | None,
        target: TransferTarget,
        byte_count: int,
    ) -> TransferObservation:
        started = time.perf_counter()
        received = 0
        try:
            async with asyncio.timeout(self.transfer_timeout):
                if self.chunk_source is not None:
                    source = self.chunk_source(proxy_name, target, self.chunk_size)
                elif client is not None:
                    source = self._http_chunks(client, target)
                else:
                    raise RuntimeError("HTTP transfer requires a client")
                async with aclosing(source) as chunks:
                    async for chunk in chunks:
                        received += min(len(chunk), byte_count - received)
                        if received >= byte_count:
                            break
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._transfer_failure(
                fingerprint,
                target.name,
                "timeout",
                "transfer_timeout",
                received,
                started,
            )
        except httpx.HTTPStatusError as error:
            return self._transfer_failure(
                fingerprint,
                target.name,
                "http_error",
                "transfer_http",
                received,
                started,
                f"HTTP {error.response.status_code}",
            )
        except httpx.HTTPError as error:
            return self._transfer_failure(
                fingerprint,
                target.name,
                "http_error",
                "transfer_http",
                received,
                started,
                type(error).__name__,
            )
        except Exception as error:
            return self._transfer_failure(
                fingerprint,
                target.name,
                "http_error",
                "transfer_error",
                received,
                started,
                type(error).__name__,
            )
        elapsed = max(time.perf_counter() - started, 1e-9)
        if received != byte_count:
            return self._transfer_failure(
                fingerprint,
                target.name,
                "short_read",
                "transfer_short_read",
                received,
                started,
            )
        return TransferObservation(
            fingerprint=fingerprint,
            target=target.name,
            status="success",
            bytes_received=received,
            elapsed_ms=max(1, round(elapsed * 1000)),
            bytes_per_second=received / elapsed,
        )

    async def _http_chunks(
        self,
        client: httpx.AsyncClient,
        target: TransferTarget,
    ) -> AsyncGenerator[bytes, None]:
        async with client.stream(
            "GET",
            str(target.url),
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                yield chunk

    @staticmethod
    def _delay_qualifies(evidence: ProbeEvidence, maximum: int) -> bool:
        if evidence.status != "success" or evidence.confirm is None:
            return False
        coarse = evidence.coarse.delay_ms
        confirm = evidence.confirm.delay_ms
        return (
            coarse is not None
            and confirm is not None
            and max(coarse, confirm) <= maximum
        )

    @classmethod
    def _control_failure(
        cls,
        phase: Literal["pre_control", "post_control"],
        observation: TransferObservation,
    ) -> ProbeRunFailure:
        diagnostic = observation.diagnostic
        cause = diagnostic.code if diagnostic is not None else observation.status
        detail = f"{observation.target}/{cause}"
        if diagnostic is not None and diagnostic.detail:
            detail = f"{detail}/{diagnostic.detail}"
        return cls._run_failure(phase, "control_transfer", detail)

    @staticmethod
    def _transfer_failure(
        fingerprint: str,
        target: str,
        status: TransferStatus,
        code: ProbeFailureCode,
        received: int = 0,
        started: float | None = None,
        detail: str = "",
    ) -> TransferObservation:
        elapsed = time.perf_counter() - started if started is not None else 0.0
        return TransferObservation(
            fingerprint=fingerprint,
            target=target,
            status=status,
            bytes_received=received,
            elapsed_ms=max(0, round(elapsed * 1000)),
            diagnostic=ProbeDiagnostic(code=code, detail=detail[:200]),
        )

    @staticmethod
    def _run_failure(
        phase: ProbeRunPhase,
        code: ProbeFailureCode,
        detail: str = "",
    ) -> ProbeRunFailure:
        return ProbeRunFailure(
            phase=phase,
            diagnostic=ProbeDiagnostic(code=code, detail=detail[:200]),
        )

    def _start_process(
        self, config_path: Path, home_dir: Path
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
    def _write_config(
        config_path: Path,
        controller_port: int,
        nodes: Sequence[ProbeableNode],
        *,
        listener_port: int | None = None,
    ) -> None:
        proxies = [
            node.proxy.mihomo_payload() | {"name": node.display_name} for node in nodes
        ]
        config: dict[str, Any] = {
            "allow-lan": False,
            "ipv6": False,
            "log-level": "silent",
            "external-controller": f"127.0.0.1:{controller_port}",
            "proxies": proxies,
        }
        if listener_port is not None:
            config["proxy-groups"] = [
                {
                    "name": MihomoProbeSession.SELECTOR_NAME,
                    "type": "select",
                    "proxies": ["DIRECT", *(node.display_name for node in nodes)],
                }
            ]
            config["listeners"] = [
                {
                    "name": "quality-http",
                    "type": "http",
                    "port": listener_port,
                    "listen": "127.0.0.1",
                    "proxy": MihomoProbeSession.SELECTOR_NAME,
                }
            ]
        config_path.write_text(
            yaml.safe_dump(
                config,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

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
        failures: list[ProbeEvidence] = []
        invocations = 0
        complete = True
        while pending:
            batch = pending.pop()
            if invocations >= self.max_validation_invocations:
                complete = False
                break
            invocations += 1
            self._write_config(config_path, controller_port, batch)
            try:
                self.validate_config(config_path, check_home)
            except Exception:
                if len(batch) == 1:
                    failures.extend(
                        ProbeEvidence.process_failure(node, "config_rejected")
                        for node in batch
                    )
                    continue
                midpoint = len(batch) // 2
                pending.extend((batch[midpoint:], batch[:midpoint]))
                continue
            accepted.extend(batch)

        position = {node.fingerprint: index for index, node in enumerate(nodes)}
        accepted.sort(key=lambda node: position[node.fingerprint])
        failures.sort(key=lambda item: position[item.fingerprint])
        return ValidatedProbeBatch(
            nodes=tuple(accepted),
            failures=tuple(failures),
            complete=complete,
        )

    async def _await_ready(
        self,
        process: subprocess.Popen[str],
        controller: str,
        request_json: JsonRequest,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if process.poll() is not None:
                raise MihomoValidationError(
                    "Mihomo probe process exited before readiness"
                )
            try:
                value = await request_json(f"{controller}/version", 1.0)
                if value.get("version"):
                    return
            except Exception as error:
                last_error = error
            await asyncio.sleep(0.05)
        raise MihomoValidationError(
            "Mihomo probe controller did not become ready"
        ) from last_error

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
