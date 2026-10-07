import asyncio
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
import yaml

from freenodes.capability import (
    CapabilityPolicy,
    CapabilityTarget,
    ProbeCandidate,
    ProbePlan,
    ProbePlanEntry,
    rank_measured_capable,
)
from freenodes.nodes import AdmittedCatalog, ClashNode, NodeProvenance
from freenodes.probe import MihomoDelayProbe, MihomoProbeSession
from freenodes.proxies import admit_proxy

TARGETS = CapabilityTarget.admit_registry(
    (
        CapabilityTarget(
            id="github", url="https://raw.example.test/project", expected_status=200
        ),
        CapabilityTarget(
            id="google", url="https://google.example.test/204", expected_status=204
        ),
        CapabilityTarget(
            id="cloudflare",
            url="https://cloudflare.example.test/204",
            expected_status=204,
        ),
    ),
    quorum=2,
)


def node(index: int, name: str) -> ClashNode:
    return ClashNode(
        fingerprint=f"fingerprint-{index}",
        display_name=name,
        proxy=admit_proxy(
            {
                "name": "source name",
                "type": "ss",
                "server": f"node-{index}.example",
                "port": 10000 + index,
                "cipher": "aes-128-gcm",
                "password": "secret",
            }
        ),
        provenance=(
            NodeProvenance(
                authority="source-a",
                site="source-a",
                source_url="https://example.test/nodes",
                observed_at=datetime(2026, 1, 1, tzinfo=UTC),
                artifact_digest="a" * 64,
                item_index=index,
            ),
        ),
    )


def plan(*nodes: ClashNode) -> ProbePlan:
    return ProbePlan(
        entries=tuple(
            ProbePlanEntry(
                ordinal=index,
                node=item,
                sources=("source-a",),
                protocol=item.proxy.type,
            )
            for index, item in enumerate(nodes)
        )
    )


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


async def test_observation_encodes_identity_status_and_expected_response():
    calls: list[tuple[str, float]] = []

    async def request(url: str, timeout: float):
        calls.append((url, timeout))
        return {"delay": 23}

    observed = await MihomoDelayProbe(timeout_ms=4000).observe(
        "http://127.0.0.1:9090",
        ProbeCandidate(fingerprint="one", proxy_name="HK / premium?#"),
        "https://target.test/204",
        request,
        204,
    )

    parsed = urlsplit(calls[0][0])
    assert observed.status == "success" and observed.delay_ms == 23
    assert unquote(parsed.path.split("/")[2]) == "HK / premium?#"
    assert parse_qs(parsed.query) == {
        "url": ["https://target.test/204"],
        "timeout": ["4000"],
        "expected": ["204"],
    }
    assert calls[0][1] == 5


async def test_wave_deadline_cancels_fixed_workers():
    started = 0

    async def request(url: str, timeout: float):
        nonlocal started
        started += 1
        await asyncio.Event().wait()

    probe = MihomoDelayProbe(concurrency=2)
    candidates = tuple(
        ProbeCandidate(fingerprint=str(index), proxy_name=str(index))
        for index in range(5)
    )
    evidence = await probe.probe_wave(
        "http://127.0.0.1:9090",
        candidates,
        "https://target.test",
        request,
        asyncio.get_running_loop().time() + 0.01,
        204,
    )

    assert started == 2
    assert {item.status for item in evidence.values()} == {"cancelled"}


async def test_session_adapts_to_yield_with_bounded_work_and_one_process(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    processes: list[FakeProcess] = []
    active = maximum = direct_calls = candidate_calls = 0

    async def request(url: str, timeout: float):
        nonlocal active, maximum, direct_calls, candidate_calls
        if url.endswith("/version"):
            return {"version": "test"}
        proxy = unquote(urlsplit(url).path.split("/")[2])
        if proxy.startswith("CONTROL / "):
            direct_calls += 1
            return {"delay": 5}
        candidate_calls += 1
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0)
            if int(proxy.removeprefix("node-")) % 3:
                raise RuntimeError("fixture rejection")
            return {"delay": 20}
        finally:
            active -= 1

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request),
        process_factory=lambda command, **kwargs: (
            processes.append(FakeProcess()) or processes[-1]
        ),
        validate_config=lambda config, home: None,
    )
    nodes = tuple(node(index, f"node-{index}") for index in range(385))

    receipt = await session.probe_capabilities(
        plan(*nodes), TARGETS, CapabilityPolicy(max_published=100)
    )

    assert receipt.status == "complete"
    assert receipt.planned == 385
    assert receipt.attempted == 384
    assert receipt.termination == "target_reached"
    assert (
        receipt.accepted_fingerprints
        == tuple(
            item.fingerprint for item in nodes if int(item.display_name[5:]) % 3 == 0
        )[:100]
    )
    assert candidate_calls + direct_calls <= (
        3 * receipt.attempted + 6 * ((receipt.attempted + 127) // 128)
    )
    assert direct_calls == 18
    assert maximum <= 64
    assert len(processes) == 1 and processes[0].returncode == 0


@pytest.mark.parametrize("ranking, expected_attempts", ((False, 512), (True, 1012)))
async def test_ranking_pilot_measures_beyond_500_without_exceeding_pool(
    tmp_path, ranking, expected_attempts
):
    calls = 0
    process = FakeProcess()

    async def request(url, timeout):
        nonlocal calls
        if url.endswith("/version"):
            return {"version": "test"}
        proxy = unquote(urlsplit(url).path.split("/")[2])
        calls += not proxy.startswith("CONTROL / ")
        return {"delay": 20}

    session = MihomoProbeSession(
        tmp_path / "mihomo.exe",
        delay_probe=MihomoDelayProbe(request_json=request),
        process_factory=lambda *args, **kwargs: process,
        validate_config=lambda *args: None,
    )
    receipt = await session.probe_capabilities(
        plan(*(node(index, f"node-{index}") for index in range(1500))),
        TARGETS,
        CapabilityPolicy(rank_latency=ranking),
    )

    assert receipt.attempted == expected_attempts
    assert len(receipt.accepted_fingerprints) == 500
    assert receipt.planned == 1500
    assert receipt.termination == "target_reached"
    assert calls == 2 * expected_attempts
    assert process.returncode == 0


@pytest.mark.parametrize(
    "count, expected, termination",
    ((2500, 2036, "target_reached"), (1600, 1600, "candidates_exhausted")),
)
async def test_ranking_reaches_coverage_before_bounded_extension(
    tmp_path, count, expected, termination
):
    process = FakeProcess()

    async def request(url, timeout):
        if url.endswith("/version"):
            return {"version": "test"}
        proxy = unquote(urlsplit(url).path.split("/")[2])
        if not proxy.startswith("CONTROL / ") and int(proxy.removeprefix("node-")) % 3:
            raise RuntimeError("candidate failed")
        return {"delay": 20}

    session = MihomoProbeSession(
        tmp_path / "mihomo.exe",
        delay_probe=MihomoDelayProbe(request_json=request),
        process_factory=lambda *args, **kwargs: process,
        validate_config=lambda *args: None,
    )
    receipt = await session.probe_capabilities(
        plan(*(node(index, f"node-{index}") for index in range(count))),
        TARGETS,
        CapabilityPolicy(rank_latency=True),
    )
    assert receipt.planned == count
    assert receipt.attempted == expected
    assert len(receipt.accepted_fingerprints) == 500
    assert receipt.termination == termination
    assert process.returncode == 0


@pytest.mark.parametrize("ranking", (False, True))
async def test_invalid_controls_stop_candidates_and_preserve_diagnosis(
    tmp_path, ranking
):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    candidate_called = False
    process = FakeProcess()

    async def request(url: str, timeout: float):
        nonlocal candidate_called
        if url.endswith("/version"):
            return {"version": "test"}
        proxy = unquote(urlsplit(url).path.split("/")[2])
        target = parse_qs(urlsplit(url).query)["url"][0]
        candidate_called |= not proxy.startswith("CONTROL / ")
        if "google" in target or "cloudflare" in target:
            raise TimeoutError
        return {"delay": 5}

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request),
        process_factory=lambda command, **kwargs: process,
        validate_config=lambda config, home: None,
    )

    receipt = await session.probe_capabilities(
        plan(node(1, "one")), TARGETS, CapabilityPolicy(rank_latency=ranking)
    )

    assert receipt.status == "inconclusive"
    assert receipt.diagnostic.code == "control_unavailable"
    assert candidate_called is False and process.returncode == 0


@pytest.mark.parametrize("ranking", (False, True))
async def test_deadline_and_cancellation_reap_process_and_workers(tmp_path, ranking):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    async def exercise(cancel: bool):
        process = FakeProcess()
        started = asyncio.Event()

        async def request(url: str, timeout: float):
            if url.endswith("/version"):
                return {"version": "test"}
            proxy = unquote(urlsplit(url).path.split("/")[2])
            if proxy.startswith("CONTROL / "):
                return {"delay": 5}
            if cancel or proxy == "node-128":
                started.set()
                await asyncio.Event().wait()
            return {"delay": 5}

        session = MihomoProbeSession(
            executable,
            delay_probe=MihomoDelayProbe(
                request_json=request,
                deadline=30 if cancel else 0.2,
            ),
            process_factory=lambda command, **kwargs: process,
            validate_config=lambda config, home: None,
        )
        task = asyncio.create_task(
            session.probe_capabilities(
                plan(
                    *(node(index, f"node-{index}") for index in range(129))
                    if not cancel
                    else (node(1, "one"),)
                ),
                TARGETS,
                CapabilityPolicy(rank_latency=ranking),
            )
        )
        await started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            receipt = await task
            assert receipt.status == "complete"
            assert receipt.planned == 129
            assert receipt.attempted == 128
            assert receipt.termination == "time_budget"
            assert receipt.deadline_reached is True
        assert process.returncode == 0

    await exercise(cancel=False)
    await exercise(cancel=True)


async def test_config_rejection_isolated_without_starting_extra_processes(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    def validate(config_path: Path, home: Path):
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if any(proxy["name"] == "bad" for proxy in content["proxies"]):
            raise ValueError("rejected")

    async def request(url: str, timeout: float):
        return {"version": "test"} if url.endswith("/version") else {"delay": 5}

    process = FakeProcess()
    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request),
        process_factory=lambda command, **kwargs: process,
        validate_config=validate,
    )

    receipt = await session.probe_capabilities(
        plan(node(1, "good"), node(2, "bad")), TARGETS, CapabilityPolicy()
    )

    assert receipt.planned == receipt.attempted == 2
    assert receipt.termination == "candidates_exhausted"
    assert receipt.accepted_fingerprints == ("fingerprint-1",)
    assert [item.status for item in receipt.decisions] == ["capable", "failed"]
    assert receipt.decisions[1].reason == "config_rejected"
    assert process.returncode == 0


NOW = datetime(2026, 9, 2, tzinfo=UTC)


async def test_real_process_proves_capability_and_releases_local_resources(
    real_mihomo_path,
):
    async def respond(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            await asyncio.sleep(0.01)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    proxy = await asyncio.start_server(respond, "0.0.0.0", 0)
    proxy_port = proxy.sockets[0].getsockname()[1]
    targets = tuple(
        CapabilityTarget(id=id_, url=url, expected_status=status)
        for id_, url, status in (
            ("github", f"http://127.0.0.2:{proxy_port}/github", 200),
            ("google", f"http://127.0.0.1:{proxy_port}/google", 200),
            ("cloudflare", f"http://127.0.0.3:{proxy_port}/cloudflare", 200),
        )
    )
    node = ClashNode(
        fingerprint=f"{1:064x}",
        display_name="FIXTURE",
        proxy=admit_proxy({"name": "FIXTURE", "type": "direct"}),
        provenance=(
            NodeProvenance(
                authority="fixture",
                site="fixture",
                source_url="https://fixture.test/nodes",
                observed_at=NOW,
                artifact_digest=f"{11:064x}",
                item_index=1,
            ),
        ),
    )
    plan = ProbePlan(
        entries=(
            ProbePlanEntry(
                ordinal=0,
                node=node,
                sources=("fixture",),
                protocol=node.proxy.type,
            ),
        )
    )

    try:
        result = await MihomoProbeSession(real_mihomo_path).probe_capabilities(
            plan, targets, CapabilityPolicy()
        )
    finally:
        proxy.close()
        await proxy.wait_closed()

    assert result.status == "complete", result.model_dump()
    assert result.accepted_fingerprints == (node.fingerprint,), result.model_dump()
    assert result.decisions[0].status == "capable"


async def test_real_mihomo_latency_ranking_selects_faster_loopback_proxy(
    real_mihomo_path,
):
    async def respond(reader, writer, *, delay):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            if header.startswith(b"CONNECT "):
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                await reader.readuntil(b"\r\n\r\n")
            await asyncio.sleep(delay)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    servers = []
    try:
        for delay in (0.01, 0.2, 0.01):
            servers.append(
                await asyncio.start_server(
                    partial(respond, delay=delay),
                    "0.0.0.0" if not servers else "127.0.0.1",
                    0,
                )
            )
        target_port = servers[0].sockets[0].getsockname()[1]
        targets = tuple(
            CapabilityTarget(
                id=f"target-{index}",
                url=f"http://127.0.0.{index}:{target_port}/",
                expected_status=200,
            )
            for index in range(1, 4)
        )
        nodes = tuple(
            node(index, f"node-{index}").model_copy(
                update={
                    "proxy": admit_proxy(
                        {
                            "name": f"node-{index}",
                            "type": "http",
                            "server": "127.0.0.1",
                            "port": server.sockets[0].getsockname()[1],
                        }
                    ),
                }
            )
            for index, server in enumerate(servers[1:], start=1)
        )
        policy = CapabilityPolicy(max_published=1, rank_latency=True)
        started = perf_counter()
        measured = await MihomoProbeSession(real_mihomo_path).probe_capabilities(
            plan(*nodes), targets, policy
        )
        selected, report = rank_measured_capable(
            AdmittedCatalog(nodes=nodes),
            measured,
            policy,
            elapsed_seconds=perf_counter() - started,
        )
    finally:
        for server in servers:
            server.close()
            await server.wait_closed()

    assert measured.status == "complete"
    assert measured.planned == measured.attempted == 2
    assert measured.accepted_fingerprints == (nodes[0].fingerprint,), (
        measured.model_dump()
    )
    assert selected.accepted_fingerprints == (nodes[1].fingerprint,)
    assert report.ranking_pool_complete is True
    assert report.selected_median_delay_ms < report.baseline_median_delay_ms
