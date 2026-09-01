"""Behavioral contract for bounded Mihomo delay probing."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
import pytest
import yaml

from src.nodes import ClashNode, NodeProvenance, admit_proxy
from src.probe import (
    MihomoDelayProbe,
    MihomoProbeSession,
    TransferTarget,
)
from src.quality import (
    ProbeCandidate,
    ProbePlan,
    ProbePlanEntry,
    QualityPolicy,
)


def candidate(index: int, name: str | None = None) -> ProbeCandidate:
    return ProbeCandidate(
        fingerprint=f"fingerprint-{index}", proxy_name=name or f"node-{index}"
    )


def test_transfer_targets_admit_https_and_reject_ambiguous_authority(tmp_path):
    primary = TransferTarget(
        name="primary",
        authority="cloudflare",
        url="https://speed.example/down?bytes=4",
    )
    fallback = TransferTarget(
        name="fallback",
        authority="hetzner",
        url="https://fallback.example/100MB.bin",
    )

    session = MihomoProbeSession(
        tmp_path / "mihomo", transfer_targets=(primary, fallback)
    )

    assert session.transfer_targets == (primary, fallback)
    with pytest.raises(ValueError, match="distinct authorities"):
        MihomoProbeSession(
            tmp_path / "mihomo",
            transfer_targets=(primary, primary.model_copy(update={"name": "other"})),
        )
    with pytest.raises(ValueError, match="HTTPS"):
        TransferTarget(
            name="public-http",
            authority="public",
            url="http://public.example/test.bin",
        )


async def test_probe_requires_two_successful_endpoints_and_encodes_proxy_name():
    calls: list[tuple[str, float]] = []

    async def request_json(url: str, timeout: float):
        calls.append((url, timeout))
        return {"delay": 123 if "gstatic" in url else 234}

    probe = MihomoDelayProbe(request_json=request_json, timeout_ms=2500)
    evidence = await probe.probe_controller(
        "http://127.0.0.1:9090",
        [candidate(1, "HK / premium?#")],
    )

    assert len(evidence) == 1
    assert evidence[0].status == "success"
    assert evidence[0].coarse.delay_ms == 123
    assert evidence[0].confirm is not None
    assert evidence[0].confirm.delay_ms == 234
    assert len(calls) == 2
    for url, timeout in calls:
        parsed = urlsplit(url)
        assert (
            unquote(parsed.path.removeprefix("/proxies/").removesuffix("/delay"))
            == "HK / premium?#"
        )
        assert parse_qs(parsed.query)["timeout"] == ["2500"]
        assert parse_qs(parsed.query)["expected"] == ["204"]
        assert timeout == pytest.approx(3.5)


async def test_confirm_is_only_requested_after_coarse_success_and_errors_are_explicit():
    calls: dict[str, int] = {}

    async def request_json(url: str, timeout: float):
        name = unquote(urlsplit(url).path.split("/")[2])
        calls[name] = calls.get(name, 0) + 1
        if name == "timeout":
            raise asyncio.TimeoutError
        if name == "malformed":
            return {"unexpected": True}
        if name == "api-error":
            raise RuntimeError("controller rejected request")
        return {"delay": 40}

    probe = MihomoDelayProbe(request_json=request_json)
    evidence = await probe.probe_controller(
        "http://127.0.0.1:9090",
        [
            candidate(1, "ok"),
            candidate(2, "timeout"),
            candidate(3, "malformed"),
            candidate(4, "api-error"),
        ],
    )

    assert [item.status for item in evidence] == [
        "success",
        "timeout",
        "api_error",
        "api_error",
    ]
    assert calls == {"ok": 2, "timeout": 1, "malformed": 1, "api-error": 1}
    assert all(item.confirm is None for item in evidence[1:])
    assert [item.coarse.diagnostic.code for item in evidence[1:]] == [
        "request_timeout",
        "controller_payload",
        "controller_error",
    ]


async def test_probe_enforces_concurrency_without_retrying_requests():
    active = 0
    maximum = 0
    calls = 0

    async def request_json(url: str, timeout: float):
        nonlocal active, maximum, calls
        calls += 1
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"delay": 10}

    candidates = [candidate(index) for index in range(8)]
    probe = MihomoDelayProbe(request_json=request_json, concurrency=2)
    evidence = await probe.probe_controller("http://127.0.0.1:9090", candidates)

    assert maximum == 2
    assert calls == 16
    assert all(item.status == "success" for item in evidence)


async def test_whole_run_deadline_cancels_outstanding_requests():
    started = asyncio.Event()

    async def request_json(url: str, timeout: float):
        started.set()
        await asyncio.Event().wait()

    probe = MihomoDelayProbe(request_json=request_json, deadline=0.02)
    evidence = await probe.probe_controller(
        "http://127.0.0.1:9090",
        [candidate(1), candidate(2)],
    )

    assert started.is_set()
    assert [item.status for item in evidence] == ["cancelled", "cancelled"]


async def test_probe_rejects_more_than_the_candidate_budget_without_network_calls():
    called = False

    async def request_json(url: str, timeout: float):
        nonlocal called
        called = True
        return {"delay": 1}

    probe = MihomoDelayProbe(request_json=request_json, max_candidates=2)

    with pytest.raises(ValueError, match="candidate budget"):
        await probe.probe_controller(
            "http://127.0.0.1:9090",
            [candidate(1), candidate(2), candidate(3)],
        )

    assert called is False


async def test_loopback_controller_requests_ignore_environment_proxies(monkeypatch):
    options = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"version": "test"}

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return Response()

    monkeypatch.setattr("src.probe.httpx.AsyncClient", Client)

    value = await MihomoDelayProbe().request_json("http://127.0.0.1:9090/version", 1.0)

    assert value["version"] == "test"
    assert options["trust_env"] is False


async def test_delay_session_reuses_one_controller_client(monkeypatch):
    clients = 0
    calls = 0

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"delay": 20}

    class Client:
        def __init__(self, **kwargs):
            nonlocal clients
            clients += 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            nonlocal calls
            calls += 1
            return Response()

    monkeypatch.setattr("src.probe.httpx.AsyncClient", Client)

    evidence = await MihomoDelayProbe().probe_controller(
        "http://127.0.0.1:9090",
        (candidate(1), candidate(2)),
    )

    assert all(item.status == "success" for item in evidence)
    assert calls == 4
    assert clients == 1


def node(index: int, name: str) -> ClashNode:
    return ClashNode(
        fingerprint=f"fingerprint-{index}",
        display_name=name,
        proxy=admit_proxy(
            {
                "name": "source name",
                "type": "ss",
                "server": "127.0.0.1",
                "port": 10000 + index,
                "cipher": "aes-128-gcm",
                "password": "secret",
            }
        ),
        provenance=(
            NodeProvenance(
                site="source-a",
                source_url="https://example.test/nodes",
                observed_at=datetime(2026, 1, 1, tzinfo=UTC),
                artifact_digest="a" * 64,
                item_index=index,
            ),
        ),
    )


def probe_plan(*nodes: ClashNode) -> ProbePlan:
    return ProbePlan(
        entries=tuple(
            ProbePlanEntry(
                ordinal=index,
                node=item,
                sources=tuple(source.site for source in item.provenance)
                or ("unknown",),
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


def policy(*, delay: int = 2500, byte_count: int = 4) -> QualityPolicy:
    return QualityPolicy(max_delay_ms=delay, transfer_bytes=byte_count)


async def request_success(url: str, timeout: float):
    return {"version": "test"} if url.endswith("/version") else {"delay": 20}


async def test_session_reaps_delay_process_before_serial_transfers(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    processes: list[FakeProcess] = []
    configs: list[dict[str, object]] = []
    events: list[tuple[str, str]] = []
    active_processes = 0
    maximum_processes = 0
    active = 0
    maximum = 0

    def process_factory(command, **kwargs):
        nonlocal active_processes, maximum_processes
        assert not processes or processes[-1].returncode == 0
        config_path = Path(command[command.index("-f") + 1])
        configs.append(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        process = FakeProcess()
        active_processes += 1
        maximum_processes = max(maximum_processes, active_processes)

        def terminate():
            nonlocal active_processes
            process.returncode = 0
            active_processes -= 1

        process.terminate = terminate
        processes.append(process)
        return process

    async def select_proxy(controller: str, proxy_name: str):
        events.append(("select", proxy_name))

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        events.append(("transfer", proxy_name))
        try:
            yield b"xx"
            await asyncio.sleep(0)
            yield b"xx"
        finally:
            active -= 1

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=process_factory,
        validate_config=lambda config, home: None,
        selector_update=select_proxy,
        chunk_source=chunks,
    )

    result = await session.probe(
        probe_plan(node(1, "one"), node(2, "two")),
        policy(),
    )

    assert result.status == "success"
    assert [item.transfer.status for item in result.evidence] == [
        "success",
        "success",
    ]
    assert events == [
        ("select", "DIRECT"),
        ("transfer", "DIRECT"),
        ("select", "one"),
        ("transfer", "one"),
        ("select", "two"),
        ("transfer", "two"),
        ("select", "DIRECT"),
        ("transfer", "DIRECT"),
    ]
    assert maximum == 1
    assert maximum_processes == 1
    assert len(processes) == 2
    assert all(process.returncode == 0 for process in processes)
    assert "proxy-groups" not in configs[0]
    assert "listeners" not in configs[0]
    assert configs[1]["proxy-groups"] == [
        {
            "name": "QUALITY-TEST",
            "type": "select",
            "proxies": ["DIRECT", "one", "two"],
        }
    ]
    assert configs[1]["listeners"] == [
        {
            "name": "quality-http",
            "type": "http",
            "port": configs[1]["listeners"][0]["port"],
            "listen": "127.0.0.1",
            "proxy": "QUALITY-TEST",
        }
    ]


async def test_process_start_failure_is_one_inconclusive_run(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    def fail_start(command, **kwargs):
        raise OSError("cannot start")

    session = MihomoProbeSession(
        executable,
        process_factory=fail_start,
        validate_config=lambda config, home: None,
    )

    result = await session.probe(
        probe_plan(node(1, "one"), node(2, "two")),
        policy(),
    )

    assert result.status == "inconclusive"
    assert result.phase == "process"
    assert result.diagnostic.code == "process_start"
    assert result.diagnostic.detail == "delay"


async def test_transfer_process_start_failure_discards_delay_evidence(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    delay_process = FakeProcess()
    starts = 0

    def process_factory(command, **kwargs):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise OSError("cannot start transfer process")
        return delay_process

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=process_factory,
        validate_config=lambda config, home: None,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "inconclusive"
    assert result.phase == "process"
    assert result.diagnostic.code == "process_start"
    assert result.diagnostic.detail == "transfer"
    assert delay_process.returncode == 0


async def test_session_isolates_config_rejection_as_node_evidence(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    def validate_config(config_path, home):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if any(proxy["name"] == "invalid" for proxy in config["proxies"]):
            raise ValueError("configuration rejected")

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        yield b"xxxx"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=validate_config,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(
        probe_plan(node(1, "valid"), node(2, "invalid")),
        policy(),
    )

    assert result.status == "success"
    assert [item.status for item in result.evidence] == ["success", "process_error"]
    assert result.evidence[1].coarse.diagnostic.code == "config_rejected"


async def test_validation_budget_exhaustion_is_inconclusive(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    process_started = False

    def process_factory(command, **kwargs):
        nonlocal process_started
        process_started = True
        return FakeProcess()

    def reject_config(config, home):
        raise ValueError("rejected")

    session = MihomoProbeSession(
        executable,
        process_factory=process_factory,
        validate_config=reject_config,
        max_validation_invocations=2,
    )

    result = await session.probe(
        probe_plan(*(node(index, f"node-{index}") for index in range(4))),
        policy(),
    )

    assert result.status == "inconclusive"
    assert result.phase == "validation"
    assert result.diagnostic.code == "validation_budget"
    assert process_started is False


async def test_slow_delay_skips_transfer_measurement(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    transfers: list[str] = []
    processes: list[FakeProcess] = []

    async def request_json(url: str, timeout: float):
        return {"version": "test"} if url.endswith("/version") else {"delay": 2501}

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        transfers.append(proxy_name)
        yield b"xxxx"

    def process_factory(command, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_json),
        process_factory=process_factory,
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "slow")), policy())

    assert result.status == "success"
    assert result.evidence[0].status == "success"
    assert result.evidence[0].transfer is None
    assert transfers == []
    assert len(processes) == 1
    assert processes[0].returncode == 0


async def test_node_transfer_failures_remain_attributable_when_controls_pass(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        if proxy_name == "http":
            request = httpx.Request("GET", "https://speed.example")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError(
                "unavailable", request=request, response=response
            )
        if proxy_name == "timeout":
            await asyncio.Event().wait()
        if proxy_name == "short":
            yield b"x"
            return
        yield b"xxxx"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
        transfer_timeout=0.02,
    )

    result = await session.probe(
        probe_plan(
            node(1, "http"),
            node(2, "timeout"),
            node(3, "short"),
            node(4, "good"),
        ),
        policy(),
    )

    assert result.status == "success"
    assert [item.transfer.status for item in result.evidence] == [
        "http_error",
        "timeout",
        "short_read",
        "success",
    ]
    assert [
        item.transfer.diagnostic.code if item.transfer.diagnostic else None
        for item in result.evidence
    ] == [
        "transfer_http",
        "transfer_timeout",
        "transfer_short_read",
        None,
    ]


async def test_closing_control_uses_fresh_transport_after_candidate_failure(
    monkeypatch,
    tmp_path,
):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    selected = "DIRECT"

    class Response:
        def __init__(self, client):
            self.client = client

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self, chunk_size):
            if selected != "DIRECT":
                self.client.candidate_failed = True
                yield b"x"
            elif self.client.candidate_failed:
                yield b"x"
            else:
                yield b"xxxx"

    class Client:
        def __init__(self, **options):
            self.candidate_failed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, method, url, **options):
            return Response(self)

    async def select_proxy(controller: str, proxy_name: str):
        nonlocal selected
        selected = proxy_name

    monkeypatch.setattr("src.probe.httpx.AsyncClient", Client)
    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=select_proxy,
    )

    result = await session.probe(probe_plan(node(1, "failed")), policy())

    assert result.status == "success"
    assert result.evidence[0].transfer.status == "short_read"


async def test_pre_control_failure_stops_before_node_measurement(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    transfers: list[str] = []

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        transfers.append(proxy_name)
        yield b"x"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "inconclusive"
    assert result.phase == "pre_control"
    assert result.diagnostic.code == "control_transfer"
    assert [target.controls_attempted for target in result.transfer_targets] == [1, 1]
    assert [target.controls_passed for target in result.transfer_targets] == [0, 0]
    assert transfers == ["DIRECT", "DIRECT"]


async def test_control_http_failure_reports_the_upstream_status(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        request = httpx.Request("GET", "https://speed.example")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError(
            "rate limited",
            request=request,
            response=response,
        )
        yield b"unreachable"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "inconclusive"
    assert result.phase == "pre_control"
    assert result.diagnostic.code == "control_transfer"
    assert result.diagnostic.detail == "hetzner-hel1/transfer_http/HTTP 429"


async def test_failed_control_uses_alternate_target_with_fresh_transport(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    direct_calls = 0

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        nonlocal direct_calls
        if proxy_name == "DIRECT":
            direct_calls += 1
            if direct_calls == 1:
                request = httpx.Request("GET", "https://speed.example")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError(
                    "temporarily unavailable",
                    request=request,
                    response=response,
                )
        yield b"xxxx"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "success"
    assert result.evidence[0].transfer.status == "success"
    assert [target.model_dump() for target in result.transfer_targets] == [
        {
            "name": "cloudflare",
            "authority": "cloudflare",
            "controls_attempted": 1,
            "controls_passed": 0,
            "candidate_attempts": 0,
        },
        {
            "name": "hetzner-hel1",
            "authority": "hetzner",
            "controls_attempted": 2,
            "controls_passed": 2,
            "candidate_attempts": 1,
        },
    ]
    assert direct_calls == 3


async def test_post_control_failure_preserves_successful_node_outcome(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    direct_calls = 0

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        nonlocal direct_calls
        if proxy_name == "DIRECT":
            direct_calls += 1
            yield b"xxxx" if direct_calls == 1 else b"x"
            return
        yield b"xxxx"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "success"
    assert result.evidence[0].transfer.status == "success"


async def test_failed_primary_target_recovers_on_alternate_target(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    attempts: list[tuple[str, str]] = []

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        attempts.append((proxy_name, target.name))
        if proxy_name != "DIRECT" and target.name == "cloudflare":
            yield b"x"
            return
        yield b"xxxx"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "success"
    assert result.evidence[0].transfer.status == "success"
    assert result.evidence[0].transfer.target == "hetzner-hel1"
    assert ("one", "cloudflare") in attempts
    assert ("one", "hetzner-hel1") in attempts
    assert [target.candidate_attempts for target in result.transfer_targets] == [1, 1]
    assert [target.controls_passed for target in result.transfer_targets] == [2, 2]


async def test_selector_failure_is_inconclusive(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    async def reject_selector(controller: str, proxy_name: str):
        raise httpx.ConnectError("controller unavailable")

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        yield b"xxxx"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=lambda command, **kwargs: FakeProcess(),
        validate_config=lambda config, home: None,
        selector_update=reject_selector,
        chunk_source=chunks,
    )

    result = await session.probe(probe_plan(node(1, "one")), policy())

    assert result.status == "inconclusive"
    assert result.phase == "selector"
    assert result.diagnostic.code == "selector_update"


async def test_session_cancellation_reaps_every_started_process(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    transfer_started = asyncio.Event()
    processes: list[FakeProcess] = []

    def process_factory(command, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    async def chunks(proxy_name: str, target: TransferTarget, chunk_size: int):
        transfer_started.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_success),
        process_factory=process_factory,
        validate_config=lambda config, home: None,
        selector_update=lambda controller, name: asyncio.sleep(0),
        chunk_source=chunks,
        transfer_timeout=30,
    )
    task = asyncio.create_task(session.probe(probe_plan(node(1, "one")), policy()))
    await transfer_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(processes) == 2
    assert all(process.returncode == 0 for process in processes)
