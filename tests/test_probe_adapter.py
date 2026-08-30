"""Behavioral contract for bounded Mihomo delay probing."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
import yaml

from src.mihomo import MihomoDelayProbe, MihomoProbeSession, ProbeCandidate
from src.nodes import ClashNode, NodeProvenance, admit_proxy


def candidate(index: int, name: str | None = None) -> ProbeCandidate:
    return ProbeCandidate(
        fingerprint=f"fingerprint-{index}", proxy_name=name or f"node-{index}"
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

    monkeypatch.setattr("src.mihomo.httpx.AsyncClient", Client)

    value = await MihomoDelayProbe().request_json("http://127.0.0.1:9090/version", 1.0)

    assert value["version"] == "test"
    assert options["trust_env"] is False


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


async def test_session_runs_one_isolated_loopback_process_and_terminates_it(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    captured: dict[str, object] = {}

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            captured["terminated"] = True
            self.returncode = 0

        def wait(self, timeout=None):
            captured["wait_timeout"] = timeout
            return self.returncode

        def kill(self):
            captured["killed"] = True
            self.returncode = -9

    def process_factory(command, **kwargs):
        captured["command"] = command
        config_path = Path(command[command.index("-f") + 1])
        captured["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return Process()

    async def request_json(url: str, timeout: float):
        if url.endswith("/version"):
            return {"version": "test", "meta": True}
        return {"delay": 20}

    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(request_json=request_json),
        process_factory=process_factory,
        validate_config=lambda config, home: None,
    )
    evidence = await session.probe([node(1, "one / exact"), node(2, "two")])

    assert [item.status for item in evidence] == ["success", "success"]
    assert captured["terminated"] is True
    config = captured["config"]
    assert config["external-controller"].startswith("127.0.0.1:")
    assert config["allow-lan"] is False
    assert "tun" not in config
    assert "mixed-port" not in config
    assert [proxy["name"] for proxy in config["proxies"]] == ["one / exact", "two"]
    command = captured["command"]
    assert command[0] == str(executable.resolve())
    assert "-d" in command and "-f" in command


async def test_session_maps_process_start_failure_to_every_candidate(tmp_path):
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")

    def fail_start(command, **kwargs):
        raise OSError("cannot start")

    session = MihomoProbeSession(
        executable,
        process_factory=fail_start,
        validate_config=lambda config, home: None,
    )
    evidence = await session.probe([node(1, "one"), node(2, "two")])

    assert [item.status for item in evidence] == ["process_error", "process_error"]
    assert all(item.confirm is None for item in evidence)
