"""Read-only source portfolio audit behavior."""

from datetime import UTC, datetime

import pytest

from src.config import Config, GitHubSourceSite, LLMConfig, SimpleSite
from src.nodes import SourceArtifact
from src.quality import (
    DelayObservation,
    ProbeDiagnostic,
    ProbeEvidence,
    ProbeRunFailure,
    ProbeRunSuccess,
    QualityPolicy,
    TransferObservation,
    TransferTargetEvidence,
)
from src.scheduler import Scheduler
from src.site_processor import DiscoveryFailure, DiscoverySuccess

NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


def artifact(source: str) -> SourceArtifact:
    identity = f"{source * 8}-{source * 4}-{source * 4}-{source * 4}-{source * 12}"
    return SourceArtifact.inline(
        site=source,
        content=(
            f"vless://{identity}@{source}.example:443?security=tls&type=tcp#{source}"
        ),
        observed_at=NOW,
        published_on=NOW.date(),
    )


def config(tmp_path) -> Config:
    return Config(
        sites=[SimpleSite(name="a", start_url="https://a.test")],
        source_candidates=[
            GitHubSourceSite(
                name="b",
                owner="upstream",
                repository="subscriptions",
                branch="main",
                path="mihomo.yaml",
            )
        ],
        output={"dir": str(tmp_path / "nodes")},
        llm=LLMConfig(),
    )


def policy() -> QualityPolicy:
    return QualityPolicy(
        min_source_nodes=1,
        min_source_qualified=1,
        min_source_pass_ratio=0.01,
        min_source_unique=1,
    )


class ProbeStub:
    def __init__(self, failed_sources=()):
        self.sources: tuple[tuple[str, ...], ...] = ()
        self.failed_sources = frozenset(failed_sources)

    async def probe(self, plan, policy):
        self.sources = tuple(entry.sources for entry in plan.entries)
        return ProbeRunSuccess(
            evidence=tuple(self._evidence(entry) for entry in plan.entries)
        )

    def _evidence(self, entry):
        if self.failed_sources.intersection(entry.sources):
            return ProbeEvidence.process_failure(entry.node, "controller_http")
        return ProbeEvidence(
            fingerprint=entry.node.fingerprint,
            proxy_name=entry.node.display_name,
            coarse=DelayObservation(endpoint="coarse", status="success", delay_ms=50),
            confirm=DelayObservation(endpoint="confirm", status="success", delay_ms=60),
            transfer=TransferObservation(
                fingerprint=entry.node.fingerprint,
                target="test",
                status="success",
                bytes_received=1024 * 1024,
                elapsed_ms=100,
                bytes_per_second=10_000_000,
            ),
        )


@pytest.mark.parametrize(
    ("candidate_available", "failed_sources", "expected_status"),
    (
        (True, (), "accepted"),
        (False, (), "insufficient_authority_diversity"),
        (True, ("b",), "insufficient_authority_diversity"),
    ),
)
async def test_audit_reports_every_source_without_writing(
    monkeypatch,
    tmp_path,
    candidate_available,
    failed_sources,
    expected_status,
):
    audit_config = config(tmp_path)
    before = audit_config.model_dump()
    sentinel = tmp_path / "unchanged"
    sentinel.write_text("keep", encoding="utf-8")

    async def discover_active(processor):
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    async def discover_candidate(client, site, *, observed_at):
        if candidate_available:
            return DiscoverySuccess(site_name="b", artifacts=(artifact("b"),))
        return DiscoveryFailure(site_name="b", errors=("token=private",))

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover_active)
    monkeypatch.setattr("src.scheduler.GitHubSourceClient.discover", discover_candidate)
    probe = ProbeStub(failed_sources)

    receipt = await Scheduler(audit_config).audit_sources(
        probe_session=probe,
        policy=policy(),
        now=NOW,
        runner_vantage="test",
    )

    assert receipt.status == expected_status
    expected_authorities = ("a", "b") if expected_status == "accepted" else ("a",)
    assert receipt.contributing_authorities == expected_authorities
    assert tuple(source.source for source in receipt.sources) == ("a", "b")
    expected_sources = (("a",), ("b",)) if candidate_available else (("a",),)
    assert probe.sources == expected_sources
    if not candidate_available:
        assert receipt.sources[1].reason == "source_unavailable"
        assert "private" not in receipt.model_dump_json()
    expected_failures = {"controller_http": 1} if failed_sources else {}
    assert receipt.probe_failures == expected_failures
    assert audit_config.model_dump() == before
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert tuple(path.name for path in tmp_path.iterdir()) == ("unchanged",)


async def test_audit_reports_underlying_control_failure(monkeypatch, tmp_path):
    async def discover_active(processor):
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    async def discover_candidate(client, site, *, observed_at):
        return DiscoverySuccess(site_name="b", artifacts=(artifact("b"),))

    class InconclusiveProbe:
        async def probe(self, plan, policy):
            return ProbeRunFailure(
                phase="post_control",
                diagnostic=ProbeDiagnostic(
                    code="control_transfer",
                    detail="transfer_timeout",
                ),
                transfer_targets=(
                    TransferTargetEvidence(
                        name="alternate",
                        authority="target-b",
                        controls_attempted=1,
                        controls_passed=0,
                        candidate_attempts=0,
                    ),
                ),
            )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover_active)
    monkeypatch.setattr("src.scheduler.GitHubSourceClient.discover", discover_candidate)

    receipt = await Scheduler(config(tmp_path)).audit_sources(
        probe_session=InconclusiveProbe(),
        policy=policy(),
        now=NOW,
        runner_vantage="test",
    )

    assert receipt.status == "measurement_inconclusive"
    assert receipt.inconclusive_nodes == receipt.selected_for_probe == 2
    assert "post_control/control_transfer: transfer_timeout" in receipt.diagnostic
    assert receipt.transfer_targets[0].authority == "target-b"
