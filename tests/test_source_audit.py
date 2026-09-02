"""Read-only capability audit behavior."""

from datetime import UTC, datetime

from src.config import Config, GitHubSourceSite, LLMConfig, SimpleSite
from src.nodes import SourceArtifact
from src.quality import CapabilityRunReceipt, NodeCapabilityDecision, ProbeDiagnostic
from src.scheduler import Scheduler
from src.site_processor import DiscoveryFailure, DiscoverySuccess

NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


def artifact(source: str) -> SourceArtifact:
    identity = f"{source * 8}-{source * 4}-{source * 4}-{source * 4}-{source * 12}"
    return SourceArtifact.inline(
        site=source,
        content=f"vless://{identity}@{source}.example:443?security=tls&type=tcp#{source}",
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


class CapableProbe:
    def __init__(self):
        self.sources: tuple[tuple[str, ...], ...] = ()
        self.targets: tuple[str, ...] = ()

    async def probe_capabilities(self, plan, targets, policy):
        self.sources = tuple(entry.sources for entry in plan.entries)
        self.targets = tuple(target.id for target in targets)
        decisions = tuple(
            NodeCapabilityDecision(
                fingerprint=entry.node.fingerprint,
                status="capable",
                successful_targets=("github", "google"),
                reason="quorum",
            )
            for entry in plan.entries
        )
        return CapabilityRunReceipt(
            status="complete",
            decisions=decisions,
            accepted_fingerprints=tuple(item.fingerprint for item in decisions),
        )


async def test_audit_measures_every_available_source_without_writing(
    monkeypatch, tmp_path
):
    audit_config = config(tmp_path)
    before = audit_config.model_dump()
    sentinel = tmp_path / "unchanged"
    sentinel.write_text("keep", encoding="utf-8")

    async def discover_active(processor):
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    async def discover_candidate(client, site, *, observed_at):
        return DiscoverySuccess(site_name="b", artifacts=(artifact("b"),))

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover_active)
    monkeypatch.setattr("src.scheduler.GitHubSourceClient.discover", discover_candidate)
    probe = CapableProbe()

    receipt = await Scheduler(audit_config).audit_sources(
        probe_session=probe,
        now=NOW,
    )

    assert receipt.status == "complete"
    assert probe.sources == (("a",), ("b",))
    assert probe.targets == ("github", "google", "cloudflare")
    assert audit_config.model_dump() == before
    assert tuple(path.name for path in tmp_path.iterdir()) == ("unchanged",)


async def test_audit_is_inconclusive_when_controls_are_unusable(monkeypatch, tmp_path):
    async def discover_active(processor):
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    async def discover_candidate(client, site, *, observed_at):
        return DiscoveryFailure(site_name="b", errors=("unavailable",))

    class InconclusiveProbe:
        async def probe_capabilities(self, plan, targets, policy):
            return CapabilityRunReceipt(
                status="inconclusive",
                diagnostic=ProbeDiagnostic(code="control_unavailable"),
            )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover_active)
    monkeypatch.setattr("src.scheduler.GitHubSourceClient.discover", discover_candidate)

    receipt = await Scheduler(config(tmp_path)).audit_sources(
        probe_session=InconclusiveProbe(),
        now=NOW,
    )

    assert receipt.status == "inconclusive"
    assert receipt.diagnostic.code == "control_unavailable"
