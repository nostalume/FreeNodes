"""One-pass deterministic discovery-to-publication behavior."""

import base64
import json
from datetime import UTC, datetime

import pytest

from src.config import Config, CrawlConfig, LLMConfig, SimpleSite
from src.mihomo import ConsumerValidation
from src.nodes import SourceArtifact
from src.publication import PublicationError
from src.quality import CapabilityRunReceipt, NodeCapabilityDecision
from src.scheduler import Scheduler
from src.site_processor import DiscoveryFailure, DiscoverySuccess

NOW = datetime(2026, 8, 29, tzinfo=UTC)


class CapableProbe:
    async def probe_capabilities(self, plan, targets, policy):
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


class FirstOnlyProbe:
    async def probe_capabilities(self, plan, targets, policy):
        decisions = tuple(
            NodeCapabilityDecision(
                fingerprint=entry.node.fingerprint,
                status="capable" if index == 0 else "failed",
                successful_targets=("github", "google") if index == 0 else (),
                failed_targets=() if index == 0 else ("github", "google"),
                reason="quorum" if index == 0 else "target_failures",
            )
            for index, entry in enumerate(plan.entries)
        )
        return CapabilityRunReceipt(
            status="complete",
            decisions=decisions,
            accepted_fingerprints=(decisions[0].fingerprint,),
        )


class StructuralValidator:
    def validate_bundle(self, root):
        assert (root / "nodes" / "merged.yaml").exists()
        assert (root / "nodes" / "v2ray.txt").exists()
        assert not (root / "nodes" / "quality-manifest.json").exists()
        return ConsumerValidation(
            profiles=("nodes/merged.yaml",),
            provider_profiles=("nodes/provider.yaml",),
            provider_names=("a",),
            group_names=("select",),
        )


def config(root, sites=("a", "b"), required=()):
    return Config(
        sites=[
            SimpleSite(
                name=name,
                start_url=f"https://{name}.test",
                required=name in required,
            )
            for name in sites
        ],
        crawl=CrawlConfig(concurrency=2),
        output={"dir": str(root / "nodes")},
        llm=LLMConfig(),
    )


def artifact(site: str, *, observed_at: datetime = NOW) -> SourceArtifact:
    payload = base64.b64encode(
        json.dumps(
            {
                "v": "2",
                "ps": site,
                "add": f"{site}.example",
                "port": "443",
                "id": "11111111-1111-1111-1111-111111111111",
                "aid": "0",
                "net": "tcp",
                "tls": "tls",
            }
        ).encode()
    ).decode()
    return SourceArtifact.inline(
        site=site,
        content=f"vmess://{payload}",
        observed_at=observed_at,
        published_on=observed_at.date(),
    )


async def test_publication_requires_capability_after_deterministic_admission(
    monkeypatch, tmp_path, capsys
):
    async def discover(self):
        if self.site.name == "b":
            return DiscoveryFailure(site_name="b", errors=("unavailable",))
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    receipt = await Scheduler(config(tmp_path)).publish_profiles(
        repository_root=tmp_path,
        validator=StructuralValidator(),
        probe_session=CapableProbe(),
        now=NOW,
    )

    assert receipt.status == "accepted"
    assert (tmp_path / "nodes" / "merged.yaml").exists()
    assert (tmp_path / "nodes" / "v2ray.txt").exists()
    assert not (tmp_path / "nodes" / "quality-manifest.json").exists()
    manifest = json.loads(
        (tmp_path / "nodes" / "publication-receipt.json").read_bytes()
    )
    assert manifest["schema"] == 3
    assert manifest["capability"]["accepted"] == 1
    assert manifest["admission"]["attempted_sources"] == 2
    assert manifest["admission"]["failed_sources"] == 1
    assert "[b] unavailable" in capsys.readouterr().out


async def test_publish_freshness_uses_explicit_run_time_not_wall_clock(
    monkeypatch, tmp_path
):
    as_of = datetime(2000, 1, 1, tzinfo=UTC)

    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name,
            artifacts=(artifact(self.site.name, observed_at=as_of),),
        )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    receipt = await Scheduler(config(tmp_path)).publish_profiles(
        repository_root=tmp_path,
        validator=StructuralValidator(),
        probe_session=CapableProbe(),
        now=as_of,
    )

    assert receipt.status == "accepted"


async def test_publication_projects_only_the_accepted_capable_identity(
    monkeypatch, tmp_path
):
    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name, artifacts=(artifact(self.site.name),)
        )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    receipt = await Scheduler(config(tmp_path)).publish_profiles(
        repository_root=tmp_path,
        validator=StructuralValidator(),
        probe_session=FirstOnlyProbe(),
        now=NOW,
    )

    manifest = json.loads(
        (tmp_path / "nodes" / "publication-receipt.json").read_bytes()
    )
    profile = (tmp_path / "nodes" / "merged.yaml").read_text(encoding="utf-8")
    assert receipt.status == "accepted"
    assert manifest["admission"]["unique_eligible"] == 2
    assert tuple(
        manifest["capability"][key]
        for key in ("attempted", "capable", "failed", "inconclusive", "accepted")
    ) == (2, 1, 1, 0, 1)
    assert "a.example" in profile and "b.example" not in profile


async def test_publish_discovers_each_source_once_and_removes_legacy_site_text(
    monkeypatch, tmp_path
):
    calls: list[str] = []

    async def discover(self):
        calls.append(self.site.name)
        return DiscoverySuccess(
            site_name=self.site.name, artifacts=(artifact(self.site.name),)
        )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)
    legacy = tmp_path / "nodes" / "a.txt"
    legacy.parent.mkdir()
    legacy.write_text("legacy", encoding="utf-8")

    receipt = await Scheduler(config(tmp_path)).publish_profiles(
        repository_root=tmp_path,
        validator=StructuralValidator(),
        probe_session=CapableProbe(),
        now=NOW,
    )

    assert receipt.status == "accepted"
    assert calls == ["a", "b"]
    assert not legacy.exists()


async def test_required_source_failure_is_local_to_that_source(monkeypatch, tmp_path):
    async def discover(self):
        if self.site.name == "b":
            return DiscoveryFailure(site_name="b", errors=("unavailable",))
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    receipt = await Scheduler(config(tmp_path, required=("b",))).publish_profiles(
        repository_root=tmp_path,
        validator=StructuralValidator(),
        probe_session=CapableProbe(),
        now=NOW,
    )

    assert receipt.status == "accepted"


async def test_zero_deterministically_eligible_nodes_preserves_snapshot(
    monkeypatch, tmp_path
):
    sentinel = tmp_path / "nodes" / "merged.yaml"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"previous")

    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name,
            artifacts=(
                SourceArtifact.inline(
                    site=self.site.name,
                    content="not a proxy",
                    observed_at=NOW,
                    published_on=NOW.date(),
                ),
            ),
        )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    with pytest.raises(PublicationError, match="admitted no nodes"):
        await Scheduler(config(tmp_path, ("a",))).publish_profiles(
            repository_root=tmp_path,
            validator=StructuralValidator(),
            probe_session=CapableProbe(),
            now=NOW,
        )

    assert sentinel.read_bytes() == b"previous"
