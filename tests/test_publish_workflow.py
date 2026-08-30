"""One-pass discovery-to-publication composition tests."""

from datetime import UTC, datetime

import pytest

from src.config import Config, CrawlConfig, LLMConfig, SimpleSite
from src.mihomo import ConsumerValidation, DelayObservation, ProbeEvidence
from src.nodes import SourceArtifact
from src.publication import PublicationError
from src.quality import QualityPolicy
from src.scheduler import Scheduler
from src.site_processor import DiscoveryFailure, DiscoverySuccess

NOW = datetime(2026, 8, 29, tzinfo=UTC)


class Validator:
    def validate_bundle(self, root):
        assert (root / "nodes" / "quality-manifest.json").exists()
        return ConsumerValidation(
            profiles=("nodes/merged.yaml",),
            provider_profiles=(),
            provider_names=(),
            group_names=("select",),
        )


class SuccessfulProbe:
    def __init__(self):
        self.received = ()

    async def probe(self, nodes):
        self.received = tuple(nodes)
        return tuple(
            ProbeEvidence(
                fingerprint=node.fingerprint,
                proxy_name=node.display_name,
                coarse=DelayObservation(
                    endpoint="coarse",
                    status="success",
                    delay_ms=50,
                ),
                confirm=DelayObservation(
                    endpoint="confirm",
                    status="success",
                    delay_ms=60,
                ),
            )
            for node in nodes
        )


def config(root, sites=("a", "b")):
    return Config(
        sites=[
            SimpleSite(name=name, start_url=f"https://{name}.test") for name in sites
        ],
        crawl=CrawlConfig(concurrency=2),
        output={"dir": str(root / "nodes")},
        llm=LLMConfig(),
    )


def artifact(site: str) -> SourceArtifact:
    # VMess retains one URI while also producing a Clash proxy.
    import base64
    import json

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
        site=site, content=f"vmess://{payload}", observed_at=NOW
    )


async def test_publish_composes_each_owner_once_and_replaces_legacy_site_text(
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
    probe = SuccessfulProbe()

    receipt = await Scheduler(config(tmp_path)).publish_profiles(
        repository_root=tmp_path,
        validator=Validator(),
        probe_session=probe,
        policy=QualityPolicy(),
        now=NOW,
    )

    assert receipt.status == "accepted"
    assert calls == ["a", "b"]
    assert len(probe.received) == 2
    assert not legacy.exists()
    assert (tmp_path / "nodes" / "merged.yaml").exists()
    assert (tmp_path / "nodes" / "v2ray.txt").exists()
    assert (tmp_path / "nodes" / "publication-receipt.json").exists()


async def test_missing_required_source_preserves_previous_snapshot(
    monkeypatch, tmp_path
):
    sentinel = tmp_path / "nodes" / "merged.yaml"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"previous")

    async def discover(self):
        if self.site.name == "b":
            return DiscoveryFailure(site_name="b", errors=("unavailable",))
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    with pytest.raises(PublicationError, match="required source"):
        await Scheduler(config(tmp_path)).publish_profiles(
            repository_root=tmp_path,
            validator=Validator(),
            probe_session=SuccessfulProbe(),
            now=NOW,
        )

    assert sentinel.read_bytes() == b"previous"


async def test_empty_quality_result_preserves_previous_snapshot(monkeypatch, tmp_path):
    sentinel = tmp_path / "nodes" / "merged.yaml"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"previous")

    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name, artifacts=(artifact(self.site.name),)
        )

    class FailedProbe:
        async def probe(self, nodes):
            return tuple(
                ProbeEvidence(
                    fingerprint=node.fingerprint,
                    proxy_name=node.display_name,
                    coarse=DelayObservation(
                        endpoint="coarse",
                        status="timeout",
                    ),
                )
                for node in nodes
            )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)

    with pytest.raises(PublicationError, match="quality policy published no nodes"):
        await Scheduler(config(tmp_path, ("a",))).publish_profiles(
            repository_root=tmp_path,
            validator=Validator(),
            probe_session=FailedProbe(),
            now=NOW,
        )

    assert sentinel.read_bytes() == b"previous"
