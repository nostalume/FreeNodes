"""Profile-validation orchestration without public cutover."""

from datetime import UTC, datetime

from src.config import Config, CrawlConfig, LLMConfig, SimpleSite
from src.mihomo import ConsumerValidation
from src.nodes import SourceArtifact
from src.scheduler import Scheduler
from src.site_processor import DiscoverySuccess


class AcceptingValidator:
    def validate_bundle(self, output_dir):
        assert (output_dir / "nodes" / "merged.yaml").exists()
        return ConsumerValidation(
            profiles=(
                "nodes/merged.yaml",
                "nodes/provider.yaml",
                "nodes/provider-cdn.yaml",
            ),
            provider_profiles=("nodes/provider.yaml", "nodes/provider-cdn.yaml"),
            provider_names=("source",),
            group_names=("auto", "manual"),
        )


async def test_scheduler_validates_profiles_without_public_cutover(
    monkeypatch, tmp_path
):
    public = tmp_path / "public-nodes"
    public.mkdir()
    sentinel = public / "merged.yaml"
    sentinel.write_text("unchanged", encoding="utf-8")
    config = Config(
        sites=[SimpleSite(name="source", start_url="https://example.test")],
        crawl=CrawlConfig(max_articles=1, timeout=30, concurrency=1),
        output={"dir": str(public)},
        llm=LLMConfig(),
    )

    async def fake_discover(self):
        artifact = SourceArtifact(
            site="source",
            source_url="inline://source/fixture",
            content="proxies:\n  - {name: One, type: ss, server: one.example, port: 8388, cipher: aes-128-gcm, password: test}\n",
            observed_at=datetime.now(UTC),
            media_type="application/yaml",
        )
        return DiscoverySuccess(site_name="source", yaml_count=1, artifacts=(artifact,))

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", fake_discover)

    receipt = await Scheduler(config).validate_profiles(
        output_parent=tmp_path / "validation-output",
        validator=AcceptingValidator(),
    )

    assert receipt.status == "consumer_validated"
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert receipt.output_dir.is_relative_to((tmp_path / "validation-output").resolve())
