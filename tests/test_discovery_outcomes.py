import pytest

from src.config import Config, CrawlConfig, LLMConfig, SimpleSite
from src.scheduler import Scheduler
from src.site_processor import DiscoveryFailure, DiscoverySuccess, SiteProcessor


@pytest.mark.asyncio
async def test_crashed_source_returns_typed_failure_with_identity(monkeypatch):
    config = Config(
        sites=(
            SimpleSite(name="healthy", start_url="https://healthy.test"),
            SimpleSite(name="broken", start_url="https://broken.test"),
        ),
        crawl=CrawlConfig(concurrency=2),
        llm=LLMConfig(),
    )

    async def discover(self: SiteProcessor):
        if self.site.name == "broken":
            raise RuntimeError("boom")
        return DiscoverySuccess(site_name=self.site.name)

    monkeypatch.setattr(SiteProcessor, "discover", discover)
    outcomes = await Scheduler(config).run()

    assert [outcome.site_name for outcome in outcomes] == ["healthy", "broken"]
    assert outcomes[0] == DiscoverySuccess(site_name="healthy")
    assert outcomes[1] == DiscoveryFailure(
        site_name="broken",
        errors=("unhandled exception: boom",),
    )
