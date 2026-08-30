"""Tests for Scheduler: site resolution, parallel dispatch, error handling.

Run: pytest tests/test_scheduler.py -v
"""

import asyncio

import pytest

from src.config import Config, CrawlConfig, LLMConfig, SimpleSite
from src.scheduler import Scheduler
from src.site_processor import DiscoveryFailure, DiscoverySuccess


@pytest.fixture
def three_site_config(tmp_path):
    return Config(
        sites=[
            SimpleSite(name="site-a", start_url="http://a.test/"),
            SimpleSite(name="site-b", start_url="http://b.test/"),
            SimpleSite(name="site-c", start_url="http://c.test/"),
        ],
        crawl=CrawlConfig(max_articles=2, timeout=5, concurrency=2),
        output={"dir": str(tmp_path / "nodes")},
        llm=LLMConfig(),
    )


class TestRun:
    async def test_run_all_sites_reports_aggregate_summary(
        self, three_site_config, monkeypatch, capsys
    ):
        """All 3 sites processed, results collected."""
        processed: list[str] = []

        async def fake_run(self):
            processed.append(self.site.name)
            return DiscoverySuccess(site_name=self.site.name, articles_processed=1)

        monkeypatch.setattr("src.scheduler.SiteProcessor.discover", fake_run)

        scheduler = Scheduler(three_site_config)
        results = await scheduler.run()
        assert len(results) == 3
        assert len(processed) == 3
        assert set(processed) == {"site-a", "site-b", "site-c"}
        output = capsys.readouterr().out
        assert "SUMMARY" in output
        assert "TOTAL" in output
        assert all(site in output for site in processed)

    async def test_run_single_target(self, three_site_config, monkeypatch):
        """Only the targeted site runs."""
        processed: list[str] = []

        async def fake_run(self):
            processed.append(self.site.name)
            return DiscoverySuccess(site_name=self.site.name)

        monkeypatch.setattr("src.scheduler.SiteProcessor.discover", fake_run)

        scheduler = Scheduler(three_site_config)
        results = await scheduler.run(target="site-b")
        assert len(results) == 1
        assert processed == ["site-b"]

    @pytest.mark.parametrize("target", ("nonexistent", "SITE-A"))
    async def test_unknown_or_wrong_case_target_performs_no_discovery(
        self, target, three_site_config, monkeypatch
    ):
        processed: list[str] = []

        async def fake_run(self):
            processed.append(self.site.name)
            return DiscoverySuccess(site_name=self.site.name)

        monkeypatch.setattr("src.scheduler.SiteProcessor.discover", fake_run)

        assert await Scheduler(three_site_config).run(target=target) == []
        assert processed == []

    async def test_handles_site_crash(self, three_site_config, monkeypatch, capsys):
        """One site crashing doesn't stop others."""
        call_count = 0

        async def fake_run(self):
            nonlocal call_count
            call_count += 1
            if self.site.name == "site-b":
                raise RuntimeError("crash!")
            return DiscoverySuccess(site_name=self.site.name, articles_processed=1)

        monkeypatch.setattr("src.scheduler.SiteProcessor.discover", fake_run)

        scheduler = Scheduler(three_site_config)
        results = await scheduler.run()
        assert len(results) == 3
        # site-b should produce an error result
        failures = [result for result in results if result.kind == "failure"]
        assert failures == [
            DiscoveryFailure(
                site_name="site-b",
                errors=("unhandled exception: crash!",),
            )
        ]
        output = capsys.readouterr().out
        assert "site-b" in output
        assert "unhandled exception: crash!" in output

    async def test_respects_concurrency_limit(self, three_site_config, monkeypatch):
        """Semaphore cap = 2, max 2 concurrent."""
        running = 0
        max_concurrent = 0

        async def fake_run(self):
            nonlocal running, max_concurrent
            running += 1
            max_concurrent = max(max_concurrent, running)
            await asyncio.sleep(0.05)
            running -= 1
            return DiscoverySuccess(site_name=self.site.name)

        monkeypatch.setattr("src.scheduler.SiteProcessor.discover", fake_run)

        scheduler = Scheduler(three_site_config)
        await scheduler.run()
        assert max_concurrent <= 2, f"concurrency exceeded limit: {max_concurrent}"
