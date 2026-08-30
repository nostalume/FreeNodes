"""Tests for llm_router: WeightedSelector, HealthTracker, LLMRouter.

Run with: pytest tests/test_llm_router.py -v
"""

import os
import time
from unittest.mock import patch

import httpx
import pytest
from openai import RateLimitError

from src.config import Config, CrawlConfig, LLMConfig, ProviderConfig, SimpleSite
from src.llm_router import (
    ExtractedLinks,
    HealthTracker,
    LLMRouter,
    RequestBudget,
    WeightedSelector,
)

# ═══════════════════════════════════════════════════════════════
# WeightedSelector
# ═══════════════════════════════════════════════════════════════


class TestWeightedSelector:
    def test_returns_none_for_empty_weights(self):
        selector = WeightedSelector()
        result = selector.pick({}, lambda n: True)
        assert result is None

    def test_returns_only_option(self):
        selector = WeightedSelector()
        result = selector.pick({"a": 100}, lambda n: True)
        assert result == "a"

    def test_returns_only_option_with_min_weight(self):
        selector = WeightedSelector()
        result = selector.pick({"a": 1}, lambda n: True)
        assert result == "a"

    def test_skips_unhealthy_providers(self):
        selector = WeightedSelector()
        healthy_set = {"a"}
        result = selector.pick(
            {"a": 50, "b": 50},
            lambda n: n in healthy_set,
        )
        assert result == "a"

    def test_returns_none_when_all_unhealthy(self):
        selector = WeightedSelector()
        result = selector.pick({"a": 50}, lambda n: False)
        assert result is None

    def test_skips_excluded_providers(self):
        selector = WeightedSelector()
        result = selector.pick(
            {"a": 50, "b": 50, "c": 50},
            lambda n: True,
            exclude={"a", "b"},
        )
        assert result == "c"

    def test_distribution_matches_weights(self):
        """Statistical: 10,000 picks should approximate the weight ratio."""
        selector = WeightedSelector()
        weights = {"a": 60, "b": 30, "c": 10}
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(10000):
            pick = selector.pick(weights, lambda n: True)
            counts[pick] += 1
        total = sum(counts.values())
        assert abs(counts["a"] / total - 0.6) < 0.03
        assert abs(counts["b"] / total - 0.3) < 0.03
        assert abs(counts["c"] / total - 0.1) < 0.03

    def test_zero_weight_never_picked(self):
        selector = WeightedSelector()
        for _ in range(1000):
            pick = selector.pick({"a": 100, "b": 0}, lambda n: True)
            assert pick == "a"

    def test_exclude_all_returns_none(self):
        selector = WeightedSelector()
        result = selector.pick({"a": 100}, lambda n: True, exclude={"a"})
        assert result is None


# ═══════════════════════════════════════════════════════════════
# HealthTracker
# ═══════════════════════════════════════════════════════════════


class TestHealthTracker:
    def test_initial_state_is_healthy(self):
        tracker = HealthTracker()
        assert tracker.is_healthy("p1") is True

    def test_disabled_after_max_failures(self):
        tracker = HealthTracker(max_failures=3, cooldown_seconds=300)
        for _ in range(3):
            tracker.record_failure("p1")
        assert tracker.is_healthy("p1") is False

    def test_still_healthy_below_threshold(self):
        tracker = HealthTracker(max_failures=3, cooldown_seconds=300)
        tracker.record_failure("p1")
        tracker.record_failure("p1")
        assert tracker.is_healthy("p1") is True

    def test_success_resets_failure_count(self):
        tracker = HealthTracker(max_failures=3, cooldown_seconds=300)
        for _ in range(2):
            tracker.record_failure("p1")
        tracker.record_success("p1")
        for _ in range(2):
            tracker.record_failure("p1")
        assert tracker.is_healthy("p1") is True  # still 2, not 4

    def test_success_clears_disabled(self):
        tracker = HealthTracker(max_failures=2, cooldown_seconds=300)
        tracker.record_failure("p1")
        tracker.record_failure("p1")
        assert tracker.is_healthy("p1") is False
        tracker.record_success("p1")
        assert tracker.is_healthy("p1") is True

    def test_auto_recovery_after_cooldown(self):
        tracker = HealthTracker(max_failures=2, cooldown_seconds=0.01)
        tracker.record_failure("p1")
        tracker.record_failure("p1")
        assert tracker.is_healthy("p1") is False
        time.sleep(0.02)
        assert tracker.is_healthy("p1") is True

    def test_multiple_providers_independent(self):
        tracker = HealthTracker(max_failures=2, cooldown_seconds=300)
        tracker.record_failure("p1")
        tracker.record_failure("p1")
        assert tracker.is_healthy("p1") is False
        assert tracker.is_healthy("p2") is True  # unaffected

    def test_health_report(self):
        tracker = HealthTracker(max_failures=3, cooldown_seconds=300)
        tracker.record_failure("p1")
        report = tracker.health_report()
        assert report["p1"]["failures"] == 1
        assert report["p1"]["healthy"] is True


class TestRequestBudget:
    def test_enforces_site_and_run_limits_exactly(self):
        budget = RequestBudget(run_limit=3, site_limit=2)

        assert budget.charge("alpha") is True
        assert budget.charge("alpha") is True
        assert budget.charge("alpha") is False
        assert budget.charge("beta") is True
        assert budget.charge("gamma") is False
        assert budget.used_total == 3
        assert budget.used_by_site == {"alpha": 2, "beta": 1}


# ═══════════════════════════════════════════════════════════════
# LLMRouter (mocked _try_provider)
# ═══════════════════════════════════════════════════════════════


class FakeResponseMessage:
    """Mock for ChatCompletion response message."""

    def __init__(self, content: str | None, reasoning: str | None = None):
        self.content = content
        self.reasoning = reasoning


class FakeChoice:
    def __init__(self, content: str | None, reasoning: str | None = None):
        self.message = FakeResponseMessage(content, reasoning)


class FakeResponse:
    def __init__(self, content: str | None, reasoning: str | None = None):
        self.choices = [FakeChoice(content, reasoning)]


class FakeCompletions:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)

    async def create(self, **kwargs):
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses: FakeResponse):
        self.chat = type(
            "Chat",
            (),
            {"completions": FakeCompletions(*responses)},
        )()


def _make_router(config: Config | None = None):
    if config is None:
        config = Config(
            sites=[SimpleSite(name="test", start_url="https://example.com")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(
                providers=[
                    ProviderConfig(
                        name="mock-a",
                        base_url="http://a.test/v1",
                        api_key_env="KEY_A",
                        models=["m-a"],
                        default_weight=60,
                    ),
                    ProviderConfig(
                        name="mock-b",
                        base_url="http://b.test/v1",
                        api_key_env="KEY_B",
                        models=["m-b"],
                        default_weight=40,
                    ),
                ],
                task_routing={
                    "test_task": {"mock-a": 60, "mock-b": 40},
                },
            ),
        )
    keys = {provider.api_key_env: "test-key" for provider in config.llm.providers}
    with patch.dict(os.environ, keys):
        return LLMRouter(config, timeout_s=5)


class TestLLMRouter:
    async def test_only_providers_with_credentials_are_admitted(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        monkeypatch.setenv("AVAILABLE_KEY", "test-key")
        config = Config(
            sites=[SimpleSite(name="t", start_url="http://x")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(
                providers=[
                    ProviderConfig(
                        name="missing",
                        base_url="http://missing/v1",
                        api_key_env="MISSING_KEY",
                        models=("m",),
                    ),
                    ProviderConfig(
                        name="available",
                        base_url="http://available/v1",
                        api_key_env="AVAILABLE_KEY",
                        models=("m",),
                    ),
                ],
                task_routing={"default": {"missing": 90, "available": 10}},
            ),
        )

        router = LLMRouter(config)

        assert router.provider_names == ("available",)

    async def test_missing_credentials_perform_no_provider_attempt(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        attempted = False

        def unexpected_client(**kwargs):
            nonlocal attempted
            attempted = True
            return FakeClient(FakeResponse("unexpected"))

        monkeypatch.setattr("src.llm_router.AsyncOpenAI", unexpected_client)
        config = Config(
            sites=[SimpleSite(name="t", start_url="http://x")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(
                providers=[
                    ProviderConfig(
                        name="missing",
                        base_url="http://missing/v1",
                        api_key_env="MISSING_KEY",
                        models=("m",),
                    )
                ],
            ),
        )
        router = LLMRouter(config)

        assert await router.ask("hi", site="t") == ""
        assert attempted is False

    async def test_rate_limit_does_not_try_another_model(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        calls: list[str] = []

        class Completions:
            async def create(self, *, model, **kwargs):
                calls.append(model)
                request = httpx.Request("POST", "https://openrouter.ai/api/v1")
                response = httpx.Response(429, request=request)
                raise RateLimitError("limited", response=response, body=None)

        client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": Completions()})()},
        )()
        monkeypatch.setattr("src.llm_router.AsyncOpenAI", lambda **kwargs: client)
        config = Config(
            sites=[SimpleSite(name="t", start_url="http://x")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(
                providers=[
                    ProviderConfig(
                        name="openrouter",
                        base_url="https://openrouter.ai/api/v1",
                        api_key_env="OPENROUTER_API_KEY",
                        models=("first", "second"),
                        default_weight=100,
                    )
                ],
            ),
        )

        assert await LLMRouter(config).ask("hi", site="t") == ""
        assert calls == ["first"]

    async def test_ask_returns_empty_when_no_providers(self):
        """Router with empty provider list returns empty string."""
        config = Config(
            sites=[SimpleSite(name="t", start_url="http://x")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(),
        )
        router = LLMRouter(config)
        result = await router.ask("hi")
        assert result == ""

    async def test_ask_returns_text_on_first_success(self, monkeypatch):
        """When the first provider succeeds, return its text immediately."""
        monkeypatch.setenv("KEY", "some-key")
        monkeypatch.setattr(
            "src.llm_router.AsyncOpenAI",
            lambda **kwargs: FakeClient(FakeResponse("ok-from-p1")),
        )

        config = Config(
            sites=[SimpleSite(name="t", start_url="http://x")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(
                providers=[
                    ProviderConfig(
                        name="p1",
                        base_url="http://a/v1",
                        api_key_env="KEY",
                        models=["m"],
                        default_weight=100,
                    ),
                ],
                task_routing={"default": {"p1": 100}},
            ),
        )
        router = LLMRouter(config, timeout_s=5)

        result = await router.ask("hi")
        assert result == "ok-from-p1"

    async def test_ask_falls_back_on_failure(self, monkeypatch):
        """Both providers tried when first fails; second succeeds."""
        config = Config(
            sites=[SimpleSite(name="t", start_url="http://x")],
            crawl=CrawlConfig(),
            output={"dir": "nodes"},
            llm=LLMConfig(
                providers=[
                    ProviderConfig(
                        name="p1",
                        base_url="http://a/v1",
                        api_key_env="K",
                        models=["m"],
                        default_weight=50,
                    ),
                    ProviderConfig(
                        name="p2",
                        base_url="http://b/v1",
                        api_key_env="K",
                        models=["m"],
                        default_weight=50,
                    ),
                ],
                task_routing={"default": {"p1": 50, "p2": 50}},
            ),
        )
        monkeypatch.setenv("K", "test-key")
        monkeypatch.setattr("src.llm_router.random.uniform", lambda start, end: start)
        clients = {
            "http://a/v1": FakeClient(FakeResponse(None)),
            "http://b/v1": FakeClient(FakeResponse("ok-from-p2")),
        }
        monkeypatch.setattr(
            "src.llm_router.AsyncOpenAI",
            lambda **kwargs: clients[kwargs["base_url"]],
        )
        router = LLMRouter(config, timeout_s=5)

        result = await router.ask("hi")
        assert result == "ok-from-p2"

    async def test_ask_returns_empty_when_all_fail(self, monkeypatch):
        """When every provider fails, return empty string (never raise)."""
        monkeypatch.setenv("KEY_A", "test-key")
        monkeypatch.setenv("KEY_B", "test-key")
        monkeypatch.setattr(
            "src.llm_router.AsyncOpenAI",
            lambda **kwargs: FakeClient(FakeResponse(None)),
        )
        router = _make_router()

        result = await router.ask("hi")
        assert result == ""


@pytest.mark.parametrize(
    ("content", "reasoning", "reasoning_model", "expected"),
    (
        ("hello", None, False, "hello"),
        (None, "reasoned answer", True, "reasoned answer"),
        (None, None, False, ""),
        ("content wins", "reasoning", True, "content wins"),
    ),
)
async def test_ask_admits_external_provider_messages(
    monkeypatch,
    content,
    reasoning,
    reasoning_model,
    expected,
):
    monkeypatch.setenv("PROVIDER_KEY", "test-key")
    monkeypatch.setattr(
        "src.llm_router.AsyncOpenAI",
        lambda **kwargs: FakeClient(FakeResponse(content, reasoning)),
    )
    config = Config(
        sites=[SimpleSite(name="source", start_url="https://source.test")],
        llm=LLMConfig(
            providers=[
                ProviderConfig(
                    name="provider",
                    base_url="https://provider.test/v1",
                    api_key_env="PROVIDER_KEY",
                    models=("model",),
                    default_weight=100,
                    is_reasoning_model=reasoning_model,
                )
            ]
        ),
    )

    assert await LLMRouter(config).ask("prompt", site="source") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (
            '{"txt": ["a.txt"], "yaml": ["b.yaml"]}',
            ExtractedLinks(txt=("a.txt",), yaml=("b.yaml",)),
        ),
        ('```json\n{"txt": ["a.txt"]}\n```', ExtractedLinks(txt=("a.txt",))),
        ("not json at all", ExtractedLinks()),
        ("", ExtractedLinks()),
    ),
)
async def test_extract_links_admits_provider_payload(raw, expected):
    router = _make_router()

    async def provider_response(
        prompt,
        task_type="default",
        max_tokens=1024,
        site="default",
    ):
        assert task_type == "extract_links"
        return raw

    router.ask = provider_response

    assert await router.extract_links("page", site="source") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (
            "Here is the regex:\n```\nhttps://node\\.example\\.com/[^\\s]+\n```",
            r"https://node\.example\.com/[^\s]+",
        ),
        (
            "```regex\nhttps://x\\.com/[a-z]+\\.(txt|yaml)\n```",
            r"https://x\.com/[a-z]+\.(txt|yaml)",
        ),
        (
            "analysis\nhttps://node\\.test\\.com/\\d+/file\\.(txt|yaml)",
            r"https://node\.test\.com/\d+/file\.(txt|yaml)",
        ),
        ("I have no idea what regex to use", None),
        ("", None),
    ),
)
async def test_generate_pattern_admits_provider_text(raw, expected):
    router = _make_router()

    async def provider_response(
        prompt,
        task_type="default",
        max_tokens=256,
        site="default",
    ):
        assert task_type == "generate_pattern"
        return raw

    router.ask = provider_response

    assert (
        await router.generate_pattern(
            ["https://node.example.com/a.txt"],
            "<p>resource</p>",
            site="source",
        )
        == expected
    )
