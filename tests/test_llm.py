from dataclasses import dataclass

import httpx
import pytest
from openai import RateLimitError

from freenodes.config import OpenRouterLimits
from freenodes.llm import (
    ExtractedLinks,
    FallbackFailure,
    FallbackText,
    FallbackUnavailable,
    OpenRouterFallback,
)


def test_deterministic_parser_classifies_and_stable_deduplicates():
    parsed = ExtractedLinks.from_text(
        "\n".join(
            (
                "https://files.test/nodes.txt",
                "https://files.test/config.yaml?token=one",
                "vmess://eyJ2IjoiMiJ9",
                "https://files.test/nodes.txt",
            )
        )
    )

    assert parsed.txt == ("https://files.test/nodes.txt",)
    assert parsed.yaml == ("https://files.test/config.yaml?token=one",)
    assert parsed.inline == ("vmess://eyJ2IjoiMiJ9",)


def test_deterministic_parser_rejects_descriptive_text():
    assert ExtractedLinks.from_text("Clash and V2Ray configuration") == ExtractedLinks()


class FakeMessage:
    def __init__(self, content: str | None, reasoning: str | None = None):
        self.content = content
        self.reasoning = reasoning


class FakeChoice:
    def __init__(self, content: str | None, reasoning: str | None = None):
        self.message = FakeMessage(content, reasoning)


class FakeResponse:
    def __init__(self, content: str | None, reasoning: str | None = None):
        self.choices = [FakeChoice(content, reasoning)]


@dataclass(frozen=True)
class Raised:
    error: Exception


class FakeCompletions:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.models: list[str] = []

    async def create(self, *, model, **kwargs):
        self.models.append(model)
        response = self.responses.pop(0)
        match response:
            case Raised(error=error):
                raise error
            case value:
                return value


class FakeClient:
    def __init__(self, *responses):
        self.completions = FakeCompletions(*responses)
        self.chat = FakeChat(self.completions)


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


def limits(*, run: int = 30, source: int = 3) -> OpenRouterLimits:
    return OpenRouterLimits(
        request_limit_per_run=run,
        request_limit_per_source=source,
        request_timeout_seconds=20,
    )


def fallback(monkeypatch, *responses, bounds=None):
    client = FakeClient(*responses)
    construction = {}

    def create_client(**kwargs):
        construction.update(kwargs)
        return client

    monkeypatch.setattr("freenodes.llm.AsyncOpenAI", create_client)
    return (
        OpenRouterFallback(bounds or limits(), credential="secret"),
        client,
        construction,
    )


async def test_missing_credential_performs_no_external_call(monkeypatch):
    attempted = False

    def unexpected_client(**kwargs):
        nonlocal attempted
        attempted = True

    monkeypatch.setattr("freenodes.llm.AsyncOpenAI", unexpected_client)

    result = await OpenRouterFallback(limits(), credential=None).ask(
        "prompt", source="alpha"
    )

    assert result == FallbackUnavailable(reason="no_credential")
    assert attempted is False


async def test_fixed_openrouter_endpoint_model_and_text_admission(monkeypatch):
    router, client, construction = fallback(monkeypatch, FakeResponse("answer"))

    result = await router.ask("prompt", source="alpha")

    assert result == FallbackText(value="answer")
    assert construction["base_url"] == "https://openrouter.ai/api/v1"
    assert construction["timeout"] == 20
    assert client.completions.models == ["openrouter/free"]


async def test_request_budget_bounds_observable_calls(monkeypatch):
    router, client, _ = fallback(
        monkeypatch,
        FakeResponse("one"),
        FakeResponse("two"),
        FakeResponse("three"),
        bounds=limits(run=3, source=2),
    )

    assert [(await router.ask("prompt", source="alpha")).kind for _ in range(3)] == [
        "text",
        "text",
        "unavailable",
    ]
    assert (await router.ask("prompt", source="beta")).kind == "text"
    assert await router.ask("prompt", source="gamma") == FallbackUnavailable(
        reason="budget_exhausted"
    )
    assert client.completions.models == ["openrouter/free"] * 3


async def test_rate_limit_is_typed_and_never_retried(monkeypatch):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    response = httpx.Response(429, request=request)
    router, client, _ = fallback(
        monkeypatch,
        Raised(RateLimitError("limited", response=response, body=None)),
    )

    assert await router.ask("prompt", source="alpha") == FallbackUnavailable(
        reason="rate_limited"
    )
    assert client.completions.models == ["openrouter/free"]


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        (FakeResponse(None, "reasoned"), FallbackText(value="reasoned")),
        (FakeResponse(None), FallbackUnavailable(reason="empty_response")),
        (object(), FallbackFailure(diagnostic="invalid_response")),
        (
            Raised(httpx.ConnectError("offline")),
            FallbackFailure(diagnostic="ConnectError"),
        ),
    ),
)
async def test_provider_outcomes_are_admitted_without_escaping(
    monkeypatch, response, expected
):
    router, _, _ = fallback(monkeypatch, response)
    assert await router.ask("prompt", source="alpha") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (
            '{"txt": ["a.txt"], "yaml": ["b.yaml"]}',
            ExtractedLinks(txt=("a.txt",), yaml=("b.yaml",)),
        ),
        ('```json\n{"txt": ["a.txt"]}\n```', ExtractedLinks(txt=("a.txt",))),
        ("not json", ExtractedLinks()),
        ("", ExtractedLinks()),
    ),
)
async def test_extract_links_admits_provider_payload(monkeypatch, raw, expected):
    router, _, _ = fallback(monkeypatch, FakeResponse(raw))
    assert await router.extract_links("page", source="alpha") == expected


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
async def test_generate_pattern_admits_provider_text(monkeypatch, raw, expected):
    router, _, _ = fallback(monkeypatch, FakeResponse(raw))
    assert (
        await router.generate_pattern(
            ["https://node.example.com/a.txt"],
            source="alpha",
        )
        == expected
    )
