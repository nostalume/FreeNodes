"""Typed password policy and isolated browser-decryption contracts."""

import asyncio
from collections import deque
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from src.config import (
    AabbPasswordSource,
    AbabPasswordSource,
    DescriptionPasswordSource,
    EmptyPasswordSource,
    PasswordCandidates,
    PasswordEvidence,
    PasswordPolicy,
    SubtitlePasswordSource,
)
from src.decryptor import DecryptionClient

URL = "https://protected.test/article"
PASTE_URL = "https://paste.to/?fixture#secret"


@dataclass
class FakePage:
    evaluations: deque[object]
    blocked: bool = False
    goto_calls: list[str] = field(default_factory=list)
    evaluate_calls: list[tuple[str, object]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    close_calls: int = 0
    close_error: Exception | None = None

    async def goto(self, url: str, *, timeout: float) -> None:
        self.goto_calls.append(url)

    async def evaluate(self, expression: str, argument: object) -> object:
        self.evaluate_calls.append((expression, argument))
        self.started.set()
        if self.blocked:
            await self.release.wait()
        return self.evaluations.popleft()

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@dataclass
class FakeContext:
    page: FakePage
    new_page_calls: int = 0
    close_calls: int = 0

    async def new_page(self) -> FakePage:
        self.new_page_calls += 1
        return self.page

    async def close(self) -> None:
        self.close_calls += 1


@dataclass
class FakeBrowser:
    contexts: deque[FakeContext]
    close_error: Exception | None = None
    close_blocked: bool = False
    new_context_calls: int = 0
    close_calls: int = 0
    terminate_calls: int = 0
    close_started: asyncio.Event = field(default_factory=asyncio.Event)
    close_release: asyncio.Event = field(default_factory=asyncio.Event)

    async def new_context(self) -> FakeContext:
        self.new_context_calls += 1
        return self.contexts.popleft()

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.close_blocked:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error

    async def terminate(self) -> None:
        self.terminate_calls += 1


@dataclass
class FakeBrowserFactory:
    browser: FakeBrowser
    error: Exception | None = None
    calls: int = 0
    proxies: list[str] = field(default_factory=list)

    async def __call__(self, *, proxy: str) -> FakeBrowser:
        self.calls += 1
        self.proxies.append(proxy)
        if self.error is not None:
            raise self.error
        return self.browser


def browser_with(*pages: FakePage, close_error: Exception | None = None):
    browser = FakeBrowser(
        deque(FakeContext(page) for page in pages),
        close_error=close_error,
    )
    return browser, FakeBrowserFactory(browser)


def candidates(*values: str) -> PasswordCandidates:
    return PasswordCandidates(values=values)


def submitted(text: str = "", html: str = "") -> dict[str, str]:
    return {"status": "submitted", "text": text, "html": html}


def test_password_patterns_are_exact_bounded_algebras():
    evidence = PasswordEvidence()
    aabb = PasswordPolicy(
        sources=(AabbPasswordSource(limit=90),),
        max_candidates=90,
    ).resolve(evidence)
    abab = PasswordPolicy(
        sources=(AbabPasswordSource(limit=90),),
        max_candidates=90,
    ).resolve(evidence)

    assert len(aabb.values) == len(set(aabb.values)) == 90
    assert len(abab.values) == len(set(abab.values)) == 90
    assert all(value[0] == value[1] and value[2] == value[3] for value in aabb.values)
    assert all(value[0] == value[2] and value[1] == value[3] for value in abab.values)
    assert all(value[0] != value[2] for value in aabb.values)
    assert all(value[0] != value[1] for value in abab.values)


def test_password_policy_preserves_source_order_and_stable_uniqueness():
    policy = PasswordPolicy(
        sources=(
            EmptyPasswordSource(),
            SubtitlePasswordSource(limit=2),
            DescriptionPasswordSource(limit=2),
            AabbPasswordSource(limit=2),
            AbabPasswordSource(limit=2),
        ),
        max_candidates=9,
    )

    resolved = policy.resolve(
        PasswordEvidence(
            subtitles="password 1122 then 3344",
            description="repeat 3344 then 5566",
        )
    )

    assert resolved.values == (
        "",
        "1122",
        "3344",
        "5566",
        "0011",
        "0022",
        "0101",
        "0202",
    )


def test_password_policy_rejects_starvation_and_duplicate_sources():
    with pytest.raises(ValidationError, match="source limits"):
        PasswordPolicy(
            sources=(AabbPasswordSource(limit=2), AbabPasswordSource(limit=2)),
            max_candidates=3,
        )
    with pytest.raises(ValidationError, match="source types"):
        PasswordPolicy(
            sources=(AabbPasswordSource(limit=1), AabbPasswordSource(limit=1)),
            max_candidates=2,
        )


async def test_password_page_resets_per_candidate_and_passes_js_data_safely():
    page = FakePage(
        deque(
            (
                submitted("wrong password"),
                submitted("vmess://eyJ2IjoiMiJ9"),
            )
        )
    )
    browser, factory = browser_with(page)
    client = DecryptionClient(proxy="http://proxy.test:8080", browser_factory=factory)

    async with client:
        outcome = await client.decrypt_page(
            URL,
            candidates("'quoted\\value", "1122"),
        )

    assert outcome.kind == "decrypted"
    assert outcome.password == "1122"
    assert page.goto_calls == [URL, URL]
    expression, argument = page.evaluate_calls[0]
    assert "'quoted\\value" not in expression
    assert argument == {
        "password": "'quoted\\value",
        "inputSelector": ".cl-input",
        "buttonSelector": ".cl-btn",
    }
    assert page.close_calls == 1
    assert browser.new_context_calls == 1
    assert browser.close_calls == 1
    assert factory.proxies == ["http://proxy.test:8080"]


@pytest.mark.parametrize(
    ("status", "code"),
    (("missing_input", "missing_input"), ("missing_button", "missing_button")),
)
async def test_password_page_preserves_structural_rejection(status, code):
    page = FakePage(deque(({"status": status, "text": "", "html": ""},)))
    browser, factory = browser_with(page)

    async with DecryptionClient(browser_factory=factory) as client:
        outcome = await client.decrypt_page(URL, candidates("1122", "3344"))

    assert outcome.kind == "rejected"
    assert outcome.code == code
    assert outcome.attempted == 1
    assert page.goto_calls == [URL]


async def test_paste_decryption_returns_admitted_page_links():
    page = FakePage(deque((submitted("https://files.test/nodes.txt"),)))
    browser, factory = browser_with(page)

    async with DecryptionClient(browser_factory=factory) as client:
        outcome = await client.decrypt_paste(PASTE_URL, candidates(""))

    assert outcome.kind == "decrypted"
    assert tuple(link.href for link in outcome.page.links) == (
        "https://files.test/nodes.txt",
    )


async def test_exhausted_candidates_remain_a_typed_rejection():
    page = FakePage(deque((submitted("wrong"), submitted("still wrong"))))
    browser, factory = browser_with(page)

    async with DecryptionClient(browser_factory=factory) as client:
        outcome = await client.decrypt_page(URL, candidates("1122", "3344"))

    assert outcome.kind == "rejected"
    assert outcome.code == "no_subscription"
    assert outcome.attempted == 2


async def test_malformed_browser_result_is_a_typed_failure():
    page = FakePage(deque(({"status": "unknown"},)))
    context = FakeContext(page)
    browser = FakeBrowser(deque((context,)))

    async with DecryptionClient(browser_factory=FakeBrowserFactory(browser)) as client:
        outcome = await client.decrypt_page(URL, candidates("1122"))

    assert outcome.kind == "failure"
    assert outcome.code == "malformed_output"
    assert page.close_calls == 1
    assert context.close_calls == 1


async def test_cleanup_failure_is_explicit_and_context_still_closes():
    page = FakePage(
        deque((submitted("vmess://success"),)),
        close_error=RuntimeError("page close failed"),
    )
    context = FakeContext(page)
    browser = FakeBrowser(deque((context,)))

    async with DecryptionClient(browser_factory=FakeBrowserFactory(browser)) as client:
        outcome = await client.decrypt_page(URL, candidates("1122"))

    assert outcome.kind == "failure"
    assert outcome.code == "cleanup_failed"
    assert "page close failed" in outcome.diagnostic
    assert context.close_calls == 1


async def test_each_target_gets_a_fresh_context_without_cookie_state():
    first = FakePage(deque((submitted("vmess://first"),)))
    second = FakePage(deque((submitted("trojan://second@example.test:443"),)))
    browser, factory = browser_with(first, second)

    async with DecryptionClient(browser_factory=factory) as client:
        first_outcome = await client.decrypt_page(URL, candidates("1122"))
        second_outcome = await client.decrypt_paste(PASTE_URL, candidates(""))

    assert first_outcome.kind == second_outcome.kind == "decrypted"
    assert browser.new_context_calls == 2
    assert first.close_calls == second.close_calls == 1


async def test_timeout_closes_page_and_context():
    page = FakePage(deque(), blocked=True)
    context = FakeContext(page)
    browser = FakeBrowser(deque((context,)))
    client = DecryptionClient(
        timeout_s=0.01,
        browser_factory=FakeBrowserFactory(browser),
    )

    async with client:
        outcome = await client.decrypt_page(URL, candidates("1122"))

    assert outcome.kind == "failure"
    assert outcome.code == "timeout"
    assert page.close_calls == 1
    assert context.close_calls == 1


async def test_cancellation_closes_target_before_propagating():
    page = FakePage(deque(), blocked=True)
    context = FakeContext(page)
    browser = FakeBrowser(deque((context,)))
    client = DecryptionClient(browser_factory=FakeBrowserFactory(browser))

    async with client:
        task = asyncio.create_task(client.decrypt_page(URL, candidates("1122")))
        await page.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert page.close_calls == 1
        assert context.close_calls == 1


async def test_browser_start_failure_is_typed():
    browser, factory = browser_with()
    factory.error = RuntimeError("browser unavailable")

    async with DecryptionClient(browser_factory=factory) as client:
        outcome = await client.decrypt_page(URL, candidates("1122"))

    assert outcome.kind == "failure"
    assert outcome.code == "browser_error"
    assert "browser unavailable" in outcome.diagnostic
    assert browser.close_calls == 0


async def test_close_failure_terminates_once_and_double_close_is_safe():
    browser, factory = browser_with(close_error=RuntimeError("close failed"))
    client = DecryptionClient(browser_factory=factory)
    await client.__aenter__()
    outcome = await client.decrypt_page(URL, candidates("1122"))
    await client.aclose()
    await client.aclose()

    assert outcome.kind == "failure"
    assert outcome.code == "browser_error"
    assert browser.close_calls == 1
    assert browser.terminate_calls == 1


async def test_close_timeout_terminates_browser():
    browser, factory = browser_with()
    browser.close_blocked = True
    client = DecryptionClient(
        close_timeout_s=0.01,
        browser_factory=factory,
    )
    await client.__aenter__()
    await client.decrypt_page(URL, candidates("1122"))

    await client.aclose()

    assert browser.close_calls == 1
    assert browser.terminate_calls == 1
