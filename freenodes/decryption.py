from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable
from typing import Literal, Protocol
from urllib.parse import unquote

from playwright.async_api import (
    Browser as PlaywrightBrowser,
)
from playwright.async_api import (
    BrowserContext as PlaywrightContext,
)
from playwright.async_api import (
    Page as PlaywrightPage,
)
from playwright.async_api import (
    Playwright,
    async_playwright,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from freenodes.config import FrozenModel, PasswordCandidates
from freenodes.web import Page, PageLink

logger = logging.getLogger(__name__)


def extract_paste_url(text: str) -> str | None:
    """Return a client-decryptable paste URL, preserving its fragment key."""
    patterns = (
        r"(https?://paste\.to/\?[a-zA-Z0-9_=-]+#[a-zA-Z0-9_-]+)",
        r'(https?://(?:dpaste|hastebin|privatebin)\.[a-zA-Z.]+/[^\s<>"\')\]]+#[a-zA-Z0-9_-]+)',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    redirect = re.search(
        r'https?://(?:www\.)?youtube\.com/redirect\?[^"\'>\s]*',
        text,
    )
    if not redirect:
        return None
    target = re.search(r"[&?]q=([^&]+)", redirect.group(0))
    if not target:
        return None
    url = unquote(target.group(1))
    return url if "paste" in url or "#" in url else None


def extract_paste_links(text: str) -> tuple[PageLink, ...]:
    """Return subscription-shaped links from decrypted paste content."""
    links: list[PageLink] = []
    for match in re.finditer(r'https?://[^\s<>"\'\[\]()]+', text):
        href = match.group(0).rstrip(".,;)")
        is_profile = href.endswith((".txt", ".yaml", ".yml", ".json", ".conf"))
        is_disguised_drive = href.endswith(".jpg") and any(
            marker in href for marker in ("dlink", "1drv", "onedrive")
        )
        if is_profile or is_disguised_drive:
            links.append(PageLink(href=href, text=href))
    return tuple(links)


DecryptionOperation = Literal["password_page", "paste"]
DecryptionFailureCode = Literal[
    "not_open",
    "browser_error",
    "timeout",
    "malformed_output",
    "cleanup_failed",
]
DecryptionRejectionCode = Literal[
    "missing_input",
    "missing_button",
    "no_subscription",
]


class DecryptedResource(FrozenModel):
    kind: Literal["decrypted"] = "decrypted"
    operation: DecryptionOperation
    page: Page
    password: str = ""
    attempted: int


class DecryptionRejected(FrozenModel):
    kind: Literal["rejected"] = "rejected"
    operation: DecryptionOperation
    code: DecryptionRejectionCode
    url: str
    attempted: int


class DecryptionFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    operation: DecryptionOperation
    code: DecryptionFailureCode
    url: str
    diagnostic: str


DecryptionOutcome = DecryptedResource | DecryptionRejected | DecryptionFailure


class BrowserPage(Protocol):
    async def goto(self, url: str, *, timeout: float) -> None: ...

    async def evaluate(self, expression: str, argument: object) -> object: ...

    async def close(self) -> None: ...


class BrowserContext(Protocol):
    async def new_page(self) -> BrowserPage: ...

    async def close(self) -> None: ...


class BrowserSession(Protocol):
    async def new_context(self) -> BrowserContext: ...

    async def close(self) -> None: ...

    async def terminate(self) -> None: ...


class BrowserFactory(Protocol):
    def __call__(self, *, proxy: str) -> Awaitable[BrowserSession]: ...


class DecryptionCapability(Protocol):
    async def decrypt_page(
        self,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome: ...

    async def decrypt_paste(
        self,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome: ...


class DecryptionOwner(DecryptionCapability, Protocol):
    async def __aenter__(self) -> DecryptionCapability: ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None: ...

    async def aclose(self) -> None: ...


class DecryptionFactory(Protocol):
    def __call__(
        self,
        *,
        proxy: str,
        timeout_s: float,
    ) -> DecryptionOwner: ...


class _BrowserEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["submitted", "missing_input", "missing_button"]
    text: str = ""
    html: str = ""


_PASSWORD_SCRIPT = """
async ({password, inputSelector, buttonSelector}) => {
    const input = document.querySelector(inputSelector);
    if (!input) return {status: "missing_input", text: "", html: ""};
    input.value = password;
    input.dispatchEvent(new Event("input", {bubbles: true}));
    input.dispatchEvent(new Event("change", {bubbles: true}));
    const button = document.querySelector(buttonSelector);
    if (!button) return {status: "missing_button", text: "", html: ""};
    button.click();
    await new Promise(resolve => setTimeout(resolve, 3000));
    return {
        status: "submitted",
        text: document.body?.innerText ?? "",
        html: document.documentElement?.outerHTML ?? "",
    };
}
"""


_PASTE_SCRIPT = """
async ({password, inputSelector, buttonText}) => {
    await new Promise(resolve => setTimeout(resolve, 3000));
    const input = document.querySelector(inputSelector);
    if (password && !input) {
        return {status: "missing_input", text: "", html: ""};
    }
    if (input) {
        input.value = password;
        input.dispatchEvent(new Event("input", {bubbles: true}));
        const buttons = Array.from(document.querySelectorAll("button"));
        const button = buttons.find(candidate =>
            candidate.textContent.trim().includes(buttonText)
        );
        if (!button) return {status: "missing_button", text: "", html: ""};
        button.click();
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
    const content = document.querySelector(
        "#cleartext, .highlight, pre, code, article"
    );
    return {
        status: "submitted",
        text: content?.textContent ?? document.body?.innerText ?? "",
        html: document.documentElement?.outerHTML ?? "",
    };
}
"""


class _PlaywrightPage:
    __slots__ = ("_page",)

    def __init__(self, page: PlaywrightPage) -> None:
        self._page = page

    async def goto(self, url: str, *, timeout: float) -> None:
        await self._page.goto(url, timeout=timeout, wait_until="domcontentloaded")

    async def evaluate(self, expression: str, argument: object) -> object:
        return await self._page.evaluate(expression, argument)

    async def close(self) -> None:
        await self._page.close()


class _PlaywrightContext:
    __slots__ = ("_context",)

    def __init__(self, context: PlaywrightContext) -> None:
        self._context = context

    async def new_page(self) -> BrowserPage:
        return _PlaywrightPage(await self._context.new_page())

    async def close(self) -> None:
        await self._context.close()


class _PlaywrightSession:
    __slots__ = ("_browser", "_runtime")

    def __init__(self, runtime: Playwright, browser: PlaywrightBrowser) -> None:
        self._runtime = runtime
        self._browser = browser

    async def new_context(self) -> BrowserContext:
        return _PlaywrightContext(
            await self._browser.new_context(ignore_https_errors=True)
        )

    async def close(self) -> None:
        await self._browser.close()
        await self._runtime.stop()

    async def terminate(self) -> None:
        await self._runtime.stop()


async def launch_browser(*, proxy: str) -> BrowserSession:
    runtime = await async_playwright().start()
    try:
        if proxy:
            browser = await runtime.chromium.launch(
                headless=True,
                proxy={"server": proxy},
            )
        else:
            browser = await runtime.chromium.launch(headless=True)
    except BaseException:
        await runtime.stop()
        raise
    return _PlaywrightSession(runtime, browser)


def create_decryption_client(
    *,
    proxy: str,
    timeout_s: float,
) -> DecryptionOwner:
    return DecryptionClient(proxy=proxy, timeout_s=timeout_s)


class DecryptionClient:
    """Reuse one browser process while isolating every target context."""

    __slots__ = (
        "_browser",
        "_browser_factory",
        "_close_timeout_s",
        "_closed",
        "_entered",
        "_proxy",
        "_timeout_s",
    )

    def __init__(
        self,
        *,
        proxy: str = "",
        timeout_s: float = 30.0,
        close_timeout_s: float = 5.0,
        browser_factory: BrowserFactory = launch_browser,
    ) -> None:
        if timeout_s <= 0 or close_timeout_s <= 0:
            raise ValueError("decryption deadlines must be positive")
        self._proxy = proxy
        self._timeout_s = timeout_s
        self._close_timeout_s = close_timeout_s
        self._browser_factory = browser_factory
        self._browser: BrowserSession | None = None
        self._entered = False
        self._closed = False

    async def __aenter__(self) -> DecryptionClient:
        if self._closed:
            raise RuntimeError("a closed DecryptionClient cannot be reopened")
        self._entered = True
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        browser = self._browser
        self._browser = None
        if browser is None:
            return
        try:
            await asyncio.wait_for(browser.close(), timeout=self._close_timeout_s)
        except asyncio.CancelledError:
            await browser.terminate()
            raise
        except Exception as error:
            logger.warning("Browser close failed; terminating: %s", error)
            try:
                await asyncio.wait_for(
                    browser.terminate(),
                    timeout=self._close_timeout_s,
                )
            except Exception as terminate_error:
                logger.error("Browser termination failed: %s", terminate_error)

    async def decrypt_page(
        self,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome:
        return await self._decrypt("password_page", url, candidates)

    async def decrypt_paste(
        self,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome:
        return await self._decrypt("paste", url, candidates)

    async def _decrypt(
        self,
        operation: DecryptionOperation,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome:
        if not self._entered or self._closed:
            return DecryptionFailure(
                operation=operation,
                code="not_open",
                url=url,
                diagnostic="DecryptionClient must be used as an async context manager",
            )

        context: BrowserContext | None = None
        page: BrowserPage | None = None
        try:
            async with asyncio.timeout(self._timeout_s):
                browser = await self._browser_session()
                context = await browser.new_context()
                page = await context.new_page()
                outcome = await self._attempt(operation, url, candidates, page)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            outcome = DecryptionFailure(
                operation=operation,
                code="timeout",
                url=url,
                diagnostic=f"{operation} exceeded its deadline",
            )
        except ValidationError as error:
            outcome = DecryptionFailure(
                operation=operation,
                code="malformed_output",
                url=url,
                diagnostic=f"invalid browser result at {error.errors()[0]['loc']}",
            )
        except Exception as error:
            outcome = DecryptionFailure(
                operation=operation,
                code="browser_error",
                url=url,
                diagnostic=str(error)[:200] or "browser operation failed",
            )
        finally:
            cleanup = await self._close_target(page, context)

        if cleanup:
            if outcome.kind == "failure":
                return outcome.model_copy(
                    update={"diagnostic": f"{outcome.diagnostic}; {cleanup}"}
                )
            return DecryptionFailure(
                operation=operation,
                code="cleanup_failed",
                url=url,
                diagnostic=cleanup,
            )
        return outcome

    async def _browser_session(self) -> BrowserSession:
        if self._browser is None:
            self._browser = await self._browser_factory(proxy=self._proxy)
        return self._browser

    async def _attempt(
        self,
        operation: DecryptionOperation,
        url: str,
        candidates: PasswordCandidates,
        page: BrowserPage,
    ) -> DecryptionOutcome:
        for attempted, password in enumerate(candidates.values, start=1):
            await page.goto(url, timeout=self._timeout_s * 1000)
            evaluation = await self._evaluate(page, operation, password)
            if evaluation.status == "missing_input":
                return DecryptionRejected(
                    operation=operation,
                    code="missing_input",
                    url=url,
                    attempted=attempted,
                )
            if evaluation.status == "missing_button":
                return DecryptionRejected(
                    operation=operation,
                    code="missing_button",
                    url=url,
                    attempted=attempted,
                )
            links = (
                extract_paste_links(evaluation.text + "\n" + evaluation.html)
                if operation == "paste"
                else ()
            )
            admitted = Page(
                url=url,
                markdown=evaluation.text,
                html=evaluation.html,
                links=links,
            )
            if admitted.has_subscription_content():
                return DecryptedResource(
                    operation=operation,
                    page=admitted,
                    password=password,
                    attempted=attempted,
                )
        return DecryptionRejected(
            operation=operation,
            code="no_subscription",
            url=url,
            attempted=len(candidates.values),
        )

    @staticmethod
    async def _evaluate(
        page: BrowserPage,
        operation: DecryptionOperation,
        password: str,
    ) -> _BrowserEvaluation:
        match operation:
            case "password_page":
                raw = await page.evaluate(
                    _PASSWORD_SCRIPT,
                    {
                        "password": password,
                        "inputSelector": ".cl-input",
                        "buttonSelector": ".cl-btn",
                    },
                )
            case "paste":
                raw = await page.evaluate(
                    _PASTE_SCRIPT,
                    {
                        "password": password,
                        "inputSelector": "#passworddecrypt",
                        "buttonText": "解密",
                    },
                )
        return _BrowserEvaluation.model_validate(raw)

    async def _close_target(
        self,
        page: BrowserPage | None,
        context: BrowserContext | None,
    ) -> str:
        failures: list[str] = []
        if page is not None:
            try:
                await asyncio.wait_for(page.close(), timeout=self._close_timeout_s)
            except Exception as error:
                failures.append(f"page close failed: {error}")
        if context is not None:
            try:
                await asyncio.wait_for(context.close(), timeout=self._close_timeout_s)
            except Exception as error:
                failures.append(f"context close failed: {error}")
        return "; ".join(failures)
