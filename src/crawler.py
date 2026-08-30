"""Typed Crawl4AI boundary plus direct file downloads."""

import re
from typing import Annotated, Literal, Protocol

import httpx
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from pydantic import BaseModel, ConfigDict, Field

from src.config import FrozenModel


class PageLink(FrozenModel):
    href: str
    text: str = ""


class Page(FrozenModel):
    url: str
    markdown: str
    html: str
    links: tuple[PageLink, ...] = Field(default=(), strict=False)
    success: bool = True
    error: str = ""

    def has_subscription_content(self) -> bool:
        """Recognize actual subscription resources, not descriptive keywords."""
        content = f"{self.markdown}\n{self.html}"
        patterns = (
            r'https?://[^"\'<\s]+\.(?:txt|yaml)',
            r"(?:vmess|vless|trojan|ss|ssr)://[a-zA-Z0-9+/=:@.#-]+",
        )
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)

    def requires_password(self) -> bool:
        indicators = (
            'class="cl-input"',
            'placeholder="在此输入密码"',
            'class="cl-btn"',
            'type="password"',
            'input[type="text"][class*="cl-input"]',
        )
        lowered = self.html.lower()
        return any(indicator.lower() in lowered for indicator in indicators)


class DownloadedText(FrozenModel):
    kind: Literal["downloaded"] = "downloaded"
    url: str
    content: str = Field(repr=False)
    byte_count: int = Field(ge=0)


class DownloadFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    code: Literal["http_error", "oversize"]
    url: str
    diagnostic: str


DownloadOutcome = Annotated[
    DownloadedText | DownloadFailure,
    Field(discriminator="kind"),
]


class WebCapability(Protocol):
    async def fetch_page(self, url: str, timeout_ms: int = 60000) -> Page: ...

    async def download_file(self, url: str) -> DownloadOutcome: ...


class _ExternalModel(BaseModel):
    """Permissive adapter for fields owned by Crawl4AI."""

    model_config = ConfigDict(extra="ignore", frozen=True, from_attributes=True)


class _CrawlMarkdown(_ExternalModel):
    raw_markdown: str = ""


class _CrawlLink(_ExternalModel):
    href: str
    text: str = ""


class _CrawlLinks(_ExternalModel):
    internal: tuple[_CrawlLink, ...] = ()
    external: tuple[_CrawlLink, ...] = ()


class _CrawlResult(_ExternalModel):
    success: bool
    error_message: str = ""
    markdown: _CrawlMarkdown | None = None
    html: str = ""
    links: _CrawlLinks = _CrawlLinks()


def admit_crawl_page(url: str, result: object) -> Page:
    """Convert one Crawl4AI result into the project-owned page model."""
    crawled = _CrawlResult.model_validate(result)
    if not crawled.success:
        return Page(
            url=url,
            success=False,
            error=crawled.error_message,
            markdown="",
            html="",
        )

    links = tuple(
        PageLink(href=link.href, text=link.text[:200])
        for link in crawled.links.internal + crawled.links.external
        if link.href and not link.href.startswith("javascript:")
    )
    markdown = crawled.markdown.raw_markdown if crawled.markdown else ""
    return Page(url=url, markdown=markdown, html=crawled.html, links=links)


class WebClient:
    """Own page-fetch and bounded textual-download policy."""

    async def fetch_page(self, url: str, timeout_ms: int = 60000) -> Page:
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        page_timeout=timeout_ms,
                    ),
                )
            return admit_crawl_page(url, result)
        except Exception as error:
            return Page(
                url=url,
                success=False,
                error=str(error),
                markdown="",
                html="",
            )

    async def download_file(
        self,
        url: str,
        *,
        max_bytes: int = 4 * 1024 * 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> DownloadOutcome:
        """Stream one textual resource through an explicit byte ceiling."""
        if max_bytes <= 0:
            raise ValueError("download limit must be positive")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=15.0, read=60.0),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                transport=transport,
            ) as client:
                async with client.stream("GET", url) as response:
                    if response.is_error:
                        return DownloadFailure(
                            code="http_error",
                            url=url,
                            diagnostic=(
                                f"download returned HTTP {response.status_code}"
                            ),
                        )
                    declared = response.headers.get("content-length")
                    if declared:
                        try:
                            if int(declared) > max_bytes:
                                return DownloadFailure(
                                    code="oversize",
                                    url=url,
                                    diagnostic="download exceeds the byte limit",
                                )
                        except ValueError:
                            pass
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes():
                        received += len(chunk)
                        if received > max_bytes:
                            return DownloadFailure(
                                code="oversize",
                                url=url,
                                diagnostic="download exceeds the byte limit",
                            )
                        chunks.append(chunk)
            body = b"".join(chunks)
            return DownloadedText(
                url=url,
                content=body.decode("utf-8", errors="replace"),
                byte_count=len(body),
            )
        except httpx.HTTPError as error:
            return DownloadFailure(
                code="http_error",
                url=url,
                diagnostic=str(error)[:200] or "download failed",
            )
