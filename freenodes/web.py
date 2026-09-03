from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Annotated, Literal, Protocol
from urllib.parse import urljoin, urlparse

import httpx
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from pydantic import BaseModel, ConfigDict, Field

from freenodes.config import FrozenModel, WebSourceBase


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

    def pattern_links(self, pattern: str) -> tuple[str, ...]:
        compiled = re.compile(pattern, re.IGNORECASE)
        values = (
            match.group(1) if compiled.groups else match.group(0)
            for match in compiled.finditer(self.html)
        )
        return tuple(
            re.sub(r'[),;.\'"]+$', "", value)
            for value in values
            if value and value.startswith("http")
        )


class Article(FrozenModel):
    url: str
    date: str
    text: str


class ArticleSelector:
    def __init__(
        self,
        source: WebSourceBase,
        *,
        limit: int,
        observed_on: date,
    ):
        self.source = source
        self.limit = limit
        self.observed_on = observed_on
        parsed = urlparse(source.start_url)
        self.base = f"{parsed.scheme}://{parsed.netloc}"

    def select(self, page: Page) -> tuple[Article, ...]:
        linked = self._from_links(page)
        return linked or self._from_markdown(page.markdown)

    def _from_links(self, page: Page) -> tuple[Article, ...]:
        articles = (
            Article(
                url=urljoin(self.base, link.href),
                date=published,
                text=link.text[:80],
            )
            for link in page.links
            if (published := self._date(link.text, link.href))
            and link.href not in {"/free-nodes/", "/", ""}
            and not self.source.excludes_article(link.href)
        )
        return self._ordered(articles)

    def _from_markdown(self, markdown: str) -> tuple[Article, ...]:
        articles: list[Article] = []
        for match in re.finditer(
            r"^## \[(.+?)\]\((https?://[^\s)]+)\)", markdown, re.MULTILINE
        ):
            text = match.group(1)
            url = match.group(2).rstrip(".,;)")
            if published := self._date(text, url):
                articles.append(Article(url=url, date=published, text=text[:80]))
        return self._ordered(articles)

    def _ordered(self, articles: Iterable[Article]) -> tuple[Article, ...]:
        ordered = sorted(articles, key=lambda value: value.date, reverse=True)
        unique = {article.url: article for article in ordered}
        return tuple(unique.values())[: self.limit]

    def _date(self, text: str, href: str) -> str | None:
        match = re.search(r"(\d{1,2})月(\d{1,2})日", text)
        if match:
            parsed = date(
                self.observed_on.year,
                int(match.group(1)),
                int(match.group(2)),
            )
            if (parsed - self.observed_on).days > 30:
                parsed = parsed.replace(year=self.observed_on.year - 1)
            return parsed.isoformat()
        for pattern, value in (
            (r"(\d{4})年(\d{1,2})月(\d{1,2})日", text),
            (r"(\d{4})/(\d{1,2})/(\d{1,2})", text),
            (r"(\d{4})-(\d{1,2})-(\d{1,2})", href),
        ):
            if match := re.search(pattern, value):
                return (
                    f"{match.group(1)}-{int(match.group(2)):02d}-"
                    f"{int(match.group(3)):02d}"
                )
        if match := re.search(r"/(\d{8})[/-]", href):
            raw = match.group(1)
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        return None


class DownloadedText(FrozenModel):
    kind: Literal["downloaded"] = "downloaded"
    url: str
    content: str = Field(repr=False)
    byte_count: int = Field(ge=0)


class DownloadFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    code: Literal["http_error", "oversize", "empty"]
    url: str
    diagnostic: str
    retryable: bool = False


DownloadOutcome = Annotated[
    DownloadedText | DownloadFailure,
    Field(discriminator="kind"),
]


class WebCapability(Protocol):
    async def fetch_page(self, url: str, timeout_ms: int = 60000) -> Page: ...

    async def download_file(
        self,
        url: str,
        *,
        max_bytes: int = 4 * 1024 * 1024,
    ) -> DownloadOutcome: ...


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
            async with (
                httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0, connect=15.0, read=60.0),
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_keepalive_connections=5, max_connections=10
                    ),
                    transport=transport,
                ) as client,
                client.stream("GET", url) as response,
            ):
                if response.is_error:
                    return DownloadFailure(
                        code="http_error",
                        url=url,
                        diagnostic=(f"download returned HTTP {response.status_code}"),
                        retryable=(
                            response.status_code in {408, 429}
                            or response.status_code >= 500
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
            if not body.strip():
                return DownloadFailure(
                    code="empty",
                    url=url,
                    diagnostic="download returned an empty body",
                )
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
                retryable=True,
            )
