import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import date

import httpx
import pytest

from freenodes.config import (
    AppConfig,
    DiscoveryLimits,
    WebSource,
)
from freenodes.llm import ExtractedLinks
from freenodes.web import (
    DownloadedText,
    DownloadFailure,
    DownloadOutcome,
    Page,
    PageLink,
    WebClient,
    admit_crawl_page,
)
from tests.discovery_support import (
    DRIVE_ID,
    NOW,
    PASSWORD_DOWNLOAD,
    SIMPLE_ARTICLE,
    SIMPLE_DOWNLOAD,
    SIMPLE_ROOT,
    DiscoveryHarness,
    FakeLinkCapability,
    FakeWebCapability,
    article_link,
    config,
    page,
    password_source,
    subscription,
    success_capabilities,
    web_source,
    youtube_source,
)


@pytest.mark.parametrize(
    ("site", "source_url"),
    (
        (web_source(), SIMPLE_DOWNLOAD),
        (password_source(), PASSWORD_DOWNLOAD),
        (youtube_source(), f"drive://{DRIVE_ID}/{DRIVE_ID}.txt"),
    ),
)
async def test_variant_yields_source_owned_artifact(site, source_url):
    capabilities = success_capabilities()

    outcome = await DiscoveryHarness(
        llm=capabilities.llm,
        youtube=capabilities.youtube,
        web=capabilities.web,
        decryption=capabilities.decryption,
        drive=capabilities.drive,
    ).run(site)

    assert outcome.kind == "success"
    assert outcome.site_name == site.name
    assert tuple(artifact.site for artifact in outcome.artifacts) == (site.name,)
    assert all(artifact.observed_at == NOW for artifact in outcome.artifacts)
    assert tuple(artifact.source_url for artifact in outcome.artifacts) == (source_url,)
    assert tuple(artifact.published_on for artifact in outcome.artifacts) == (
        date(2026, 8, 29),
    )


async def test_simple_inline_node_is_admitted_without_a_download():
    site = WebSource(name="inline-source", start_url=SIMPLE_ROOT)
    inline = subscription("inline.example", "inline")
    web = FakeWebCapability(
        pages={
            SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
            SIMPLE_ARTICLE: page(SIMPLE_ARTICLE, markdown="inline payload"),
        }
    )

    outcome = await DiscoveryHarness(
        llm=FakeLinkCapability(deque((ExtractedLinks(inline=(inline,)),))),
        web=web,
    ).run(site)

    assert outcome.kind == "success"
    assert tuple(artifact.content for artifact in outcome.artifacts) == (
        inline.encode(),
    )
    assert web.downloaded == []


async def test_simple_discovery_selects_newest_non_navigation_articles():
    root = "https://articles.test/blog/"
    newest = "https://articles.test/newest"
    middle = "https://articles.test/middle"
    older = "https://articles.test/older"
    newest_download = "https://files.test/newest.txt"
    middle_download = "https://files.test/middle.txt"
    site = WebSource(
        name="article-source",
        start_url=root,
        resource_pattern=r"https://files\.test/[a-z]+\.txt",
    )
    run_config = config(site).model_copy(
        update={
            "discovery": DiscoveryLimits(
                article_limit=2,
                source_concurrency=1,
            )
        }
    )
    web = FakeWebCapability(
        pages={
            root: page(
                root,
                links=(
                    PageLink(href="/older", text="8月27日 older"),
                    PageLink(href="/category/news", text="8月30日 navigation"),
                    PageLink(href="/middle", text="2026/8/29 middle"),
                    PageLink(href="/newest", text="8月30日 newest"),
                ),
            ),
            newest: page(newest, html=newest_download),
            middle: page(middle, html=middle_download),
        },
        downloads={
            newest_download: subscription("newest.example", "newest"),
            middle_download: subscription("middle.example", "middle"),
        },
    )

    outcome = await DiscoveryHarness(web=web).run(site, limits=run_config.discovery)

    assert outcome.kind == "success"
    assert web.fetched == [root, newest, middle]
    assert older not in web.fetched
    assert tuple(artifact.source_url for artifact in outcome.artifacts) == (
        newest_download,
        middle_download,
    )
    assert tuple(artifact.published_on for artifact in outcome.artifacts) == (
        date(2026, 8, 30),
        date(2026, 8, 29),
    )


async def test_simple_discovery_uses_dated_markdown_when_links_are_absent():
    root = "https://markdown.test/blog/"
    article = "https://markdown.test/2026/08/29/nodes"
    download = "https://files.test/markdown.txt"
    site = WebSource(
        name="markdown-source",
        start_url=root,
        resource_pattern=r"https://files\.test/markdown\.txt",
    )
    web = FakeWebCapability(
        pages={
            root: page(root, markdown=f"## [8月29日 nodes]({article})"),
            article: page(article, html=download),
        },
        downloads={download: subscription("markdown.example", "markdown")},
    )

    outcome = await DiscoveryHarness(web=web).run(site)

    assert outcome.kind == "success"
    assert web.fetched == [root, article]
    assert tuple(artifact.source_url for artifact in outcome.artifacts) == (download,)


async def test_simple_discovery_retries_failed_downloads(monkeypatch):
    download = "https://files.test/unavailable.txt"
    site = WebSource(
        name="retry-source",
        start_url=SIMPLE_ROOT,
        resource_pattern=r"https://files\.test/unavailable\.txt",
    )

    @dataclass
    class FailingWeb(FakeWebCapability):
        attempts: int = 0

        async def download_file(self, url: str) -> DownloadOutcome:
            self.attempts += 1
            return DownloadFailure(
                code="http_error",
                url=url,
                diagnostic="upstream unavailable",
                retryable=True,
            )

    async def no_wait(delay: float) -> None:
        return None

    monkeypatch.setattr("freenodes.discovery.asyncio.sleep", no_wait)
    web = FailingWeb(
        pages={
            SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
            SIMPLE_ARTICLE: page(SIMPLE_ARTICLE, html=download),
        }
    )

    outcome = await DiscoveryHarness(web=web).run(site)

    assert outcome.kind == "failure"
    assert web.attempts == 3
    assert any("upstream unavailable" in error for error in outcome.errors)


async def test_simple_discovery_does_not_retry_terminal_download_failure():
    download = "https://files.test/forbidden.txt"
    site = WebSource(
        name="terminal-source",
        start_url=SIMPLE_ROOT,
        resource_pattern=r"https://files\.test/forbidden\.txt",
    )

    @dataclass
    class TerminalWeb(FakeWebCapability):
        attempts: int = 0

        async def download_file(self, url: str) -> DownloadOutcome:
            self.attempts += 1
            return DownloadFailure(
                code="http_error",
                url=url,
                diagnostic="download returned HTTP 403",
                retryable=False,
            )

    web = TerminalWeb(
        pages={
            SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
            SIMPLE_ARTICLE: page(SIMPLE_ARTICLE, html=download),
        }
    )

    outcome = await DiscoveryHarness(web=web).run(site)

    assert outcome.kind == "failure"
    assert web.attempts == 1


async def test_simple_discovery_stops_network_work_at_artifact_limit():
    downloads = tuple(f"https://files.test/{index}.txt" for index in range(3))
    site = WebSource(
        name="bounded-source",
        start_url=SIMPLE_ROOT,
        resource_pattern=r"https://files\.test/\d+\.txt",
    )
    run_config = AppConfig(
        sources=(site,),
        discovery=DiscoveryLimits(
            article_limit=1,
            artifact_limit_per_source=1,
        ),
    )
    web = FakeWebCapability(
        pages={
            SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
            SIMPLE_ARTICLE: page(SIMPLE_ARTICLE, html=" ".join(downloads)),
        },
        downloads={
            url: subscription("bounded.example", str(index))
            for index, url in enumerate(downloads)
        },
    )

    outcome = await DiscoveryHarness(web=web).run(site, limits=run_config.discovery)

    assert outcome.kind == "success"
    assert web.downloaded == [downloads[0]]
    assert len(outcome.artifacts) == 1


URL = "https://files.test/nodes.txt"


def test_crawl_page_admits_library_owned_link_metadata():
    result = {
        "success": True,
        "html": "<a href='/daily'>Daily</a>",
        "markdown": {"raw_markdown": "[Daily](/daily)"},
        "links": {
            "internal": [
                {
                    "href": "https://source.test/daily",
                    "text": "Daily",
                    "title": "library-owned title",
                    "base_domain": "source.test",
                }
            ],
            "external": [],
        },
    }

    page = admit_crawl_page("https://source.test", result)

    assert page == Page(
        url="https://source.test",
        markdown="[Daily](/daily)",
        html="<a href='/daily'>Daily</a>",
        links=(PageLink(href="https://source.test/daily", text="Daily"),),
    )


async def test_download_admits_text_with_observed_byte_count():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content="节点".encode())
    )

    outcome = await WebClient().download_file(URL, transport=transport)

    assert outcome == DownloadedText(url=URL, content="节点", byte_count=6)


async def test_download_http_failure_is_explicit():
    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="missing"))

    outcome = await WebClient().download_file(URL, transport=transport)

    assert outcome.kind == "failure"
    assert outcome.code == "http_error"
    assert outcome.url == URL
    assert outcome.retryable is False


async def test_download_marks_transient_http_status_for_retry():
    transport = httpx.MockTransport(lambda request: httpx.Response(503))

    outcome = await WebClient().download_file(URL, transport=transport)

    assert outcome.kind == "failure"
    assert outcome.retryable is True


async def test_download_rejects_empty_success_body():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b""))

    outcome = await WebClient().download_file(URL, transport=transport)

    assert outcome.kind == "failure"
    assert outcome.code == "empty"
    assert outcome.url == URL
    assert outcome.retryable is False


async def test_download_limit_is_enforced_while_reading():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"12345")
    )

    outcome = await WebClient().download_file(
        URL,
        max_bytes=4,
        transport=transport,
    )

    assert outcome.kind == "failure"
    assert outcome.code == "oversize"


async def test_download_cancellation_propagates():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await WebClient().download_file(
            URL,
            transport=httpx.MockTransport(handler),
        )
