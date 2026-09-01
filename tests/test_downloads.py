"""Bounded typed direct-download boundary."""

import asyncio

import httpx
import pytest

from src.crawler import DownloadedText, Page, PageLink, WebClient, admit_crawl_page

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
