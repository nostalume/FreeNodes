"""Bounded typed direct-download boundary."""

import asyncio

import httpx
import pytest

from src.crawler import DownloadedText, WebClient

URL = "https://files.test/nodes.txt"


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
