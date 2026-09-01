"""Behavior of the path-specific GitHub subscription boundary."""

import asyncio
from datetime import UTC, date, datetime

import httpx
import pytest

from src.config import GitHubSourceSite
from src.crawler import DownloadedText, DownloadFailure, DownloadOutcome
from src.github_source import GitHubCommitClient, GitHubSourceClient

SHA = "a" * 40
NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


def site(path: str = "mihomo.yaml") -> GitHubSourceSite:
    return GitHubSourceSite(
        name="candidate",
        owner="upstream",
        repository="subscriptions",
        branch="main",
        path=path,
    )


def commit_response(
    request: httpx.Request,
    *,
    committed_at: str = "2026-08-29T08:30:00Z",
) -> httpx.Response:
    assert request.url.params["sha"] == "main"
    assert request.url.params["path"] in {"mihomo.yaml", "base64.txt"}
    assert request.url.params["per_page"] == "1"
    return httpx.Response(
        200,
        json=[{"sha": SHA, "commit": {"committer": {"date": committed_at}}}],
    )


class StubWeb:
    def __init__(self, outcome: DownloadOutcome | None = None):
        self.outcome = outcome
        self.request: tuple[str, int] | None = None

    async def download_file(
        self,
        url: str,
        *,
        max_bytes: int = 4 * 1024 * 1024,
    ) -> DownloadOutcome:
        self.request = (url, max_bytes)
        return self.outcome or DownloadedText(
            url=url,
            content="proxies: []",
            byte_count=11,
        )


@pytest.mark.parametrize(
    ("path", "content", "txt_count", "yaml_count"),
    (("mihomo.yaml", "proxies: []", 0, 1), ("base64.txt", "dm1lc3M6Ly8=", 1, 0)),
)
async def test_discovery_binds_content_to_path_commit(
    path: str,
    content: str,
    txt_count: int,
    yaml_count: int,
):
    source = site(path)
    immutable_url = source.raw_url_at(SHA)
    web = StubWeb(
        DownloadedText(
            url=immutable_url,
            content=content,
            byte_count=len(content.encode()),
        )
    )
    client = GitHubSourceClient(
        web,
        commits=GitHubCommitClient(httpx.MockTransport(commit_response)),
        clock=lambda: NOW,
    )

    outcome = await client.discover(source)

    assert outcome.kind == "success"
    assert (outcome.txt_count, outcome.yaml_count) == (txt_count, yaml_count)
    assert web.request == (immutable_url, 4 * 1024 * 1024)
    assert outcome.artifacts[0].source_url == immutable_url
    assert outcome.artifacts[0].published_on == date(2026, 8, 29)
    assert outcome.artifacts[0].observed_at == NOW


@pytest.mark.parametrize(
    ("response", "error"),
    (
        (httpx.Response(429, text="secret-token"), "HTTP 429"),
        (httpx.Response(200, json=[]), "no matching commit"),
        (httpx.Response(200, json=[{"sha": "bad"}]), "response is invalid"),
        (
            httpx.Response(
                200,
                json=[
                    {
                        "sha": SHA,
                        "commit": {"committer": {"date": "2026-08-29T08:30:00"}},
                    }
                ],
            ),
            "response is invalid",
        ),
        (
            httpx.Response(
                200,
                json=[
                    {
                        "sha": SHA,
                        "commit": {"committer": {"date": "2026-09-01T00:00:00Z"}},
                    }
                ],
            ),
            "later than observation",
        ),
    ),
)
async def test_commit_boundary_returns_safe_typed_failure(
    response: httpx.Response,
    error: str,
):
    web = StubWeb()
    client = GitHubSourceClient(
        web,
        commits=GitHubCommitClient(httpx.MockTransport(lambda request: response)),
        clock=lambda: NOW,
    )

    outcome = await client.discover(site())

    assert outcome.kind == "failure"
    assert error in outcome.errors[0]
    assert "secret-token" not in outcome.errors[0]
    assert web.request is None


@pytest.mark.parametrize("code", ("http_error", "oversize", "empty"))
async def test_download_failure_stops_admission(code: str):
    source = site()
    web = StubWeb(
        DownloadFailure(
            code=code,
            url=source.raw_url_at(SHA),
            diagnostic=f"{code} download",
        )
    )
    client = GitHubSourceClient(
        web,
        commits=GitHubCommitClient(httpx.MockTransport(commit_response)),
        clock=lambda: NOW,
    )

    outcome = await client.discover(source)

    assert outcome.kind == "failure"
    assert outcome.errors == (f"{code} download",)


async def test_download_must_return_the_commit_pinned_identity():
    client = GitHubSourceClient(
        StubWeb(
            DownloadedText(url=site().raw_url, content="proxies: []", byte_count=11)
        ),
        commits=GitHubCommitClient(httpx.MockTransport(commit_response)),
        clock=lambda: NOW,
    )

    outcome = await client.discover(site())

    assert outcome.kind == "failure"
    assert "identity" in outcome.errors[0]


async def test_commit_request_cancellation_propagates():
    async def cancel(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client = GitHubSourceClient(
        StubWeb(),
        commits=GitHubCommitClient(httpx.MockTransport(cancel)),
        clock=lambda: NOW,
    )

    with pytest.raises(asyncio.CancelledError):
        await client.discover(site())
