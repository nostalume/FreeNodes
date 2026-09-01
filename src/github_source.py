"""Typed admission of immutable subscription files hosted by GitHub."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Literal

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
)

from src.config import FrozenModel, GitHubSourceSite
from src.crawler import WebCapability
from src.nodes import PublishedInstant, SourceArtifact
from src.site_processor import DiscoveryFailure, DiscoveryOutcome, DiscoverySuccess


class GitHubCommit(FrozenModel):
    kind: Literal["commit"] = "commit"
    sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    committed_at: AwareDatetime


class GitHubCommitFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    diagnostic: str


GitHubCommitOutcome = Annotated[
    GitHubCommit | GitHubCommitFailure,
    Field(discriminator="kind"),
]


class _GitHubExternalModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _GitHubCommitter(_GitHubExternalModel):
    date: AwareDatetime


class _GitHubCommitBody(_GitHubExternalModel):
    committer: _GitHubCommitter


class _GitHubCommitItem(_GitHubExternalModel):
    sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    commit: _GitHubCommitBody


class _GitHubCommitList(RootModel[tuple[_GitHubCommitItem, ...]]):
    pass


class GitHubCommitClient:
    """Read one path-specific commit from GitHub's public REST boundary."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    async def latest(
        self,
        site: GitHubSourceSite,
        *,
        observed_at: datetime,
    ) -> GitHubCommitOutcome:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(20.0, connect=10.0),
                transport=self.transport,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "FreeNodes-source-audit",
                    "X-GitHub-Api-Version": "2026-03-10",
                },
            ) as client:
                response = await client.get(
                    site.commits_api_url,
                    params={"sha": site.branch, "path": site.path, "per_page": 1},
                )
        except httpx.HTTPError:
            return GitHubCommitFailure(diagnostic="GitHub commits request failed")
        if response.is_error:
            return GitHubCommitFailure(
                diagnostic=f"GitHub commits request returned HTTP {response.status_code}"
            )
        try:
            commits = _GitHubCommitList.model_validate_json(response.content).root
        except (ValidationError, ValueError):
            return GitHubCommitFailure(diagnostic="GitHub commits response is invalid")
        if not commits:
            return GitHubCommitFailure(
                diagnostic="GitHub commits response contains no matching commit"
            )
        commit = commits[0]
        if commit.commit.committer.date > observed_at:
            return GitHubCommitFailure(
                diagnostic="GitHub commit time is later than observation time"
            )
        return GitHubCommit(
            sha=commit.sha,
            committed_at=commit.commit.committer.date,
        )


class GitHubSourceClient:
    """Bind immutable raw bytes to their path-specific upstream commit."""

    def __init__(
        self,
        web: WebCapability,
        *,
        commits: GitHubCommitClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.web = web
        self.commits = commits or GitHubCommitClient()
        self.clock = clock

    async def discover(
        self,
        site: GitHubSourceSite,
        *,
        observed_at: datetime | None = None,
    ) -> DiscoveryOutcome:
        observed_at = observed_at or self.clock()
        commit = await self.commits.latest(site, observed_at=observed_at)
        if commit.kind == "failure":
            return DiscoveryFailure(site_name=site.name, errors=(commit.diagnostic,))
        source_url = site.raw_url_at(commit.sha)
        downloaded = await self.web.download_file(source_url, max_bytes=4 * 1024 * 1024)
        if downloaded.kind == "failure":
            return DiscoveryFailure(
                site_name=site.name, errors=(downloaded.diagnostic,)
            )
        if downloaded.url != source_url:
            return DiscoveryFailure(
                site_name=site.name,
                errors=("download identity does not match the committed source",),
            )
        yaml_count = int(site.path.lower().endswith((".yaml", ".yml")))
        artifact = SourceArtifact(
            authority=site.authority,
            site=site.name,
            source_url=source_url,
            content=downloaded.content,
            observed_at=observed_at,
            publication_time=PublishedInstant(at=commit.committed_at),
            media_type="application/yaml" if yaml_count else "text/plain",
        )
        return DiscoverySuccess(
            site_name=site.name,
            txt_count=1 - yaml_count,
            yaml_count=yaml_count,
            total_bytes=downloaded.byte_count,
            artifacts=(artifact,),
        )
