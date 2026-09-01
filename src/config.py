"""Strict admission for declarative crawler configuration."""

import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)


class FrozenModel(BaseModel):
    """Project model with strict, immutable, closed fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PasswordEvidence(FrozenModel):
    subtitles: str = ""
    description: str = ""


class PasswordCandidates(FrozenModel):
    values: tuple[str, ...] = Field(min_length=1, strict=False, repr=False)

    @field_validator("values")
    @classmethod
    def validate_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("password candidates must be unique")
        return values


class EmptyPasswordSource(FrozenModel):
    type: Literal["empty"] = "empty"
    limit: Literal[1] = 1


class SubtitlePasswordSource(FrozenModel):
    type: Literal["subtitles"] = "subtitles"
    limit: int = Field(default=5, gt=0)


class DescriptionPasswordSource(FrozenModel):
    type: Literal["description"] = "description"
    limit: int = Field(default=5, gt=0)


class AabbPasswordSource(FrozenModel):
    type: Literal["aabb"] = "aabb"
    limit: int = Field(default=20, gt=0, le=90)


class AbabPasswordSource(FrozenModel):
    type: Literal["abab"] = "abab"
    limit: int = Field(default=20, gt=0, le=90)


PasswordSource = Annotated[
    EmptyPasswordSource
    | SubtitlePasswordSource
    | DescriptionPasswordSource
    | AabbPasswordSource
    | AbabPasswordSource,
    Field(discriminator="type"),
]


class PasswordPolicy(FrozenModel):
    sources: tuple[PasswordSource, ...] = Field(min_length=1, strict=False)
    max_candidates: int = Field(gt=0, le=181)

    @model_validator(mode="after")
    def validate_capacity(self) -> "PasswordPolicy":
        source_types = tuple(source.type for source in self.sources)
        if len(source_types) != len(set(source_types)):
            raise ValueError("password source types must be unique")
        if sum(source.limit for source in self.sources) > self.max_candidates:
            raise ValueError("password source limits exceed the total policy bound")
        if not set(source_types).intersection({"empty", "aabb", "abab"}):
            raise ValueError("password policy requires one unconditional source")
        return self

    def resolve(self, evidence: PasswordEvidence) -> PasswordCandidates:
        admitted: list[str] = []
        seen: set[str] = set()
        for source in self.sources:
            values = self._source_values(source, evidence)[: source.limit]
            for value in values:
                if value in seen:
                    continue
                seen.add(value)
                admitted.append(value)
        return PasswordCandidates(values=tuple(admitted))

    @staticmethod
    def _source_values(
        source: PasswordSource,
        evidence: PasswordEvidence,
    ) -> tuple[str, ...]:
        match source.type:
            case "empty":
                return ("",)
            case "subtitles":
                return _four_digit_passwords(evidence.subtitles)
            case "description":
                return _four_digit_passwords(evidence.description)
            case "aabb":
                return tuple(
                    f"{first}{first}{second}{second}"
                    for first in range(10)
                    for second in range(10)
                    if first != second
                )
            case "abab":
                return tuple(
                    f"{first}{second}{first}{second}"
                    for first in range(10)
                    for second in range(10)
                    if first != second
                )


def _four_digit_passwords(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"(?<!\d)\d{4}(?!\d)", text)))


PASSWORD_PAGE_POLICY = PasswordPolicy(
    sources=(
        SubtitlePasswordSource(),
        DescriptionPasswordSource(),
        AabbPasswordSource(),
        AbabPasswordSource(),
    ),
    max_candidates=50,
)


PASTE_PASSWORD_POLICY = PasswordPolicy(
    sources=(
        EmptyPasswordSource(),
        SubtitlePasswordSource(),
        DescriptionPasswordSource(),
        AabbPasswordSource(),
        AbabPasswordSource(),
    ),
    max_candidates=51,
)


class SiteBase(FrozenModel):
    name: str
    start_url: str
    required: bool = False
    description: str = ""
    link_pattern: str | None = None
    exclude_patterns: tuple[str, ...] = Field(
        default=("category", "page-"),
        strict=False,
    )

    @field_validator("start_url")
    @classmethod
    def validate_start_url(cls, value: str) -> str:
        TypeAdapter(HttpUrl).validate_python(value)
        return value


class SimpleSite(SiteBase):
    type: Literal["simple"] = "simple"


class PasswordSite(SiteBase):
    type: Literal["yt_pwd", "youtube_password"] = "yt_pwd"
    password_policy: PasswordPolicy = PASSWORD_PAGE_POLICY
    paste_policy: PasswordPolicy = PASTE_PASSWORD_POLICY


class DriveSite(SiteBase):
    type: Literal["cloud_drive"] = "cloud_drive"
    password_policy: PasswordPolicy = PASTE_PASSWORD_POLICY


class GitHubSourceSite(FrozenModel):
    type: Literal["github"] = "github"
    name: str
    owner: str
    repository: str
    branch: str
    path: str
    required: bool = False
    description: str = ""

    @field_validator("name", "owner", "repository", "branch")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value or value != value.strip() or re.search(r"[/\\?#]", value):
            raise ValueError("must be one non-empty path component")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        parts = value.split("/")
        if (
            not value
            or value != value.strip()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("path must be a relative normalized GitHub file path")
        return value

    @property
    def raw_url(self) -> str:
        return self.raw_url_at(self.branch)

    def raw_url_at(self, revision: str) -> str:
        path = "/".join(quote(part, safe="") for part in self.path.split("/"))
        return (
            "https://raw.githubusercontent.com/"
            f"{quote(self.owner, safe='')}/{quote(self.repository, safe='')}/"
            f"{quote(revision, safe='')}/{path}"
        )

    @property
    def commits_api_url(self) -> str:
        return (
            "https://api.github.com/repos/"
            f"{quote(self.owner, safe='')}/{quote(self.repository, safe='')}/commits"
        )


CrawlerSite = SimpleSite | PasswordSite | DriveSite


Site = Annotated[
    CrawlerSite | GitHubSourceSite,
    Field(discriminator="type"),
]


class ProviderConfig(FrozenModel):
    name: str
    base_url: str
    api_key_env: str
    models: tuple[str, ...] = Field(strict=False)
    is_reasoning_model: bool = False
    default_weight: int = 10


class LLMConfig(FrozenModel):
    providers: tuple[ProviderConfig, ...] = Field(default=(), strict=False)
    task_routing: dict[str, dict[str, int]] = Field(default_factory=dict)
    max_requests_per_run: int = Field(default=30, gt=0)
    max_requests_per_site: int = Field(default=3, gt=0)


class CrawlConfig(FrozenModel):
    max_articles: int = Field(default=3, gt=0)
    timeout: int = Field(default=30, gt=0)
    concurrency: int = Field(default=3, gt=0)
    proxy: str = ""
    max_source_artifacts: int = Field(default=12, gt=0)
    max_source_bytes: int = Field(default=16 * 1024 * 1024, gt=0)
    max_run_source_bytes: int = Field(default=64 * 1024 * 1024, gt=0)


class OutputConfig(FrozenModel):
    dir: Path = Field(default=Path("nodes"), strict=False)


class RepositoryIdentity(FrozenModel):
    owner: str = "nostalume"
    repository: str = "FreeNodes"


class Config(FrozenModel):
    sites: tuple[Site, ...] = Field(strict=False)
    source_candidates: tuple[GitHubSourceSite, ...] = Field(default=(), strict=False)
    crawl: CrawlConfig = CrawlConfig()
    output: OutputConfig = OutputConfig()
    llm: LLMConfig = LLMConfig()
    repository: RepositoryIdentity = RepositoryIdentity()

    @model_validator(mode="after")
    def validate_source_names(self) -> "Config":
        names = tuple(site.name for site in (*self.sites, *self.source_candidates))
        if len(names) != len(set(names)):
            raise ValueError("source names must be unique")
        return self


def load_config(path: str | Path = "config.yaml") -> Config:
    """Decode YAML once and admit its complete shape."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(raw)
