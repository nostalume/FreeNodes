import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

import yaml
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _path_component(value: str) -> str:
    if not value or value != value.strip() or re.search(r"[/\\?#]", value):
        raise ValueError("must be one non-empty path component")
    return value


PathComponent = Annotated[str, AfterValidator(_path_component)]


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


class URLSource(FrozenModel):
    name: str
    start_url: str

    @field_validator("start_url")
    @classmethod
    def validate_start_url(cls, value: str) -> str:
        TypeAdapter(HttpUrl).validate_python(value)
        return value

    def excludes_article(self, url: str) -> bool:
        return False


class WebSourceBase(URLSource):
    resource_pattern: str | None = None
    article_exclusions: tuple[str, ...] = Field(
        default=("category", "page-"),
        strict=False,
    )

    def excludes_article(self, url: str) -> bool:
        return any(pattern in url for pattern in self.article_exclusions)


class WebSource(WebSourceBase):
    kind: Literal["web"] = "web"


class PasswordPageSource(WebSourceBase):
    kind: Literal["password_page"] = "password_page"
    password_policy: PasswordPolicy
    paste_policy: PasswordPolicy


class YouTubeResourceSource(URLSource):
    kind: Literal["youtube_resources"] = "youtube_resources"
    password_policy: PasswordPolicy


class GitHubFileSource(FrozenModel):
    kind: Literal["github_file"] = "github_file"
    name: PathComponent
    owner: PathComponent
    repository: PathComponent
    branch: PathComponent
    path: str

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

    @property
    def authority(self) -> str:
        return f"github:{self.owner.casefold()}/{self.repository.casefold()}"


DiscoverySource = WebSource | PasswordPageSource | YouTubeResourceSource


Source = Annotated[
    DiscoverySource | GitHubFileSource,
    Field(discriminator="kind"),
]


class OpenRouterLimits(FrozenModel):
    request_limit_per_run: int = Field(default=30, gt=0)
    request_limit_per_source: int = Field(default=3, gt=0)
    request_timeout_seconds: int = Field(default=20, gt=0)


class DiscoveryLimits(FrozenModel):
    source_concurrency: int = Field(default=3, gt=0)
    article_limit: int = Field(default=3, gt=0)
    request_timeout_seconds: int = Field(default=30, gt=0)
    proxy_url: str | None = None
    artifact_limit_per_source: int = Field(default=12, gt=0)
    byte_limit_per_source: int = Field(default=16 * 1024 * 1024, gt=0)
    byte_limit_per_run: int = Field(default=64 * 1024 * 1024, gt=0)

    @field_validator("proxy_url")
    @classmethod
    def validate_proxy_url(cls, value: str | None) -> str | None:
        if value is not None:
            TypeAdapter(AnyUrl).validate_python(value)
        return value


class PublicationPolicy(FrozenModel):
    stale_after_hours: int = Field(default=24, gt=0, strict=True)
    expires_after_hours: int = Field(default=48, gt=0, strict=True)
    node_limit: int = Field(default=500, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_freshness_window(self) -> "PublicationPolicy":
        if self.expires_after_hours <= self.stale_after_hours:
            raise ValueError("expiry must be later than stale admission")
        return self


class RepositoryIdentity(FrozenModel):
    owner: PathComponent = "nostalume"
    name: PathComponent = "FreeNodes"


class AppConfig(FrozenModel):
    sources: tuple[Source, ...] = Field(strict=False)
    audit_sources: tuple[GitHubFileSource, ...] = Field(default=(), strict=False)
    discovery: DiscoveryLimits = DiscoveryLimits()
    openrouter: OpenRouterLimits = OpenRouterLimits()
    publication: PublicationPolicy = PublicationPolicy()
    repository: RepositoryIdentity = RepositoryIdentity()

    @model_validator(mode="after")
    def validate_source_names(self) -> "AppConfig":
        all_sources = (*self.sources, *self.audit_sources)
        names = tuple(source.name for source in all_sources)
        if len(names) != len(set(names)):
            raise ValueError("source names must be unique")
        github_paths = tuple(
            (source.authority, source.branch, source.path)
            for source in all_sources
            if source.kind == "github_file"
        )
        if len(github_paths) != len(set(github_paths)):
            raise ValueError("GitHub source paths must be unique")
        return self


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
