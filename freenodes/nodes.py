from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from itertools import chain
from typing import Annotated, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import AwareDatetime, Field, TypeAdapter, field_validator, model_validator

from freenodes.config import FrozenModel
from freenodes.proxies import (
    Proxy,
    decode_proxies,
    is_global_endpoint,
)

BYTES_ADAPTER = TypeAdapter(bytes)


class PublishedInstant(FrozenModel):
    kind: Literal["instant"] = "instant"
    at: AwareDatetime


class PublishedDate(FrozenModel):
    kind: Literal["date"] = "date"
    on: date


class UnknownPublicationTime(FrozenModel):
    kind: Literal["unknown"] = "unknown"


PublicationTime = Annotated[
    PublishedInstant | PublishedDate | UnknownPublicationTime,
    Field(discriminator="kind"),
]


def _publication_date(value: PublicationTime) -> date | None:
    match value:
        case PublishedInstant(at=published_at):
            return published_at.date()
        case PublishedDate(on=published_on):
            return published_on
        case UnknownPublicationTime():
            return None


class SourceArtifact(FrozenModel):
    authority: str = ""
    site: str
    source_url: str
    content: bytes = Field(repr=False)
    observed_at: AwareDatetime
    publication_time: PublicationTime = UnknownPublicationTime()
    media_type: str | None = None

    @field_validator("site", "source_url")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("content", mode="before")
    @classmethod
    def encode_content(cls, value: bytes | str) -> bytes:
        return BYTES_ADAPTER.validate_python(value)

    @property
    def authority_id(self) -> str:
        return self.authority or self.site

    @property
    def published_on(self) -> date | None:
        return _publication_date(self.publication_time)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @classmethod
    def inline(
        cls,
        *,
        site: str,
        content: str,
        observed_at: datetime,
        published_on: date | None = None,
        published_at: datetime | None = None,
        authority: str = "",
        media_type: str = "text/plain",
    ) -> SourceArtifact:
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        if published_at is not None:
            publication_time: PublicationTime = PublishedInstant(at=published_at)
        elif published_on is not None:
            publication_time = PublishedDate(on=published_on)
        else:
            publication_time = UnknownPublicationTime()
        return cls(
            authority=authority,
            site=site,
            source_url=f"inline://{site}/{digest}",
            content=encoded,
            observed_at=observed_at,
            publication_time=publication_time,
            media_type=media_type,
        )


class NodeProvenance(FrozenModel):
    authority: str
    site: str
    source_url: str
    observed_at: AwareDatetime
    publication_time: PublicationTime = UnknownPublicationTime()
    artifact_digest: str
    item_index: int = Field(ge=0)

    @property
    def published_on(self) -> date | None:
        return _publication_date(self.publication_time)


class NodeBase(FrozenModel):
    fingerprint: str
    display_name: str
    provenance: tuple[NodeProvenance, ...] = Field(default=(), strict=False)


class ClashNode(NodeBase):
    kind: Literal["clash"] = "clash"
    proxy: Proxy = Field(repr=False)


class UriNode(NodeBase):
    kind: Literal["uri"] = "uri"
    uri: str = Field(repr=False)


class DualNode(NodeBase):
    kind: Literal["dual"] = "dual"
    proxy: Proxy = Field(repr=False)
    uri: str = Field(repr=False)


Node = Annotated[ClashNode | UriNode | DualNode, Field(discriminator="kind")]
ProbeableNode = ClashNode | DualNode
UriCapableNode = UriNode | DualNode
NodeT = TypeVar("NodeT", bound=ClashNode | UriNode | DualNode)


class Rejection(FrozenModel):
    code: str
    site: str
    source_url: str
    item_index: int | None = None
    message: str = ""


class SourceReceipt(FrozenModel):
    authority: str
    site: str
    source_url: str
    artifact_digest: str
    observed_at: AwareDatetime
    publication_time: PublicationTime = UnknownPublicationTime()
    freshness: Literal["current", "stale", "expired", "future", "unknown"]

    @property
    def published_on(self) -> date | None:
        return _publication_date(self.publication_time)


class CodeCount(FrozenModel):
    code: str
    count: int = Field(ge=0, strict=True)


class SourceAdmissionSummary(FrozenModel):
    source: str
    authority: str
    status: Literal["available", "empty", "failed"]
    artifacts: int = Field(ge=0, strict=True)
    candidate_records: int = Field(ge=0, strict=True)
    unique_eligible: int = Field(ge=0, strict=True)
    rejection_codes: tuple[CodeCount, ...] = Field(default=(), strict=False)


class AdmissionCounts(FrozenModel):
    attempted_sources: int = Field(ge=0, strict=True)
    failed_sources: int = Field(ge=0, strict=True)
    empty_sources: int = Field(ge=0, strict=True)
    sources_with_artifacts: int = Field(ge=0, strict=True)
    discovered_artifacts: int = Field(ge=0, strict=True)
    rejected_artifacts: int = Field(ge=0, strict=True)
    decoded_artifacts: int = Field(ge=0, strict=True)
    candidate_records: int = Field(ge=0, strict=True)
    rejected_records: int = Field(ge=0, strict=True)
    eligible_occurrences: int = Field(ge=0, strict=True)
    unique_eligible: int = Field(ge=0, strict=True)
    duplicate_occurrences: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_accounting(self) -> AdmissionCounts:
        if self.attempted_sources != (
            self.failed_sources + self.empty_sources + self.sources_with_artifacts
        ):
            raise ValueError("source accounting does not balance")
        if self.discovered_artifacts != (
            self.rejected_artifacts + self.decoded_artifacts
        ):
            raise ValueError("artifact accounting does not balance")
        if self.candidate_records != (
            self.rejected_records + self.eligible_occurrences
        ):
            raise ValueError("candidate accounting does not balance")
        if self.eligible_occurrences != (
            self.unique_eligible + self.duplicate_occurrences
        ):
            raise ValueError("deduplication accounting does not balance")
        return self


class AdmissionSummary(FrozenModel):
    counts: AdmissionCounts
    rejection_codes: tuple[CodeCount, ...] = Field(default=(), strict=False)
    sources: tuple[SourceAdmissionSummary, ...] = Field(default=(), strict=False)


class AdmittedCatalog(FrozenModel):
    nodes: tuple[Node, ...] = Field(default=(), strict=False)
    rejections: tuple[Rejection, ...] = Field(default=(), strict=False)
    receipts: tuple[SourceReceipt, ...] = Field(default=(), strict=False)
    summary: AdmissionSummary | None = None

    @property
    def accepted_count(self) -> int:
        return len(self.nodes)

    @property
    def rejected_count(self) -> int:
        if self.summary is None:
            return len(self.rejections)
        return (
            self.summary.counts.rejected_artifacts
            + self.summary.counts.rejected_records
        )

    def latest_source_dates(self) -> dict[str, date | None]:
        receipts = ((item.site, item.published_on) for item in self.receipts)
        provenance = (
            (item.site, item.published_on)
            for node in self.nodes
            for item in node.provenance
        )
        latest: dict[str, date | None] = {}
        for source, published_on in chain(receipts, provenance):
            previous = latest.setdefault(source, None)
            if published_on is not None and (
                previous is None or published_on > previous
            ):
                latest[source] = published_on
        return latest

    @property
    def clash_nodes(self) -> tuple[ProbeableNode, ...]:
        selected: list[ProbeableNode] = []
        for node in self.nodes:
            match node.kind:
                case "clash" | "dual":
                    selected.append(node)
                case "uri":
                    continue
        return tuple(selected)

    @property
    def uri_nodes(self) -> tuple[UriCapableNode, ...]:
        selected: list[UriCapableNode] = []
        for node in self.nodes:
            match node.kind:
                case "uri" | "dual":
                    selected.append(node)
                case "clash":
                    continue
        return tuple(selected)

    @property
    def uri_count(self) -> int:
        return len(self.uri_nodes)

    @property
    def clash_count(self) -> int:
        return len(self.clash_nodes)


def _keyword_matches(name: str, keyword: str) -> bool:
    if keyword.isascii():
        return (
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])",
                name,
                flags=re.IGNORECASE,
            )
            is not None
        )
    return keyword.casefold() in name.casefold()


def detect_regions(
    names: Sequence[str],
    region_keywords: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    regions: dict[str, list[str]] = {}
    for name in names:
        label = next(
            (
                region
                for region, keywords in region_keywords.items()
                if any(_keyword_matches(name, keyword) for keyword in keywords)
            ),
            "🌍 其他",
        )
        regions.setdefault(label, []).append(name)
    return {key: regions[key] for key in sorted(regions)}


def _looks_like_yaml(artifact: SourceArtifact, content: str) -> bool:
    media_type = (artifact.media_type or "").casefold()
    return "yaml" in media_type or bool(re.search(r"(?m)^\s*proxies\s*:", content))


def _artifact_candidates(
    artifact: SourceArtifact,
) -> tuple[list[tuple[Proxy | None, str | None, str, str]], list[Rejection]]:
    content = artifact.content.decode("utf-8", errors="replace")
    document = decode_proxies(
        content,
        yaml_hint=_looks_like_yaml(artifact, content),
    )
    candidates = [
        (candidate.proxy, candidate.uri, candidate.name, candidate.fingerprint)
        for candidate in document.candidates
    ]
    rejections = [
        Rejection(
            code=issue.code,
            site=artifact.site,
            source_url=artifact.source_url,
            item_index=issue.item_index,
        )
        for issue in document.issues
    ]
    return candidates, rejections


def _new_node(
    *,
    fingerprint: str,
    display_name: str,
    proxy: Proxy | None,
    uri: str | None,
    provenance: tuple[NodeProvenance, ...],
) -> Node:
    if proxy is not None and uri is not None:
        return DualNode(
            fingerprint=fingerprint,
            display_name=display_name,
            proxy=proxy,
            uri=uri,
            provenance=provenance,
        )
    if proxy is not None:
        return ClashNode(
            fingerprint=fingerprint,
            display_name=display_name,
            proxy=proxy,
            provenance=provenance,
        )
    assert uri is not None
    return UriNode(
        fingerprint=fingerprint,
        display_name=display_name,
        uri=uri,
        provenance=provenance,
    )


class _NodeBuilder:
    __slots__ = (
        "display_name",
        "fingerprint",
        "provenance",
        "provenance_keys",
        "proxy",
        "uri",
    )

    def __init__(
        self,
        *,
        fingerprint: str,
        display_name: str,
        proxy: Proxy | None,
        uri: str | None,
    ) -> None:
        self.fingerprint = fingerprint
        self.display_name = display_name
        self.proxy = proxy
        self.uri = uri
        self.provenance: list[NodeProvenance] = []
        self.provenance_keys: set[tuple[str, str, str]] = set()

    def absorb(
        self,
        proxy: Proxy | None,
        uri: str | None,
        provenance: NodeProvenance,
    ) -> None:
        self.proxy = self.proxy or proxy
        self.uri = self.uri or uri
        key = (provenance.authority, provenance.site, provenance.artifact_digest)
        if key in self.provenance_keys:
            return
        self.provenance_keys.add(key)
        self.provenance.append(provenance)

    def freeze(self) -> Node:
        return _new_node(
            fingerprint=self.fingerprint,
            display_name=self.display_name,
            proxy=self.proxy,
            uri=self.uri,
            provenance=tuple(self.provenance),
        )


def _freshness(
    publication_time: PublicationTime,
    observed_at: datetime,
    *,
    stale_after: timedelta,
    expires_after: timedelta,
) -> tuple[Literal["current", "stale", "expired", "future", "unknown"], str | None]:
    match publication_time:
        case PublishedInstant(at=published_at):
            age = observed_at - published_at
            if age.total_seconds() < 0:
                return "future", "clock_inversion"
            if age <= stale_after:
                return "current", None
            if age <= expires_after:
                return "stale", None
            return "expired", "source_expired"
        case PublishedDate(on=published_on):
            days = (observed_at.date() - published_on).days
            if days < 0:
                return "future", "clock_inversion"
            if days == 0:
                return "current", None
            if days <= 2:
                return "stale", None
            return "expired", "source_expired"
        case UnknownPublicationTime():
            return "unknown", None


def _code_counts(values: Counter[str]) -> tuple[CodeCount, ...]:
    return tuple(
        CodeCount(code=code, count=count) for code, count in sorted(values.items())
    )


def admit_artifacts(
    artifacts: Sequence[SourceArtifact],
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(hours=24),
    expires_after: timedelta = timedelta(hours=48),
) -> AdmittedCatalog:
    observed_now = now or datetime.now(UTC)
    if observed_now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    records: dict[str, _NodeBuilder] = {}
    rejections: list[Rejection] = []
    rejection_samples: Counter[tuple[str, str]] = Counter()
    receipts: list[SourceReceipt] = []
    code_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    source_unique: dict[str, set[str]] = defaultdict(set)
    source_codes: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_artifacts = 0
    decoded_artifacts = 0
    candidate_records = 0
    rejected_records = 0
    eligible_occurrences = 0

    def retain_sample(rejection: Rejection) -> None:
        key = (rejection.site, rejection.code)
        if rejection_samples[key] < 3:
            rejections.append(rejection)
            rejection_samples[key] += 1

    for artifact in artifacts:
        artifact_digest = artifact.digest
        freshness, rejection_code = _freshness(
            artifact.publication_time,
            observed_now,
            stale_after=stale_after,
            expires_after=expires_after,
        )
        receipts.append(
            SourceReceipt(
                authority=artifact.authority_id,
                site=artifact.site,
                source_url=artifact.source_url,
                artifact_digest=artifact_digest,
                observed_at=artifact.observed_at,
                publication_time=artifact.publication_time,
                freshness=freshness,
            )
        )
        if rejection_code:
            rejected_artifacts += 1
            code_counts[rejection_code] += 1
            source_codes[artifact.site][rejection_code] += 1
            retain_sample(
                Rejection(
                    code=rejection_code,
                    site=artifact.site,
                    source_url=artifact.source_url,
                )
            )
            continue

        decoded_artifacts += 1
        candidates, artifact_rejections = _artifact_candidates(artifact)
        rejected_records += len(artifact_rejections)
        candidate_records += len(candidates) + len(artifact_rejections)
        candidate_counts[artifact.site] += len(candidates) + len(artifact_rejections)
        for rejection in artifact_rejections:
            retain_sample(rejection)
            code_counts[rejection.code] += 1
            source_codes[artifact.site][rejection.code] += 1
        for index, (proxy, uri, name, fingerprint) in enumerate(candidates):
            server = (
                proxy.endpoint() if proxy is not None else urlsplit(uri or "").hostname
            )
            if server is not None and not is_global_endpoint(server):
                rejected_records += 1
                rejection = Rejection(
                    code="endpoint_scope",
                    site=artifact.site,
                    source_url=artifact.source_url,
                    item_index=index,
                )
                retain_sample(rejection)
                code_counts[rejection.code] += 1
                source_codes[artifact.site][rejection.code] += 1
                continue
            eligible_occurrences += 1
            provenance = NodeProvenance(
                authority=artifact.authority_id,
                site=artifact.site,
                source_url=artifact.source_url,
                observed_at=artifact.observed_at,
                publication_time=artifact.publication_time,
                artifact_digest=artifact_digest,
                item_index=index,
            )
            if fingerprint in records:
                code_counts["duplicate"] += 1
                source_codes[artifact.site]["duplicate"] += 1
            builder = records.setdefault(
                fingerprint,
                _NodeBuilder(
                    fingerprint=fingerprint,
                    display_name=name,
                    proxy=proxy,
                    uri=uri,
                ),
            )
            builder.absorb(proxy, uri, provenance)
            source_unique[artifact.site].add(fingerprint)

    source_names = tuple(dict.fromkeys(artifact.site for artifact in artifacts))
    counts = AdmissionCounts(
        attempted_sources=len(source_names),
        failed_sources=0,
        empty_sources=0,
        sources_with_artifacts=len(source_names),
        discovered_artifacts=len(artifacts),
        rejected_artifacts=rejected_artifacts,
        decoded_artifacts=decoded_artifacts,
        candidate_records=candidate_records,
        rejected_records=rejected_records,
        eligible_occurrences=eligible_occurrences,
        unique_eligible=len(records),
        duplicate_occurrences=eligible_occurrences - len(records),
    )
    summary = AdmissionSummary(
        counts=counts,
        rejection_codes=_code_counts(code_counts),
        sources=tuple(
            SourceAdmissionSummary(
                source=source,
                authority=next(
                    artifact.authority_id
                    for artifact in artifacts
                    if artifact.site == source
                ),
                status="available",
                artifacts=sum(artifact.site == source for artifact in artifacts),
                candidate_records=candidate_counts[source],
                unique_eligible=len(source_unique[source]),
                rejection_codes=_code_counts(source_codes[source]),
            )
            for source in source_names
        ),
    )
    return AdmittedCatalog(
        nodes=tuple(builder.freeze() for builder in records.values()),
        rejections=tuple(rejections),
        receipts=tuple(receipts),
        summary=summary,
    )
