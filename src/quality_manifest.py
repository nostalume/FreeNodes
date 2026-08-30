"""Render and admit the redacted public quality manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import (
    AwareDatetime,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import FrozenModel
from src.nodes import NodeCatalog
from src.profiles import OutputBundle, PublicEntryRegistry, render_profiles
from src.quality import (
    DailyReliability,
    QualityHistory,
    QualityPolicy,
    QualitySelection,
)


class QualityManifestError(RuntimeError):
    pass


class ManifestHistoryDay(FrozenModel):
    day: date = Field(strict=False)
    successes: int = Field(ge=0, strict=True)
    attempts: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_counts(self) -> "ManifestHistoryDay":
        if self.successes > self.attempts:
            raise ValueError("successes exceed attempts")
        return self


class ManifestHistoryRecord(FrozenModel):
    id: str = Field(min_length=24, max_length=24)
    days: tuple[ManifestHistoryDay, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_days(self) -> "ManifestHistoryRecord":
        days = tuple(item.day for item in self.days)
        if len(set(days)) != len(days):
            raise ValueError("duplicate history day")
        return self


class ManifestTool(FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ManifestPolicy(FrozenModel):
    max_candidates: int = Field(gt=0, strict=True)
    max_published: int = Field(gt=0, strict=True)
    max_per_source: int = Field(gt=0, strict=True)
    max_delay_ms: int = Field(gt=0, strict=True)
    history_days: int = Field(gt=0, strict=True)
    required_endpoints: Literal[2]


class ManifestCounts(FrozenModel):
    admitted: int = Field(ge=0, strict=True)
    selected_for_probe: int = Field(ge=0, strict=True)
    probe_success: int = Field(ge=0, strict=True)
    published: int = Field(gt=0, strict=True)
    excluded: int = Field(ge=0, strict=True)


SourceState = Literal["current", "stale", "expired", "future", "failed"]


class ManifestSource(FrozenModel):
    name: str = Field(min_length=1)
    state: SourceState


class ManifestPublishedNode(FrozenModel):
    id: str = Field(min_length=24, max_length=24)
    worst_delay_ms: int = Field(ge=0, strict=True)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)


_EXCLUSION_STATUSES = {
    "not_probeable",
    "not_selected",
    "timeout",
    "api_error",
    "process_error",
    "cancelled",
    "slow",
    "global_quota",
    "source_quota",
}


class QualityManifestV1(FrozenModel):
    schema_version: Literal[1] = Field(alias="schema")
    status: Literal["quality_verified"]
    generated_at: str = Field(min_length=1)
    tool: ManifestTool
    runner_vantage: str = Field(min_length=1)
    policy: ManifestPolicy
    counts: ManifestCounts
    exclusions: dict[str, int]
    sources: tuple[ManifestSource, ...] = Field(strict=False)
    published: tuple[ManifestPublishedNode, ...] = Field(strict=False)
    history: tuple[ManifestHistoryRecord, ...] = Field(strict=False)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        TypeAdapter(AwareDatetime).validate_python(value)
        return value

    @field_validator("exclusions")
    @classmethod
    def validate_exclusions(cls, value: dict[str, int]) -> dict[str, int]:
        if not set(value).issubset(_EXCLUSION_STATUSES):
            raise ValueError("unknown exclusion status")
        if any(count <= 0 for count in value.values()):
            raise ValueError("exclusion counts must be positive")
        return value

    @model_validator(mode="after")
    def reconcile(self) -> "QualityManifestV1":
        counts = self.counts
        if counts.published + counts.excluded != counts.admitted:
            raise ValueError("published and excluded counts do not match admitted")
        if counts.published != len(self.published):
            raise ValueError("published count does not match published records")
        if counts.excluded != sum(self.exclusions.values()):
            raise ValueError("excluded count does not match exclusion records")
        if not (
            counts.published
            <= counts.probe_success
            <= counts.selected_for_probe
            <= counts.admitted
        ):
            raise ValueError("probe counts are inconsistent")
        source_names = tuple(item.name for item in self.sources)
        published_ids = tuple(item.id for item in self.published)
        history_ids = tuple(item.id for item in self.history)
        if len(set(source_names)) != len(source_names):
            raise ValueError("duplicate source name")
        if len(set(published_ids)) != len(published_ids):
            raise ValueError("duplicate published node id")
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("duplicate history node id")
        return self


def opaque_node_id(fingerprint: str) -> str:
    return hashlib.sha256(f"quality-v1:{fingerprint}".encode()).hexdigest()[:24]


def load_quality_history(
    path: Path, catalog: NodeCatalog, policy: QualityPolicy, *, as_of: date
) -> dict[str, QualityHistory]:
    """Map redacted prior history back to current semantic fingerprints."""
    if not path.is_file():
        return {}
    try:
        manifest = QualityManifestV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        fingerprint_by_id = {
            opaque_node_id(node.fingerprint): node.fingerprint for node in catalog.nodes
        }
        cutoff = as_of - timedelta(days=policy.history_days - 1)
        result: dict[str, QualityHistory] = {}
        for record in manifest.history:
            fingerprint = fingerprint_by_id.get(record.id)
            if fingerprint is None:
                continue
            days = tuple(
                DailyReliability(
                    day=item.day, successes=item.successes, attempts=item.attempts
                )
                for item in record.days
                if cutoff <= item.day <= as_of
            )
            result[fingerprint] = QualityHistory(fingerprint=fingerprint, days=days)
        return result
    except (OSError, ValueError, ValidationError) as error:
        raise QualityManifestError("existing quality history is invalid") from error


def render_quality_bundle(
    selection: QualitySelection,
    policy: QualityPolicy,
    registry: PublicEntryRegistry | None = None,
    *,
    generated_at: datetime,
    runner_vantage: str,
    failed_sources: Sequence[str] = (),
    history: Mapping[str, QualityHistory] | None = None,
) -> OutputBundle:
    """Render profiles from published nodes and attach one redacted manifest."""
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if not runner_vantage.strip():
        raise ValueError("runner_vantage must not be empty")

    profile_bundle = render_profiles(selection.published, registry)
    assessments = selection.assessments
    exclusion_counts = dict(sorted(Counter(selection.exclusions.values()).items()))
    selected_for_probe = sum(
        item.status not in {"not_selected", "not_probeable"} for item in assessments
    )
    probe_success = sum(item.observed_delay_ms() is not None for item in assessments)
    source_states: dict[str, SourceState] = {
        receipt.site: receipt.freshness for receipt in selection.published.receipts
    }
    for source in failed_sources:
        source_states.setdefault(source, "failed")

    prior_history = history or {}
    cutoff = generated_at.date() - timedelta(days=policy.history_days - 1)
    history_records: list[ManifestHistoryRecord] = []
    for assessment in sorted(assessments, key=lambda item: item.fingerprint):
        fingerprint = assessment.fingerprint
        days_by_date = {
            sample.day: sample
            for sample in prior_history.get(
                fingerprint, QualityHistory(fingerprint=fingerprint)
            ).days
            if cutoff <= sample.day <= generated_at.date()
        }
        if assessment.status not in {"not_selected", "not_probeable"}:
            days_by_date[generated_at.date()] = DailyReliability(
                day=generated_at.date(),
                successes=int(assessment.observed_delay_ms() is not None),
                attempts=1,
            )
        if days_by_date:
            history_records.append(
                ManifestHistoryRecord(
                    id=opaque_node_id(fingerprint),
                    days=tuple(
                        ManifestHistoryDay(
                            day=day,
                            successes=sample.successes,
                            attempts=sample.attempts,
                        )
                        for day, sample in sorted(days_by_date.items())
                    ),
                )
            )

    assessment_index = selection.assessment_index()
    published = tuple(
        ManifestPublishedNode(
            id=opaque_node_id(node.fingerprint),
            worst_delay_ms=assessment_index[node.fingerprint].required_delay_ms(),
            reliability=assessment_index[node.fingerprint].reliability_score(),
        )
        for node in selection.published.nodes
    )
    manifest = QualityManifestV1(
        schema=1,
        status="quality_verified",
        generated_at=generated_at.astimezone(UTC).isoformat(),
        tool=ManifestTool(name="freenodespider", version="0.1.0"),
        runner_vantage=runner_vantage,
        policy=ManifestPolicy(
            max_candidates=policy.max_candidates,
            max_published=policy.max_published,
            max_per_source=policy.max_per_source,
            max_delay_ms=policy.max_delay_ms,
            history_days=policy.history_days,
            required_endpoints=2,
        ),
        counts=ManifestCounts(
            admitted=len(assessments),
            selected_for_probe=selected_for_probe,
            probe_success=probe_success,
            published=len(published),
            excluded=len(selection.exclusions),
        ),
        exclusions=exclusion_counts,
        sources=tuple(
            ManifestSource(name=name, state=state)
            for name, state in sorted(source_states.items())
        ),
        published=published,
        history=tuple(history_records),
    )
    manifest_bytes = (
        json.dumps(
            manifest.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    files = dict(profile_bundle.files)
    files["nodes/quality-manifest.json"] = manifest_bytes
    return OutputBundle.from_files(
        files,
        accepted_count=profile_bundle.accepted_count,
        clash_count=profile_bundle.clash_count,
        uri_count=profile_bundle.uri_count,
        aggregate_files=profile_bundle.aggregate_files
        + ("nodes/quality-manifest.json",),
    )
