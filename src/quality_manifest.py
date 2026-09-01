"""Render and admit the redacted public quality manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime
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
from src.profiles import OutputBundle, PublicEntryRegistry, render_profiles
from src.quality import (
    ProbeFailureCode,
    QualityPolicy,
    QualitySelection,
    SourceDecision,
    SourceHistory,
    SourceObservation,
    SourceReason,
    TransferTargetEvidence,
)


class QualityManifestError(RuntimeError):
    pass


class SourceAuditReceipt(FrozenModel):
    schema_version: Literal[1] = Field(default=1, alias="schema")
    status: Literal[
        "accepted",
        "measurement_inconclusive",
        "no_qualified_nodes",
        "insufficient_authority_diversity",
    ]
    generated_at: AwareDatetime
    runner_vantage: str = Field(min_length=1)
    tool_version: Literal["0.1.0"] = "0.1.0"
    policy: QualityPolicy
    admitted_nodes: int = Field(ge=0)
    selected_for_probe: int = Field(ge=0)
    qualified_nodes: int = Field(ge=0)
    failed_nodes: int = Field(ge=0)
    inconclusive_nodes: int = Field(ge=0)
    not_probed_nodes: int = Field(ge=0)
    eligible_sources: int = Field(ge=0)
    contributing_authorities: tuple[str, ...] = Field(default=(), strict=False)
    transfer_targets: tuple[TransferTargetEvidence, ...] = Field(
        default=(), strict=False
    )
    probe_failures: dict[ProbeFailureCode, int] = Field(default_factory=dict)
    sources: tuple[SourceDecision, ...] = Field(strict=False)
    diagnostic: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def reconcile_counts(self) -> "SourceAuditReceipt":
        if self.selected_for_probe != (
            self.qualified_nodes + self.failed_nodes + self.inconclusive_nodes
        ):
            raise ValueError("audit probe outcomes do not match selected probes")
        if self.admitted_nodes != self.selected_for_probe + self.not_probed_nodes:
            raise ValueError("audit admitted count does not reconcile")
        return self

    @classmethod
    def from_selection(
        cls,
        selection: QualitySelection,
        *,
        generated_at: datetime,
        runner_vantage: str,
        policy: QualityPolicy,
        selected_for_probe: int,
        transfer_targets: tuple[TransferTargetEvidence, ...] = (),
    ) -> "SourceAuditReceipt":
        qualified = selection.qualified_count
        inconclusive = selection.inconclusive_count
        failed = selected_for_probe - qualified - inconclusive
        eligible = sum(item.status == "eligible" for item in selection.sources)
        authorities = selection.contributing_authorities
        failures = Counter(
            code
            for assessment in selection.assessments
            if (code := assessment.failure_code()) is not None
        )
        return cls(
            status=(
                "accepted"
                if qualified > 0 and len(authorities) >= 2
                else (
                    "no_qualified_nodes"
                    if qualified == 0
                    else "insufficient_authority_diversity"
                )
            ),
            generated_at=generated_at,
            runner_vantage=runner_vantage,
            policy=policy,
            admitted_nodes=len(selection.catalog.nodes),
            selected_for_probe=selected_for_probe,
            qualified_nodes=qualified,
            failed_nodes=failed,
            inconclusive_nodes=inconclusive,
            not_probed_nodes=len(selection.catalog.nodes) - selected_for_probe,
            eligible_sources=eligible,
            contributing_authorities=authorities,
            transfer_targets=transfer_targets,
            probe_failures=dict(sorted(failures.items())),
            sources=selection.sources,
        )

    @classmethod
    def measurement_inconclusive(
        cls,
        *,
        generated_at: datetime,
        runner_vantage: str,
        policy: QualityPolicy,
        admitted_nodes: int,
        selected_for_probe: int,
        diagnostic: str,
        transfer_targets: tuple[TransferTargetEvidence, ...] = (),
    ) -> "SourceAuditReceipt":
        return cls(
            status="measurement_inconclusive",
            generated_at=generated_at,
            runner_vantage=runner_vantage,
            policy=policy,
            admitted_nodes=admitted_nodes,
            selected_for_probe=selected_for_probe,
            qualified_nodes=0,
            failed_nodes=0,
            inconclusive_nodes=selected_for_probe,
            not_probed_nodes=admitted_nodes - selected_for_probe,
            eligible_sources=0,
            transfer_targets=transfer_targets,
            sources=(),
            diagnostic=diagnostic,
        )


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


SourceState = Literal[
    "current",
    "stale",
    "expired",
    "future",
    "unknown",
    "failed",
]


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
    "transfer_failed",
    "inconclusive",
    "slow",
    "source_ineligible",
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
    probe_failures: dict[ProbeFailureCode, int] = Field(default_factory=dict)
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

    @field_validator("probe_failures")
    @classmethod
    def validate_probe_failures(
        cls, value: dict[ProbeFailureCode, int]
    ) -> dict[ProbeFailureCode, int]:
        if any(count <= 0 for count in value.values()):
            raise ValueError("probe failure counts must be positive")
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


class ManifestPolicyV2(FrozenModel):
    max_candidates: int = Field(gt=0, strict=True)
    max_full_probes: int = Field(gt=0, strict=True)
    max_probe_per_source: int = Field(gt=0, strict=True)
    max_published: int = Field(gt=0, strict=True)
    max_per_source: int = Field(gt=0, strict=True)
    max_delay_ms: int = Field(gt=0, strict=True)
    transfer_bytes: int = Field(gt=0, strict=True)
    max_source_age_days: int = Field(ge=0, strict=True)
    min_source_nodes: int = Field(gt=0, strict=True)
    min_source_qualified: int = Field(gt=0, strict=True)
    min_source_pass_ratio: float = Field(gt=0, le=1)
    min_source_unique: int = Field(gt=0, strict=True)
    source_history_size: int = Field(gt=0, strict=True)
    source_history_successes: int = Field(gt=0, strict=True)
    quarantine_failures: int = Field(gt=0, strict=True)
    removal_failures: int = Field(gt=0, strict=True)
    removal_days: int = Field(gt=0, strict=True)
    required_endpoints: Literal[2]


class ManifestSourceDecision(FrozenModel):
    name: str = Field(min_length=1)
    status: Literal["eligible", "quarantined", "rejected"]
    reason: SourceReason
    reliability: float = Field(ge=0, le=1)
    observation: SourceObservation
    history: tuple[SourceObservation, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_history(self) -> "ManifestSourceDecision":
        if not self.history or self.history[-1] != self.observation:
            raise ValueError("source history must end with its current observation")
        return self


class ManifestPublishedNodeV2(ManifestPublishedNode):
    bytes_per_second: float = Field(gt=0)


class QualityManifestV2(FrozenModel):
    schema_version: Literal[2] = Field(alias="schema")
    status: Literal["quality_verified"]
    generated_at: str = Field(min_length=1)
    tool: ManifestTool
    runner_vantage: str = Field(min_length=1)
    policy: ManifestPolicyV2
    counts: ManifestCounts
    exclusions: dict[str, int]
    probe_failures: dict[ProbeFailureCode, int] = Field(default_factory=dict)
    sources: tuple[ManifestSourceDecision, ...] = Field(strict=False)
    published: tuple[ManifestPublishedNodeV2, ...] = Field(strict=False)

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

    @field_validator("probe_failures")
    @classmethod
    def validate_probe_failures(
        cls, value: dict[ProbeFailureCode, int]
    ) -> dict[ProbeFailureCode, int]:
        if any(count <= 0 for count in value.values()):
            raise ValueError("probe failure counts must be positive")
        return value

    @model_validator(mode="after")
    def reconcile(self) -> "QualityManifestV2":
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
        if len({item.name for item in self.sources}) != len(self.sources):
            raise ValueError("duplicate source name")
        if len({item.id for item in self.published}) != len(self.published):
            raise ValueError("duplicate published node id")
        return self


class ManifestCountsV3(ManifestCounts):
    failed: int = Field(ge=0, strict=True)
    inconclusive: int = Field(ge=0, strict=True)
    not_probed: int = Field(ge=0, strict=True)


class ManifestTransferTarget(FrozenModel):
    name: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    controls_attempted: int = Field(ge=0, strict=True)
    controls_passed: int = Field(ge=0, strict=True)
    candidate_attempts: int = Field(ge=0, strict=True)


class ManifestPublishedNodeV3(ManifestPublishedNodeV2):
    transfer_target: str = Field(min_length=1)


class QualityManifestV3(QualityManifestV2):
    schema_version: Literal[3] = Field(alias="schema")
    decision: Literal["publishable", "insufficient_authority_diversity"]
    counts: ManifestCountsV3
    contributing_authorities: tuple[str, ...] = Field(strict=False)
    transfer_targets: tuple[ManifestTransferTarget, ...] = Field(strict=False)
    published: tuple[ManifestPublishedNodeV3, ...] = Field(strict=False)

    @model_validator(mode="after")
    def reconcile_measurement_outcomes(self) -> "QualityManifestV3":
        counts = self.counts
        if counts.selected_for_probe != (
            counts.probe_success + counts.failed + counts.inconclusive
        ):
            raise ValueError("probe outcome counts do not match selected probes")
        if counts.admitted != counts.selected_for_probe + counts.not_probed:
            raise ValueError("admitted count does not match probed and unprobed nodes")
        if any(
            target.controls_passed > target.controls_attempted
            for target in self.transfer_targets
        ):
            raise ValueError("target control counts are inconsistent")
        return self


QualityManifest = QualityManifestV1 | QualityManifestV2 | QualityManifestV3
_MANIFEST_ADAPTER = TypeAdapter(QualityManifest)


def admit_quality_manifest_json(value: str | bytes) -> QualityManifest:
    return _MANIFEST_ADAPTER.validate_json(value)


def opaque_node_id(fingerprint: str) -> str:
    return hashlib.sha256(f"quality-v1:{fingerprint}".encode()).hexdigest()[:24]


def load_quality_history(
    path: Path, policy: QualityPolicy, *, as_of: date
) -> dict[str, SourceHistory]:
    """Admit source history; schema 1 starts a new source-quality baseline."""
    if not path.is_file():
        return {}
    try:
        manifest = admit_quality_manifest_json(path.read_bytes())
        if manifest.schema_version == 1:
            return {}
        return {
            item.name: SourceHistory(
                source=item.name,
                observations=tuple(
                    observation
                    for observation in item.history
                    if observation.day <= as_of
                )[-policy.source_history_size :],
            )
            for item in manifest.sources
        }
    except (OSError, ValueError, ValidationError) as error:
        raise QualityManifestError("existing quality history is invalid") from error


def render_quality_bundle(
    selection: QualitySelection,
    policy: QualityPolicy,
    registry: PublicEntryRegistry | None = None,
    *,
    generated_at: datetime,
    runner_vantage: str,
    history: Mapping[str, SourceHistory] | None = None,
    transfer_targets: tuple[TransferTargetEvidence, ...] = (),
) -> OutputBundle:
    """Render profiles from published nodes and attach one redacted manifest."""
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if not runner_vantage.strip():
        raise ValueError("runner_vantage must not be empty")

    profile_bundle = render_profiles(selection.published, registry)
    assessments = selection.assessments
    exclusion_counts = dict(sorted(Counter(selection.exclusions.values()).items()))
    probe_failures = dict(
        sorted(
            Counter(
                code
                for assessment in assessments
                if (code := assessment.failure_code()) is not None
            ).items()
        )
    )
    selected_for_probe = sum(
        item.status not in {"not_selected", "not_probeable"} for item in assessments
    )
    probe_success = selection.qualified_count
    inconclusive = selection.inconclusive_count
    failed = selected_for_probe - probe_success - inconclusive
    prior_history = history or {}
    sources = tuple(
        ManifestSourceDecision(
            name=decision.source,
            status=decision.status,
            reason=decision.reason,
            reliability=decision.reliability,
            observation=decision.observation,
            history=prior_history.get(
                decision.source, SourceHistory(source=decision.source)
            ).updated(
                decision.observation,
                limit=policy.source_history_size,
            ),
        )
        for decision in sorted(selection.sources, key=lambda item: item.source)
    )

    assessment_index = selection.assessment_index()
    published = tuple(
        ManifestPublishedNodeV3(
            id=opaque_node_id(node.fingerprint),
            worst_delay_ms=assessment_index[node.fingerprint].required_delay_ms(),
            bytes_per_second=assessment_index[node.fingerprint].required_throughput(),
            transfer_target=assessment_index[
                node.fingerprint
            ].required_transfer_target(),
            reliability=assessment_index[node.fingerprint].reliability_score(),
        )
        for node in selection.published.nodes
    )
    manifest = QualityManifestV3(
        schema=3,
        status="quality_verified",
        decision=(
            "publishable"
            if len(selection.contributing_authorities) >= 2
            else "insufficient_authority_diversity"
        ),
        generated_at=generated_at.astimezone(UTC).isoformat(),
        tool=ManifestTool(name="freenodespider", version="0.1.0"),
        runner_vantage=runner_vantage,
        policy=ManifestPolicyV2(
            max_candidates=policy.max_candidates,
            max_full_probes=policy.max_full_probes,
            max_probe_per_source=policy.max_probe_per_source,
            max_published=policy.max_published,
            max_per_source=policy.max_per_source,
            max_delay_ms=policy.max_delay_ms,
            transfer_bytes=policy.transfer_bytes,
            max_source_age_days=policy.max_source_age_days,
            min_source_nodes=policy.min_source_nodes,
            min_source_qualified=policy.min_source_qualified,
            min_source_pass_ratio=policy.min_source_pass_ratio,
            min_source_unique=policy.min_source_unique,
            source_history_size=policy.source_history_size,
            source_history_successes=policy.source_history_successes,
            quarantine_failures=policy.quarantine_failures,
            removal_failures=policy.removal_failures,
            removal_days=policy.removal_days,
            required_endpoints=2,
        ),
        counts=ManifestCountsV3(
            admitted=len(assessments),
            selected_for_probe=selected_for_probe,
            probe_success=probe_success,
            failed=failed,
            inconclusive=inconclusive,
            not_probed=len(assessments) - selected_for_probe,
            published=len(published),
            excluded=len(selection.exclusions),
        ),
        exclusions=exclusion_counts,
        probe_failures=probe_failures,
        sources=sources,
        contributing_authorities=selection.contributing_authorities,
        transfer_targets=tuple(
            ManifestTransferTarget(
                name=target.name,
                authority=target.authority,
                controls_attempted=target.controls_attempted,
                controls_passed=target.controls_passed,
                candidate_attempts=target.candidate_attempts,
            )
            for target in transfer_targets
        ),
        published=published,
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
