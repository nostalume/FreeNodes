"""Pure candidate allocation, quality assessment, and publication quotas."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import date, timedelta
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from src.config import FrozenModel
from src.nodes import Node, NodeCatalog, ProbeableNode

ProbeStatus = Literal["success", "timeout", "api_error", "process_error", "cancelled"]
ProbeFailureCode = Literal[
    "request_timeout",
    "controller_http",
    "controller_payload",
    "controller_error",
    "run_deadline",
    "config_rejected",
    "process_start",
    "controller_start",
    "transfer_timeout",
    "transfer_http",
    "transfer_short_read",
    "transfer_error",
    "transfer_deadline",
    "listener_start",
    "validation_budget",
    "selector_update",
    "control_transfer",
]

ProbeRunPhase = Literal[
    "validation",
    "process",
    "readiness",
    "selector",
    "pre_control",
    "post_control",
]


class ProbeCandidate(FrozenModel):
    fingerprint: str = Field(min_length=1)
    proxy_name: str = Field(min_length=1, repr=False)


class ProbePlanEntry(FrozenModel):
    ordinal: int = Field(ge=0, strict=True)
    node: ProbeableNode = Field(repr=False)
    sources: tuple[str, ...] = Field(min_length=1, strict=False)
    protocol: str


class ProbePlan(FrozenModel):
    candidate_ceiling: int = Field(default=4000, gt=0)
    full_probe_limit: int = Field(default=256, gt=0, le=256)
    source_probe_limit: int = Field(default=32, gt=0, le=32)
    entries: tuple[ProbePlanEntry, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def validate_order(self) -> "ProbePlan":
        ordinals = tuple(item.ordinal for item in self.entries)
        if ordinals != tuple(range(len(self.entries))):
            raise ValueError("probe plan ordinals must be contiguous")
        fingerprints = tuple(item.node.fingerprint for item in self.entries)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("probe plan fingerprints must be unique")
        if len(self.entries) > min(self.candidate_ceiling, self.full_probe_limit):
            raise ValueError("probe plan exceeds its admitted limit")
        return self

    @property
    def nodes(self) -> tuple[ProbeableNode, ...]:
        return tuple(item.node for item in self.entries)


class ProbeDiagnostic(FrozenModel):
    code: ProbeFailureCode
    detail: str = Field(default="", max_length=200)


class DelayObservation(FrozenModel):
    endpoint: str
    status: ProbeStatus
    delay_ms: int | None = Field(default=None, ge=0, strict=True)
    diagnostic: ProbeDiagnostic | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> "DelayObservation":
        if (self.status == "success") != (self.delay_ms is not None):
            raise ValueError("only successful observations carry delay evidence")
        if (self.status == "success") == (self.diagnostic is not None):
            raise ValueError("only failed observations carry a diagnostic")
        return self

    @classmethod
    def failed(
        cls,
        endpoint: str,
        status: ProbeStatus,
        code: ProbeFailureCode,
        detail: str = "",
    ) -> "DelayObservation":
        return cls(
            endpoint=endpoint,
            status=status,
            diagnostic=ProbeDiagnostic(code=code, detail=detail[:200]),
        )


class ProbeEvidence(FrozenModel):
    fingerprint: str
    proxy_name: str = Field(repr=False)
    coarse: DelayObservation
    confirm: DelayObservation | None = None
    transfer: TransferObservation | None = None

    @model_validator(mode="after")
    def validate_transfer_identity(self) -> "ProbeEvidence":
        if self.transfer is not None and self.transfer.fingerprint != self.fingerprint:
            raise ValueError(
                "transfer evidence fingerprint does not match delay evidence"
            )
        return self

    @property
    def status(self) -> ProbeStatus:
        if self.coarse.status != "success":
            return self.coarse.status
        if self.confirm is None:
            return "cancelled"
        return self.confirm.status

    @property
    def diagnostic(self) -> ProbeDiagnostic | None:
        if self.coarse.status != "success":
            return self.coarse.diagnostic
        return self.confirm.diagnostic if self.confirm is not None else None

    @classmethod
    def process_failure(
        cls, node: ProbeableNode, code: ProbeFailureCode
    ) -> "ProbeEvidence":
        return cls(
            fingerprint=node.fingerprint,
            proxy_name=node.display_name,
            coarse=DelayObservation.failed("mihomo-process", "process_error", code),
        )


TransferStatus = Literal[
    "success",
    "inconclusive",
    "timeout",
    "http_error",
    "short_read",
    "process_error",
    "cancelled",
]


class TransferObservation(FrozenModel):
    fingerprint: str
    target: str = Field(min_length=1)
    status: TransferStatus
    bytes_received: int = Field(ge=0, strict=True)
    elapsed_ms: int = Field(ge=0, strict=True)
    bytes_per_second: float | None = Field(default=None, gt=0)
    diagnostic: ProbeDiagnostic | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> "TransferObservation":
        if (self.status == "success") != (self.bytes_per_second is not None):
            raise ValueError("only successful transfers carry throughput")
        if (self.status == "success") == (self.diagnostic is not None):
            raise ValueError("only failed transfers carry a diagnostic")
        return self


class TransferTargetEvidence(FrozenModel):
    name: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    controls_attempted: int = Field(ge=0, strict=True)
    controls_passed: int = Field(ge=0, strict=True)
    candidate_attempts: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_controls(self) -> "TransferTargetEvidence":
        if self.controls_passed > self.controls_attempted:
            raise ValueError("passed controls exceed attempted controls")
        return self


class ValidatedProbeBatch(FrozenModel):
    nodes: tuple[ProbeableNode, ...] = Field(strict=False)
    failures: tuple[ProbeEvidence, ...] = Field(strict=False)
    complete: bool = True


class ProbeRunSuccess(FrozenModel):
    status: Literal["success"] = "success"
    evidence: tuple[ProbeEvidence, ...] = Field(default=(), strict=False)
    transfer_targets: tuple[TransferTargetEvidence, ...] = Field(
        default=(), strict=False
    )


class ProbeRunFailure(FrozenModel):
    status: Literal["inconclusive"] = "inconclusive"
    phase: ProbeRunPhase
    diagnostic: ProbeDiagnostic
    transfer_targets: tuple[TransferTargetEvidence, ...] = Field(
        default=(), strict=False
    )


ProbeRunResult = Annotated[
    ProbeRunSuccess | ProbeRunFailure,
    Field(discriminator="status"),
]


class QualityError(ValueError):
    pass


class QualityPolicy(FrozenModel):
    max_candidates: int = Field(default=4000, gt=0)
    max_full_probes: int = Field(default=256, gt=0, le=256)
    max_probe_per_source: int = Field(default=32, gt=0, le=32)
    max_published: int = Field(default=500, gt=0)
    max_per_source: int = Field(default=100, gt=0)
    max_delay_ms: int = Field(default=2500, gt=0)
    transfer_bytes: int = Field(default=1024 * 1024, gt=0)
    max_source_age_days: int = Field(default=1, ge=0)
    min_source_nodes: int = Field(default=20, gt=0)
    min_source_qualified: int = Field(default=8, gt=0)
    min_source_pass_ratio: float = Field(default=0.4, gt=0, le=1)
    min_source_unique: int = Field(default=5, gt=0)
    source_history_size: int = Field(default=4, gt=0)
    source_history_successes: int = Field(default=3, gt=0)
    quarantine_failures: int = Field(default=2, gt=0)
    removal_failures: int = Field(default=3, gt=0)
    removal_days: int = Field(default=7, gt=0)

    @model_validator(mode="after")
    def validate_source_policy(self) -> "QualityPolicy":
        if self.source_history_successes > self.source_history_size:
            raise ValueError("source history successes exceed its window")
        if self.quarantine_failures >= self.removal_failures:
            raise ValueError("source quarantine must precede removal")
        return self


SourceOutcome = Literal["eligible", "failed", "unavailable"]
SourceReason = Literal[
    "current",
    "retained",
    "source_unavailable",
    "source_stale",
    "insufficient_population",
    "insufficient_qualified",
    "low_pass_ratio",
    "insufficient_unique",
    "unstable_history",
    "repeated_failure",
    "unrecovered",
]


class SourceObservation(FrozenModel):
    day: date = Field(strict=False)
    outcome: SourceOutcome
    reason: SourceReason
    admitted_unique: int = Field(ge=0)
    sampled: int = Field(ge=0)
    qualified: int = Field(ge=0)
    pass_ratio: float = Field(ge=0, le=1)
    unique_contribution: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "SourceObservation":
        expected = self.qualified / self.sampled if self.sampled else 0.0
        if self.qualified > self.sampled or abs(self.pass_ratio - expected) > 1e-12:
            raise ValueError("source sample counts and pass ratio disagree")
        if self.unique_contribution > self.admitted_unique:
            raise ValueError("unique contribution exceeds admitted nodes")
        return self


class SourceHistory(FrozenModel):
    source: str
    observations: tuple[SourceObservation, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def validate_observations(self) -> "SourceHistory":
        days = tuple(item.day for item in self.observations)
        if days != tuple(sorted(set(days))):
            raise ValueError("source observation days must be unique and ordered")
        return self

    def updated(
        self, observation: SourceObservation, *, limit: int
    ) -> tuple[SourceObservation, ...]:
        by_day = {item.day: item for item in self.observations}
        by_day[observation.day] = observation
        return tuple(item for _, item in sorted(by_day.items()))[-limit:]

    def score(self, *, limit: int) -> float:
        recent = self.observations[-limit:]
        if not recent:
            return -1.0
        return sum(item.outcome == "eligible" for item in recent) / len(recent)


class SourceDecisionBase(FrozenModel):
    source: str
    reason: SourceReason
    reliability: float = Field(ge=0, le=1)
    observation: SourceObservation


class EligibleSource(SourceDecisionBase):
    status: Literal["eligible"] = "eligible"


class QuarantinedSource(SourceDecisionBase):
    status: Literal["quarantined"] = "quarantined"


class RejectedSource(SourceDecisionBase):
    status: Literal["rejected"] = "rejected"


SourceDecision = Annotated[
    EligibleSource | QuarantinedSource | RejectedSource,
    Field(discriminator="status"),
]


class AssessmentBase(FrozenModel):
    fingerprint: str
    status: str

    def observed_delay_ms(self) -> int | None:
        return None

    def reliability_score(self) -> float | None:
        return None

    def required_delay_ms(self) -> int:
        raise QualityError("assessment has no successful delay observation")

    def required_throughput(self) -> float:
        raise QualityError("assessment has no successful transfer observation")

    def required_transfer_target(self) -> str:
        raise QualityError("assessment has no successful transfer target")

    def summary_status(self) -> str:
        return self.status

    def failure_code(self) -> ProbeFailureCode | None:
        return None


class UnobservedAssessment(AssessmentBase):
    status: Literal["not_probeable", "not_selected"]


class FailedAssessment(AssessmentBase):
    status: Literal[
        "timeout", "api_error", "process_error", "cancelled", "transfer_failed"
    ]
    diagnostic: ProbeFailureCode

    def summary_status(self) -> str:
        return f"{self.status}/{self.diagnostic}"

    def failure_code(self) -> ProbeFailureCode:
        return self.diagnostic


class InconclusiveAssessment(AssessmentBase):
    status: Literal["inconclusive"] = "inconclusive"
    diagnostic: ProbeFailureCode

    def failure_code(self) -> ProbeFailureCode:
        return self.diagnostic


class ObservedDelayAssessment(AssessmentBase):
    worst_delay_ms: int = Field(ge=0, strict=True)

    def observed_delay_ms(self) -> int:
        return self.worst_delay_ms

    def required_delay_ms(self) -> int:
        return self.worst_delay_ms


class SlowAssessment(ObservedDelayAssessment):
    status: Literal["slow"] = "slow"


class ScoredAssessment(ObservedDelayAssessment):
    status: Literal[
        "eligible",
        "published",
        "global_quota",
        "source_quota",
    ]
    transfer_target: str = Field(min_length=1)
    bytes_per_second: float = Field(gt=0)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)

    def reliability_score(self) -> float | None:
        return self.reliability

    def required_throughput(self) -> float:
        return self.bytes_per_second

    def required_transfer_target(self) -> str:
        return self.transfer_target


QualityAssessment = Annotated[
    UnobservedAssessment
    | FailedAssessment
    | InconclusiveAssessment
    | SlowAssessment
    | ScoredAssessment,
    Field(discriminator="status"),
]

_ASSESSMENT_ADAPTER = TypeAdapter(QualityAssessment)


def admit_assessment(value: object) -> QualityAssessment:
    return _ASSESSMENT_ADAPTER.validate_python(value)


class QualitySelection(FrozenModel):
    catalog: NodeCatalog
    published: NodeCatalog
    assessments: tuple[QualityAssessment, ...] = Field(strict=False)
    sources: tuple[SourceDecision, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def validate_ledger(self) -> "QualitySelection":
        assessment_ids = tuple(item.fingerprint for item in self.assessments)
        if len(set(assessment_ids)) != len(assessment_ids):
            raise ValueError("duplicate assessment fingerprint")
        catalog_ids = {node.fingerprint for node in self.catalog.nodes}
        if set(assessment_ids) != catalog_ids:
            raise ValueError("assessment ledger does not cover the admitted catalog")
        if any(item.status == "eligible" for item in self.assessments):
            raise ValueError("selection contains unresolved eligible assessment")
        published_ids = {node.fingerprint for node in self.published.nodes}
        ledger_published = {
            item.fingerprint for item in self.assessments if item.status == "published"
        }
        if published_ids != ledger_published:
            raise ValueError("published catalog and assessment ledger disagree")
        source_names = tuple(item.source for item in self.sources)
        if len(set(source_names)) != len(source_names):
            raise ValueError("duplicate source decision")
        return self

    def assessment_index(self) -> dict[str, QualityAssessment]:
        return {item.fingerprint: item for item in self.assessments}

    @property
    def qualified_count(self) -> int:
        qualified = {"published", "global_quota", "source_quota"}
        return sum(item.status in qualified for item in self.assessments)

    @property
    def inconclusive_count(self) -> int:
        return sum(item.status == "inconclusive" for item in self.assessments)

    @property
    def contributing_authorities(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    provenance.site
                    for node in self.published.nodes
                    for provenance in node.provenance
                    if provenance.site
                }
            )
        )

    @property
    def exclusions(self) -> dict[str, str]:
        return {
            item.fingerprint: item.status
            for item in self.assessments
            if item.status != "published"
        }


def _sources(node: Node) -> tuple[str, ...]:
    sites = tuple(sorted({item.site for item in node.provenance if item.site}))
    return sites or ("unknown",)


def plan_probe_candidates(
    catalog: NodeCatalog,
    policy: QualityPolicy,
    *,
    sample_overflow: bool = False,
) -> ProbePlan:
    """Build one stable, source-fair full-probe plan before network I/O."""
    grouped: dict[str, dict[str, list[ProbeableNode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    memberships: dict[str, tuple[str, ...]] = {}
    probeable_count = 0
    for node in catalog.nodes:
        match node.kind:
            case "clash" | "dual":
                pass
            case "uri":
                continue
        probeable_count += 1
        if not sample_overflow and probeable_count > policy.max_candidates:
            raise QualityError("probe candidate ceiling exceeded")
        sources = _sources(node)
        bucket = grouped[sources[0]][node.proxy.type]
        bucket.append(node)
        bucket.sort(key=lambda item: item.fingerprint)
        memberships[node.fingerprint] = sources
        if len(bucket) > policy.max_probe_per_source:
            memberships.pop(bucket.pop().fingerprint)
        if len(memberships) > policy.max_candidates:
            raise QualityError("probe candidate ceiling exceeded")

    source_queues: dict[str, deque[ProbeableNode]] = {}
    for source, protocols in grouped.items():
        ordered = deque[ProbeableNode]()
        names = sorted(protocols)
        while names:
            remaining: list[str] = []
            for protocol in names:
                bucket = protocols[protocol]
                if bucket:
                    ordered.append(bucket.pop(0))
                if bucket:
                    remaining.append(protocol)
            names = remaining
        source_queues[source] = ordered

    limit = min(policy.max_candidates, policy.max_full_probes)
    selected: list[ProbeableNode] = []
    source_counts: dict[str, int] = defaultdict(int)
    source_names = sorted(source_queues)
    while source_names and len(selected) < limit:
        remaining_sources: list[str] = []
        for source in source_names:
            queue = source_queues[source]
            while queue and len(selected) < limit:
                node = queue.popleft()
                sources = memberships[node.fingerprint]
                if any(
                    source_counts[item] >= policy.max_probe_per_source
                    for item in sources
                ):
                    continue
                selected.append(node)
                for item in sources:
                    source_counts[item] += 1
                break
            if queue and source_counts[source] < policy.max_probe_per_source:
                remaining_sources.append(source)
        source_names = remaining_sources
    used: set[str] = set()
    for index, node in enumerate(selected):
        if node.display_name in used:
            selected[index] = node.model_copy(
                update={"display_name": f"{node.display_name} · {node.fingerprint[:8]}"}
            )
        used.add(selected[index].display_name)
    return ProbePlan(
        candidate_ceiling=policy.max_candidates,
        full_probe_limit=policy.max_full_probes,
        source_probe_limit=policy.max_probe_per_source,
        entries=tuple(
            ProbePlanEntry(
                ordinal=index,
                node=node,
                sources=memberships[node.fingerprint],
                protocol=node.proxy.type,
            )
            for index, node in enumerate(selected)
        ),
    )


def _decide_source(
    source: str,
    observation: SourceObservation,
    history: SourceHistory,
    policy: QualityPolicy,
) -> SourceDecision:
    window = history.updated(observation, limit=policy.source_history_size)
    successes = sum(item.outcome == "eligible" for item in window)
    reliability = successes / len(window)
    failures = 0
    severe = 0
    for item in reversed(window):
        if item.outcome == "eligible":
            break
        failures += 1
        severe += item.outcome == "unavailable" or item.qualified == 0
    last_success = next(
        (item.day for item in reversed(window) if item.outcome == "eligible"), None
    )
    if not history.observations:
        if observation.outcome == "eligible":
            return EligibleSource(
                source=source,
                reason="current",
                reliability=reliability,
                observation=observation,
            )
        return RejectedSource(
            source=source,
            reason=observation.reason,
            reliability=reliability,
            observation=observation,
        )
    if observation.outcome != "eligible" and last_success is not None:
        if observation.day - last_success >= timedelta(days=policy.removal_days):
            return RejectedSource(
                source=source,
                reason="unrecovered",
                reliability=reliability,
                observation=observation,
            )
    if failures >= policy.removal_failures:
        return RejectedSource(
            source=source,
            reason="repeated_failure",
            reliability=reliability,
            observation=observation,
        )
    if severe >= policy.quarantine_failures:
        return QuarantinedSource(
            source=source,
            reason="unstable_history",
            reliability=reliability,
            observation=observation,
        )
    required = min(policy.source_history_successes, max(1, len(window) - 1))
    if successes >= required:
        reason: SourceReason = (
            "current" if observation.outcome == "eligible" else "retained"
        )
        return EligibleSource(
            source=source,
            reason=reason,
            reliability=reliability,
            observation=observation,
        )
    return QuarantinedSource(
        source=source,
        reason="unstable_history",
        reliability=reliability,
        observation=observation,
    )


def _observe_source(
    source: str,
    *,
    day: date,
    fingerprints: AbstractSet[str],
    published_on: date | None,
    sample: AbstractSet[str],
    qualified: AbstractSet[str],
    claimed: AbstractSet[str],
    unavailable: AbstractSet[str],
    policy: QualityPolicy,
) -> SourceObservation:
    sampled = len(sample)
    passed = len(sample & qualified)
    ratio = passed / sampled if sampled else 0.0
    unique = len(fingerprints - claimed)
    if source in unavailable:
        outcome, reason = "unavailable", "source_unavailable"
    elif (
        published_on is None
        or not 0 <= (day - published_on).days <= policy.max_source_age_days
    ):
        outcome, reason = "failed", "source_stale"
    elif len(fingerprints) < policy.min_source_nodes:
        outcome, reason = "failed", "insufficient_population"
    elif passed < policy.min_source_qualified:
        outcome, reason = "failed", "insufficient_qualified"
    elif ratio < policy.min_source_pass_ratio:
        outcome, reason = "failed", "low_pass_ratio"
    elif unique < policy.min_source_unique:
        outcome, reason = "failed", "insufficient_unique"
    else:
        outcome, reason = "eligible", "current"
    return SourceObservation(
        day=day,
        outcome=outcome,
        reason=reason,
        admitted_unique=len(fingerprints),
        sampled=sampled,
        qualified=passed,
        pass_ratio=ratio,
        unique_contribution=unique,
    )


QualifiedNodes = dict[str, tuple[ProbeableNode, ScoredAssessment]]


def _qualify_nodes(
    catalog: NodeCatalog,
    evidence: Mapping[str, ProbeEvidence],
    policy: QualityPolicy,
) -> tuple[dict[str, QualityAssessment], QualifiedNodes]:
    assessments: dict[str, QualityAssessment] = {}
    qualified: QualifiedNodes = {}
    for node in catalog.nodes:
        item = evidence.get(node.fingerprint)
        if node.kind == "uri":
            assessment: QualityAssessment = UnobservedAssessment(
                fingerprint=node.fingerprint, status="not_probeable"
            )
        elif item is None:
            assessment = UnobservedAssessment(
                fingerprint=node.fingerprint, status="not_selected"
            )
        elif item.status != "success":
            if item.diagnostic is None:
                raise QualityError("failed probe evidence has no diagnostic")
            assessment = FailedAssessment(
                fingerprint=node.fingerprint,
                status=item.status,
                diagnostic=item.diagnostic.code,
            )
        else:
            coarse = item.coarse.delay_ms
            confirm = item.confirm.delay_ms if item.confirm else None
            if coarse is None or confirm is None:
                raise QualityError("successful evidence has incomplete delay values")
            worst_delay = max(coarse, confirm)
            if worst_delay > policy.max_delay_ms:
                assessments[node.fingerprint] = SlowAssessment(
                    fingerprint=node.fingerprint,
                    worst_delay_ms=worst_delay,
                )
                continue
            transfer = item.transfer
            if transfer is None:
                raise QualityError("successful delay evidence has no transfer outcome")
            if transfer.status == "inconclusive":
                if transfer.diagnostic is None:
                    raise QualityError("inconclusive transfer has no diagnostic")
                assessments[node.fingerprint] = InconclusiveAssessment(
                    fingerprint=node.fingerprint,
                    diagnostic=transfer.diagnostic.code,
                )
                continue
            if transfer.status != "success":
                if transfer.diagnostic is None:
                    raise QualityError("failed transfer has no diagnostic")
                assessments[node.fingerprint] = FailedAssessment(
                    fingerprint=node.fingerprint,
                    status="transfer_failed",
                    diagnostic=transfer.diagnostic.code,
                )
                continue
            if transfer.bytes_received != policy.transfer_bytes:
                raise QualityError("successful transfer has an unexpected byte count")
            if transfer.bytes_per_second is None:
                raise QualityError("successful transfer has no throughput")
            assessment = ScoredAssessment(
                fingerprint=node.fingerprint,
                status="eligible",
                worst_delay_ms=worst_delay,
                transfer_target=transfer.target,
                bytes_per_second=transfer.bytes_per_second,
            )
            qualified[node.fingerprint] = (node, assessment)
        assessments[node.fingerprint] = assessment
    return assessments, qualified


def _source_indexes(
    catalog: NodeCatalog,
    plan: ProbePlan,
    day: date,
    max_age_days: int,
) -> tuple[dict[str, set[str]], dict[str, date | None], dict[str, set[str]]]:
    nodes: dict[str, set[str]] = defaultdict(set)
    published = catalog.latest_source_dates()
    for node in catalog.nodes:
        for provenance in node.provenance:
            if provenance.published_on is not None:
                if 0 <= (day - provenance.published_on).days <= max_age_days:
                    nodes[provenance.site].add(node.fingerprint)
    samples: dict[str, set[str]] = defaultdict(set)
    for entry in plan.entries:
        for source in entry.sources:
            samples[source].add(entry.node.fingerprint)
    return nodes, published, samples


def _decide_sources(
    catalog: NodeCatalog,
    plan: ProbePlan,
    qualified: set[str],
    conclusive: set[str],
    policy: QualityPolicy,
    history: Mapping[str, SourceHistory],
    unavailable: set[str],
    day: date,
) -> tuple[SourceDecision, ...]:
    source_nodes, source_days, samples = _source_indexes(
        catalog, plan, day, policy.max_source_age_days
    )
    source_names = set(source_days) | set(samples) | unavailable

    def pass_ratio(source: str) -> float:
        current_sample = samples[source] & source_nodes[source] & conclusive
        return (
            len(current_sample & qualified) / len(current_sample)
            if current_sample
            else 0
        )

    ranked = sorted(
        source_names,
        key=lambda source: (
            -history.get(source, SourceHistory(source=source)).score(
                limit=policy.source_history_size
            ),
            -pass_ratio(source),
            source,
        ),
    )
    claimed: set[str] = set()
    decisions: list[SourceDecision] = []
    for source in ranked:
        observation = _observe_source(
            source,
            day=day,
            fingerprints=source_nodes[source],
            published_on=source_days.get(source),
            sample=samples[source] & source_nodes[source] & conclusive,
            qualified=qualified,
            claimed=claimed,
            unavailable=unavailable,
            policy=policy,
        )
        decision = _decide_source(
            source,
            observation,
            history.get(source, SourceHistory(source=source)),
            policy,
        )
        decisions.append(decision)
        if decision.status == "eligible":
            claimed.update(source_nodes[source])
    return tuple(decisions)


def _select_nodes(
    catalog: NodeCatalog,
    qualified: QualifiedNodes,
    decisions: tuple[SourceDecision, ...],
    assessments: dict[str, QualityAssessment],
    policy: QualityPolicy,
    day: date,
) -> NodeCatalog:
    source_order = {
        decision.source: (index, decision) for index, decision in enumerate(decisions)
    }
    candidates = []
    for fingerprint, (node, assessment) in qualified.items():
        choices = [
            source_order[source] for source in _sources(node) if source in source_order
        ]
        source_index, decision = min(choices, key=lambda item: item[0])
        candidates.append(
            (
                -decision.reliability,
                assessment.worst_delay_ms,
                -assessment.bytes_per_second,
                fingerprint,
                source_index,
                decision.source,
                node,
                assessment,
            )
        )
    published: list[ProbeableNode] = []
    source_counts: dict[str, int] = defaultdict(int)
    for _, _, _, fingerprint, _, source, node, assessment in sorted(candidates):
        if len(published) >= policy.max_published:
            status = "global_quota"
        elif source_counts[source] >= policy.max_per_source:
            status = "source_quota"
        else:
            published.append(node)
            source_counts[source] += 1
            status = "published"
        assessments[fingerprint] = assessment.model_copy(
            update={
                "status": status,
                "reliability": source_order[source][1].reliability,
            }
        )
    return NodeCatalog(
        nodes=tuple(published),
        rejections=catalog.rejections,
        receipts=catalog.receipts,
    )


def assess_quality(
    catalog: NodeCatalog,
    plan: ProbePlan,
    evidence: Sequence[ProbeEvidence],
    policy: QualityPolicy,
    *,
    history: Mapping[str, SourceHistory] | None = None,
    unavailable_sources: Sequence[str] = (),
    as_of: date | None = None,
) -> QualitySelection:
    """Qualify nodes, decide sources once, then apply publication quotas."""
    catalog_by_id = {node.fingerprint: node for node in catalog.nodes}
    expected = tuple(entry.node.fingerprint for entry in plan.entries)
    observed = tuple(item.fingerprint for item in evidence)
    if observed != expected:
        raise QualityError("probe evidence does not cover the probe plan")
    if not set(expected).issubset(catalog_by_id):
        raise QualityError("probe plan does not belong to the catalog")
    evidence_by_id = dict(zip(observed, evidence, strict=True))
    assessments, qualified = _qualify_nodes(catalog, evidence_by_id, policy)
    conclusive = {
        fingerprint
        for fingerprint, assessment in assessments.items()
        if assessment.status not in {"not_probeable", "not_selected", "inconclusive"}
    }
    day = as_of or date.today()
    decisions = _decide_sources(
        catalog,
        plan,
        set(qualified),
        conclusive,
        policy,
        history or {},
        set(unavailable_sources),
        day,
    )
    published = _select_nodes(catalog, qualified, decisions, assessments, policy, day)
    return QualitySelection(
        catalog=catalog,
        published=published,
        assessments=tuple(assessments[node.fingerprint] for node in catalog.nodes),
        sources=decisions,
    )
