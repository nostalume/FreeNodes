"""Pure candidate allocation, quality assessment, and publication quotas."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from src.config import FrozenModel
from src.mihomo import ProbeEvidence
from src.nodes import Node, NodeCatalog, ProbeableNode


class QualityError(ValueError):
    pass


class QualityPolicy(FrozenModel):
    max_candidates: int = Field(default=4000, gt=0)
    max_published: int = Field(default=500, gt=0)
    max_per_source: int = Field(default=100, gt=0)
    max_delay_ms: int = Field(default=2500, gt=0)
    history_days: int = Field(default=7, gt=0)


class DailyReliability(FrozenModel):
    day: date
    successes: int = Field(ge=0)
    attempts: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "DailyReliability":
        if self.successes > self.attempts:
            raise ValueError("successes exceed attempts")
        return self


class QualityHistory(FrozenModel):
    fingerprint: str
    days: tuple[DailyReliability, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def validate_days(self) -> "QualityHistory":
        dates = tuple(sample.day for sample in self.days)
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate reliability day")
        return self

    def score(self, *, as_of: date, history_days: int) -> float:
        cutoff = as_of - timedelta(days=history_days - 1)
        recent = tuple(sample for sample in self.days if cutoff <= sample.day <= as_of)
        attempts = sum(sample.attempts for sample in recent)
        if attempts == 0:
            return -1.0
        return sum(sample.successes for sample in recent) / attempts


class AssessmentBase(FrozenModel):
    fingerprint: str

    def observed_delay_ms(self) -> int | None:
        return None

    def reliability_score(self) -> float | None:
        return None

    def required_delay_ms(self) -> int:
        raise QualityError("assessment has no successful delay observation")


class UnobservedAssessment(AssessmentBase):
    status: Literal["not_probeable", "not_selected"]


class FailedAssessment(AssessmentBase):
    status: Literal["timeout", "api_error", "process_error", "cancelled"]


class ScoredAssessment(AssessmentBase):
    status: Literal[
        "slow",
        "eligible",
        "published",
        "global_quota",
        "source_quota",
    ]
    worst_delay_ms: int = Field(ge=0, strict=True)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)

    def observed_delay_ms(self) -> int:
        return self.worst_delay_ms

    def reliability_score(self) -> float | None:
        return self.reliability

    def required_delay_ms(self) -> int:
        return self.worst_delay_ms


QualityAssessment = Annotated[
    UnobservedAssessment | FailedAssessment | ScoredAssessment,
    Field(discriminator="status"),
]

_ASSESSMENT_ADAPTER = TypeAdapter(QualityAssessment)


def admit_assessment(value: object) -> QualityAssessment:
    return _ASSESSMENT_ADAPTER.validate_python(value)


class QualitySelection(FrozenModel):
    catalog: NodeCatalog
    published: NodeCatalog
    assessments: tuple[QualityAssessment, ...] = Field(strict=False)

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
        return self

    def assessment_index(self) -> dict[str, QualityAssessment]:
        return {item.fingerprint: item for item in self.assessments}

    @property
    def exclusions(self) -> dict[str, str]:
        return {
            item.fingerprint: item.status
            for item in self.assessments
            if item.status != "published"
        }


def _source(node: Node) -> str:
    sites = {item.site for item in node.provenance if item.site}
    return min(sites) if sites else "unknown"


def select_probe_candidates(
    catalog: NodeCatalog,
    policy: QualityPolicy,
) -> tuple[ProbeableNode, ...]:
    """Round-robin sources, with stable ordering inside every source."""
    grouped: dict[str, deque[ProbeableNode]] = defaultdict(deque)
    probeable = sorted(
        catalog.clash_nodes,
        key=lambda node: (_source(node), node.fingerprint),
    )
    for node in probeable:
        grouped[_source(node)].append(node)

    selected: list[ProbeableNode] = []
    source_names = sorted(grouped)
    while source_names and len(selected) < policy.max_candidates:
        remaining_sources: list[str] = []
        for source_name in source_names:
            queue = grouped[source_name]
            if queue and len(selected) < policy.max_candidates:
                selected.append(queue.popleft())
            if queue:
                remaining_sources.append(source_name)
        source_names = remaining_sources
    return tuple(selected)


def assess_quality(
    catalog: NodeCatalog,
    evidence: Sequence[ProbeEvidence],
    policy: QualityPolicy,
    *,
    history: Mapping[str, QualityHistory] | None = None,
    as_of: date | None = None,
) -> QualitySelection:
    """Apply decisive evidence, stable ranking, and source/global quotas."""
    catalog_by_id = {node.fingerprint: node for node in catalog.nodes}
    probeable_by_id = {node.fingerprint: node for node in catalog.clash_nodes}
    evidence_by_id: dict[str, ProbeEvidence] = {}
    for item in evidence:
        if item.fingerprint not in catalog_by_id:
            raise QualityError("probe evidence does not belong to the catalog")
        if item.fingerprint in evidence_by_id:
            raise QualityError("duplicate probe evidence")
        evidence_by_id[item.fingerprint] = item

    observed_day = as_of or date.today()
    prior = history or {}
    assessments: dict[str, QualityAssessment] = {}
    eligible: list[tuple[float, int, str, ProbeableNode, ScoredAssessment]] = []

    for node in catalog.nodes:
        item = evidence_by_id.get(node.fingerprint)
        record = prior.get(node.fingerprint)
        reliability = (
            record.score(as_of=observed_day, history_days=policy.history_days)
            if record is not None
            else -1.0
        )
        retained_reliability = None if reliability < 0 else reliability
        probeable_node = probeable_by_id.get(node.fingerprint)
        if probeable_node is None:
            assessment: QualityAssessment = UnobservedAssessment(
                fingerprint=node.fingerprint,
                status="not_probeable",
            )
        elif item is None:
            assessment = UnobservedAssessment(
                fingerprint=node.fingerprint,
                status="not_selected",
            )
        elif item.status != "success":
            assessment = FailedAssessment(
                fingerprint=node.fingerprint,
                status=item.status,
            )
        else:
            coarse_delay = item.coarse.delay_ms
            confirm_delay = item.confirm.delay_ms if item.confirm else None
            if coarse_delay is None or confirm_delay is None:
                raise QualityError("successful evidence has incomplete delay values")
            worst_delay = max(coarse_delay, confirm_delay)
            status = "eligible" if worst_delay <= policy.max_delay_ms else "slow"
            assessment = ScoredAssessment(
                fingerprint=node.fingerprint,
                status=status,
                worst_delay_ms=worst_delay,
                reliability=retained_reliability,
            )
            if status == "eligible":
                eligible.append(
                    (
                        -reliability,
                        worst_delay,
                        node.fingerprint,
                        probeable_node,
                        assessment,
                    )
                )
        assessments[node.fingerprint] = assessment

    eligible.sort(key=lambda item: item[:3])
    published: list[ProbeableNode] = []
    source_counts: dict[str, int] = defaultdict(int)
    for _, _, fingerprint, node, assessment in eligible:
        source_name = _source(node)
        if len(published) >= policy.max_published:
            status = "global_quota"
        elif source_counts[source_name] >= policy.max_per_source:
            status = "source_quota"
        else:
            published.append(node)
            source_counts[source_name] += 1
            status = "published"
        assessments[fingerprint] = ScoredAssessment(
            fingerprint=fingerprint,
            status=status,
            worst_delay_ms=assessment.worst_delay_ms,
            reliability=assessment.reliability,
        )

    assessment_ledger = tuple(assessments[node.fingerprint] for node in catalog.nodes)
    return QualitySelection(
        catalog=catalog,
        published=NodeCatalog(
            nodes=tuple(published),
            rejections=catalog.rejections,
            receipts=catalog.receipts,
        ),
        assessments=assessment_ledger,
    )
