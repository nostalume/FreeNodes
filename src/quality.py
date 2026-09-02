"""Typed capability evidence, classification, and bounded candidate planning."""

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, HttpUrl, model_validator

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
    "validation_budget",
    "control_unavailable",
]


class CapabilityTarget(FrozenModel):
    id: str = Field(min_length=1)
    url: HttpUrl
    expected_status: int = Field(ge=100, le=599, strict=True)

    @model_validator(mode="after")
    def require_https(self) -> Self:
        if self.url.scheme != "https":
            raise ValueError("capability target must use HTTPS")
        return self

    @property
    def authority(self) -> str:
        if self.url.host is None:
            raise ValueError("capability target has no authority")
        return self.url.host

    @classmethod
    def admit_registry(
        cls, targets: Sequence[Self], *, quorum: int
    ) -> tuple[Self, ...]:
        admitted = tuple(targets)
        if quorum < 1 or quorum > len(admitted):
            raise ValueError("capability quorum is infeasible")
        if len({target.id for target in admitted}) != len(admitted):
            raise ValueError("capability targets require distinct ids")
        if len({target.authority for target in admitted}) != len(admitted):
            raise ValueError("capability targets require distinct authorities")
        return admitted


DEFAULT_CAPABILITY_TARGETS = CapabilityTarget.admit_registry(
    (
        CapabilityTarget(
            id="github",
            url=HttpUrl(
                "https://raw.githubusercontent.com/nostalume/FreeNodes/main/AGENTS.md"
            ),
            expected_status=200,
        ),
        CapabilityTarget(
            id="google",
            url=HttpUrl("https://www.gstatic.com/generate_204"),
            expected_status=204,
        ),
        CapabilityTarget(
            id="cloudflare",
            url=HttpUrl("https://cp.cloudflare.com/generate_204"),
            expected_status=204,
        ),
    ),
    quorum=2,
)


class TargetControlWindow(FrozenModel):
    target_id: str = Field(min_length=1)
    opening: Literal["success", "failure"]
    closing: Literal["success", "failure"]


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
    full_probe_limit: int = Field(default=4000, gt=0, le=4000)
    source_probe_limit: int = Field(default=32, gt=0, le=32)
    entries: tuple[ProbePlanEntry, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
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
    def validate_measurement(self) -> Self:
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
    ) -> Self:
        return cls(
            endpoint=endpoint,
            status=status,
            diagnostic=ProbeDiagnostic(code=code, detail=detail[:200]),
        )


class TargetObservation(FrozenModel):
    target_id: str = Field(min_length=1)
    status: Literal["success", "timeout", "request_error", "cancelled"]
    delay_ms: int | None = Field(default=None, gt=0, strict=True)
    diagnostic: ProbeDiagnostic | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if (self.status == "success") != (self.delay_ms is not None):
            raise ValueError("only successful target observations carry delay")
        if (self.status == "success") == (self.diagnostic is not None):
            raise ValueError("only failed target observations carry a diagnostic")
        return self

    @classmethod
    def from_delay(cls, target_id: str, value: DelayObservation) -> Self:
        status = (
            value.status
            if value.status in {"success", "timeout", "cancelled"}
            else "request_error"
        )
        return cls(
            target_id=target_id,
            status=status,
            delay_ms=value.delay_ms,
            diagnostic=value.diagnostic,
        )


class NodeCapabilityDecision(FrozenModel):
    fingerprint: str = Field(min_length=1)
    status: Literal["capable", "failed", "inconclusive"]
    successful_targets: tuple[str, ...] = Field(default=(), strict=False)
    failed_targets: tuple[str, ...] = Field(default=(), strict=False)
    target_delays: tuple[tuple[str, int], ...] = Field(default=(), strict=False)
    reason: Literal[
        "quorum",
        "invalid_control_window",
        "target_failures",
        "incomplete_evidence",
        "config_rejected",
    ]


class CapabilityRunReceipt(FrozenModel):
    status: Literal["complete", "inconclusive"]
    decisions: tuple[NodeCapabilityDecision, ...] = Field(default=(), strict=False)
    accepted_fingerprints: tuple[str, ...] = Field(default=(), strict=False)
    controls: tuple[TargetControlWindow, ...] = Field(default=(), strict=False)
    deadline_reached: bool = False
    diagnostic: ProbeDiagnostic | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if (self.status == "complete") == (self.diagnostic is not None):
            raise ValueError("only inconclusive capability runs carry a diagnostic")
        capable = set(self.capable_fingerprints)
        if not set(self.accepted_fingerprints).issubset(capable):
            raise ValueError("accepted fingerprints must be capable")
        if self.status == "inconclusive" and self.accepted_fingerprints:
            raise ValueError("inconclusive capability runs cannot accept nodes")
        return self

    @property
    def attempted(self) -> int:
        return len(self.decisions)

    @property
    def capable_fingerprints(self) -> tuple[str, ...]:
        return tuple(
            item.fingerprint for item in self.decisions if item.status == "capable"
        )


class ValidatedProbeBatch(FrozenModel):
    nodes: tuple[ProbeableNode, ...] = Field(strict=False)
    failures: tuple[NodeCapabilityDecision, ...] = Field(strict=False)
    complete: bool = True


class QualityError(ValueError):
    pass


class QualityPolicy(FrozenModel):
    max_candidates: int = Field(default=4000, gt=0)
    max_full_probes: int = Field(default=4000, gt=0, le=4000)
    max_probe_per_source: int = Field(default=32, gt=0, le=32)
    max_published: int = Field(default=500, gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.max_full_probes > self.max_candidates:
            raise ValueError("full probes exceed candidate ceiling")
        return self


def classify_capability(
    fingerprint: str,
    observations: Sequence[TargetObservation],
    controls: Sequence[TargetControlWindow],
    *,
    quorum: int = 2,
) -> NodeCapabilityDecision:
    windows = {item.target_id: item for item in controls}
    if quorum < 1 or len(windows) != len(controls):
        raise QualityError("invalid capability quorum or duplicate control window")
    valid = tuple(
        target_id
        for target_id, item in windows.items()
        if item.opening == item.closing == "success"
    )
    if len(valid) < quorum:
        invalid = tuple(target_id for target_id in windows if target_id not in valid)
        return NodeCapabilityDecision(
            fingerprint=fingerprint,
            status="inconclusive",
            failed_targets=invalid,
            reason="invalid_control_window",
        )
    measured = {item.target_id: item for item in observations}
    if len(measured) != len(observations) or not set(measured).issubset(windows):
        raise QualityError("duplicate or unknown target observation")
    successes = tuple(
        target_id
        for target_id in valid
        if measured.get(target_id) and measured[target_id].status == "success"
    )
    failures = tuple(
        target_id
        for target_id in valid
        if measured.get(target_id) and measured[target_id].status == "request_error"
    )
    delays = tuple(
        (target_id, delay)
        for target_id in successes
        if (delay := measured[target_id].delay_ms) is not None
    )
    if len(successes) >= quorum:
        status, reason = "capable", "quorum"
    elif len(valid) - len(failures) < quorum:
        status, reason = "failed", "target_failures"
    else:
        status, reason = "inconclusive", "incomplete_evidence"
    return NodeCapabilityDecision(
        fingerprint=fingerprint,
        status=status,
        successful_targets=successes,
        failed_targets=failures,
        target_delays=delays,
        reason=reason,
    )


def select_capable(
    catalog: NodeCatalog,
    decisions: Sequence[NodeCapabilityDecision],
) -> NodeCatalog:
    indexed = {node.fingerprint: node for node in catalog.nodes}
    capable = tuple(item.fingerprint for item in decisions if item.status == "capable")
    if len(set(capable)) != len(capable) or not set(capable).issubset(indexed):
        raise QualityError("capability decisions do not belong to the catalog")
    selected = NodeCatalog(
        nodes=tuple(
            indexed[fingerprint]
            for fingerprint in capable
            if indexed[fingerprint].kind != "uri"
        ),
        rejections=catalog.rejections,
        receipts=catalog.receipts,
        summary=catalog.summary,
    )
    if not selected.clash_nodes or not selected.uri_nodes:
        raise QualityError("capable catalog requires Clash and URI projections")
    return selected


def _sources(node: Node) -> tuple[str, ...]:
    sites = tuple(sorted({item.site for item in node.provenance if item.site}))
    return sites or ("unknown",)


def plan_probe_candidates(
    catalog: NodeCatalog,
    policy: QualityPolicy,
    *,
    sample_overflow: bool = False,
) -> ProbePlan:
    """Build one stable, source-fair capability plan before network I/O."""
    grouped: dict[str, dict[str, list[ProbeableNode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    memberships: dict[str, tuple[str, ...]] = {}
    probeable_count = 0
    for node in catalog.nodes:
        if node.kind == "uri":
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
        if not sample_overflow and len(memberships) > policy.max_candidates:
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

    selected: list[ProbeableNode] = []
    source_counts: dict[str, int] = defaultdict(int)
    source_names = sorted(source_queues)
    limit = min(policy.max_candidates, policy.max_full_probes)
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
