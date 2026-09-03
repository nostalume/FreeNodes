from collections import defaultdict, deque
from collections.abc import Sequence
from ipaddress import ip_address
from typing import Literal, Self

from pydantic import Field, HttpUrl, model_validator

from freenodes.config import FrozenModel
from freenodes.nodes import (
    AdmittedCatalog,
    Node,
    ProbeableNode,
)

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
    def require_secure_target(self) -> Self:
        host = self.url.host or ""
        try:
            loopback = ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if self.url.scheme != "https" and not loopback:
            raise ValueError("capability target must use HTTPS or loopback HTTP")
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


class CapabilityError(ValueError):
    pass


class CapabilityPolicy(FrozenModel):
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
        raise CapabilityError("invalid capability quorum or duplicate control window")
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
        raise CapabilityError("duplicate or unknown target observation")
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


class CapableCatalog(FrozenModel):
    admitted: AdmittedCatalog = Field(repr=False)
    receipt: CapabilityRunReceipt = Field(repr=False)
    nodes: tuple[ProbeableNode, ...] = Field(strict=False)

    @staticmethod
    def _selected(
        admitted: AdmittedCatalog,
        receipt: CapabilityRunReceipt,
    ) -> tuple[ProbeableNode, ...]:
        if receipt.status != "complete":
            raise CapabilityError("capability measurement is inconclusive")
        accepted = receipt.accepted_fingerprints
        indexed = {node.fingerprint: node for node in admitted.nodes}
        measured = tuple(item.fingerprint for item in receipt.decisions)
        if (
            len(set(measured)) != len(measured)
            or not set(measured).issubset(indexed)
            or len(set(accepted)) != len(accepted)
            or not set(accepted).issubset(indexed)
        ):
            raise CapabilityError("capability receipt does not belong to the catalog")
        selected: list[ProbeableNode] = []
        for fingerprint in accepted:
            node = indexed[fingerprint]
            match node.kind:
                case "uri":
                    raise CapabilityError("accepted node has no testable proxy")
                case "clash" | "dual":
                    selected.append(node)
        if not selected or not any(node.kind == "dual" for node in selected):
            raise CapabilityError("capable catalog requires Clash and URI projections")
        return tuple(selected)

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.nodes != self._selected(self.admitted, self.receipt):
            raise ValueError("capable nodes must exactly match the capability receipt")
        return self

    @classmethod
    def from_measurement(
        cls,
        admitted: AdmittedCatalog,
        receipt: CapabilityRunReceipt,
    ) -> Self:
        return cls(
            admitted=admitted,
            receipt=receipt,
            nodes=cls._selected(admitted, receipt),
        )


def _sources(node: Node) -> tuple[str, ...]:
    sites = tuple(sorted({item.site for item in node.provenance if item.site}))
    return sites or ("unknown",)


class ProbePlanner:
    def __init__(self, catalog: AdmittedCatalog, policy: CapabilityPolicy):
        self.catalog = catalog
        self.policy = policy

    def plan(self) -> ProbePlan:
        grouped, memberships = self._group()
        selected = self._select(self._queues(grouped), memberships)
        return ProbePlan(
            candidate_ceiling=self.policy.max_candidates,
            full_probe_limit=self.policy.max_full_probes,
            source_probe_limit=self.policy.max_probe_per_source,
            entries=tuple(
                ProbePlanEntry(
                    ordinal=index,
                    node=node,
                    sources=memberships[node.fingerprint],
                    protocol=node.proxy.type,
                )
                for index, node in enumerate(self._unique_names(selected))
            ),
        )

    def _group(
        self,
    ) -> tuple[
        dict[str, dict[str, list[ProbeableNode]]],
        dict[str, tuple[str, ...]],
    ]:
        grouped: dict[str, dict[str, list[ProbeableNode]]] = defaultdict(
            lambda: defaultdict(list)
        )
        memberships: dict[str, tuple[str, ...]] = {}
        for node in self.catalog.clash_nodes:
            sources = _sources(node)
            grouped[sources[0]][node.proxy.type].append(node)
            memberships[node.fingerprint] = sources
        return grouped, memberships

    def _queues(
        self,
        grouped: dict[str, dict[str, list[ProbeableNode]]],
    ) -> dict[str, deque[ProbeableNode]]:
        queues: dict[str, deque[ProbeableNode]] = {}
        for source, protocols in grouped.items():
            buckets = {
                protocol: deque(
                    sorted(nodes, key=lambda node: node.fingerprint)[
                        : self.policy.max_probe_per_source
                    ]
                )
                for protocol, nodes in protocols.items()
            }
            ordered = deque[ProbeableNode]()
            while buckets:
                for protocol in sorted(buckets):
                    ordered.append(buckets[protocol].popleft())
                    if not buckets[protocol]:
                        del buckets[protocol]
            queues[source] = ordered
        return queues

    def _select(
        self,
        queues: dict[str, deque[ProbeableNode]],
        memberships: dict[str, tuple[str, ...]],
    ) -> list[ProbeableNode]:
        selected: list[ProbeableNode] = []
        source_counts: dict[str, int] = defaultdict(int)
        sources = sorted(queues)
        limit = min(self.policy.max_candidates, self.policy.max_full_probes)
        while sources and len(selected) < limit:
            remaining: list[str] = []
            for source in sources:
                self._take(queues[source], memberships, source_counts, selected, limit)
                if (
                    queues[source]
                    and source_counts[source] < self.policy.max_probe_per_source
                ):
                    remaining.append(source)
            sources = remaining
        return selected

    def _take(
        self,
        queue: deque[ProbeableNode],
        memberships: dict[str, tuple[str, ...]],
        source_counts: dict[str, int],
        selected: list[ProbeableNode],
        limit: int,
    ) -> None:
        while queue and len(selected) < limit:
            node = queue.popleft()
            sources = memberships[node.fingerprint]
            if any(
                source_counts[source] >= self.policy.max_probe_per_source
                for source in sources
            ):
                continue
            selected.append(node)
            for source in sources:
                source_counts[source] += 1
            return

    @staticmethod
    def _unique_names(selected: list[ProbeableNode]) -> list[ProbeableNode]:
        used: set[str] = set()
        for index, node in enumerate(selected):
            if node.display_name in used:
                selected[index] = node.model_copy(
                    update={
                        "display_name": f"{node.display_name} · {node.fingerprint[:8]}"
                    }
                )
            used.add(selected[index].display_name)
        return selected


def plan_probe_candidates(
    catalog: AdmittedCatalog,
    policy: CapabilityPolicy,
) -> ProbePlan:
    return ProbePlanner(catalog, policy).plan()
