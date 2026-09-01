"""Deterministic node-quality policy tests."""

from datetime import UTC, date, datetime, timedelta
from typing import Literal

import pytest

import src.quality as quality
from src.nodes import (
    DualNode,
    Node,
    NodeCatalog,
    NodeProvenance,
    PublishedDate,
    SourceReceipt,
    UnknownPublicationTime,
    UriNode,
    admit_proxy,
)
from src.quality import (
    DelayObservation,
    ProbeDiagnostic,
    ProbeEvidence,
    ProbePlan,
    ProbePlanEntry,
    QualityPolicy,
    SourceHistory,
    SourceObservation,
    TransferObservation,
    assess_quality,
)

TEST_DAY = date(2026, 8, 29)


def relaxed_policy(**values) -> QualityPolicy:
    return QualityPolicy(
        min_source_nodes=values.pop("min_source_nodes", 1),
        min_source_qualified=values.pop("min_source_qualified", 1),
        min_source_pass_ratio=values.pop("min_source_pass_ratio", 0.01),
        min_source_unique=values.pop("min_source_unique", 1),
        **values,
    )


def node(
    index: int,
    site: str,
    *,
    proxy: bool = True,
    protocol: Literal["ss", "direct"] = "ss",
) -> Node:
    values = {
        "fingerprint": f"{index:064x}",
        "display_name": f"node-{index}",
        "provenance": (
            NodeProvenance(
                authority=site,
                site=site,
                source_url=f"https://{site}.test/nodes",
                observed_at=datetime(2026, 8, 29, tzinfo=UTC),
                publication_time=PublishedDate(on=TEST_DAY),
                artifact_digest="a" * 64,
                item_index=index,
            ),
        ),
    }
    if proxy:
        proxy_value = (
            {"name": f"node-{index}", "type": "direct"}
            if protocol == "direct"
            else {
                "name": f"node-{index}",
                "type": "ss",
                "server": "example.com",
                "port": 10000 + index,
                "cipher": "aes-128-gcm",
                "password": "secret",
            }
        )
        return DualNode(
            **values,
            proxy=admit_proxy(proxy_value),
            uri=f"ss://node-{index}",
        )
    return UriNode(
        **values,
        uri="ssr://opaque",
    )


def success(endpoint: str, delay_ms: int) -> DelayObservation:
    return DelayObservation(endpoint=endpoint, status="success", delay_ms=delay_ms)


def failure(
    endpoint: str,
    status: Literal["timeout", "api_error"],
) -> DelayObservation:
    code = "request_timeout" if status == "timeout" else "controller_error"
    return DelayObservation(
        endpoint=endpoint,
        status=status,
        diagnostic=ProbeDiagnostic(code=code),
    )


def failed_evidence(item: Node) -> ProbeEvidence:
    return evidence(item, failure("coarse", "timeout"), None)


def inconclusive_transfer_evidence(item: Node) -> ProbeEvidence:
    return ProbeEvidence(
        fingerprint=item.fingerprint,
        proxy_name=item.display_name,
        coarse=success("coarse", 100),
        confirm=success("confirm", 120),
        transfer=TransferObservation(
            fingerprint=item.fingerprint,
            target="primary+alternate",
            status="inconclusive",
            bytes_received=0,
            elapsed_ms=10,
            diagnostic=ProbeDiagnostic(code="control_transfer"),
        ),
    )


def evidence(
    item: Node,
    coarse: DelayObservation,
    confirm: DelayObservation | None,
) -> ProbeEvidence:
    return ProbeEvidence(
        fingerprint=item.fingerprint,
        proxy_name=item.display_name,
        coarse=coarse,
        confirm=confirm,
    )


def successful_evidence(
    item: Node,
    coarse_ms: int = 100,
    confirm_ms: int = 120,
    *,
    throughput: float = 1_000_000,
) -> ProbeEvidence:
    return ProbeEvidence(
        fingerprint=item.fingerprint,
        proxy_name=item.display_name,
        coarse=success("coarse", coarse_ms),
        confirm=success("confirm", confirm_ms),
        transfer=TransferObservation(
            fingerprint=item.fingerprint,
            target="test",
            status="success",
            bytes_received=1024 * 1024,
            elapsed_ms=1000,
            bytes_per_second=throughput,
        ),
    )


def assess(
    catalog: NodeCatalog,
    observations: list[ProbeEvidence],
    policy: QualityPolicy | None = None,
    *,
    history: dict[str, SourceHistory] | None = None,
    unavailable_sources: tuple[str, ...] = (),
    as_of: date = TEST_DAY,
):
    active_policy = policy or relaxed_policy()
    nodes = {item.fingerprint: item for item in catalog.clash_nodes}
    plan = ProbePlan(
        candidate_ceiling=active_policy.max_candidates,
        full_probe_limit=active_policy.max_full_probes,
        source_probe_limit=active_policy.max_probe_per_source,
        entries=tuple(
            ProbePlanEntry(
                ordinal=index,
                node=nodes[item.fingerprint],
                sources=tuple(
                    sorted({part.site for part in nodes[item.fingerprint].provenance})
                ),
                protocol=nodes[item.fingerprint].proxy.type,
            )
            for index, item in enumerate(observations)
        ),
    )
    return assess_quality(
        catalog,
        plan,
        observations,
        active_policy,
        history=history,
        unavailable_sources=unavailable_sources,
        as_of=as_of,
    )


def test_probe_allocation_is_fair_deterministic_and_bounded():
    nodes = [*(node(index, "a") for index in range(1, 6)), node(20, "b"), node(30, "c")]
    policy = QualityPolicy(
        max_candidates=4000,
        max_full_probes=4,
        max_probe_per_source=2,
    )

    forward = quality.plan_probe_candidates(NodeCatalog(nodes=tuple(nodes)), policy)
    reverse = quality.plan_probe_candidates(
        NodeCatalog(nodes=tuple(reversed(nodes))), policy
    )

    assert [item.node.fingerprint for item in forward.entries] == [
        item.node.fingerprint for item in reverse.entries
    ]
    assert [
        (item.sources[0], int(item.node.fingerprint, 16)) for item in forward.entries
    ] == [
        ("a", 1),
        ("b", 20),
        ("c", 30),
        ("a", 2),
    ]


def test_unprobeable_nodes_do_not_consume_the_probe_budget():
    catalog = NodeCatalog(nodes=(node(1, "a", proxy=False), node(2, "b")))

    selected = quality.plan_probe_candidates(
        catalog,
        QualityPolicy(max_candidates=1, max_full_probes=1),
    )

    assert [item.node.display_name for item in selected.entries] == ["node-2"]


def test_probe_plan_rejects_catalog_over_candidate_ceiling():
    catalog = NodeCatalog(nodes=tuple(node(index, "a") for index in range(1, 4)))

    with pytest.raises(quality.QualityError, match="candidate ceiling"):
        quality.plan_probe_candidates(catalog, QualityPolicy(max_candidates=2))


def test_probe_plan_deterministically_samples_audit_catalog_overflow():
    nodes = tuple(node(index, "a") for index in range(1, 4))
    policy = QualityPolicy(
        max_candidates=2,
        max_full_probes=2,
        max_probe_per_source=2,
    )

    forward = quality.plan_probe_candidates(
        NodeCatalog(nodes=nodes),
        policy,
        sample_overflow=True,
    )
    reverse = quality.plan_probe_candidates(
        NodeCatalog(nodes=tuple(reversed(nodes))),
        policy,
        sample_overflow=True,
    )

    assert tuple(entry.node.fingerprint for entry in forward.entries) == tuple(
        entry.node.fingerprint for entry in reverse.entries
    )
    assert tuple(int(entry.node.fingerprint, 16) for entry in forward.entries) == (1, 2)


def test_probe_plan_stratifies_protocols_and_caps_each_source():
    mixed = [
        node(1, "a"),
        node(2, "a"),
        node(3, "a", protocol="direct"),
        node(4, "a", protocol="direct"),
    ]
    forward = quality.plan_probe_candidates(
        NodeCatalog(nodes=tuple(mixed)), QualityPolicy()
    )
    reverse = quality.plan_probe_candidates(
        NodeCatalog(nodes=tuple(reversed(mixed))), QualityPolicy()
    )

    assert [entry.node.fingerprint for entry in forward.entries] == [
        entry.node.fingerprint for entry in reverse.entries
    ]
    assert [entry.protocol for entry in forward.entries] == [
        "direct",
        "ss",
        "direct",
        "ss",
    ]

    many = tuple(node(index, "a") for index in range(1, 41)) + tuple(
        node(index, "b") for index in range(41, 81)
    )
    bounded = quality.plan_probe_candidates(NodeCatalog(nodes=many), QualityPolicy())
    counts = {
        site: sum(site in entry.sources for entry in bounded.entries)
        for site in ("a", "b")
    }

    assert counts == {"a": 32, "b": 32}


def test_probe_plan_charges_every_provenance_membership_to_its_source_cap():
    shared = tuple(
        node(index, "a").model_copy(
            update={
                "provenance": (
                    *node(index, "a").provenance,
                    *node(index, "b").provenance,
                )
            }
        )
        for index in range(1, 3)
    )
    catalog = NodeCatalog(nodes=(*shared, node(3, "b"), node(4, "b")))

    plan = quality.plan_probe_candidates(
        catalog,
        QualityPolicy(max_full_probes=4, max_probe_per_source=2),
    )

    assert [int(entry.node.fingerprint, 16) for entry in plan.entries] == [1, 3]
    assert sum("a" in entry.sources for entry in plan.entries) == 1
    assert sum("b" in entry.sources for entry in plan.entries) == 2


def test_both_endpoints_and_inclusive_delay_boundary_are_required():
    nodes = [node(index, "a") for index in range(1, 6)]
    observations = [
        successful_evidence(nodes[0], 100, 2500),
        evidence(
            nodes[1],
            success("coarse", 100),
            success("confirm", 2501),
        ),
        evidence(nodes[2], failure("coarse", "timeout"), None),
        evidence(
            nodes[3],
            success("coarse", 100),
            failure("confirm", "api_error"),
        ),
    ]

    result = assess(NodeCatalog(nodes=tuple(nodes)), observations, relaxed_policy())

    assert [item.fingerprint for item in result.published.nodes] == [
        nodes[0].fingerprint
    ]
    assert result.exclusions == {
        nodes[1].fingerprint: "slow",
        nodes[2].fingerprint: "timeout",
        nodes[3].fingerprint: "api_error",
        nodes[4].fingerprint: "not_selected",
    }


def test_delay_success_without_transfer_evidence_is_rejected():
    candidate = node(1, "a")

    with pytest.raises(quality.QualityError, match="transfer"):
        assess(
            NodeCatalog(nodes=(candidate,)),
            [
                evidence(
                    candidate,
                    success("coarse", 100),
                    success("confirm", 120),
                )
            ],
            QualityPolicy(),
        )


def test_slow_delay_is_excluded_without_transfer_measurement():
    candidate = node(1, "a")

    result = assess(
        NodeCatalog(nodes=(candidate,)),
        [
            evidence(
                candidate,
                success("coarse", 2500),
                success("confirm", 2501),
            )
        ],
        relaxed_policy(),
    )

    assert result.exclusions[candidate.fingerprint] == "slow"


def test_failed_transfer_is_excluded_with_its_typed_diagnostic():
    candidate = node(1, "a")
    observation = successful_evidence(candidate).model_copy(
        update={
            "transfer": TransferObservation(
                fingerprint=candidate.fingerprint,
                target="test",
                status="short_read",
                bytes_received=512,
                elapsed_ms=10,
                diagnostic=ProbeDiagnostic(code="transfer_short_read"),
            )
        }
    )

    result = assess(NodeCatalog(nodes=(candidate,)), [observation], relaxed_policy())

    assert result.exclusions[candidate.fingerprint] == "transfer_failed"
    assert result.assessments[0].failure_code() == "transfer_short_read"


def test_throughput_breaks_equal_source_reliability_and_delay_ties():
    slower = node(1, "a")
    faster = node(2, "a")

    result = assess(
        NodeCatalog(nodes=(slower, faster)),
        [
            successful_evidence(slower, throughput=500_000),
            successful_evidence(faster, throughput=2_000_000),
        ],
        relaxed_policy(max_published=1),
    )

    assert result.published.nodes == (faster,)


def test_source_below_minimum_population_does_not_erase_passed_nodes():
    candidates = tuple(node(index, "a") for index in range(1, 20))

    result = assess(
        NodeCatalog(nodes=candidates),
        [successful_evidence(candidate) for candidate in candidates],
        QualityPolicy(),
    )

    assert len(result.published.nodes) == 19
    assert result.sources[0].reason == "insufficient_population"


def test_source_threshold_boundaries_are_inclusive():
    candidates = tuple(node(index, "a") for index in range(1, 21))
    observations = [
        successful_evidence(candidate) if index < 8 else failed_evidence(candidate)
        for index, candidate in enumerate(candidates)
    ]

    result = assess(NodeCatalog(nodes=candidates), observations, QualityPolicy())

    assert len(result.published.nodes) == 8
    assert result.sources[0].status == "eligible"
    assert result.sources[0].observation.pass_ratio == 0.4


def test_source_with_eight_qualified_but_subthreshold_ratio_is_rejected():
    candidates = tuple(node(index, "a") for index in range(1, 22))
    observations = [
        successful_evidence(candidate) if index < 8 else failed_evidence(candidate)
        for index, candidate in enumerate(candidates)
    ]

    result = assess(NodeCatalog(nodes=candidates), observations, QualityPolicy())

    assert len(result.published.nodes) == 8
    assert result.sources[0].reason == "low_pass_ratio"


def test_source_older_than_the_age_boundary_is_rejected():
    candidates = tuple(
        node(index, "a").model_copy(
            update={
                "provenance": tuple(
                    item.model_copy(
                        update={
                            "publication_time": PublishedDate(
                                on=TEST_DAY - timedelta(days=2)
                            )
                        }
                    )
                    for item in node(index, "a").provenance
                )
            }
        )
        for index in range(1, 21)
    )
    observations = [successful_evidence(candidate) for candidate in candidates]

    result = assess(NodeCatalog(nodes=candidates), observations, QualityPolicy())

    assert len(result.published.nodes) == 20
    assert result.sources[0].reason == "source_stale"


def test_inconclusive_nodes_do_not_reduce_source_productivity():
    candidates = tuple(node(index, "a") for index in range(1, 21))
    observations = [
        successful_evidence(candidate)
        if index < 8
        else inconclusive_transfer_evidence(candidate)
        for index, candidate in enumerate(candidates)
    ]

    result = assess(NodeCatalog(nodes=candidates), observations, QualityPolicy())

    assert result.sources[0].observation.sampled == 8
    assert result.sources[0].observation.pass_ratio == 1
    assert len(result.published.nodes) == 8


@pytest.mark.parametrize(
    ("published_on", "freshness"),
    ((TEST_DAY - timedelta(days=3), "expired"), (None, "unknown")),
)
def test_unadmitted_source_receipt_remains_stale_instead_of_unavailable(
    published_on,
    freshness,
):
    receipt = SourceReceipt(
        authority="a",
        site="a",
        source_url="https://a.test/expired",
        artifact_digest="a" * 64,
        observed_at=datetime(2026, 8, 29, tzinfo=UTC),
        publication_time=(
            PublishedDate(on=published_on)
            if published_on is not None
            else UnknownPublicationTime()
        ),
        freshness=freshness,
    )

    result = assess(NodeCatalog(receipts=(receipt,)), [], QualityPolicy())

    assert result.sources[0].reason == "source_stale"


def test_old_artifacts_cannot_inflate_a_current_source_population():
    current = node(1, "a")
    old = tuple(
        node(index, "a").model_copy(
            update={
                "provenance": tuple(
                    item.model_copy(
                        update={
                            "publication_time": PublishedDate(
                                on=TEST_DAY - timedelta(days=2)
                            )
                        }
                    )
                    for item in node(index, "a").provenance
                )
            }
        )
        for index in range(2, 21)
    )
    candidates = (current, *old)

    result = assess(
        NodeCatalog(nodes=candidates),
        [successful_evidence(candidate) for candidate in candidates],
        QualityPolicy(),
    )

    assert result.sources[0].reason == "insufficient_population"
    assert result.sources[0].observation.admitted_unique == 1


def test_redundant_source_does_not_claim_shared_nodes_twice():
    source_a = tuple(node(index, "a") for index in range(1, 21))
    shared = tuple(
        candidate.model_copy(
            update={
                "provenance": (
                    *candidate.provenance,
                    *node(index, "b").provenance,
                )
            }
        )
        for index, candidate in enumerate(source_a[:16], start=1)
    )
    catalog = NodeCatalog(
        nodes=(*shared, *source_a[16:], *(node(i, "b") for i in range(21, 25)))
    )
    observations = [successful_evidence(candidate) for candidate in catalog.nodes]

    result = assess(catalog, observations, QualityPolicy())
    decisions = {item.source: item for item in result.sources}

    assert decisions["a"].status == "eligible"
    assert decisions["b"].status == "rejected"
    assert decisions["b"].reason == "insufficient_unique"
    assert decisions["b"].observation.unique_contribution == 4


def source_observation(day: date, outcome: str, qualified: int) -> SourceObservation:
    return SourceObservation(
        day=day,
        outcome=outcome,
        reason="current" if outcome == "eligible" else "insufficient_qualified",
        admitted_unique=20,
        sampled=20,
        qualified=qualified,
        pass_ratio=qualified / 20,
        unique_contribution=20,
    )


def test_source_retains_one_failure_then_quarantines_and_rejects_decay():
    candidates = tuple(node(index, "a") for index in range(1, 21))
    failures = [failed_evidence(candidate) for candidate in candidates]
    prior_success = source_observation(TEST_DAY - timedelta(days=1), "eligible", 8)

    retained = assess(
        NodeCatalog(nodes=candidates),
        failures,
        QualityPolicy(),
        history={"a": SourceHistory(source="a", observations=(prior_success,))},
    )
    quarantined = assess(
        NodeCatalog(nodes=candidates),
        failures,
        QualityPolicy(),
        history={
            "a": SourceHistory(
                source="a",
                observations=(
                    prior_success.model_copy(
                        update={"day": TEST_DAY - timedelta(days=2)}
                    ),
                    source_observation(TEST_DAY - timedelta(days=1), "failed", 0),
                ),
            )
        },
    )
    rejected = assess(
        NodeCatalog(nodes=candidates),
        failures,
        QualityPolicy(),
        history={
            "a": SourceHistory(
                source="a",
                observations=(
                    prior_success.model_copy(
                        update={"day": TEST_DAY - timedelta(days=3)}
                    ),
                    source_observation(TEST_DAY - timedelta(days=2), "failed", 0),
                    source_observation(TEST_DAY - timedelta(days=1), "failed", 0),
                ),
            )
        },
    )

    assert (retained.sources[0].status, retained.sources[0].reason) == (
        "eligible",
        "retained",
    )
    assert quarantined.sources[0].status == "quarantined"
    assert rejected.sources[0].status == "rejected"
    assert rejected.sources[0].reason == "repeated_failure"


def test_source_is_removal_eligible_after_seven_days_without_recovery():
    candidates = tuple(node(index, "a") for index in range(1, 21))
    history = SourceHistory(
        source="a",
        observations=(source_observation(TEST_DAY - timedelta(days=7), "eligible", 8),),
    )

    result = assess(
        NodeCatalog(nodes=candidates),
        [failed_evidence(candidate) for candidate in candidates],
        QualityPolicy(),
        history={"a": history},
    )

    assert result.sources[0].status == "rejected"
    assert result.sources[0].reason == "unrecovered"


def test_source_and_global_quotas_apply_after_stable_quality_ordering():
    nodes = [node(1, "a"), node(2, "a"), node(3, "a"), node(4, "b"), node(5, "b")]
    delays = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60}
    observations = [
        successful_evidence(
            item, delays[int(item.fingerprint, 16)], delays[int(item.fingerprint, 16)]
        )
        for item in nodes
    ]
    policy = relaxed_policy(max_published=3, max_per_source=2)

    result = assess(NodeCatalog(nodes=tuple(nodes)), observations, policy)

    assert [int(item.fingerprint, 16) for item in result.published.nodes] == [5, 4, 3]
    assert result.exclusions[nodes[1].fingerprint] == "global_quota"
    assert result.exclusions[nodes[0].fingerprint] == "global_quota"


def test_recent_source_reliability_precedes_delay():
    fast = node(1, "a")
    reliable = node(2, "b")
    today = date(2026, 8, 29)
    history = {
        "a": SourceHistory(
            source="a",
            observations=(
                SourceObservation(
                    day=today - timedelta(days=1),
                    outcome="failed",
                    reason="low_pass_ratio",
                    admitted_unique=1,
                    sampled=1,
                    qualified=0,
                    pass_ratio=0,
                    unique_contribution=1,
                ),
            ),
        ),
        "b": SourceHistory(
            source="b",
            observations=(
                SourceObservation(
                    day=today - timedelta(days=1),
                    outcome="eligible",
                    reason="current",
                    admitted_unique=1,
                    sampled=1,
                    qualified=1,
                    pass_ratio=1,
                    unique_contribution=1,
                ),
            ),
        ),
    }

    result = assess(
        NodeCatalog(nodes=(fast, reliable)),
        [successful_evidence(fast, 10, 10), successful_evidence(reliable, 200, 200)],
        relaxed_policy(max_published=1),
        history=history,
        as_of=today,
    )

    assert result.published.nodes == (reliable,)
    assert result.exclusions[fast.fingerprint] == "global_quota"


def test_every_admitted_node_is_published_or_has_one_exclusion():
    nodes = (node(1, "a"), node(2, "b", proxy=False), node(3, "c"))
    result = assess(
        NodeCatalog(nodes=nodes), [successful_evidence(nodes[0])], QualityPolicy()
    )

    accounted = {item.fingerprint for item in result.published.nodes} | set(
        result.exclusions
    )
    assert accounted == {item.fingerprint for item in nodes}
    assert result.exclusions[nodes[1].fingerprint] == "not_probeable"
    assert result.exclusions[nodes[2].fingerprint] == "not_selected"
