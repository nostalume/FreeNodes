from datetime import timedelta

import pytest

from freenodes.capability import (
    CapabilityError,
    CapabilityPolicy,
    CapabilityRunReceipt,
    NodeCapabilityDecision,
    plan_probe_candidates,
    rank_measured_capable,
)
from freenodes.nodes import AdmittedCatalog
from freenodes.publication import (
    QualityHistory,
    QualityHistoryEntry,
    QualityHistoryGeneration,
)
from tests.test_node_admission import NOW, node


def measurement(catalog, delays, *, termination="candidates_exhausted", limit=2):
    decisions = tuple(
        NodeCapabilityDecision(
            fingerprint=item.fingerprint,
            status="capable" if values else "failed",
            successful_targets=("github", "google") if values else (),
            target_delays=tuple(zip(("github", "google"), values, strict=False)),
            reason="quorum" if values else "target_failures",
        )
        for item, values in zip(catalog.nodes, delays, strict=True)
    )
    return CapabilityRunReceipt(
        status="complete",
        planned=len(decisions) + (termination == "time_budget"),
        termination=termination,
        decisions=decisions,
        accepted_fingerprints=tuple(
            item.fingerprint for item in decisions if item.status == "capable"
        )[:limit],
        deadline_reached=termination == "time_budget",
    )


def test_rank_uses_median_not_single_minimum_and_is_order_invariant():
    catalog = AdmittedCatalog(nodes=tuple(node(index, "a") for index in range(1, 6)))
    run = measurement(catalog, ((1, 999), (60, 80), (40, 40), (), (40, 40)))
    policy = CapabilityPolicy(max_published=2, rank_latency=True)
    selected, report = rank_measured_capable(catalog, run, policy, elapsed_seconds=1)

    assert selected.accepted_fingerprints == (
        catalog.nodes[2].fingerprint,
        catalog.nodes[4].fingerprint,
    )
    assert report.baseline_median_delay_ms == 285
    assert report.selected_median_delay_ms == 40
    assert report.source_counts == {"a": (2, 2)}
    assert report.ranking_pool_complete is True
    assert set(selected.accepted_fingerprints) <= set(run.capable_fingerprints)
    reordered, _ = rank_measured_capable(
        catalog.model_copy(update={"nodes": tuple(reversed(catalog.nodes))}),
        run.model_copy(update={"decisions": tuple(reversed(run.decisions))}),
        policy,
        elapsed_seconds=1,
    )
    assert reordered.accepted_fingerprints == selected.accepted_fingerprints


def test_rank_preserves_source_protocol_fairness_before_latency():
    catalog = AdmittedCatalog(
        nodes=(node(1, "a"), node(2, "a"), node(3, "a", "ss"), node(4, "b"))
    )
    run = measurement(catalog, ((2, 2), (1, 1), (100, 100), (200, 200)), limit=3)
    selected, report = rank_measured_capable(
        catalog,
        run,
        CapabilityPolicy(max_published=3, rank_latency=True),
        elapsed_seconds=1,
    )
    assert selected == run
    assert report.selection == "no_improvement"
    assert report.source_counts == {"a": (3, 3)}
    assert report.protocol_counts == {"direct": (2, 2), "ss": (1, 1)}


def test_history_prefers_previous_success_but_uses_current_not_old_delay():
    catalog = AdmittedCatalog(nodes=tuple(node(index, "a") for index in range(1, 4)))
    history = QualityHistory(
        generations=(
            QualityHistoryGeneration(
                observed_at=NOW - timedelta(hours=4),
                entries=tuple(
                    QualityHistoryEntry(
                        fingerprint=item.fingerprint,
                        outcome="capable",
                        delay_ms=1,
                        consecutive_failures=0,
                    )
                    for item in catalog.nodes[:2]
                ),
            ),
        )
    )
    run = measurement(catalog, ((500, 500), (100, 100), (1, 1)), limit=1)
    selected, _ = rank_measured_capable(
        catalog,
        run,
        CapabilityPolicy(max_published=1, rank_latency=True),
        history=history,
        elapsed_seconds=1,
    )
    assert selected.accepted_fingerprints == (catalog.nodes[1].fingerprint,)


def test_deadline_falls_back_without_claiming_full_ranking_or_mutating_receipt():
    catalog = AdmittedCatalog(nodes=(node(1, "a"), node(2, "a")))
    run = measurement(
        catalog, ((900, 900), (10, 10)), termination="time_budget", limit=1
    )
    selected, report = rank_measured_capable(
        catalog,
        run,
        CapabilityPolicy(max_published=1, rank_latency=True),
        elapsed_seconds=1,
    )
    assert selected == run
    assert report.ranking_pool_complete is False
    assert report.selection == "incomplete_pool"
    assert report.selected_median_delay_ms == report.baseline_median_delay_ms == 900


def test_rank_falls_back_when_protocol_presence_would_be_lost():
    catalog = AdmittedCatalog(nodes=(node(1, "a", "ss"), node(2, "b"), node(3, "a")))
    run = measurement(catalog, ((900, 900), (100, 100), (10, 10)), limit=2)
    selected, report = rank_measured_capable(
        catalog,
        run,
        CapabilityPolicy(max_published=2, rank_latency=True),
        elapsed_seconds=1,
    )
    assert selected == run
    assert report.selection == "coverage_regression"


def test_ranked_target_stop_is_a_complete_extension():
    catalog = AdmittedCatalog(nodes=(node(1, "a"), node(2, "a")))
    run = measurement(
        catalog, ((900, 900), (10, 10)), termination="target_reached", limit=1
    )
    selected, report = rank_measured_capable(
        catalog,
        run,
        CapabilityPolicy(max_published=1, rank_latency=True),
        elapsed_seconds=1,
    )
    assert selected.accepted_fingerprints == (catalog.nodes[1].fingerprint,)
    assert report.ranking_pool_complete
    assert report.selection == "ranked"


def test_fewer_capable_nodes_are_not_padded_with_failed_or_unmeasured_nodes():
    catalog = AdmittedCatalog(nodes=(node(1, "a"), node(2, "a"), node(3, "a")))
    run = measurement(catalog, ((), (10, 10), ()), limit=500)
    selected, _ = rank_measured_capable(
        catalog, run, CapabilityPolicy(rank_latency=True), elapsed_seconds=1
    )
    assert selected.accepted_fingerprints == (catalog.nodes[1].fingerprint,)


def test_inconclusive_measurement_cannot_rank():
    from freenodes.capability import ProbeDiagnostic

    with pytest.raises(CapabilityError, match="complete measurement"):
        rank_measured_capable(
            AdmittedCatalog(),
            CapabilityRunReceipt(
                status="inconclusive",
                diagnostic=ProbeDiagnostic(code="control_unavailable"),
            ),
            CapabilityPolicy(rank_latency=True),
            elapsed_seconds=1,
        )


def test_foreign_measurement_is_rejected_before_ranking():
    catalog = AdmittedCatalog(nodes=(node(1, "a"),))
    foreign = AdmittedCatalog(nodes=(node(2, "a"),))
    with pytest.raises(CapabilityError, match="belong to the catalog"):
        rank_measured_capable(
            catalog,
            measurement(foreign, ((10, 10),)),
            CapabilityPolicy(rank_latency=True),
            elapsed_seconds=1,
        )


def test_all_quarantined_source_is_omitted_instead_of_creating_empty_queue():
    catalog = AdmittedCatalog(nodes=(node(1, "a"), node(2, "b")))
    history = QualityHistory(
        generations=(
            QualityHistoryGeneration(
                observed_at=NOW,
                entries=(
                    QualityHistoryEntry(
                        fingerprint=catalog.nodes[0].fingerprint,
                        outcome="failed",
                        consecutive_failures=2,
                    ),
                ),
            ),
        )
    )
    planned = plan_probe_candidates(
        catalog, CapabilityPolicy(rank_latency=True), history
    )
    assert tuple(entry.node.fingerprint for entry in planned.entries) == (
        catalog.nodes[1].fingerprint,
    )
