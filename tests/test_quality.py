"""Deterministic node-quality policy tests."""

from datetime import UTC, date, datetime, timedelta
from typing import Literal

from src.mihomo import DelayObservation, ProbeEvidence
from src.nodes import (
    DualNode,
    Node,
    NodeCatalog,
    NodeProvenance,
    UriNode,
    admit_proxy,
)
from src.quality import (
    DailyReliability,
    QualityHistory,
    QualityPolicy,
    assess_quality,
    select_probe_candidates,
)


def node(index: int, site: str, *, proxy: bool = True) -> Node:
    values = {
        "fingerprint": f"{index:064x}",
        "display_name": f"node-{index}",
        "provenance": (
            NodeProvenance(
                site=site,
                source_url=f"https://{site}.test/nodes",
                observed_at=datetime(2026, 8, 29, tzinfo=UTC),
                artifact_digest="a" * 64,
                item_index=index,
            ),
        ),
    }
    if proxy:
        return DualNode(
            **values,
            proxy=admit_proxy(
                {
                    "name": f"node-{index}",
                    "type": "ss",
                    "server": "example.com",
                    "port": 10000 + index,
                    "cipher": "aes-128-gcm",
                    "password": "secret",
                }
            ),
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
    return DelayObservation(endpoint=endpoint, status=status)


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
) -> ProbeEvidence:
    return evidence(
        item,
        success("coarse", coarse_ms),
        success("confirm", confirm_ms),
    )


def test_probe_allocation_is_fair_deterministic_and_bounded():
    nodes = [*(node(index, "a") for index in range(1, 6)), node(20, "b"), node(30, "c")]
    policy = QualityPolicy(max_candidates=4)

    forward = select_probe_candidates(NodeCatalog(nodes=tuple(nodes)), policy)
    reverse = select_probe_candidates(NodeCatalog(nodes=tuple(reversed(nodes))), policy)

    assert [item.fingerprint for item in forward] == [
        item.fingerprint for item in reverse
    ]
    assert [
        (item.provenance[0].site, int(item.fingerprint, 16)) for item in forward
    ] == [
        ("a", 1),
        ("b", 20),
        ("c", 30),
        ("a", 2),
    ]


def test_unprobeable_nodes_do_not_consume_the_probe_budget():
    catalog = NodeCatalog(nodes=(node(1, "a", proxy=False), node(2, "b")))

    selected = select_probe_candidates(catalog, QualityPolicy(max_candidates=1))

    assert [item.display_name for item in selected] == ["node-2"]


def test_both_endpoints_and_inclusive_delay_boundary_are_required():
    nodes = [node(index, "a") for index in range(1, 6)]
    observations = [
        successful_evidence(nodes[0], 100, 2500),
        successful_evidence(nodes[1], 100, 2501),
        evidence(nodes[2], failure("coarse", "timeout"), None),
        evidence(
            nodes[3],
            success("coarse", 100),
            failure("confirm", "api_error"),
        ),
    ]

    result = assess_quality(
        NodeCatalog(nodes=tuple(nodes)), observations, QualityPolicy()
    )

    assert [item.fingerprint for item in result.published.nodes] == [
        nodes[0].fingerprint
    ]
    assert result.exclusions == {
        nodes[1].fingerprint: "slow",
        nodes[2].fingerprint: "timeout",
        nodes[3].fingerprint: "api_error",
        nodes[4].fingerprint: "not_selected",
    }


def test_source_and_global_quotas_apply_after_stable_quality_ordering():
    nodes = [node(1, "a"), node(2, "a"), node(3, "a"), node(4, "b"), node(5, "b")]
    delays = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60}
    observations = [
        successful_evidence(
            item, delays[int(item.fingerprint, 16)], delays[int(item.fingerprint, 16)]
        )
        for item in nodes
    ]
    policy = QualityPolicy(max_published=3, max_per_source=2)

    result = assess_quality(NodeCatalog(nodes=tuple(nodes)), observations, policy)

    assert [int(item.fingerprint, 16) for item in result.published.nodes] == [5, 4, 3]
    assert result.exclusions[nodes[1].fingerprint] == "global_quota"
    assert result.exclusions[nodes[0].fingerprint] == "global_quota"


def test_recent_reliability_precedes_delay_and_old_history_is_ignored():
    fast = node(1, "a")
    reliable = node(2, "b")
    today = date(2026, 8, 29)
    history = {
        fast.fingerprint: QualityHistory(
            fingerprint=fast.fingerprint,
            days=(
                DailyReliability(
                    day=today - timedelta(days=8), successes=10, attempts=10
                ),
            ),
        ),
        reliable.fingerprint: QualityHistory(
            fingerprint=reliable.fingerprint,
            days=(DailyReliability(day=today, successes=9, attempts=10),),
        ),
    }

    result = assess_quality(
        NodeCatalog(nodes=(fast, reliable)),
        [successful_evidence(fast, 10, 10), successful_evidence(reliable, 200, 200)],
        QualityPolicy(max_published=1, history_days=7),
        history=history,
        as_of=today,
    )

    assert result.published.nodes == (reliable,)
    assert result.exclusions[fast.fingerprint] == "global_quota"


def test_every_admitted_node_is_published_or_has_one_exclusion():
    nodes = (node(1, "a"), node(2, "b", proxy=False), node(3, "c"))
    result = assess_quality(
        NodeCatalog(nodes=nodes), [successful_evidence(nodes[0])], QualityPolicy()
    )

    accounted = {item.fingerprint for item in result.published.nodes} | set(
        result.exclusions
    )
    assert accounted == {item.fingerprint for item in nodes}
    assert result.exclusions[nodes[1].fingerprint] == "not_probeable"
    assert result.exclusions[nodes[2].fingerprint] == "not_selected"
