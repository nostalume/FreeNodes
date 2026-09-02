"""Behavioral contract for deterministic, bounded capability planning."""

from datetime import UTC, datetime

from src.nodes import ClashNode, NodeCatalog, NodeProvenance, UriNode, admit_proxy
from src.quality import QualityPolicy, plan_probe_candidates

NOW = datetime(2026, 9, 2, tzinfo=UTC)


def node(index: int, source: str, protocol: str = "direct") -> ClashNode:
    payload: dict[str, object] = {"name": f"node-{index}", "type": protocol}
    if protocol == "ss":
        payload |= {
            "server": f"node-{index}.example",
            "port": 443,
            "cipher": "aes-128-gcm",
            "password": "secret",
        }
    return ClashNode(
        fingerprint=f"{index:064x}",
        display_name=f"node-{index}",
        proxy=admit_proxy(payload),
        provenance=(
            NodeProvenance(
                authority=source,
                site=source,
                source_url=f"https://{source}.test/nodes",
                observed_at=NOW,
                artifact_digest=f"{index + 1:064x}",
                item_index=index,
            ),
        ),
    )


def test_plan_is_stable_source_fair_and_protocol_interleaved():
    catalog = NodeCatalog(
        nodes=(
            node(4, "b", "ss"),
            node(2, "a", "ss"),
            node(3, "b"),
            node(1, "a"),
        )
    )
    policy = QualityPolicy(max_candidates=4, max_full_probes=4)

    first = plan_probe_candidates(catalog, policy)
    reordered = plan_probe_candidates(
        catalog.model_copy(update={"nodes": tuple(reversed(catalog.nodes))}), policy
    )

    assert tuple(entry.node.fingerprint for entry in first.entries) == tuple(
        entry.node.fingerprint for entry in reordered.entries
    )
    assert tuple(entry.sources for entry in first.entries[:2]) == (("a",), ("b",))
    assert {entry.protocol for entry in first.entries} == {"direct", "ss"}


def test_plan_excludes_uri_only_and_bounds_probe_effects_on_catalog_overflow():
    uri = UriNode(
        fingerprint="f" * 64,
        display_name="opaque",
        uri="ssr://opaque",
        provenance=node(9, "uri").provenance,
    )
    catalog = NodeCatalog(nodes=(node(1, "a"), uri, node(2, "b")))

    policy = QualityPolicy(max_candidates=1, max_full_probes=1)
    planned = plan_probe_candidates(catalog, policy)
    reordered = plan_probe_candidates(
        catalog.model_copy(update={"nodes": tuple(reversed(catalog.nodes))}), policy
    )

    assert len(planned.entries) == 1
    assert planned.entries[0].node.kind != "uri"
    assert planned.entries[0].node.fingerprint == reordered.entries[0].node.fingerprint


def test_plan_caps_each_source_and_makes_duplicate_proxy_names_safe():
    first = node(1, "a").model_copy(update={"display_name": "same"})
    second = node(2, "a").model_copy(update={"display_name": "same"})
    third = node(3, "a")

    planned = plan_probe_candidates(
        NodeCatalog(nodes=(first, second, third)),
        QualityPolicy(
            max_candidates=3,
            max_full_probes=3,
            max_probe_per_source=2,
        ),
    )

    assert len(planned.entries) == 2
    assert len({entry.node.display_name for entry in planned.entries}) == 2
