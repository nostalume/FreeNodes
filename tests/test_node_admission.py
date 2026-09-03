from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

import freenodes.nodes as nodes
from freenodes import proxies
from freenodes.capability import CapabilityPolicy, plan_probe_candidates
from freenodes.nodes import AdmittedCatalog, ClashNode, NodeProvenance, UriNode
from freenodes.proxies import admit_proxy

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
    catalog = AdmittedCatalog(
        nodes=(
            node(4, "b", "ss"),
            node(2, "a", "ss"),
            node(3, "b"),
            node(1, "a"),
        )
    )
    policy = CapabilityPolicy(max_candidates=4, max_full_probes=4)

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
    catalog = AdmittedCatalog(nodes=(node(1, "a"), uri, node(2, "b")))

    policy = CapabilityPolicy(max_candidates=1, max_full_probes=1)
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
        AdmittedCatalog(nodes=(first, second, third)),
        CapabilityPolicy(
            max_candidates=3,
            max_full_probes=3,
            max_probe_per_source=2,
        ),
    )

    assert len(planned.entries) == 2
    assert len({entry.node.display_name for entry in planned.entries}) == 2


NOW = datetime(2026, 8, 29, 4, 0, tzinfo=UTC)


def artifact(content: str, **overrides):
    values = {
        "site": "source-a",
        "source_url": "https://example.test/subscription",
        "content": content,
        "observed_at": NOW,
        "media_type": "text/plain",
    }
    values.update(overrides)
    return nodes.SourceArtifact(**values)


def test_yaml_provider_and_full_config_are_admitted_across_documents():
    source = artifact(
        """proxies:
  - {name: A, type: trojan, server: one.example, port: 443, password: alpha}
---
proxy-providers: {}
proxies:
  - {name: B, type: vmess, server: two.example, port: 443, uuid: 11111111-1111-1111-1111-111111111111}
""",
        media_type="application/yaml",
    )

    catalog = nodes.admit_artifacts([source], now=NOW)

    assert catalog.accepted_count == 2
    assert catalog.rejected_count == 0
    assert {node.display_name for node in catalog.nodes} == {"A", "B"}
    assert {type(node.proxy) for node in catalog.clash_nodes} == {
        proxies.TrojanProxy,
        proxies.VmessProxy,
    }


def test_artifact_requires_admitted_timezone_and_unknown_proxy_variant_is_rejected():
    with pytest.raises(ValidationError, match="observed_at"):
        artifact(
            "trojan://password@one.example:443#One", observed_at=datetime(2026, 8, 29)
        )

    catalog = nodes.admit_artifacts(
        [
            artifact(
                "proxies:\n  - {name: unknown, type: mystery, server: one.example, port: 443}\n"
            )
        ],
        now=NOW,
    )

    assert catalog.accepted_count == 0
    assert catalog.rejections[0].code == "unsupported_proxy_type"


def test_same_endpoint_with_different_auth_or_transport_is_preserved():
    source = artifact(
        """proxies:
  - {name: A, type: vmess, server: same.example, port: 443, uuid: 11111111-1111-1111-1111-111111111111, network: ws}
  - {name: B, type: vmess, server: same.example, port: 443, uuid: 22222222-2222-2222-2222-222222222222, network: grpc}
""",
        media_type="application/yaml",
    )

    assert nodes.admit_artifacts([source], now=NOW).accepted_count == 2


def test_semantic_duplicate_merges_provenance_and_ignores_display_name():
    first = artifact(
        "proxies:\n  - {name: First, type: trojan, server: same.example, port: 443, password: secret}\n",
    )
    second = artifact(
        "proxies:\n  - {name: Renamed, type: trojan, server: same.example, port: 443, password: secret}\n",
        site="source-b",
        source_url="https://other.example/sub",
    )

    catalog = nodes.admit_artifacts([first, second], now=NOW)

    assert catalog.accepted_count == 1
    assert len(catalog.nodes[0].provenance) == 2
    assert {item.site for item in catalog.nodes[0].provenance} == {
        "source-a",
        "source-b",
    }


def test_uri_and_base64_uri_containers_have_exact_counts():
    import base64

    uri_a = "vless://11111111-1111-1111-1111-111111111111@one.example:443?security=tls&type=ws#One"
    uri_b = "trojan://password@two.example:443?security=tls#Two"
    encoded = base64.b64encode(f"{uri_a}\n{uri_b}\n".encode()).decode()

    catalog = nodes.admit_artifacts(
        [artifact(uri_a), artifact(encoded, source_url="https://example.test/base64")],
        now=NOW,
    )

    assert catalog.accepted_count == 2
    assert catalog.uri_count == 2
    assert catalog.rejected_count == 0


def test_uri_identity_keeps_unprojected_transport_parameters():
    first = "vless://11111111-1111-1111-1111-111111111111@one.example:443?security=tls&type=grpc&serviceName=alpha#A"
    second = "vless://11111111-1111-1111-1111-111111111111@one.example:443?security=tls&type=grpc&serviceName=beta#B"

    catalog = nodes.admit_artifacts([artifact(f"{first}\n{second}\n")], now=NOW)

    assert catalog.accepted_count == 2


def test_arbitrary_text_and_malformed_yaml_are_decisively_rejected():
    catalog = nodes.admit_artifacts(
        [
            artifact('{"status":"ok","data":[1,2,3]}'),
            artifact("proxies: [", source_url="https://example.test/broken.yaml"),
        ],
        now=NOW,
    )

    assert catalog.accepted_count == 0
    assert {rejection.code for rejection in catalog.rejections} == {
        "malformed_node",
        "malformed_yaml",
    }


def test_structurally_invalid_proxy_is_rejected_without_secret_in_repr():
    secret = "do-not-log-this-password"
    catalog = nodes.admit_artifacts(
        [
            artifact(
                f"proxies:\n  - {{name: bad, type: trojan, server: x, port: 443, password: {secret}}}\n"
            )
        ],
        now=NOW,
    )

    assert catalog.accepted_count == 0
    assert catalog.rejections[0].code == "invalid_server"
    assert secret not in repr(catalog)
    assert secret not in repr(catalog.rejections)


def test_invalid_reality_public_key_is_rejected_before_mihomo():
    catalog = nodes.admit_artifacts(
        [
            artifact(
                """proxies:
  - name: bad-reality
    type: vless
    server: one.example
    port: 443
    uuid: 11111111-1111-1111-1111-111111111111
    tls: true
    reality-opts: {public-key: enabled}
"""
            )
        ],
        now=NOW,
    )

    assert catalog.accepted_count == 0
    assert catalog.rejections[0].code == "invalid_reality_public_key"


def test_source_publication_time_controls_freshness_before_parsing():
    valid = "trojan://password@one.example:443#One"
    stale = artifact(
        valid,
        publication_time=nodes.PublishedDate(on=(NOW - timedelta(days=3)).date()),
    )
    future = artifact(
        valid,
        publication_time=nodes.PublishedDate(on=(NOW + timedelta(days=1)).date()),
        source_url="inline://future",
    )

    catalog = nodes.admit_artifacts([stale, future], now=NOW)

    assert catalog.accepted_count == 0
    assert [item.code for item in catalog.rejections] == [
        "source_expired",
        "clock_inversion",
    ]
    assert [item.freshness for item in catalog.receipts] == ["expired", "future"]


def test_missing_source_publication_time_remains_unknown_not_fake_current():
    source = artifact("trojan://password@one.example:443#One")

    catalog = nodes.admit_artifacts([source], now=NOW)

    assert catalog.accepted_count == 1
    assert catalog.receipts[0].freshness == "unknown"


@pytest.mark.parametrize(
    ("days", "freshness", "rejection"),
    (
        (-1, "future", "clock_inversion"),
        (0, "current", None),
        (1, "stale", None),
        (2, "stale", None),
        (3, "expired", "source_expired"),
    ),
)
def test_date_only_publication_time_uses_calendar_boundaries(
    days, freshness, rejection
):
    source = artifact(
        "trojan://password@one.example:443#One",
        publication_time=nodes.PublishedDate(on=(NOW - timedelta(days=days)).date()),
    )

    catalog = nodes.admit_artifacts([source], now=NOW)

    assert catalog.receipts[0].freshness == freshness
    assert ([item.code for item in catalog.rejections] or [None]) == [rejection]


@pytest.mark.parametrize(
    ("age", "freshness", "rejection"),
    (
        (timedelta(hours=-1), "future", "clock_inversion"),
        (timedelta(hours=24), "current", None),
        (timedelta(hours=24, microseconds=1), "stale", None),
        (timedelta(hours=48), "stale", None),
        (timedelta(hours=48, microseconds=1), "expired", "source_expired"),
    ),
)
def test_exact_publication_time_uses_hour_boundaries(age, freshness, rejection):
    source = artifact(
        "trojan://password@one.example:443#One",
        publication_time=nodes.PublishedInstant(at=NOW - age),
    )

    catalog = nodes.admit_artifacts([source], now=NOW)

    assert catalog.receipts[0].freshness == freshness
    assert ([item.code for item in catalog.rejections] or [None]) == [rejection]


def test_admission_accounts_every_candidate_and_rejects_non_global_literal():
    duplicate = "trojan://password@one.example:443#One"
    catalog = nodes.admit_artifacts(
        [
            artifact(f"{duplicate}\n{duplicate}\nnot-a-node"),
            artifact(
                """proxies:
  - {name: local, type: trojan, server: 127.0.0.1, port: 443, password: secret}
""",
                source_url="https://example.test/local.yaml",
                media_type="application/yaml",
            ),
        ],
        now=NOW,
    )

    assert catalog.accepted_count == 1
    assert catalog.summary is not None
    assert catalog.summary.counts.candidate_records == 4
    assert catalog.summary.counts.rejected_records == 2
    assert catalog.summary.counts.eligible_occurrences == 2
    assert catalog.summary.counts.unique_eligible == 1
    assert catalog.summary.counts.duplicate_occurrences == 1
    assert {item.code for item in catalog.summary.rejection_codes} >= {
        "duplicate",
        "endpoint_scope",
        "malformed_node",
    }


def test_inline_uri_is_a_source_artifact_not_a_fake_download_url():
    source = nodes.SourceArtifact.inline(
        site="inline-source",
        content="trojan://password@one.example:443#One",
        observed_at=NOW,
        published_on=NOW.date(),
    )

    catalog = nodes.admit_artifacts([source], now=NOW)

    assert source.source_url.startswith("inline://")
    assert catalog.accepted_count == 1
    assert catalog.nodes[0].provenance[0].source_url == source.source_url
