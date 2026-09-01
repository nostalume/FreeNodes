"""Contracts for typed source artifacts and the admitted node catalog."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

import src.nodes as nodes

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
        nodes.TrojanProxy,
        nodes.VmessProxy,
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
        "unsupported_content",
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
    stale = artifact(valid, published_on=(NOW - timedelta(days=3)).date())
    future = artifact(
        valid,
        published_on=(NOW + timedelta(days=1)).date(),
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


def test_duplicate_names_are_assigned_deterministically():
    source = artifact(
        """proxies:
  - {name: Shared, type: trojan, server: one.example, port: 443, password: a}
  - {name: Shared, type: trojan, server: two.example, port: 443, password: b}
"""
    )

    names = [
        node.display_name for node in nodes.admit_artifacts([source], now=NOW).nodes
    ]

    assert names == ["Shared", "Shared_2"]


def test_artifact_content_is_hashed_once_for_all_admitted_nodes(monkeypatch):
    content = """proxies:
  - {name: one, type: ss, server: one.example, port: 443, cipher: aes-128-gcm, password: secret}
  - {name: two, type: ss, server: two.example, port: 443, cipher: aes-128-gcm, password: secret}
"""
    encoded = content.encode("utf-8")
    real_sha256 = nodes.hashlib.sha256
    content_hashes = 0

    def observed_sha256(value=b""):
        nonlocal content_hashes
        if value == encoded:
            content_hashes += 1
        return real_sha256(value)

    monkeypatch.setattr(nodes.hashlib, "sha256", observed_sha256)
    source = nodes.SourceArtifact(
        site="source",
        source_url="https://example.test/nodes.yaml",
        content=content,
        observed_at=NOW,
        published_on=NOW.date(),
        media_type="application/yaml",
    )

    catalog = nodes.admit_artifacts((source,), now=NOW)

    assert catalog.accepted_count == 2
    assert content_hashes == 1
