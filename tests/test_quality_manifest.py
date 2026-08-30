"""Redacted quality-bearing profile bundle contracts."""

import json
from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from src.config import RepositoryIdentity
from src.mihomo import DelayObservation, ProbeEvidence
from src.nodes import (
    ClashNode,
    DualNode,
    Node,
    NodeCatalog,
    NodeProvenance,
    SourceReceipt,
    admit_proxy,
)
from src.profiles import PublicEntryRegistry
from src.quality import QualityPolicy, assess_quality
from src.quality_manifest import (
    QualityManifestV1,
    load_quality_history,
    render_quality_bundle,
)

NOW = datetime(2026, 8, 29, 4, 0, tzinfo=UTC)
SECRET = "manifest-must-not-leak-this-password"


def node(index: int, site: str) -> DualNode:
    return DualNode(
        fingerprint=f"{index:064x}",
        display_name=f"SECRET NODE {index}",
        proxy=admit_proxy(
            {
                "name": f"SECRET NODE {index}",
                "type": "trojan",
                "server": "secret.example",
                "port": 443,
                "password": SECRET,
            }
        ),
        uri=f"trojan://{SECRET}@secret.example:443#SECRET-NODE-{index}",
        provenance=(
            NodeProvenance(
                site=site,
                source_url=f"https://{site}.test/{SECRET}",
                observed_at=NOW,
                artifact_digest=f"{index:064x}",
                item_index=index,
            ),
        ),
    )


def observed(item: Node, delay: int) -> ProbeEvidence:
    return ProbeEvidence(
        fingerprint=item.fingerprint,
        proxy_name=item.display_name,
        coarse=DelayObservation(
            endpoint="coarse",
            status="success",
            delay_ms=delay,
        ),
        confirm=DelayObservation(
            endpoint="confirm",
            status="success",
            delay_ms=delay + 10,
        ),
    )


def quality_bundle():
    fast = node(1, "source-a")
    slow = node(2, "source-b")
    catalog = NodeCatalog(
        nodes=(fast, slow),
        receipts=(
            SourceReceipt(
                site="source-a",
                source_url=f"https://a.test/{SECRET}",
                artifact_digest="a" * 64,
                observed_at=NOW,
                freshness="current",
            ),
            SourceReceipt(
                site="source-b",
                source_url=f"https://b.test/{SECRET}",
                artifact_digest="b" * 64,
                observed_at=NOW,
                freshness="stale",
            ),
        ),
    )
    policy = QualityPolicy(max_delay_ms=2500)
    selection = assess_quality(
        catalog, [observed(fast, 50), observed(slow, 2600)], policy, as_of=NOW.date()
    )
    bundle = render_quality_bundle(
        selection,
        policy,
        PublicEntryRegistry.from_identity(
            RepositoryIdentity(owner="owner", repository="repo")
        ),
        generated_at=NOW,
        runner_vantage="github-actions",
        failed_sources=("source-c",),
    )
    return selection, bundle


def test_quality_bundle_renders_only_published_nodes_in_every_profile():
    selection, bundle = quality_bundle()

    assert bundle.accepted_count == 1
    assert bundle.clash_count == 1
    assert bundle.uri_count == 1
    standalone = yaml.safe_load(bundle.files["nodes/merged.yaml"])
    plain = bundle.files["nodes/merged.txt"].decode()
    providers = [
        yaml.safe_load(content)
        for name, content in bundle.files.items()
        if name.startswith("nodes/source-") and name.endswith(".yaml")
    ]
    assert [proxy["name"] for proxy in standalone["proxies"]] == ["SECRET NODE 1"]
    assert "SECRET-NODE-1" in plain and "SECRET-NODE-2" not in plain
    assert sum(len(provider["proxies"]) for provider in providers) == 1
    assert len(selection.published.nodes) == 1


def test_quality_manifest_reconciles_counts_and_contains_no_node_secrets():
    _, bundle = quality_bundle()
    raw = bundle.files["nodes/quality-manifest.json"]
    manifest = json.loads(raw)

    assert manifest["schema"] == 1
    assert manifest["status"] == "quality_verified"
    assert manifest["counts"] == {
        "admitted": 2,
        "excluded": 1,
        "probe_success": 2,
        "published": 1,
        "selected_for_probe": 2,
    }
    assert manifest["exclusions"] == {"slow": 1}
    assert len(manifest["published"]) == 1
    assert manifest["published"][0]["worst_delay_ms"] == 60
    assert len(manifest["published"][0]["id"]) == 24
    assert SECRET.encode() not in raw
    assert b"SECRET NODE" not in raw
    assert b"secret.example" not in raw


def test_quality_manifest_schema_rejects_unknown_and_inconsistent_data():
    _, bundle = quality_bundle()
    payload = json.loads(bundle.files["nodes/quality-manifest.json"])
    QualityManifestV1.model_validate(payload)

    payload["counts"]["published"] = 2
    with pytest.raises(ValidationError, match="published"):
        QualityManifestV1.model_validate(payload)
    payload["counts"]["published"] = 1
    payload["credential"] = "must-not-be-admitted"
    with pytest.raises(ValidationError, match="credential"):
        QualityManifestV1.model_validate(payload)


def test_manifest_reports_source_freshness_and_failures_without_source_urls():
    _, bundle = quality_bundle()
    raw = bundle.files["nodes/quality-manifest.json"]
    manifest = json.loads(raw)

    assert manifest["sources"] == [
        {"name": "source-a", "state": "current"},
        {"name": "source-b", "state": "stale"},
        {"name": "source-c", "state": "failed"},
    ]
    assert b"https://" not in raw


def test_redacted_daily_history_round_trips_to_current_fingerprints(tmp_path):
    selection, bundle = quality_bundle()
    manifest_path = tmp_path / "quality-manifest.json"
    manifest_path.write_bytes(bundle.files["nodes/quality-manifest.json"])

    history = load_quality_history(
        manifest_path,
        NodeCatalog(
            nodes=tuple(
                ClashNode(
                    fingerprint=fingerprint,
                    display_name="different display name",
                    proxy=admit_proxy(
                        {"name": "different display name", "type": "direct"}
                    ),
                )
                for fingerprint in (
                    assessment.fingerprint for assessment in selection.assessments
                )
            )
        ),
        QualityPolicy(),
        as_of=NOW.date(),
    )

    assert set(history) == {
        assessment.fingerprint for assessment in selection.assessments
    }
    assert all(record.days[0].day == NOW.date() for record in history.values())
    assert all(record.days[0].attempts == 1 for record in history.values())
