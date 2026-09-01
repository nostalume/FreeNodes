"""Redacted quality-bearing profile bundle contracts."""

import json
from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from src.config import RepositoryIdentity
from src.nodes import (
    DualNode,
    Node,
    NodeCatalog,
    NodeProvenance,
    SourceReceipt,
    admit_proxy,
)
from src.profiles import PublicEntryRegistry
from src.publication import render_publication_report
from src.quality import (
    DelayObservation,
    ProbeDiagnostic,
    ProbeEvidence,
    QualityPolicy,
    TransferObservation,
    TransferTargetEvidence,
    assess_quality,
    plan_probe_candidates,
)
from src.quality_manifest import (
    QualityManifestV1,
    QualityManifestV3,
    admit_quality_manifest_json,
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
                published_on=NOW.date(),
                artifact_digest=f"{index:064x}",
                item_index=index,
            ),
        ),
    )


def observed(item: Node, delay: int) -> ProbeEvidence:
    transfer = None
    if delay + 10 <= 2500:
        transfer = TransferObservation(
            fingerprint=item.fingerprint,
            target="test",
            status="success",
            bytes_received=1024 * 1024,
            elapsed_ms=100,
            bytes_per_second=10_000_000,
        )
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
        transfer=transfer,
    )


def manifest_policy() -> QualityPolicy:
    return QualityPolicy(
        min_source_nodes=1,
        min_source_qualified=1,
        min_source_pass_ratio=0.01,
        min_source_unique=1,
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
    policy = manifest_policy()
    plan = plan_probe_candidates(catalog, policy)
    selection = assess_quality(
        catalog,
        plan,
        [observed(fast, 50), observed(slow, 2600)],
        policy,
        unavailable_sources=("source-c",),
        as_of=NOW.date(),
    )
    bundle = render_quality_bundle(
        selection,
        policy,
        PublicEntryRegistry.from_identity(
            RepositoryIdentity(owner="owner", repository="repo")
        ),
        generated_at=NOW,
        runner_vantage="github-actions",
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


def test_publication_report_uses_the_admitted_quality_manifest(tmp_path):
    _, bundle = quality_bundle()
    manifest = tmp_path / "nodes" / "quality-manifest.json"
    manifest.parent.mkdir()
    manifest.write_bytes(bundle.files["nodes/quality-manifest.json"])

    report = render_publication_report(tmp_path)

    assert "Decision: insufficient_authority_diversity" in report
    assert "published 1 of 2 admitted nodes" in report
    assert "Node evidence: passed 1, failed 1, inconclusive 0, not probed 0" in report


def test_quality_manifest_reconciles_counts_and_contains_no_node_secrets():
    _, bundle = quality_bundle()
    raw = bundle.files["nodes/quality-manifest.json"]
    manifest = json.loads(raw)

    assert manifest["schema"] == 3
    assert manifest["status"] == "quality_verified"
    assert manifest["decision"] == "insufficient_authority_diversity"
    assert manifest["contributing_authorities"] == ["source-a"]
    assert manifest["counts"] == {
        "admitted": 2,
        "excluded": 1,
        "failed": 1,
        "inconclusive": 0,
        "not_probed": 0,
        "probe_success": 1,
        "published": 1,
        "selected_for_probe": 2,
    }
    assert manifest["exclusions"] == {"slow": 1}
    assert len(manifest["published"]) == 1
    assert manifest["published"][0]["worst_delay_ms"] == 60
    assert manifest["published"][0]["bytes_per_second"] == 10_000_000
    assert manifest["published"][0]["transfer_target"] == "test"
    assert len(manifest["published"][0]["id"]) == 24
    assert SECRET.encode() not in raw
    assert b"SECRET NODE" not in raw
    assert b"secret.example" not in raw


def test_quality_manifest_reports_typed_probe_failure_codes():
    fast = node(1, "source-a")
    failed = node(2, "source-b")
    catalog = NodeCatalog(nodes=(fast, failed))
    policy = manifest_policy()
    plan = plan_probe_candidates(catalog, policy)
    selection = assess_quality(
        catalog,
        plan,
        [
            observed(fast, 50),
            ProbeEvidence(
                fingerprint=failed.fingerprint,
                proxy_name=failed.display_name,
                coarse=DelayObservation(
                    endpoint="coarse",
                    status="api_error",
                    diagnostic=ProbeDiagnostic(
                        code="controller_http", detail="HTTP 503"
                    ),
                ),
            ),
        ],
        policy,
        as_of=NOW.date(),
    )

    bundle = render_quality_bundle(
        selection,
        policy,
        generated_at=NOW,
        runner_vantage="github-actions",
    )

    manifest = json.loads(bundle.files["nodes/quality-manifest.json"])
    assert manifest["probe_failures"] == {"controller_http": 1}


def test_quality_manifest_reports_transfer_target_controls_and_attempts():
    selection, _ = quality_bundle()
    bundle = render_quality_bundle(
        selection,
        manifest_policy(),
        generated_at=NOW,
        runner_vantage="github-actions",
        transfer_targets=(
            TransferTargetEvidence(
                name="primary",
                authority="target-a",
                controls_attempted=2,
                controls_passed=2,
                candidate_attempts=1,
            ),
        ),
    )

    manifest = json.loads(bundle.files["nodes/quality-manifest.json"])
    assert manifest["transfer_targets"] == [
        {
            "name": "primary",
            "authority": "target-a",
            "controls_attempted": 2,
            "controls_passed": 2,
            "candidate_attempts": 1,
        }
    ]


def test_quality_manifest_schema_rejects_unknown_and_inconsistent_data():
    _, bundle = quality_bundle()
    payload = json.loads(bundle.files["nodes/quality-manifest.json"])
    QualityManifestV3.model_validate(payload)

    payload["counts"]["published"] = 2
    with pytest.raises(ValidationError, match="published"):
        QualityManifestV3.model_validate(payload)
    payload["counts"]["published"] = 1
    payload["credential"] = "must-not-be-admitted"
    with pytest.raises(ValidationError, match="credential"):
        QualityManifestV3.model_validate(payload)


def test_manifest_reports_source_freshness_and_failures_without_source_urls():
    _, bundle = quality_bundle()
    raw = bundle.files["nodes/quality-manifest.json"]
    manifest = json.loads(raw)

    assert [(item["name"], item["status"]) for item in manifest["sources"]] == [
        ("source-a", "eligible"),
        ("source-b", "rejected"),
        ("source-c", "rejected"),
    ]
    assert b"https://" not in raw


def test_redacted_source_history_round_trips_without_node_identity(tmp_path):
    selection, bundle = quality_bundle()
    manifest_path = tmp_path / "quality-manifest.json"
    manifest_path.write_bytes(bundle.files["nodes/quality-manifest.json"])

    history = load_quality_history(
        manifest_path,
        manifest_policy(),
        as_of=NOW.date(),
    )

    assert set(history) == {"source-a", "source-b", "source-c"}
    assert all(record.observations[-1].day == NOW.date() for record in history.values())
    assert all(SECRET not in record.model_dump_json() for record in history.values())


def test_schema_one_manifest_is_admitted_as_an_empty_source_history(tmp_path):
    _, bundle = quality_bundle()
    payload = json.loads(bundle.files["nodes/quality-manifest.json"])
    payload["schema"] = 1
    payload.pop("decision")
    payload.pop("contributing_authorities")
    payload["counts"] = {
        "admitted": 2,
        "selected_for_probe": 2,
        "probe_success": 1,
        "published": 1,
        "excluded": 1,
    }
    payload.pop("transfer_targets")
    payload["policy"] = {
        "max_candidates": 4000,
        "max_published": 500,
        "max_per_source": 100,
        "max_delay_ms": 2500,
        "history_days": 7,
        "required_endpoints": 2,
    }
    payload["sources"] = [{"name": "source-a", "state": "current"}]
    payload["published"] = [
        {
            "id": payload["published"][0]["id"],
            "worst_delay_ms": 60,
            "reliability": None,
        }
    ]
    payload["history"] = []
    path = tmp_path / "quality-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert isinstance(admit_quality_manifest_json(path.read_bytes()), QualityManifestV1)
    assert load_quality_history(path, manifest_policy(), as_of=NOW.date()) == {}


def test_manifest_is_identical_when_catalog_and_receipts_are_reordered():
    selection, expected = quality_bundle()
    catalog = NodeCatalog(
        nodes=tuple(reversed(selection.catalog.nodes)),
        receipts=tuple(reversed(selection.catalog.receipts)),
    )
    policy = manifest_policy()
    plan = plan_probe_candidates(catalog, policy)
    by_id = {
        node.fingerprint: observed(node, 50 if int(node.fingerprint, 16) == 1 else 2600)
        for node in catalog.nodes
    }
    reordered = assess_quality(
        catalog,
        plan,
        [by_id[entry.node.fingerprint] for entry in plan.entries],
        policy,
        unavailable_sources=("source-c",),
        as_of=NOW.date(),
    )
    actual = render_quality_bundle(
        reordered,
        policy,
        PublicEntryRegistry.from_identity(
            RepositoryIdentity(owner="owner", repository="repo")
        ),
        generated_at=NOW,
        runner_vantage="github-actions",
    )

    assert (
        actual.files["nodes/quality-manifest.json"]
        == expected.files["nodes/quality-manifest.json"]
    )


def test_passed_nodes_keep_fresh_provenance_from_nonpreferred_sources():
    source_a = tuple(node(index, "source-a") for index in range(1, 21))
    shared = tuple(
        candidate.model_copy(
            update={
                "provenance": (
                    *candidate.provenance,
                    *node(index, "source-b").provenance,
                )
            }
        )
        for index, candidate in enumerate(source_a[:16], start=1)
    )
    catalog = NodeCatalog(
        nodes=(
            *shared,
            *source_a[16:],
            *(node(index, "source-b") for index in range(21, 25)),
        )
    )
    policy = QualityPolicy()
    plan = plan_probe_candidates(catalog, policy)
    by_id = {
        candidate.fingerprint: observed(candidate, 50) for candidate in catalog.nodes
    }
    selection = assess_quality(
        catalog,
        plan,
        [by_id[entry.node.fingerprint] for entry in plan.entries],
        policy,
        as_of=NOW.date(),
    )

    bundle = render_quality_bundle(
        selection,
        policy,
        generated_at=NOW,
        runner_vantage="github-actions",
    )

    assert "nodes/source-a.yaml" in bundle.files
    assert "nodes/source-b.yaml" in bundle.files
