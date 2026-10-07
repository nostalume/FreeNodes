from datetime import UTC, datetime, timedelta

import pytest

from freenodes.capability import (
    CapabilityRunReceipt,
    NodeCapabilityDecision,
    ProbeDiagnostic,
)
from freenodes.config import AppConfig, GitHubFileSource, PublicationPolicy
from freenodes.discovery import DiscoverySuccess
from freenodes.latency_pilot import ComparisonReport, compare
from freenodes.nodes import SourceArtifact
from freenodes.publication import PublicationError
from freenodes.verification import PublicVerificationReceipt
from tests.support import ConsumerValidator, make_application, snapshot


@pytest.mark.parametrize(
    "defect",
    (
        None,
        "coverage",
        "latency",
        "deadline",
        "control",
        "consumer",
        "cdn",
        "stale",
        "foreign",
    ),
)
async def test_private_runner_comparison_fails_closed_and_redacts_report(
    tmp_path, monkeypatch, defect
):
    now = datetime.now(UTC)
    sites = tuple(
        GitHubFileSource(
            name=name, owner=name, repository="nodes", branch="main", path="nodes.txt"
        )
        for name in ("a", "b")
    )
    config = AppConfig(sources=sites, publication=PublicationPolicy(node_limit=2))
    calls = []

    async def discover(client, site, *, observed_at):
        return DiscoverySuccess(
            site_name=site.name,
            artifacts=(
                SourceArtifact.inline(
                    site=site.name,
                    content="\n".join(
                        f"trojan://hidden@{site.name}{index}.example:443#Node"
                        for index in range(3)
                    ),
                    observed_at=observed_at,
                ),
            ),
        )

    class MeasuredProbe:
        async def probe_capabilities(self, plan, targets, policy):
            calls.append((len(plan.entries), policy.rank_latency))
            if defect == "control":
                return CapabilityRunReceipt(
                    status="inconclusive",
                    diagnostic=ProbeDiagnostic(code="control_unavailable"),
                )
            decisions = tuple(
                NodeCapabilityDecision(
                    fingerprint="unmeasured"
                    if defect == "foreign" and not index
                    else entry.node.fingerprint,
                    status="failed" if failed else "capable",
                    reason="target_failures" if failed else "quorum",
                    successful_targets=() if failed else ("github", "google"),
                    failed_targets=("github", "google") if failed else (),
                    target_delays=()
                    if failed
                    else (("github", delay), ("google", delay)),
                )
                for index, entry in enumerate(plan.entries)
                for delay in (1000 if index < 2 or defect == "latency" else 10,)
                for failed in (
                    defect == "coverage" and policy.rank_latency and index > 0,
                )
            )
            deadline = defect == "deadline" and policy.rank_latency
            return CapabilityRunReceipt(
                status="complete",
                planned=len(decisions) + int(deadline),
                termination="time_budget" if deadline else "candidates_exhausted",
                deadline_reached=deadline,
                decisions=decisions,
                accepted_fingerprints=tuple(
                    d.fingerprint for d in decisions if d.status == "capable"
                )[:2],
            )

    async def observe():
        generation = (now - timedelta(days=3) if defect == "stale" else now).isoformat()
        return PublicVerificationReceipt(
            direct="current",
            cdn="degraded" if defect == "cdn" else "current",
            direct_generation=generation,
            cdn_generation=None if defect == "cdn" else generation,
        )

    monkeypatch.setattr("freenodes.application.GitHubSourceClient.discover", discover)
    public = tmp_path / "nodes"
    public.mkdir()
    (public / "publication-receipt.json").write_bytes(b"previous receipt untouched")
    (tmp_path / "IMPORT.md").write_bytes(b"previous import untouched")
    before = snapshot(tmp_path)
    report_path = tmp_path / ".cache" / "comparison.json"
    report = await compare(
        make_application(config),
        repository_root=tmp_path,
        report_path=report_path,
        probe_session=MeasuredProbe(),
        validator=ConsumerValidator(fail=defect == "consumer"),
        observe_public=observe,
    )

    assert report.status == ("passed" if defect is None else "failed")
    assert report.checks["public_snapshot_unchanged"] is True
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    raw = report_path.read_text()
    assert "hidden" not in raw and "trojan://" not in raw
    assert ComparisonReport.model_validate_json(raw) == report
    assert not list(report_path.parent.glob("private-profiles-*"))
    failed_check = {
        "coverage": "coverage_target_met",
        "latency": "latency_improved",
        "deadline": "rank_pool_complete",
        "cdn": "public_current",
        "stale": "publication_fresh",
        "control": "execution_complete",
        "consumer": "execution_complete",
        "foreign": "execution_complete",
    }
    if defect is not None:
        assert report.checks[failed_check[defect]] is False
    if defect not in ("control", "consumer", "foreign"):
        assert calls == [(6, True)]


@pytest.mark.parametrize("path", ("nodes/report.json", "IMPORT.md"))
async def test_comparison_refuses_public_report_path_before_discovery(tmp_path, path):
    config = AppConfig(sources=[])
    with pytest.raises(PublicationError, match=r"outside|private"):
        await compare(
            make_application(config),
            repository_root=tmp_path,
            report_path=tmp_path / path,
            probe_session=None,
            validator=None,
            observe_public=None,
        )
