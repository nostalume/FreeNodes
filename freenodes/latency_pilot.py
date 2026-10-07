"""Bounded, non-publishing runner comparison: python -m freenodes.latency_pilot."""

import asyncio
import hashlib
import subprocess
import sys
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal, Self

import httpx
from pydantic import AwareDatetime, Field

from freenodes.application import Application, CapabilityProbeSession, RunContext
from freenodes.capability import (
    DEFAULT_CAPABILITY_TARGETS,
    CapabilityPolicy,
    CapableCatalog,
    LatencyRankingReport,
    _sources,
)
from freenodes.config import FrozenModel, GitHubFileSource, load_config
from freenodes.github import GitHubCommitClient, GitHubSourceClient
from freenodes.mihomo import MihomoValidator, acquire_pinned_mihomo
from freenodes.probe import MihomoProbeSession
from freenodes.profiles import SubscriptionURLs, render_profiles
from freenodes.publication import (
    BundleValidator,
    PublicationError,
    load_quality_history,
    validate_bundle_output_parent,
    write_validated_bundle,
)
from freenodes.verification import PublicVerificationReceipt, verify_remote_entries


class SelectionMetrics(FrozenModel):
    planned: int
    attempted: int
    accepted: int
    median_delay_ms: float | None
    elapsed_seconds: float
    source_counts: dict[str, int]
    protocol_counts: dict[str, int]

    @classmethod
    def from_catalog(cls, catalog: CapableCatalog, elapsed: float) -> Self:
        accepted = set(catalog.receipt.accepted_fingerprints)
        delays = [
            median(delay for _, delay in item.target_delays)
            for item in catalog.receipt.decisions
            if item.fingerprint in accepted and item.target_delays
        ]
        return cls(
            planned=catalog.receipt.planned or 0,
            attempted=catalog.receipt.attempted,
            accepted=len(catalog.nodes),
            median_delay_ms=float(median(delays)) if delays else None,
            elapsed_seconds=elapsed,
            source_counts=dict(Counter(_sources(node)[0] for node in catalog.nodes)),
            protocol_counts=dict(Counter(node.proxy.type for node in catalog.nodes)),
        )


class ComparisonReport(FrozenModel):
    schema_version: Literal[1] = Field(default=1, alias="schema")
    status: Literal["passed", "failed"]
    observed_at: AwareDatetime
    scope: Literal["active_github_sources"] = "active_github_sources"
    snapshots: tuple[tuple[str, str, str], ...] = Field(default=(), strict=False)
    baseline: SelectionMetrics | None = None
    ranked: SelectionMetrics | None = None
    checks: dict[str, bool]
    diagnostic: str = ""
    public: PublicVerificationReceipt | None = None


def comparison_checks(
    baseline: SelectionMetrics | None,
    ranked: SelectionMetrics | None,
    ranking: LatencyRankingReport | None,
    public: PublicVerificationReceipt | None,
    *,
    now: datetime,
    stale_hours: int,
) -> dict[str, bool]:
    if baseline is None or ranked is None or ranking is None or public is None:
        return {"execution_complete": False}
    delay = ranked.median_delay_ms
    before = ranking.baseline_median_delay_ms
    baseline_delay = baseline.median_delay_ms
    age = now - datetime.fromisoformat(public.direct_generation)
    return {
        "execution_complete": True,
        "rank_pool_complete": ranking.ranking_pool_complete,
        "coverage_not_reduced": ranked.accepted >= baseline.accepted,
        "latency_improved": (
            delay is not None
            and before is not None
            and baseline_delay is not None
            and delay < before
            and delay <= baseline_delay
        ),
        "source_coverage_preserved": baseline.source_counts.keys()
        <= ranked.source_counts.keys(),
        "protocol_coverage_preserved": baseline.protocol_counts.keys()
        <= ranked.protocol_counts.keys(),
        "public_current": public.direct == public.cdn == "current",
        "publication_fresh": timedelta(0) <= age <= timedelta(hours=stale_hours),
    }


def public_digests(root: Path) -> dict[str, str]:
    paths = [path for path in (root / "nodes").rglob("*") if path.is_file()]
    if (root / "IMPORT.md").is_file():
        paths.append(root / "IMPORT.md")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


async def compare(
    application: Application,
    *,
    repository_root: Path,
    report_path: Path,
    probe_session: CapabilityProbeSession,
    validator: BundleValidator,
    observe_public: Callable[[], Awaitable[PublicVerificationReceipt]],
) -> ComparisonReport:
    root = repository_root.resolve()
    report_path = report_path.resolve()
    validate_bundle_output_parent(report_path.parent, root / "nodes")
    if report_path == root / "IMPORT.md" or report_path.exists():
        raise PublicationError("pilot report must be a new private file")
    before = public_digests(root)
    baseline = ranked = ranking = public = None
    snapshots = ()
    diagnostic = ""
    stage = "discovery"
    now = datetime.now(UTC)
    context = RunContext(
        sites=tuple(
            site for site in application.sources if isinstance(site, GitHubFileSource)
        ),
        observed_at=now,
        publication=application.publication,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        discovered = await asyncio.wait_for(application._discover(context), 180)
        admission = application._admit_available(discovered)
        if admission.kind == "failure":
            raise PublicationError("pilot source admission failed")
        snapshots = tuple(
            (
                artifact.site,
                hashlib.sha256(artifact.content).hexdigest(),
                hashlib.sha256(artifact.source_url.encode()).hexdigest(),
            )
            for outcome in discovered.outcomes
            if outcome.kind == "success"
            for artifact in outcome.artifacts
        )
        history = load_quality_history(root)
        with TemporaryDirectory(
            prefix="private-profiles-", dir=report_path.parent
        ) as scratch:
            for mode in ("first_capable", "ranked"):
                stage = mode
                policy = CapabilityPolicy(
                    max_published=application.publication.node_limit,
                    rank_latency=mode == "ranked",
                )
                started = perf_counter()
                catalog, _, report = await asyncio.wait_for(
                    application._measure(
                        admission.catalog,
                        probe_session,
                        policy,
                        history=history,
                        targets=DEFAULT_CAPABILITY_TARGETS,
                    ),
                    420,
                )
                metrics = SelectionMetrics.from_catalog(
                    catalog, perf_counter() - started
                )
                write_validated_bundle(
                    catalog=catalog,
                    bundle=render_profiles(catalog, application.registry),
                    output_parent=Path(scratch) / mode,
                    validator=validator,
                    latency_ranking=report,
                )
                if mode == "first_capable":
                    baseline = metrics
                else:
                    ranked, ranking = metrics, report
        stage = "public_observation"
        public = await asyncio.wait_for(observe_public(), 180)
    except Exception as error:
        diagnostic = f"{stage}: {type(error).__name__}"
        if isinstance(
            error, PublicationError
        ) and "capability measurement is inconclusive:" in str(error):
            diagnostic += " (" + str(error).split(":")[-1].strip() + ")"
    checks = comparison_checks(
        baseline,
        ranked,
        ranking,
        public,
        now=datetime.now(UTC),
        stale_hours=application.publication.stale_after_hours,
    )
    checks["public_snapshot_unchanged"] = public_digests(root) == before
    result = ComparisonReport(
        status="passed" if all(checks.values()) and not diagnostic else "failed",
        observed_at=now,
        snapshots=snapshots,
        baseline=baseline,
        ranked=ranked,
        checks=checks,
        diagnostic=diagnostic,
        public=public,
    )
    with report_path.open("x", encoding="utf-8") as output:
        output.write(result.model_dump_json(indent=2, by_alias=True) + "\n")
    return result


class GhMetadataTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET" or request.url.host != "api.github.com":
            raise httpx.RequestError(
                "pilot only permits GitHub metadata GET", request=request
            )
        endpoint = (
            request.url.path.lstrip("/") + "?" + request.url.query.decode("ascii")
        )
        process = await asyncio.create_subprocess_exec(
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            "GET",
            endpoint,
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: " + request.headers["X-GitHub-Api-Version"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            body, _ = await asyncio.wait_for(process.communicate(), 20)
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        if process.returncode or len(body) > 1024 * 1024:
            raise httpx.RequestError(
                "pilot GitHub metadata request failed", request=request
            )
        return httpx.Response(200, content=body, request=request)


async def main() -> int:
    config = load_config()
    acquired = acquire_pinned_mihomo(Path(".cache/mihomo"))
    application = Application(
        config.sources,
        config.discovery,
        openrouter=config.openrouter,
        publication=config.publication,
        repository=config.repository,
        github_factory=lambda web: GitHubSourceClient(
            web, commits=GitHubCommitClient(transport=GhMetadataTransport())
        ),
    )
    result = await compare(
        application,
        repository_root=Path.cwd(),
        report_path=Path(".cache/latency-comparison/report.json"),
        probe_session=MihomoProbeSession(acquired.executable),
        validator=MihomoValidator(acquired.executable),
        observe_public=lambda: verify_remote_entries(
            SubscriptionURLs.from_identity(config.repository),
            acquired.executable,
            attempts=1,
            retry_delay=0,
        ),
    )
    print(result.model_dump_json(indent=2, by_alias=True))
    return 0 if result.status == "passed" else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
