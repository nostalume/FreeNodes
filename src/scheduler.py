"""Scheduler — parallel site dispatching with shared resource management."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field

from src.config import Config, FrozenModel, PublicationPolicy, Site
from src.crawler import WebCapability, WebClient
from src.decryptor import DecryptionFactory, create_decryption_client
from src.drive import DriveFactory, create_drive_client
from src.github_source import GitHubSourceClient
from src.llm_router import LLMRouter
from src.nodes import (
    AdmissionCounts,
    AdmissionSummary,
    NodeCatalog,
    SourceAdmissionSummary,
    SourceArtifact,
    admit_artifacts,
)
from src.profiles import PublicEntryRegistry, render_profiles, site_slug
from src.publication import (
    BundleValidator,
    PublicationCapability,
    PublicationError,
    PublicationReceipt,
    ValidationReceipt,
    publish_bundle,
    validate_bundle_output_parent,
    write_validated_bundle,
)
from src.quality import (
    DEFAULT_CAPABILITY_TARGETS,
    CapabilityRunReceipt,
    CapabilityTarget,
    ProbePlan,
    QualityError,
    QualityPolicy,
    plan_probe_candidates,
    select_capable,
)
from src.site_processor import (
    DiscoveryFailure,
    DiscoveryOutcome,
    SiteProcessor,
)
from src.youtube import YouTubeCapability, YouTubeClient

logger = logging.getLogger(__name__)


class CapabilityProbeSession(Protocol):
    async def probe_capabilities(
        self,
        plan: ProbePlan,
        targets: tuple[CapabilityTarget, ...],
        policy: QualityPolicy,
    ) -> CapabilityRunReceipt: ...


class YouTubeFactory(Protocol):
    def __call__(
        self,
        *,
        proxy: str,
        concurrency: int,
    ) -> YouTubeCapability: ...


class WebFactory(Protocol):
    def __call__(self) -> WebCapability: ...


class GitHubFactory(Protocol):
    def __call__(self, web: WebCapability) -> GitHubSourceClient: ...


class DiscoveryBudget:
    """Admit bounded completed source content for one scheduler run."""

    def __init__(self, config: Config):
        self.source_artifacts = config.crawl.max_source_artifacts
        self.source_bytes = config.crawl.max_source_bytes
        self.run_bytes = config.crawl.max_run_source_bytes
        self.retained_bytes = 0

    def admit(self, outcome: DiscoveryOutcome) -> DiscoveryOutcome:
        if outcome.kind == "failure":
            return outcome
        if len(outcome.artifacts) > self.source_artifacts:
            return self._failure(outcome, "source artifact count limit exceeded")
        if outcome.total_bytes > self.source_bytes:
            return self._failure(outcome, "source byte limit exceeded")
        if self.retained_bytes + outcome.total_bytes > self.run_bytes:
            return self._failure(outcome, "run source byte limit exceeded")
        self.retained_bytes += outcome.total_bytes
        return outcome

    @staticmethod
    def _failure(outcome: DiscoveryOutcome, error: str) -> DiscoveryFailure:
        return DiscoveryFailure(
            site_name=outcome.site_name,
            articles_processed=outcome.articles_processed,
            txt_count=0,
            yaml_count=0,
            total_bytes=0,
            pattern_generated=outcome.pattern_generated,
            active_pattern=outcome.active_pattern,
            errors=(*outcome.errors, error),
        )


def create_youtube_client(
    *,
    proxy: str,
    concurrency: int,
) -> YouTubeCapability:
    return YouTubeClient(proxy=proxy, concurrency=concurrency)


class RunContext(FrozenModel):
    """Selected sources and the observation time shared by one run."""

    kind: Literal["context"] = "context"
    sites: tuple[Site, ...] = Field(strict=False)
    observed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    publication: PublicationPolicy = PublicationPolicy()


class DiscoveredRun(FrozenModel):
    """Discovery outcomes tied to the context that produced them."""

    kind: Literal["discovered"] = "discovered"
    context: RunContext
    outcomes: tuple[DiscoveryOutcome, ...] = Field(strict=False)
    completed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class AdmittedRun(FrozenModel):
    """A catalog tied to the discovery evidence it admitted."""

    kind: Literal["admitted"] = "admitted"
    context: RunContext
    catalog: NodeCatalog
    unavailable_sources: tuple[str, ...] = Field(default=(), strict=False)


class RunFailure(FrozenModel):
    """Typed failure at a boundary between run states."""

    kind: Literal["failure"] = "failure"
    code: str
    message: str
    sites: tuple[str, ...] = Field(default=(), strict=False)


class Scheduler:
    """Dispatch multiple sites concurrently with a shared LLMRouter.

    Usage:
        config = load_config()
        scheduler = Scheduler(config)
        results = await scheduler.run()          # all sites
        results = await scheduler.run("nodefree")  # single site
    """

    def __init__(
        self,
        config: Config,
        *,
        youtube_factory: YouTubeFactory = create_youtube_client,
        web_factory: WebFactory = WebClient,
        github_factory: GitHubFactory = GitHubSourceClient,
        decryption_factory: DecryptionFactory = create_decryption_client,
        drive_factory: DriveFactory = create_drive_client,
    ):
        self.config = config
        self.llm = LLMRouter(config)
        self.youtube_factory = youtube_factory
        self.web_factory = web_factory
        self.github_factory = github_factory
        self.decryption_factory = decryption_factory
        self.drive_factory = drive_factory
        self.registry = PublicEntryRegistry.from_identity(config.repository)

    async def run(self, target: str | None = None) -> list[DiscoveryOutcome]:
        """Run all (or a single) sites, respecting concurrency limit.

        Args:
            target: Optional site name. When set, only that site runs.

        Returns:
            One typed discovery outcome per selected site.
        """
        start = self._begin_run(target)
        if start.kind == "failure":
            logger.error(start.message)
            return []

        discovered = await self._discover(start)
        self._print_summary(discovered.outcomes)
        admitted = self._admit_available(discovered, allow_empty=True)
        if admitted.kind == "admitted":
            summary = admitted.catalog.summary
            assert summary is not None
            counts = summary.counts
            print(
                "ADMISSION: "
                f"{counts.unique_eligible} unique, "
                f"{counts.rejected_records} rejected, "
                f"{counts.duplicate_occurrences} duplicate occurrence(s)"
            )
        return list(discovered.outcomes)

    async def audit_sources(
        self,
        *,
        probe_session: CapabilityProbeSession,
        policy: QualityPolicy | None = None,
        now: datetime | None = None,
        targets: tuple[CapabilityTarget, ...] = DEFAULT_CAPABILITY_TARGETS,
    ) -> CapabilityRunReceipt:
        """Assess active and candidate sources without persistent effects."""
        observed_at = now or datetime.now(UTC)
        context = RunContext(
            sites=(*self.config.sites, *self.config.source_candidates),
            observed_at=observed_at,
            publication=self.config.publication,
        )
        discovered = await self._discover(context)
        admitted = self._admit_available(discovered, allow_empty=True)
        if admitted.kind == "failure":
            raise PublicationError(admitted.message)
        quality_policy = policy or QualityPolicy()
        plan = plan_probe_candidates(
            admitted.catalog,
            quality_policy,
        )
        return await probe_session.probe_capabilities(plan, targets, quality_policy)

    async def _discover(self, context: RunContext) -> DiscoveredRun:
        budget = DiscoveryBudget(self.config)
        youtube = self.youtube_factory(
            proxy=self.config.crawl.proxy,
            concurrency=self.config.crawl.concurrency,
        )
        web = self.web_factory()
        github = self.github_factory(web)
        concurrency = self.config.crawl.concurrency
        pending: dict[int, asyncio.Task[DiscoveryOutcome]] = {}
        committed: list[DiscoveryOutcome] = []
        next_start = 0
        next_commit = 0

        def refill() -> None:
            nonlocal next_start
            while next_start < len(context.sites) and len(pending) < concurrency:
                site = context.sites[next_start]
                pending[next_start] = asyncio.create_task(
                    self._discover_site(
                        site,
                        context.observed_at,
                        youtube,
                        web,
                        github,
                    )
                )
                next_start += 1

        refill()
        try:
            while next_commit < len(context.sites):
                outcome = await pending.pop(next_commit)
                committed.append(budget.admit(outcome))
                next_commit += 1
                while next_commit in pending and pending[next_commit].done():
                    committed.append(budget.admit(await pending.pop(next_commit)))
                    next_commit += 1
                refill()
        except asyncio.CancelledError:
            for task in pending.values():
                task.cancel()
            await asyncio.gather(*pending.values(), return_exceptions=True)
            raise
        return DiscoveredRun(context=context, outcomes=tuple(committed))

    async def _discover_site(
        self,
        site: Site,
        observed_at: datetime,
        youtube: YouTubeCapability,
        web: WebCapability,
        github: GitHubSourceClient,
    ) -> DiscoveryOutcome:
        try:
            match site.type:
                case "github":
                    return await github.discover(site, observed_at=observed_at)
                case "simple" | "yt_pwd" | "youtube_password" | "cloud_drive":
                    return await SiteProcessor(
                        site,
                        self.config,
                        self.llm,
                        youtube,
                        web,
                        self.decryption_factory,
                        self.drive_factory,
                    ).discover()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("Site %s crashed: %s", site.name, error)
            return DiscoveryFailure(
                site_name=site.name,
                errors=(f"unhandled exception: {error}",),
            )

    async def publish_profiles(
        self,
        *,
        repository_root: Path,
        validator: BundleValidator,
        probe_session: CapabilityProbeSession,
        policy: QualityPolicy | None = None,
        targets: tuple[CapabilityTarget, ...] = DEFAULT_CAPABILITY_TARGETS,
        runner_vantage: str = "local",
        registry: PublicEntryRegistry | None = None,
        now: datetime | None = None,
        base_revision: str | None = None,
    ) -> PublicationReceipt:
        """Publish one bounded snapshot from deterministic source evidence."""
        observed_at = now or datetime.now(UTC)
        start = self._begin_run(None, observed_at=observed_at)
        if start.kind == "failure":
            raise PublicationError(start.message)
        discovered = await self._discover(start)
        self._print_summary(discovered.outcomes)
        admission = self._admit_available(discovered)
        del discovered
        if admission.kind == "failure":
            raise PublicationError(admission.message)
        quality_policy = policy or QualityPolicy(
            max_published=admission.context.publication.max_nodes
        )
        plan = plan_probe_candidates(admission.catalog, quality_policy)
        measurement = await probe_session.probe_capabilities(
            plan, targets, quality_policy
        )
        if measurement.status != "complete":
            detail = (
                measurement.diagnostic.code if measurement.diagnostic else "unknown"
            )
            raise PublicationError(f"capability measurement is inconclusive: {detail}")
        accepted = set(measurement.accepted_fingerprints)
        try:
            catalog = select_capable(
                admission.catalog,
                tuple(
                    item
                    for item in measurement.decisions
                    if item.fingerprint in accepted
                ),
            )
            capability = PublicationCapability.from_run(
                measurement,
                targets,
                runner_vantage,
            )
        except (QualityError, ValueError) as error:
            raise PublicationError(str(error)) from error
        summary = admission.catalog.summary
        assert summary is not None
        bundle = render_profiles(catalog, registry or self.registry)
        previous_managed = tuple(
            f"nodes/{site_slug(site.name)}.{extension}"
            for site in admission.context.sites
            for extension in ("txt", "yaml")
        )
        return publish_bundle(
            bundle,
            repository_root.resolve(),
            validator=validator,
            now=observed_at,
            previous_managed=previous_managed,
            admission_summary=summary,
            selection_limit=quality_policy.max_published,
            base_revision=base_revision,
            capability=capability,
        )

    async def validate_profiles(
        self,
        *,
        output_parent: Path,
        validator: BundleValidator,
        target: str | None = None,
        registry: PublicEntryRegistry | None = None,
    ) -> ValidationReceipt:
        """Run discovery through consumer validation without public cutover."""
        public_dir = self.config.output.dir
        validation_parent = validate_bundle_output_parent(output_parent, public_dir)

        start = self._begin_run(target)
        if start.kind == "failure":
            raise PublicationError(start.message)
        discovered = await self._discover(start)
        admission = self._admit_available(discovered)
        del discovered
        if admission.kind == "failure":
            raise PublicationError(admission.message)
        catalog = admission.catalog

        bundle = render_profiles(catalog, registry or self.registry)
        return write_validated_bundle(
            catalog=catalog,
            bundle=bundle,
            output_parent=validation_parent,
            validator=validator,
        )

    def _begin_run(
        self,
        target: str | None,
        *,
        observed_at: datetime | None = None,
    ) -> RunContext | RunFailure:
        sites = self._resolve_sites(target)
        if sites:
            return RunContext(
                sites=sites,
                observed_at=observed_at or datetime.now(UTC),
                publication=self.config.publication,
            )
        purpose = "profile validation" if target else "publication"
        return RunFailure(
            code="no_sites",
            message=f"no sites selected for {purpose}",
        )

    @staticmethod
    def _admit_available(
        discovered: DiscoveredRun,
        *,
        allow_empty: bool = False,
    ) -> AdmittedRun | RunFailure:
        artifacts: list[SourceArtifact] = []
        failures: list[str] = []
        unavailable: list[str] = []
        for outcome in discovered.outcomes:
            if outcome.kind == "failure":
                unavailable.append(outcome.site_name)
                failures.extend(
                    f"{outcome.site_name}: {error}" for error in outcome.errors
                )
                continue
            if not outcome.artifacts:
                unavailable.append(outcome.site_name)
            artifacts.extend(outcome.artifacts)
            failures.extend(f"{outcome.site_name}: {error}" for error in outcome.errors)
        if not artifacts and not allow_empty:
            detail = "; ".join(failures[:3]) or "discovery returned no artifacts"
            return RunFailure(
                code="no_source_artifacts",
                message=f"discovery has no source artifacts: {detail}",
            )

        policy = discovered.context.publication
        catalog = admit_artifacts(
            artifacts,
            now=discovered.context.observed_at,
            stale_after=timedelta(hours=policy.stale_after_hours),
            expires_after=timedelta(hours=policy.expires_after_hours),
        )
        admitted_summary = catalog.summary
        assert admitted_summary is not None
        admitted_sources = {item.source: item for item in admitted_summary.sources}
        sites = {site.name: site for site in discovered.context.sites}

        def authority(source: str) -> str:
            site = sites[source]
            match site.type:
                case "github":
                    return site.authority
                case "simple" | "yt_pwd" | "youtube_password" | "cloud_drive":
                    return site.name

        failed_sources = sum(
            outcome.kind == "failure" for outcome in discovered.outcomes
        )
        empty_sources = sum(
            outcome.kind == "success" and not outcome.artifacts
            for outcome in discovered.outcomes
        )
        source_summaries: list[SourceAdmissionSummary] = []
        for outcome in discovered.outcomes:
            admitted_source = admitted_sources.get(outcome.site_name)
            if admitted_source is not None:
                source_summaries.append(admitted_source)
                continue
            source_summaries.append(
                SourceAdmissionSummary(
                    source=outcome.site_name,
                    authority=authority(outcome.site_name),
                    status="failed" if outcome.kind == "failure" else "empty",
                    artifacts=0,
                    candidate_records=0,
                    unique_eligible=0,
                )
            )
        admitted_counts = admitted_summary.counts
        counts = AdmissionCounts(
            attempted_sources=len(discovered.outcomes),
            failed_sources=failed_sources,
            empty_sources=empty_sources,
            sources_with_artifacts=(
                len(discovered.outcomes) - failed_sources - empty_sources
            ),
            discovered_artifacts=admitted_counts.discovered_artifacts,
            rejected_artifacts=admitted_counts.rejected_artifacts,
            decoded_artifacts=admitted_counts.decoded_artifacts,
            candidate_records=admitted_counts.candidate_records,
            rejected_records=admitted_counts.rejected_records,
            eligible_occurrences=admitted_counts.eligible_occurrences,
            unique_eligible=admitted_counts.unique_eligible,
            duplicate_occurrences=admitted_counts.duplicate_occurrences,
        )
        summary = AdmissionSummary(
            counts=counts,
            rejection_codes=admitted_summary.rejection_codes,
            sources=tuple(source_summaries),
        )
        catalog = catalog.model_copy(update={"summary": summary})
        if catalog.accepted_count == 0 and not allow_empty:
            return RunFailure(
                code="admission_empty",
                message=(
                    "profile validation admitted no nodes "
                    f"({catalog.rejected_count} rejection(s))"
                ),
            )
        return AdmittedRun(
            context=discovered.context,
            catalog=catalog,
            unavailable_sources=tuple(sorted(set(unavailable))),
        )

    def _resolve_sites(self, target: str | None) -> list[Site]:
        """Resolve a target name to admitted site configurations."""
        if target:
            matches = [site for site in self.config.sites if site.name == target]
            if not matches:
                logger.warning(f"Unknown target '{target}', ignoring")
            return matches
        return list(self.config.sites)

    @staticmethod
    def _print_summary(results: list[DiscoveryOutcome] | tuple[DiscoveryOutcome, ...]):
        """Print a summary table of all site results."""
        print(f"\n{'=' * 70}")
        print(f"{'SUMMARY':^70}")
        print(f"{'=' * 70}")
        print(
            f"{'SITE':16s} {'ARTICLES':10s} {'TXT':6s} {'YAML':6s} {'BYTES':12s} {'PATTERN':12s}"
        )
        print("-" * 70)
        total_articles = 0
        total_txt = 0
        total_yaml = 0
        total_bytes = 0
        for r in results:
            active_pattern = r.active_pattern
            pattern = (
                "generated (run)"
                if r.pattern_generated
                else (active_pattern[:20] if active_pattern else "—")
            )
            print(
                f"{r.site_name:16s} {r.articles_processed:4d}        {r.txt_count:4d}   {r.yaml_count:4d}  {r.total_bytes:8d}B  {pattern:12s}"
            )
            total_articles += r.articles_processed
            total_txt += r.txt_count
            total_yaml += r.yaml_count
            total_bytes += r.total_bytes
        print("-" * 70)
        print(
            f"{'TOTAL':16s} {total_articles:4d}        {total_txt:4d}   {total_yaml:4d}  {total_bytes:8d}B"
        )
        print(f"{'=' * 70}")

        errors = [r for r in results if r.errors]
        if errors:
            print(f"\n⚠ {len(errors)} site(s) had errors:")
            for r in errors:
                for e in r.errors:
                    print(f"  [{r.site_name}] {e}")
