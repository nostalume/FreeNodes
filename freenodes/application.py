import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field

from freenodes.capability import (
    DEFAULT_CAPABILITY_TARGETS,
    CapabilityError,
    CapabilityPolicy,
    CapabilityRunReceipt,
    CapabilityTarget,
    CapableCatalog,
    ProbePlan,
    plan_probe_candidates,
)
from freenodes.config import (
    DiscoveryLimits,
    FrozenModel,
    GitHubFileSource,
    OpenRouterLimits,
    PublicationPolicy,
    RepositoryIdentity,
    Source,
)
from freenodes.decryption import DecryptionFactory, create_decryption_client
from freenodes.discovery import (
    DiscoveryFailure,
    DiscoveryOutcome,
    DiscoveryRequest,
    SourceDiscovery,
)
from freenodes.drive import DriveFactory, create_drive_client
from freenodes.github import GitHubSourceClient
from freenodes.llm import OpenRouterFallback
from freenodes.nodes import (
    AdmissionCounts,
    AdmissionSummary,
    AdmittedCatalog,
    SourceAdmissionSummary,
    SourceArtifact,
    admit_artifacts,
)
from freenodes.profiles import SubscriptionURLs, render_profiles, site_slug
from freenodes.publication import (
    BundleValidator,
    PublicationCapability,
    PublicationError,
    PublicationReceipt,
    ValidationReceipt,
    publish_bundle,
    validate_bundle_output_parent,
    write_validated_bundle,
)
from freenodes.web import WebCapability, WebClient
from freenodes.youtube import YouTubeCapability, YouTubeClient

logger = logging.getLogger(__name__)


class CapabilityProbeSession(Protocol):
    async def probe_capabilities(
        self,
        plan: ProbePlan,
        targets: tuple[CapabilityTarget, ...],
        policy: CapabilityPolicy,
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
    def __init__(self, limits: DiscoveryLimits):
        self.source_artifacts = limits.artifact_limit_per_source
        self.source_bytes = limits.byte_limit_per_source
        self.run_bytes = limits.byte_limit_per_run
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
    kind: Literal["context"] = "context"
    sites: tuple[Source, ...] = Field(strict=False)
    observed_at: AwareDatetime
    publication: PublicationPolicy = PublicationPolicy()


class DiscoveredRun(FrozenModel):
    kind: Literal["discovered"] = "discovered"
    context: RunContext
    outcomes: tuple[DiscoveryOutcome, ...] = Field(strict=False)


class AdmittedRun(FrozenModel):
    kind: Literal["admitted"] = "admitted"
    context: RunContext
    catalog: AdmittedCatalog
    unavailable_sources: tuple[str, ...] = Field(default=(), strict=False)


class RunFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    code: str
    message: str
    sites: tuple[str, ...] = Field(default=(), strict=False)


class Application:
    def __init__(
        self,
        sources: tuple[Source, ...],
        discovery: DiscoveryLimits,
        *,
        candidate_sources: tuple[GitHubFileSource, ...] = (),
        openrouter: OpenRouterLimits,
        publication: PublicationPolicy,
        repository: RepositoryIdentity,
        llm: OpenRouterFallback | None = None,
        youtube_factory: YouTubeFactory = create_youtube_client,
        web_factory: WebFactory = WebClient,
        github_factory: GitHubFactory = GitHubSourceClient,
        decryption_factory: DecryptionFactory = create_decryption_client,
        drive_factory: DriveFactory = create_drive_client,
    ):
        self.sources = sources
        self.candidate_sources = candidate_sources
        self.discovery = discovery
        self.publication = publication
        self.llm = llm or OpenRouterFallback(openrouter, credential=None)
        self.youtube_factory = youtube_factory
        self.web_factory = web_factory
        self.github_factory = github_factory
        self.decryption_factory = decryption_factory
        self.drive_factory = drive_factory
        self.registry = SubscriptionURLs.from_identity(repository)

    async def run(self, target: str | None = None) -> list[DiscoveryOutcome]:
        start = self._begin_run(target)
        if start.kind == "failure":
            logger.error(start.message)
            return []

        discovered = await self._discover(start)
        return list(discovered.outcomes)

    async def audit_sources(
        self,
        *,
        probe_session: CapabilityProbeSession,
        policy: CapabilityPolicy | None = None,
        now: datetime | None = None,
        targets: tuple[CapabilityTarget, ...] = DEFAULT_CAPABILITY_TARGETS,
    ) -> CapabilityRunReceipt:
        observed_at = now or datetime.now(UTC)
        context = RunContext(
            sites=(*self.sources, *self.candidate_sources),
            observed_at=observed_at,
            publication=self.publication,
        )
        discovered = await self._discover(context)
        admitted = self._admit_available(discovered, allow_empty=True)
        if admitted.kind == "failure":
            raise PublicationError(admitted.message)
        quality_policy = policy or CapabilityPolicy()
        plan = plan_probe_candidates(
            admitted.catalog,
            quality_policy,
        )
        return await probe_session.probe_capabilities(plan, targets, quality_policy)

    async def _discover(self, context: RunContext) -> DiscoveredRun:
        budget = DiscoveryBudget(self.discovery)
        youtube = self.youtube_factory(
            proxy=self.discovery.proxy_url or "",
            concurrency=self.discovery.source_concurrency,
        )
        web = self.web_factory()
        github = self.github_factory(web)
        concurrency = self.discovery.source_concurrency
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
        site: Source,
        observed_at: datetime,
        youtube: YouTubeCapability,
        web: WebCapability,
        github: GitHubSourceClient,
    ) -> DiscoveryOutcome:
        try:
            match site.kind:
                case "github_file":
                    return await github.discover(site, observed_at=observed_at)
                case "web" | "password_page" | "youtube_resources":
                    return await SourceDiscovery(
                        site,
                        DiscoveryRequest(
                            limits=self.discovery,
                            observed_at=observed_at,
                        ),
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

    async def publish(
        self,
        *,
        repository_root: Path,
        validator: BundleValidator,
        probe_session: CapabilityProbeSession,
        policy: CapabilityPolicy | None = None,
        targets: tuple[CapabilityTarget, ...] = DEFAULT_CAPABILITY_TARGETS,
        runner_vantage: str = "local",
        registry: SubscriptionURLs | None = None,
        now: datetime | None = None,
        base_revision: str | None = None,
    ) -> PublicationReceipt:
        observed_at = now or datetime.now(UTC)
        start = self._begin_run(None, observed_at=observed_at)
        if start.kind == "failure":
            raise PublicationError(start.message)
        discovered = await self._discover(start)
        admission = self._admit_available(discovered)
        del discovered
        if admission.kind == "failure":
            raise PublicationError(admission.message)
        quality_policy = policy or CapabilityPolicy(
            max_published=admission.context.publication.node_limit
        )
        catalog, measurement = await self._measure(
            admission.catalog,
            probe_session,
            quality_policy,
            targets,
        )
        try:
            capability = PublicationCapability.from_run(
                measurement,
                targets,
                runner_vantage,
            )
        except (CapabilityError, ValueError) as error:
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
        probe_session: CapabilityProbeSession,
        target: str | None = None,
        registry: SubscriptionURLs | None = None,
        policy: CapabilityPolicy | None = None,
        targets: tuple[CapabilityTarget, ...] = DEFAULT_CAPABILITY_TARGETS,
    ) -> ValidationReceipt:
        public_dir = Path("nodes")
        validation_parent = validate_bundle_output_parent(output_parent, public_dir)

        start = self._begin_run(target)
        if start.kind == "failure":
            raise PublicationError(start.message)
        discovered = await self._discover(start)
        admission = self._admit_available(discovered)
        del discovered
        if admission.kind == "failure":
            raise PublicationError(admission.message)
        quality_policy = policy or CapabilityPolicy(
            max_published=admission.context.publication.node_limit
        )
        catalog, _ = await self._measure(
            admission.catalog,
            probe_session,
            quality_policy,
            targets,
        )

        bundle = render_profiles(catalog, registry or self.registry)
        return write_validated_bundle(
            catalog=catalog,
            bundle=bundle,
            output_parent=validation_parent,
            validator=validator,
        )

    @staticmethod
    async def _measure(
        admitted: AdmittedCatalog,
        probe_session: CapabilityProbeSession,
        policy: CapabilityPolicy,
        targets: tuple[CapabilityTarget, ...],
    ) -> tuple[CapableCatalog, CapabilityRunReceipt]:
        plan = plan_probe_candidates(admitted, policy)
        measurement = await probe_session.probe_capabilities(plan, targets, policy)
        if measurement.status != "complete":
            detail = (
                measurement.diagnostic.code if measurement.diagnostic else "unknown"
            )
            raise PublicationError(f"capability measurement is inconclusive: {detail}")
        try:
            return CapableCatalog.from_measurement(admitted, measurement), measurement
        except CapabilityError as error:
            raise PublicationError(str(error)) from error

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
                publication=self.publication,
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
            match site.kind:
                case "github_file":
                    return site.authority
                case "web" | "password_page" | "youtube_resources":
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

    def _resolve_sites(self, target: str | None) -> list[Source]:
        if target:
            matches = [site for site in self.sources if site.name == target]
            if not matches:
                logger.warning(f"Unknown target '{target}', ignoring")
            return matches
        return list(self.sources)
