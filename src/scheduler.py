"""Scheduler — parallel site dispatching with shared resource management."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from src.config import Config, FrozenModel, Site
from src.crawler import WebCapability, WebClient
from src.decryptor import DecryptionFactory, create_decryption_client
from src.drive import DriveFactory, create_drive_client
from src.llm_router import LLMRouter
from src.mihomo import ProbeEvidence
from src.nodes import NodeCatalog, ProbeableNode, SourceArtifact, admit_artifacts
from src.profiles import OutputBundle, PublicEntryRegistry, render_profiles, site_slug
from src.publication import (
    BundleValidator,
    PublicationError,
    PublicationReceipt,
    ValidationReceipt,
    publish_bundle,
    validate_bundle_output_parent,
    write_validated_bundle,
)
from src.quality import (
    QualityHistory,
    QualityPolicy,
    QualitySelection,
    assess_quality,
    select_probe_candidates,
)
from src.quality_manifest import load_quality_history, render_quality_bundle
from src.site_processor import (
    DiscoveryFailure,
    DiscoveryOutcome,
    SiteProcessor,
)
from src.youtube import YouTubeCapability, YouTubeClient

logger = logging.getLogger(__name__)


class ProbeSession(Protocol):
    async def probe(
        self,
        nodes: tuple[ProbeableNode, ...],
    ) -> tuple[ProbeEvidence, ...]: ...


class YouTubeFactory(Protocol):
    def __call__(
        self,
        *,
        proxy: str,
        concurrency: int,
    ) -> YouTubeCapability: ...


class WebFactory(Protocol):
    def __call__(self) -> WebCapability: ...


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


class DiscoveredRun(FrozenModel):
    """Discovery outcomes tied to the context that produced them."""

    kind: Literal["discovered"] = "discovered"
    context: RunContext
    outcomes: tuple[DiscoveryOutcome, ...] = Field(strict=False)
    completed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class AdmittedRun(FrozenModel):
    """A catalog tied to the discovery evidence it admitted."""

    kind: Literal["admitted"] = "admitted"
    discovered: DiscoveredRun
    catalog: NodeCatalog

    @property
    def context(self) -> RunContext:
        return self.discovered.context


class ProbedRun(FrozenModel):
    """Probe evidence bound to the exact admitted candidates and policy."""

    kind: Literal["probed"] = "probed"
    admitted: AdmittedRun
    policy: QualityPolicy
    history: tuple[QualityHistory, ...] = Field(default=(), strict=False)
    evidence: tuple[ProbeEvidence, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_evidence(self) -> "ProbedRun":
        candidates = select_probe_candidates(self.admitted.catalog, self.policy)
        expected = tuple(node.fingerprint for node in candidates)
        observed = tuple(item.fingerprint for item in self.evidence)
        if observed != expected:
            raise ValueError("probe evidence does not cover the selected candidates")
        candidate_names = {node.fingerprint: node.display_name for node in candidates}
        if any(
            candidate_names[item.fingerprint] != item.proxy_name
            for item in self.evidence
        ):
            raise ValueError("probe evidence name does not match its candidate")
        history_ids = tuple(item.fingerprint for item in self.history)
        catalog_ids = {node.fingerprint for node in self.admitted.catalog.nodes}
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("duplicate quality history fingerprint")
        if not set(history_ids).issubset(catalog_ids):
            raise ValueError("quality history does not belong to the admitted catalog")
        return self

    def history_index(self) -> dict[str, QualityHistory]:
        return {item.fingerprint: item for item in self.history}


class SelectedRun(FrozenModel):
    """One quality decision ledger bound to the probe evidence it consumes."""

    kind: Literal["selected"] = "selected"
    probed: ProbedRun
    selection: QualitySelection

    @model_validator(mode="after")
    def validate_selection(self) -> "SelectedRun":
        if self.selection.catalog != self.probed.admitted.catalog:
            raise ValueError("selection does not belong to the probed catalog")
        return self


class RenderedRun(FrozenModel):
    """Public bytes bound to the selected quality decision."""

    kind: Literal["rendered"] = "rendered"
    selected: SelectedRun
    bundle: OutputBundle


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
        decryption_factory: DecryptionFactory = create_decryption_client,
        drive_factory: DriveFactory = create_drive_client,
    ):
        self.config = config
        self.llm = LLMRouter(config)
        self.youtube_factory = youtube_factory
        self.web_factory = web_factory
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
        return list(discovered.outcomes)

    async def _discover(self, context: RunContext) -> DiscoveredRun:
        semaphore = asyncio.Semaphore(self.config.crawl.concurrency)
        youtube = self.youtube_factory(
            proxy=self.config.crawl.proxy,
            concurrency=self.config.crawl.concurrency,
        )
        web = self.web_factory()
        outcomes = tuple(
            await asyncio.gather(
                *(
                    self._discover_site(site, semaphore, youtube, web)
                    for site in context.sites
                )
            )
        )
        return DiscoveredRun(context=context, outcomes=outcomes)

    async def _discover_site(
        self,
        site: Site,
        semaphore: asyncio.Semaphore,
        youtube: YouTubeCapability,
        web: WebCapability,
    ) -> DiscoveryOutcome:
        async with semaphore:
            try:
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
        probe_session: ProbeSession,
        policy: QualityPolicy | None = None,
        registry: PublicEntryRegistry | None = None,
        now: datetime | None = None,
        runner_vantage: str = "github-actions",
    ) -> PublicationReceipt:
        """Discover, admit, probe, select, render, validate, and promote once."""
        observed_at = now or datetime.now(UTC)
        start = self._begin_run(None, observed_at=observed_at)
        if start.kind == "failure":
            raise PublicationError(start.message)
        discovered = await self._discover(start)
        admission = self._require_admission(discovered)
        if admission.kind == "failure":
            raise PublicationError(admission.message)
        quality_policy = policy or QualityPolicy()
        repository = repository_root.resolve()
        history = load_quality_history(
            repository / "nodes" / "quality-manifest.json",
            admission.catalog,
            quality_policy,
            as_of=observed_at.date(),
        )
        probed = await self._probe(
            admission,
            probe_session,
            quality_policy,
            tuple(history.values()),
        )
        if probed.kind == "failure":
            raise PublicationError(probed.message)
        selected = self._select(probed)
        if selected.kind == "failure":
            raise PublicationError(selected.message)
        rendered = self._render(
            selected,
            registry or self.registry,
            runner_vantage,
        )
        previous_managed = tuple(
            f"nodes/{site_slug(site.name)}.{extension}"
            for site in admission.context.sites
            for extension in ("txt", "yaml")
        )
        return publish_bundle(
            rendered.bundle,
            repository,
            validator=validator,
            now=observed_at,
            previous_managed=previous_managed,
        )

    @staticmethod
    async def _probe(
        admitted: AdmittedRun,
        probe_session: ProbeSession,
        policy: QualityPolicy,
        history: tuple[QualityHistory, ...],
    ) -> ProbedRun | RunFailure:
        candidates = select_probe_candidates(admitted.catalog, policy)
        evidence = await probe_session.probe(candidates)
        try:
            return ProbedRun(
                admitted=admitted,
                policy=policy,
                history=history,
                evidence=evidence,
            )
        except ValidationError as error:
            return RunFailure(
                code="invalid_probe_evidence",
                message=f"probe evidence is invalid: {error.errors()[0]['msg']}",
            )

    @staticmethod
    def _select(probed: ProbedRun) -> SelectedRun | RunFailure:
        selection = assess_quality(
            probed.admitted.catalog,
            probed.evidence,
            probed.policy,
            history=probed.history_index(),
            as_of=probed.admitted.context.observed_at.date(),
        )
        if not selection.published.nodes:
            status_counts: dict[str, int] = {}
            for assessment in selection.assessments:
                status_counts[assessment.status] = (
                    status_counts.get(assessment.status, 0) + 1
                )
            status_summary = ", ".join(
                f"{status}={count}" for status, count in sorted(status_counts.items())
            )
            return RunFailure(
                code="quality_selection_empty",
                message=f"quality policy published no nodes ({status_summary})",
            )
        return SelectedRun(probed=probed, selection=selection)

    @staticmethod
    def _render(
        selected: SelectedRun,
        registry: PublicEntryRegistry,
        runner_vantage: str,
    ) -> RenderedRun:
        context = selected.probed.admitted.context
        bundle = render_quality_bundle(
            selected.selection,
            selected.probed.policy,
            registry,
            generated_at=context.observed_at,
            runner_vantage=runner_vantage,
            history=selected.probed.history_index(),
        )
        return RenderedRun(selected=selected, bundle=bundle)

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
            )
        purpose = "profile validation" if target else "publication"
        return RunFailure(
            code="no_sites",
            message=f"no sites selected for {purpose}",
        )

    @staticmethod
    def _require_admission(discovered: DiscoveredRun) -> AdmittedRun | RunFailure:
        required = {site.name for site in discovered.context.sites if site.required}
        missing: list[str] = []
        for outcome in discovered.outcomes:
            if outcome.site_name in required and (
                outcome.kind == "failure" or not outcome.artifacts
            ):
                missing.append(outcome.site_name)
        if missing:
            sites = tuple(sorted(missing))
            return RunFailure(
                code="required_discovery_unavailable",
                message="required source discovery unavailable: " + ", ".join(sites),
                sites=sites,
            )

        admitted = Scheduler._admit_available(discovered)
        if admitted.kind == "failure":
            return admitted
        catalog = admitted.catalog
        admitted_sites = {
            provenance.site for node in catalog.nodes for provenance in node.provenance
        }
        missing_admission = tuple(sorted(required - admitted_sites))
        if missing_admission:
            return RunFailure(
                code="required_admission_empty",
                message="required source admitted no nodes: "
                + ", ".join(missing_admission),
                sites=missing_admission,
            )
        return admitted

    @staticmethod
    def _admit_available(discovered: DiscoveredRun) -> AdmittedRun | RunFailure:
        artifacts: list[SourceArtifact] = []
        failures: list[str] = []
        for outcome in discovered.outcomes:
            if outcome.kind == "failure":
                failures.extend(
                    f"{outcome.site_name}: {error}" for error in outcome.errors
                )
                continue
            artifacts.extend(outcome.artifacts)
            failures.extend(f"{outcome.site_name}: {error}" for error in outcome.errors)
        if not artifacts:
            detail = "; ".join(failures[:3]) or "discovery returned no artifacts"
            return RunFailure(
                code="no_source_artifacts",
                message=f"discovery has no source artifacts: {detail}",
            )

        catalog = admit_artifacts(artifacts, now=discovered.completed_at)
        if catalog.accepted_count == 0:
            return RunFailure(
                code="admission_empty",
                message=(
                    "profile validation admitted no nodes "
                    f"({catalog.rejected_count} rejection(s))"
                ),
            )
        return AdmittedRun(discovered=discovered, catalog=catalog)

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
