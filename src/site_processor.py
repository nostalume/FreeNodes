"""SiteProcessor — full lifecycle for a single crawl target.

Handles three site types:
  - simple: blog pages with direct subscription links
  - yt_pwd: YouTube video password → blog decryption
  - cloud_drive: YouTube channel → cloud drive zip → extract
"""

import asyncio
import re
from datetime import UTC, date, datetime
from typing import Annotated, Literal
from urllib.parse import urljoin, urlparse

from pydantic import Field

from src.config import (
    Config,
    CrawlerSite,
    DriveSite,
    FrozenModel,
    PasswordEvidence,
    PasswordPolicy,
    PasswordSite,
)
from src.crawler import (
    DownloadFailure,
    DownloadOutcome,
    Page,
    WebCapability,
)
from src.decryptor import (
    DecryptionCapability,
    DecryptionFactory,
    create_decryption_client,
    extract_paste_url,
)
from src.drive import (
    DriveCapability,
    DriveFactory,
    DriveFiles,
    create_drive_client,
)
from src.llm_router import ExtractedLinks, LLMRouter
from src.nodes import PublishedDate, SourceArtifact
from src.youtube import (
    GoogleDriveResource,
    PasteResource,
    YouTubeCapability,
    extract_date_from_title,
    select_video_resource,
)


class DiscoveryBase(FrozenModel):
    site_name: str
    articles_processed: int = 0
    txt_count: int = 0
    yaml_count: int = 0
    total_bytes: int = 0
    pattern_generated: bool = False
    active_pattern: str | None = None
    errors: tuple[str, ...] = Field(default=(), strict=False)
    artifacts: tuple[SourceArtifact, ...] = Field(default=(), strict=False, repr=False)


class DiscoverySuccess(DiscoveryBase):
    kind: Literal["success"] = "success"


class DiscoveryFailure(DiscoveryBase):
    kind: Literal["failure"] = "failure"


DiscoveryOutcome = Annotated[
    DiscoverySuccess | DiscoveryFailure,
    Field(discriminator="kind"),
]


class Article(FrozenModel):
    """A dated article selected for discovery."""

    url: str
    date: str
    text: str


class DatedVideo(FrozenModel):
    """A channel video whose title supplied a publication date."""

    url: str
    date: str
    title: str


class _DiscoveryProgress(FrozenModel):
    site_name: str
    articles_processed: int = 0
    txt_count: int = 0
    yaml_count: int = 0
    total_bytes: int = 0
    pattern_generated: bool = False
    active_pattern: str | None = None
    errors: tuple[str, ...] = Field(default=(), strict=False)
    artifacts: tuple[SourceArtifact, ...] = Field(default=(), strict=False, repr=False)
    artifact_limit: int = Field(default=12, gt=0, repr=False)
    byte_limit: int = Field(default=16 * 1024 * 1024, gt=0, repr=False)

    def with_error(self, message: str) -> "_DiscoveryProgress":
        return self.model_copy(update={"errors": self.errors + (message,)})

    def with_artifact(
        self,
        artifact: SourceArtifact,
        *,
        content_kind: Literal["txt", "yaml", "other"],
    ) -> "_DiscoveryProgress":
        if len(self.artifacts) >= self.artifact_limit:
            return self.with_error("source artifact count limit exceeded")
        byte_count = len(artifact.content)
        if self.total_bytes + byte_count > self.byte_limit:
            return self.with_error("source byte limit exceeded")
        return self.model_copy(
            update={
                "artifacts": self.artifacts + (artifact,),
                "txt_count": self.txt_count + int(content_kind == "txt"),
                "yaml_count": self.yaml_count + int(content_kind == "yaml"),
                "total_bytes": self.total_bytes + byte_count,
            }
        )

    def summarize(
        self,
        *,
        articles_processed: int,
        pattern_generated: bool = False,
        active_pattern: str | None = None,
    ) -> "_DiscoveryProgress":
        return self.model_copy(
            update={
                "articles_processed": articles_processed,
                "pattern_generated": pattern_generated,
                "active_pattern": active_pattern,
            }
        )

    def finish(self) -> DiscoveryOutcome:
        if self.artifacts:
            return DiscoverySuccess(
                site_name=self.site_name,
                articles_processed=self.articles_processed,
                txt_count=self.txt_count,
                yaml_count=self.yaml_count,
                total_bytes=self.total_bytes,
                pattern_generated=self.pattern_generated,
                active_pattern=self.active_pattern,
                errors=tuple(self.errors),
                artifacts=tuple(self.artifacts),
            )
        errors = self.errors or ("discovery returned no artifacts",)
        return DiscoveryFailure(
            site_name=self.site_name,
            articles_processed=self.articles_processed,
            txt_count=self.txt_count,
            yaml_count=self.yaml_count,
            total_bytes=self.total_bytes,
            pattern_generated=self.pattern_generated,
            active_pattern=self.active_pattern,
            errors=errors,
            artifacts=tuple(self.artifacts),
        )


class ResolvedLinks(FrozenModel):
    kind: Literal["links"] = "links"
    values: tuple[str, ...] = Field(default=(), strict=False)


class LinkResolutionFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    diagnostic: str


LinkResolution = Annotated[
    ResolvedLinks | LinkResolutionFailure,
    Field(discriminator="kind"),
]


class DirectLinks(FrozenModel):
    values: tuple[str, ...] = Field(default=(), strict=False)
    generated_pattern: str | None = None


class FetchedArticles(FrozenModel):
    kind: Literal["articles"] = "articles"
    selected_count: int = 0
    pages: tuple["FetchedArticle", ...] = Field(default=(), strict=False, repr=False)
    errors: tuple[str, ...] = Field(default=(), strict=False)


class ArticleFetchFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    diagnostic: str


FetchedArticleOutcome = Annotated[
    FetchedArticles | ArticleFetchFailure,
    Field(discriminator="kind"),
]


class FetchedArticle(FrozenModel):
    article: Article
    page: Page = Field(repr=False)


class DatedResource(FrozenModel):
    value: str = Field(repr=False)
    published_on: date


class ArticleResources(FrozenModel):
    articles_processed: int = 0
    downloads: tuple[DatedResource, ...] = Field(default=(), strict=False)
    inline: tuple[DatedResource, ...] = Field(default=(), strict=False, repr=False)
    unresolved: tuple[FetchedArticle, ...] = Field(default=(), strict=False, repr=False)
    errors: tuple[str, ...] = Field(default=(), strict=False)
    pattern_generated: bool = False

    def add_resolution(
        self,
        resolution: LinkResolution,
        *,
        published_on: date,
    ) -> "ArticleResources":
        if resolution.kind == "failure":
            return self.model_copy(
                update={"errors": self.errors + (resolution.diagnostic,)}
            )
        downloads, inline = SiteProcessor._separate_inline_nodes(
            list(resolution.values)
        )
        return self.model_copy(
            update={
                "downloads": self.downloads
                + tuple(
                    DatedResource(value=value, published_on=published_on)
                    for value in downloads
                ),
                "inline": self.inline
                + tuple(
                    DatedResource(value=value, published_on=published_on)
                    for value in inline
                ),
            }
        )


class SiteProcessor:
    """Process a single site end-to-end: blog/videos → links → download → save."""

    def __init__(
        self,
        site: CrawlerSite,
        config: Config,
        llm: LLMRouter,
        youtube: YouTubeCapability,
        web: WebCapability,
        decryption_factory: DecryptionFactory = create_decryption_client,
        drive_factory: DriveFactory = create_drive_client,
    ):
        self.config = config
        self.site = site
        self.max_articles = config.crawl.max_articles
        self.llm = llm
        self.youtube = youtube
        self.web = web
        self.decryption_factory = decryption_factory
        self.drive_factory = drive_factory
        self._base = self._derive_base(site.start_url)
        self._configured_pattern = site.link_pattern
        self._generated_pattern: str | None = None

    # ── Public ──

    async def discover(self) -> DiscoveryOutcome:
        """Discover immutable artifacts without writing public output files."""
        match self.site.type:
            case "cloud_drive":
                return await self.discover_channel_resources(self.site)
            case "yt_pwd" | "youtube_password":
                return await self.discover_password_site(self.site)
            case "simple":
                return await self.discover_simple()

    async def discover_simple(self) -> DiscoveryOutcome:
        fetched = await self._fetch_article_pages()
        if fetched.kind == "failure":
            return DiscoveryFailure(
                site_name=self.site.name,
                errors=(fetched.diagnostic,),
            )
        resources = await self._resolve_direct_articles(fetched)
        return (await self._materialize_articles(resources)).finish()

    async def discover_password_site(self, site: PasswordSite) -> DiscoveryOutcome:
        fetched = await self._fetch_article_pages()
        if fetched.kind == "failure":
            return DiscoveryFailure(
                site_name=self.site.name,
                errors=(fetched.diagnostic,),
            )
        resources = await self._resolve_direct_articles(fetched)
        async with self.decryption_factory(
            proxy=self.config.crawl.proxy,
            timeout_s=float(self.config.crawl.timeout),
        ) as decryption:
            resolved = await self._resolve_password_articles(
                resources,
                decryption,
                site,
            )
        return (await self._materialize_articles(resolved)).finish()

    async def discover_channel_resources(self, site: DriveSite) -> DiscoveryOutcome:
        async with (
            self.decryption_factory(
                proxy=self.config.crawl.proxy,
                timeout_s=float(self.config.crawl.timeout),
            ) as decryption,
            self.drive_factory(
                proxy=self.config.crawl.proxy,
                timeout_s=float(self.config.crawl.timeout),
            ) as drive,
        ):
            progress = await self._run_cloud_drive(
                decryption,
                drive,
                site.password_policy,
            )
        return progress.finish()

    async def _run_cloud_drive(
        self,
        decryption: DecryptionCapability,
        drive: DriveCapability,
        policy: PasswordPolicy,
    ) -> _DiscoveryProgress:
        """Resolve each dated video to one typed resource and admit its content."""
        result = _DiscoveryProgress(
            site_name=self.site.name,
            artifact_limit=self.config.crawl.max_source_artifacts,
            byte_limit=self.config.crawl.max_source_bytes,
        )
        print(f"\n{'=' * 60}")
        print(f"SITE: {self.site.name} (cloud_drive)")
        print(f"URL:  {self.site.start_url}")
        print(f"{'=' * 60}")

        print("\n[1/3] Listing channel videos...")
        listing = await self.youtube.list_channel_videos(
            self.site.start_url,
            limit=self.max_articles,
        )
        if listing.kind == "failure":
            return result.with_error(
                f"youtube list {listing.code}: {listing.diagnostic}"
            )
        if listing.kind == "empty":
            return result.with_error("no videos found")
        videos = listing.videos
        print(f"       got {len(videos)} videos")

        picked: list[DatedVideo] = []
        for video in videos:
            published = extract_date_from_title(video.title)
            if published:
                picked.append(
                    DatedVideo(
                        url=video.url,
                        date=published,
                        title=video.title,
                    )
                )
        picked.sort(key=lambda video: video.date, reverse=True)
        picked = picked[: self.max_articles]

        for picked_video in picked:
            print(f"       [{picked_video.date}] {picked_video.title[:60]}")
        if not picked:
            return result.with_error("no dated videos found")

        print(f"\n[2/3] Processing {len(picked)} videos...")
        for picked_video in picked:
            print(f"  → {picked_video.title[:60]}")
            published_on = date.fromisoformat(picked_video.date)
            try:
                video = await self.youtube.get_video_details(picked_video.url)
                if video.kind == "failure":
                    result = result.with_error(
                        f"video details {video.code}: {video.diagnostic}"
                    )
                    continue
                resource = select_video_resource(video.url, video.resources)
                match resource.kind:
                    case "google_drive":
                        result = await self._admit_drive_resource(
                            result,
                            drive,
                            resource,
                            published_on=published_on,
                        )
                    case "paste":
                        result = await self._admit_paste_resource(
                            result,
                            decryption,
                            resource,
                            policy,
                            PasswordEvidence(
                                subtitles=video.subtitles_text,
                                description=video.description,
                            ),
                            published_on=published_on,
                        )
                    case "unsupported":
                        result = result.with_error(
                            f"unsupported {resource.provider} resource "
                            f"({resource.reason}): {resource.url}"
                        )
                    case "missing":
                        result = result.with_error(
                            f"video resource missing: {resource.video_url}"
                        )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                result = result.with_error(
                    f"video {picked_video.title[:40]}...: {error}"
                )
                continue

        return result.summarize(articles_processed=len(picked))

    async def _admit_drive_resource(
        self,
        progress: _DiscoveryProgress,
        drive: DriveCapability,
        resource: GoogleDriveResource,
        *,
        published_on: date,
    ) -> _DiscoveryProgress:
        outcome = await drive.download_archive(resource.file_id)
        if outcome.kind == "failure":
            return progress.with_error(f"drive {outcome.code}: {outcome.diagnostic}")
        if outcome.kind == "empty":
            return progress.with_error(f"drive {outcome.code}: {outcome.file_id}")
        return self._admit_drive_files(
            progress,
            outcome,
            published_on=published_on,
        )

    def _admit_drive_files(
        self,
        progress: _DiscoveryProgress,
        outcome: DriveFiles,
        *,
        published_on: date,
    ) -> _DiscoveryProgress:
        for resource in outcome.files:
            content_kind: Literal["txt", "yaml", "other"] = (
                "yaml" if resource.media_type == "application/yaml" else "txt"
            )
            progress = progress.with_artifact(
                SourceArtifact(
                    authority=self.site.name,
                    site=self.site.name,
                    source_url=f"drive://{outcome.file_id}/{resource.name}",
                    content=resource.text,
                    observed_at=datetime.now(UTC),
                    publication_time=PublishedDate(on=published_on),
                    media_type=resource.media_type,
                ),
                content_kind=content_kind,
            )
            print(f"    extracted: {resource.name} ({len(resource.content)}B)")
        return progress

    async def _admit_paste_resource(
        self,
        progress: _DiscoveryProgress,
        decryption: DecryptionCapability,
        resource: PasteResource,
        policy: PasswordPolicy,
        evidence: PasswordEvidence,
        *,
        published_on: date,
    ) -> _DiscoveryProgress:
        decrypted = await decryption.decrypt_paste(
            resource.url,
            policy.resolve(evidence),
        )
        if decrypted.kind == "failure":
            return progress.with_error(
                f"paste {decrypted.code}: {decrypted.diagnostic}"
            )
        if decrypted.kind == "rejected":
            return progress.with_error(
                f"paste {decrypted.code} after {decrypted.attempted} candidates"
            )
        links = tuple(link.href for link in decrypted.page.links)
        if not links:
            return progress.with_error("paste decrypted without resource links")
        for url in links:
            downloaded = await self.web.download_file(url)
            if downloaded.kind == "failure":
                progress = progress.with_error(
                    f"paste download {downloaded.code}: {downloaded.diagnostic}"
                )
                continue
            content_kind: Literal["txt", "yaml", "other"] = (
                "yaml" if self._is_yaml_content(downloaded.content) else "txt"
            )
            progress = progress.with_artifact(
                SourceArtifact(
                    authority=self.site.name,
                    site=self.site.name,
                    source_url=url,
                    content=downloaded.content,
                    observed_at=datetime.now(UTC),
                    publication_time=PublishedDate(on=published_on),
                    media_type=(
                        "application/yaml" if content_kind == "yaml" else "text/plain"
                    ),
                ),
                content_kind=content_kind,
            )
        return progress

    async def _fetch_article_pages(self) -> FetchedArticleOutcome:
        print(f"\n{'=' * 60}")
        print(f"SITE: {self.site.name} ({self.site.start_url})")
        print(f"Cfg:  pattern={self._configured_pattern or 'null (LLM)'}")
        print(f"{'=' * 60}")
        print("\n[1/4] Fetching blog page...")
        blog = await self.web.fetch_page(self.site.start_url)
        if not blog.success:
            return ArticleFetchFailure(
                diagnostic=f"blog fetch failed: {blog.error[:100]}"
            )
        print(f"       got {len(blog.links)} links, {len(blog.markdown)} chars")

        print(f"\n[2/4] Picking newest {self.max_articles} articles...")
        articles = self._pick_articles(blog)
        for article in articles:
            print(f"       [{article.date}] {article.url}")
        if not articles:
            return ArticleFetchFailure(diagnostic="no articles found")

        pages: list[FetchedArticle] = []
        errors: list[str] = []
        print(f"\n[3/4] Fetching {len(articles)} articles...")
        for index, article in enumerate(articles, start=1):
            print(f"  [{index}/{len(articles)}] {article.url}")
            page = await self.web.fetch_page(article.url, timeout_ms=60000)
            if page.success:
                pages.append(FetchedArticle(article=article, page=page))
            else:
                errors.append(f"article fetch failed: {page.error[:80]}")
        return FetchedArticles(
            selected_count=len(articles),
            pages=tuple(pages),
            errors=tuple(errors),
        )

    async def _resolve_direct_articles(
        self,
        fetched: FetchedArticles,
    ) -> ArticleResources:
        downloads: dict[str, DatedResource] = {}
        inline: list[DatedResource] = []
        unresolved: list[FetchedArticle] = []
        pattern_generated = False
        for fetched_article in fetched.pages:
            published_on = date.fromisoformat(fetched_article.article.date)
            direct = await self._extract_links(fetched_article.page)
            if not direct.values:
                unresolved.append(fetched_article)
                continue
            direct_downloads, direct_inline = self._separate_inline_nodes(
                list(direct.values)
            )
            for value in direct_downloads:
                downloads.setdefault(
                    value,
                    DatedResource(value=value, published_on=published_on),
                )
            inline.extend(
                DatedResource(value=value, published_on=published_on)
                for value in direct_inline
            )
            pattern_generated = pattern_generated or bool(direct.generated_pattern)
        return ArticleResources(
            articles_processed=fetched.selected_count,
            downloads=tuple(downloads.values()),
            inline=tuple(inline),
            unresolved=tuple(unresolved),
            errors=fetched.errors,
            pattern_generated=pattern_generated,
        )

    async def _resolve_password_articles(
        self,
        resources: ArticleResources,
        decryption: DecryptionCapability,
        site: PasswordSite,
    ) -> ArticleResources:
        resolved = resources.model_copy(update={"unresolved": ()})
        for fetched_article in resources.unresolved:
            resolution = await self._try_youtube_password_flow(
                fetched_article.page,
                decryption=decryption,
                policy=site.password_policy,
                paste_policy=site.paste_policy,
            )
            resolved = resolved.add_resolution(
                resolution,
                published_on=date.fromisoformat(fetched_article.article.date),
            )
        return resolved

    async def _materialize_articles(
        self,
        resources: ArticleResources,
    ) -> _DiscoveryProgress:
        active_pattern = self._generated_pattern or self._configured_pattern
        result = _DiscoveryProgress(
            site_name=self.site.name,
            errors=resources.errors,
            artifact_limit=self.config.crawl.max_source_artifacts,
            byte_limit=self.config.crawl.max_source_bytes,
        ).summarize(
            articles_processed=resources.articles_processed,
            pattern_generated=resources.pattern_generated,
            active_pattern=active_pattern,
        )
        total_links = len(resources.downloads)
        print(
            f"\n       total: {total_links} unique links, "
            f"{len(resources.inline)} inline payloads"
        )
        if not total_links and not resources.inline:
            return result.with_error("no subscription links found")

        print(
            f"\n[4/4] Downloading until {result.artifact_limit} artifacts "
            f"are admitted from {total_links} links "
            "(up to 3 attempts for transient failures)..."
        )
        for payload in resources.inline:
            result = result.with_artifact(
                SourceArtifact.inline(
                    site=self.site.name,
                    content=payload.value,
                    observed_at=datetime.now(UTC),
                    published_on=payload.published_on,
                ),
                content_kind="txt",
            )

        for resource in resources.downloads:
            if len(result.artifacts) >= result.artifact_limit:
                break
            url = resource.value
            downloaded = await self._download_retry(url)
            if downloaded.kind == "failure":
                result = result.with_error(
                    f"download {downloaded.code}: {downloaded.diagnostic}"
                )
                print(f"  FAIL: {url}")
                continue
            body = downloaded.content
            content_kind: Literal["txt", "yaml", "other"] = (
                "yaml" if self._is_yaml_content(body) else "txt"
            )
            print(f"  OK  {content_kind}: {url} ({len(body)}B)")
            result = result.with_artifact(
                SourceArtifact(
                    authority=self.site.name,
                    site=self.site.name,
                    source_url=url,
                    content=body,
                    observed_at=datetime.now(UTC),
                    publication_time=PublishedDate(on=resource.published_on),
                    media_type=(
                        "application/yaml" if content_kind == "yaml" else "text/plain"
                    ),
                ),
                content_kind=content_kind,
            )
        return result

    # ── YouTube password flow ──

    async def _try_youtube_password_flow(
        self,
        page: Page,
        *,
        decryption: DecryptionCapability,
        policy: PasswordPolicy,
        paste_policy: PasswordPolicy,
    ) -> LinkResolution:
        """Resolve paste or video evidence into explicit subscription links."""
        paste_url = extract_paste_url(page.html) or extract_paste_url(page.markdown)
        if paste_url:
            print(f"    found paste.to URL: {paste_url[:50]}...")
            paste = await decryption.decrypt_paste(
                paste_url,
                paste_policy.resolve(
                    PasswordEvidence(description=page.markdown),
                ),
            )
            if paste.kind == "decrypted":
                if paste.page.links:
                    print(f"    paste decrypted → {len(paste.page.links)} links")
                    return ResolvedLinks(
                        values=tuple(link.href for link in paste.page.links),
                    )
                return LinkResolutionFailure(
                    diagnostic="paste decrypted without resource links"
                )
            if paste.kind == "failure":
                return LinkResolutionFailure(
                    diagnostic=f"paste {paste.code}: {paste.diagnostic}"
                )
            return LinkResolutionFailure(
                diagnostic=(f"paste {paste.code} after {paste.attempted} candidates")
            )

        if not page.requires_password():
            print("    page not password-protected, skipping decrypt")
            return LinkResolutionFailure(
                diagnostic="page has no direct links and is not password-protected"
            )

        yt_links = self._youtube_links(page.html, page.markdown)
        if not yt_links:
            print("    no YouTube links found in page")
            candidates = policy.resolve(PasswordEvidence())
        else:
            evidence = PasswordEvidence()
            diagnoses: list[str] = []
            for yt_url in yt_links[:2]:
                print(f"    checking YouTube: {yt_url[:50]}...")
                try:
                    video = await self.youtube.get_video_details(yt_url)
                    if video.kind == "failure":
                        diagnoses.append(
                            f"youtube details {video.code}: {video.diagnostic}"
                        )
                        continue
                    evidence = PasswordEvidence(
                        subtitles=video.subtitles_text,
                        description=video.description,
                    )
                    candidates = policy.resolve(evidence)
                    print(f"    admitted {len(candidates.values)} password candidates")
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    diagnoses.append(f"youtube details failed: {error}")
                    continue
            else:
                return LinkResolutionFailure(
                    diagnostic="; ".join(diagnoses)
                    or "youtube details returned no usable evidence"
                )

        print(f"    trying {len(candidates.values)} password candidates...")
        decrypted = await decryption.decrypt_page(page.url, candidates)
        if decrypted.kind == "decrypted":
            exact = self._exact_links(
                decrypted.page.html,
                decrypted.page.markdown,
            )
            if exact:
                return ResolvedLinks(values=exact)
            llm_result = await self.llm.extract_links(
                decrypted.page.markdown,
                site=self.site.name,
            )
            links = llm_result.downloads + llm_result.inline
            print(
                f"    decrypt success with password {decrypted.password} → {len(links)} links"
            )
            if links:
                return ResolvedLinks(values=links)
            return LinkResolutionFailure(
                diagnostic="decrypted page contained no subscription links"
            )

        if decrypted.kind == "rejected":
            print(f"    decrypt rejected after {decrypted.attempted} candidates")
            return LinkResolutionFailure(
                diagnostic=(
                    f"decryption {decrypted.code} after "
                    f"{decrypted.attempted} candidates"
                )
            )
        print(f"    decrypt {decrypted.code}: {decrypted.diagnostic}")
        return LinkResolutionFailure(
            diagnostic=f"decryption {decrypted.code}: {decrypted.diagnostic}"
        )

    # ── Article selection ──

    def _pick_articles(self, blog: Page) -> list[Article]:
        """Select the *max_articles* newest articles from a blog listing page.

        Rule-only: tries several date formats found across different blog sites.
        Falls back to scanning markdown content for article links when ``links`` is empty.
        """
        # Strategy 1: Use Crawl4AI's parsed links
        articles = self._pick_articles_from_links(blog)
        if articles:
            return articles

        # Strategy 2: Fallback — parse markdown for [title](url) article links
        articles = self._pick_articles_from_markdown(blog.markdown)
        return articles[: self.max_articles]

    def _pick_articles_from_links(self, blog: Page) -> list[Article]:
        """Extract article links from Crawl4AI's parsed link list."""
        articles: list[Article] = []
        for link in blog.links:
            text = link.text
            href = link.href
            d = self._parse_article_date(text, href)
            if d is None:
                continue

            # Exclude non-article links: exact matches + configurable substring patterns
            if href in ("/free-nodes/", "/", ""):
                continue
            exclusions = self.site.exclude_patterns or ("category", "page-")
            if any(pattern in href for pattern in exclusions):
                continue

            full = urljoin(self._base, href)
            articles.append(Article(url=full, date=d, text=text[:80]))

        seen: set[str] = set()
        unique: list[Article] = []
        for article in sorted(articles, key=lambda value: value.date, reverse=True):
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)
        return unique[: self.max_articles]

    def _pick_articles_from_markdown(self, markdown: str) -> list[Article]:
        """Fallback: parse markdown headings for ``[title](url)`` article links with Chinese dates.

        Handles WordPress blogs where articles are rendered in headings
        but not captured in Crawl4AI's ``links`` structure (for example, yudou).
        """
        articles: list[Article] = []
        # Match markdown headings: ## [title text](url)
        for m in re.finditer(
            r"^## \[(.+?)\]\((https?://[^\s)]+)\)", markdown, re.MULTILINE
        ):
            text = m.group(1)
            url = m.group(2).rstrip(".,;)")
            d = self._parse_article_date(text, url)
            if d:
                articles.append(Article(url=url, date=d, text=text[:80]))

        seen: set[str] = set()
        unique: list[Article] = []
        for article in sorted(articles, key=lambda value: value.date, reverse=True):
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)
        return unique[: self.max_articles]

    # ── Link extraction: rule-first + LLM fallback + self-heal ──

    async def _extract_links(self, page: Page) -> DirectLinks:
        """Resolve direct links and retain only valid run-local regex state."""
        html = page.html
        if self._generated_pattern:
            matched = self._extract_by_pattern(html, self._generated_pattern)
            if matched:
                print(f"    regex hit: {len(matched)} links (0 LLM)")
                return DirectLinks(values=tuple(matched))
            self._generated_pattern = None
            print("    generated regex missed; discarding run-local reuse")

        if self._configured_pattern:
            matched = self._extract_by_pattern(html, self._configured_pattern)
            if matched:
                print(f"    regex hit: {len(matched)} links (0 LLM)")
                return DirectLinks(values=tuple(matched))
            print("    configured regex missed; keeping declarative pattern")

        exact = self._exact_links(page.html, page.markdown)
        if exact:
            print(f"    exact parser hit: {len(exact)} links (0 LLM)")
            return DirectLinks(values=exact)

        llm_result = await self.llm.extract_links(page.markdown, site=self.site.name)
        values = llm_result.downloads + llm_result.inline
        if not values:
            return DirectLinks()

        print(f"    LLM found {len(values)} links")
        for link in values[:3]:
            print(f"       {link}")
        if not llm_result.downloads:
            return DirectLinks(values=values)

        new_pattern = await self.llm.generate_pattern(
            list(llm_result.downloads),
            html,
            site=self.site.name,
        )
        if not new_pattern:
            print("    LLM could not generate a reusable pattern")
            return DirectLinks(values=values)

        print(f"    generated pattern: {new_pattern[:80]}...")
        if LLMRouter.verify_pattern(
            new_pattern,
            list(llm_result.downloads),
            html,
        ):
            self._generated_pattern = new_pattern
            print("    pattern verified for run-local reuse")
            return DirectLinks(values=values, generated_pattern=new_pattern)
        print("    pattern rejected by verification")
        return DirectLinks(values=values)

    # ── Helpers ──

    @staticmethod
    def _youtube_links(*texts: str) -> tuple[str, ...]:
        pattern = re.compile(
            r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
        )
        return tuple(
            dict.fromkeys(
                match.group() for text in texts for match in pattern.finditer(text)
            )
        )

    @staticmethod
    def _exact_links(*texts: str) -> tuple[str, ...]:
        admitted = tuple(ExtractedLinks.from_text(text) for text in texts)
        return tuple(
            dict.fromkeys(
                link for result in admitted for link in result.downloads + result.inline
            )
        )

    @staticmethod
    def _separate_inline_nodes(values: list[str]) -> tuple[list[str], list[str]]:
        """Separate network download URLs from inline subscription payloads."""
        downloads: list[str] = []
        inline: list[str] = []
        protocol_prefixes = (
            "vmess://",
            "vless://",
            "trojan://",
            "ss://",
            "ssr://",
            "socks://",
            "socks5://",
            "hysteria://",
            "hysteria2://",
            "hy2://",
            "tuic://",
        )
        for value in values:
            stripped = value.strip()
            lines = [line.strip() for line in stripped.splitlines() if line.strip()]
            if lines and all(
                line.lower().startswith(protocol_prefixes) for line in lines
            ):
                inline.append("\n".join(lines))
            elif stripped.startswith(("http://", "https://")):
                downloads.append(stripped)
        return downloads, inline

    @staticmethod
    def _is_yaml_content(body: str) -> bool:
        """Detect if downloaded content is Clash YAML vs plain-text node list.

        YAML configs contain YAML structure keys like ``proxies:``, ``mixed-port:``,
        or ``proxy-groups:`` at the start of lines. Plain node lists contain
        protocol links (vmess://, ss://) or base64.
        """
        for line in body.splitlines()[:20]:
            line = line.strip()
            if line.startswith(
                (
                    "proxies:",
                    "proxy-groups:",
                    "mixed-port:",
                    "allow-lan:",
                    "mode:",
                    "rules:",
                    "dns:",
                )
            ):
                return True
        return False

    @staticmethod
    def _parse_article_date(text: str, href: str) -> str | None:
        """Extract date from article title text or URL href.

        Tries these patterns in order:
          1. ``X月X日`` in title text                    → 2026-06-18
          2. ``YYYY年MM月DD日`` in title text              → 2026-06-18
          3. ``YYYY/MM/DD`` in title text                 → 2026-06-18
          4. ``YYYY-MM-DD`` in href URL                   → 2026-06-18
          5. ``YYYYMMDD`` (8-digit) in href URL            → 2026-06-18
        """
        m = re.search(r"(\d{1,2})月(\d{1,2})日", text)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            today = date.today()
            parsed = date(today.year, month, day)
            # Cross-year boundary: if parsed date is > 1 month in the future,
            # it's likely from last year (e.g. "12月30日" seen on Jan 1st)
            diff_days = (parsed - today).days
            if diff_days > 30:
                parsed = parsed.replace(year=today.year - 1)
            return parsed.isoformat()
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", text)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", href)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        m = re.search(r"/(\d{8})[/-]", href)
        if m:
            raw = m.group(1)
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        return None

    @staticmethod
    def _extract_by_pattern(html: str, pattern: str) -> list[str]:
        """Extract subscription URLs matching *pattern*."""
        compiled = re.compile(pattern, re.IGNORECASE)
        cleaned: list[str] = []
        for match in compiled.finditer(html):
            value = match.group(1) if compiled.groups else match.group(0)
            if value and value.startswith("http"):
                cleaned.append(re.sub(r'[),;.\'"]+$', "", value))
        return cleaned

    async def _download_retry(self, url: str, retries: int = 3) -> DownloadOutcome:
        """Retry typed HTTP failures while preserving the final diagnosis."""
        failure = DownloadFailure(
            code="http_error",
            url=url,
            diagnostic="download was not attempted",
        )
        for attempt in range(retries):
            try:
                outcome = await self.web.download_file(url)
                if outcome.kind == "downloaded":
                    return outcome
                failure = outcome
                if not failure.retryable or attempt == retries - 1:
                    break
                print(f"    retry {attempt + 1}/{retries} {url} ({failure.diagnostic})")
                await asyncio.sleep(1.5)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failure = DownloadFailure(
                    code="http_error",
                    url=url,
                    diagnostic=str(error)[:200] or "download failed",
                )
                break
        return failure

    @staticmethod
    def _derive_base(start_url: str) -> str:
        """Extract scheme + host from a URL for resolving relative links."""
        parsed = urlparse(start_url)
        return f"{parsed.scheme}://{parsed.netloc}"
