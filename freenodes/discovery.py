import asyncio
import re
from datetime import date
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from freenodes.config import (
    DiscoveryLimits,
    DiscoverySource,
    FrozenModel,
    PasswordCandidates,
    PasswordEvidence,
    PasswordPageSource,
    PasswordPolicy,
    YouTubeResourceSource,
)
from freenodes.decryption import (
    DecryptionCapability,
    DecryptionFactory,
    create_decryption_client,
    extract_paste_url,
)
from freenodes.drive import (
    DriveCapability,
    DriveFactory,
    DriveFiles,
    create_drive_client,
)
from freenodes.llm import ExtractedLinks, OpenRouterFallback
from freenodes.nodes import PublishedDate, SourceArtifact
from freenodes.web import (
    Article,
    ArticleSelector,
    DownloadFailure,
    DownloadOutcome,
    Page,
    WebCapability,
)
from freenodes.youtube import (
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


class DatedVideo(FrozenModel):
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
        return self.model_copy(update={"errors": (*self.errors, message)})

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
                "artifacts": (*self.artifacts, artifact),
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


class DiscoveryRequest(FrozenModel):
    limits: DiscoveryLimits
    observed_at: AwareDatetime


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
                update={"errors": (*self.errors, resolution.diagnostic)}
            )
        downloads, inline = SourceDiscovery._separate_inline_nodes(
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


class SourceDiscovery:
    def __init__(
        self,
        site: DiscoverySource,
        request: DiscoveryRequest,
        llm: OpenRouterFallback,
        youtube: YouTubeCapability,
        web: WebCapability,
        decryption_factory: DecryptionFactory = create_decryption_client,
        drive_factory: DriveFactory = create_drive_client,
    ):
        self.limits = request.limits
        self.observed_at = request.observed_at
        self.site = site
        self.max_articles = request.limits.article_limit
        self.llm = llm
        self.youtube = youtube
        self.web = web
        self.decryption_factory = decryption_factory
        self.drive_factory = drive_factory
        match site.kind:
            case "web" | "password_page":
                self._configured_pattern = site.resource_pattern
                self.articles = ArticleSelector(
                    site,
                    limit=self.max_articles,
                    observed_on=self.observed_at.date(),
                )
            case "youtube_resources":
                self._configured_pattern = None
                self.articles = None
        self._generated_pattern: str | None = None

    async def discover(self) -> DiscoveryOutcome:
        match self.site.kind:
            case "youtube_resources":
                return await self.discover_channel_resources(self.site)
            case "password_page":
                return await self.discover_password_site(self.site)
            case "web":
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

    async def discover_password_site(
        self, site: PasswordPageSource
    ) -> DiscoveryOutcome:
        fetched = await self._fetch_article_pages()
        if fetched.kind == "failure":
            return DiscoveryFailure(
                site_name=self.site.name,
                errors=(fetched.diagnostic,),
            )
        resources = await self._resolve_direct_articles(fetched)
        async with self.decryption_factory(
            proxy=self.limits.proxy_url or "",
            timeout_s=float(self.limits.request_timeout_seconds),
        ) as decryption:
            resolved = await self._resolve_password_articles(
                resources,
                decryption,
                site,
            )
        return (await self._materialize_articles(resolved)).finish()

    async def discover_channel_resources(
        self, site: YouTubeResourceSource
    ) -> DiscoveryOutcome:
        async with (
            self.decryption_factory(
                proxy=self.limits.proxy_url or "",
                timeout_s=float(self.limits.request_timeout_seconds),
            ) as decryption,
            self.drive_factory(
                proxy=self.limits.proxy_url or "",
                timeout_s=float(self.limits.request_timeout_seconds),
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
        result = _DiscoveryProgress(
            site_name=self.site.name,
            artifact_limit=self.limits.artifact_limit_per_source,
            byte_limit=self.limits.byte_limit_per_source,
        )
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
        picked: list[DatedVideo] = []
        for video in videos:
            published = extract_date_from_title(video.title, self.observed_at.date())
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

        if not picked:
            return result.with_error("no dated videos found")

        for picked_video in picked:
            published_on = date.fromisoformat(picked_video.date)
            try:
                result = await self._admit_video_resource(
                    result,
                    picked_video,
                    decryption,
                    drive,
                    policy,
                    published_on,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                result = result.with_error(
                    f"video {picked_video.title[:40]}...: {error}"
                )
                continue

        return result.summarize(articles_processed=len(picked))

    async def _admit_video_resource(
        self,
        progress: _DiscoveryProgress,
        picked: DatedVideo,
        decryption: DecryptionCapability,
        drive: DriveCapability,
        policy: PasswordPolicy,
        published_on: date,
    ) -> _DiscoveryProgress:
        video = await self.youtube.get_video_details(picked.url)
        if video.kind == "failure":
            return progress.with_error(
                f"video details {video.code}: {video.diagnostic}"
            )
        resource = select_video_resource(video.url, video.resources)
        match resource.kind:
            case "google_drive":
                return await self._admit_drive_resource(
                    progress,
                    drive,
                    resource,
                    published_on=published_on,
                )
            case "paste":
                return await self._admit_paste_resource(
                    progress,
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
                return progress.with_error(
                    f"unsupported {resource.provider} resource "
                    f"({resource.reason}): {resource.url}"
                )
            case "missing":
                return progress.with_error(
                    f"video resource missing: {resource.video_url}"
                )

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
                    observed_at=self.observed_at,
                    publication_time=PublishedDate(on=published_on),
                    media_type=resource.media_type,
                ),
                content_kind=content_kind,
            )
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
                    observed_at=self.observed_at,
                    publication_time=PublishedDate(on=published_on),
                    media_type=(
                        "application/yaml" if content_kind == "yaml" else "text/plain"
                    ),
                ),
                content_kind=content_kind,
            )
        return progress

    async def _fetch_article_pages(self) -> FetchedArticleOutcome:
        blog = await self.web.fetch_page(self.site.start_url)
        if not blog.success:
            return ArticleFetchFailure(
                diagnostic=f"blog fetch failed: {blog.error[:100]}"
            )
        if self.articles is None:
            return ArticleFetchFailure(diagnostic="source does not expose articles")
        articles = self.articles.select(blog)
        if not articles:
            return ArticleFetchFailure(diagnostic="no articles found")

        pages: list[FetchedArticle] = []
        errors: list[str] = []
        for article in articles:
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
        site: PasswordPageSource,
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
            artifact_limit=self.limits.artifact_limit_per_source,
            byte_limit=self.limits.byte_limit_per_source,
        ).summarize(
            articles_processed=resources.articles_processed,
            pattern_generated=resources.pattern_generated,
            active_pattern=active_pattern,
        )
        if not resources.downloads and not resources.inline:
            return result.with_error("no subscription links found")

        for payload in resources.inline:
            result = result.with_artifact(
                SourceArtifact.inline(
                    site=self.site.name,
                    content=payload.value,
                    observed_at=self.observed_at,
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
                continue
            body = downloaded.content
            content_kind: Literal["txt", "yaml", "other"] = (
                "yaml" if self._is_yaml_content(body) else "txt"
            )
            result = result.with_artifact(
                SourceArtifact(
                    authority=self.site.name,
                    site=self.site.name,
                    source_url=url,
                    content=body,
                    observed_at=self.observed_at,
                    publication_time=PublishedDate(on=resource.published_on),
                    media_type=(
                        "application/yaml" if content_kind == "yaml" else "text/plain"
                    ),
                ),
                content_kind=content_kind,
            )
        return result

    async def _try_youtube_password_flow(
        self,
        page: Page,
        *,
        decryption: DecryptionCapability,
        policy: PasswordPolicy,
        paste_policy: PasswordPolicy,
    ) -> LinkResolution:
        paste_url = extract_paste_url(page.html) or extract_paste_url(page.markdown)
        if paste_url:
            return await self._resolve_paste_page(
                paste_url, page, decryption, paste_policy
            )

        if not page.requires_password():
            return LinkResolutionFailure(
                diagnostic="page has no direct links and is not password-protected"
            )

        yt_links = self._youtube_links(page.html, page.markdown)
        if not yt_links:
            candidates = policy.resolve(PasswordEvidence())
        else:
            evidence = PasswordEvidence()
            diagnoses: list[str] = []
            for yt_url in yt_links[:2]:
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

        return await self._decrypt_page(page, decryption, candidates)

    async def _resolve_paste_page(
        self,
        paste_url: str,
        page: Page,
        decryption: DecryptionCapability,
        policy: PasswordPolicy,
    ) -> LinkResolution:
        paste = await decryption.decrypt_paste(
            paste_url,
            policy.resolve(PasswordEvidence(description=page.markdown)),
        )
        if paste.kind == "decrypted":
            if paste.page.links:
                return ResolvedLinks(
                    values=tuple(link.href for link in paste.page.links)
                )
            return LinkResolutionFailure(
                diagnostic="paste decrypted without resource links"
            )
        if paste.kind == "failure":
            return LinkResolutionFailure(
                diagnostic=f"paste {paste.code}: {paste.diagnostic}"
            )
        return LinkResolutionFailure(
            diagnostic=f"paste {paste.code} after {paste.attempted} candidates"
        )

    async def _decrypt_page(
        self,
        page: Page,
        decryption: DecryptionCapability,
        candidates: PasswordCandidates,
    ) -> LinkResolution:
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
                source=self.site.name,
            )
            links = llm_result.downloads + llm_result.inline
            if links:
                return ResolvedLinks(values=links)
            return LinkResolutionFailure(
                diagnostic="decrypted page contained no subscription links"
            )

        if decrypted.kind == "rejected":
            return LinkResolutionFailure(
                diagnostic=(
                    f"decryption {decrypted.code} after "
                    f"{decrypted.attempted} candidates"
                )
            )
        return LinkResolutionFailure(
            diagnostic=f"decryption {decrypted.code}: {decrypted.diagnostic}"
        )

    async def _extract_links(self, page: Page) -> DirectLinks:
        html = page.html
        if self._generated_pattern:
            matched = page.pattern_links(self._generated_pattern)
            if matched:
                return DirectLinks(values=tuple(matched))
            self._generated_pattern = None

        if self._configured_pattern:
            matched = page.pattern_links(self._configured_pattern)
            if matched:
                return DirectLinks(values=tuple(matched))

        exact = self._exact_links(page.html, page.markdown)
        if exact:
            return DirectLinks(values=exact)

        llm_result = await self.llm.extract_links(page.markdown, source=self.site.name)
        values = llm_result.downloads + llm_result.inline
        if not values:
            return DirectLinks()

        if not llm_result.downloads:
            return DirectLinks(values=values)

        new_pattern = await self.llm.generate_pattern(
            list(llm_result.downloads),
            source=self.site.name,
        )
        if not new_pattern:
            return DirectLinks(values=values)

        if OpenRouterFallback.verify_pattern(
            new_pattern,
            list(llm_result.downloads),
            html,
        ):
            self._generated_pattern = new_pattern
            return DirectLinks(values=values, generated_pattern=new_pattern)
        return DirectLinks(values=values)

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

    async def _download_retry(self, url: str, retries: int = 3) -> DownloadOutcome:
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
