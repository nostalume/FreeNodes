"""Vertical characterization of every configured source variant."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pytest

from src.config import (
    Config,
    CrawlConfig,
    DriveSite,
    LLMConfig,
    PasswordCandidates,
    PasswordSite,
    SimpleSite,
    Site,
)
from src.crawler import DownloadedText, DownloadFailure, DownloadOutcome, Page, PageLink
from src.decryptor import (
    DecryptedResource,
    DecryptionOutcome,
    DecryptionRejected,
)
from src.drive import DriveFailure, DriveFile, DriveFiles, DriveOutcome
from src.llm_router import ExtractedLinks
from src.mihomo import ConsumerValidation, DelayObservation, ProbeEvidence
from src.scheduler import Scheduler
from src.site_processor import SiteProcessor
from src.youtube import (
    ChannelEmpty,
    ChannelOutcome,
    ChannelVideos,
    DetailsOutcome,
    SubtitlesAvailable,
    SubtitlesEmpty,
    VideoDetails,
    VideoReference,
    YouTubeCapability,
    YouTubeFailure,
    classify_video_resources,
)

SIMPLE_ROOT = "https://simple.test/blog/"
SIMPLE_ARTICLE = "https://simple.test/2026/08/29/nodes"
SIMPLE_DOWNLOAD = "https://files.test/simple.txt"
PASSWORD_ROOT = "https://password.test/blog/"
PASSWORD_ARTICLE = "https://password.test/2026/08/29/nodes"
PASSWORD_DOWNLOAD = "https://files.test/password.txt"
PASSWORD_VIDEO = "https://youtube.com/watch?v=password123"
DRIVE_CHANNEL = "https://youtube.com/@drive-source"
DRIVE_VIDEO = "https://youtube.com/watch?v=drive123"
DRIVE_ID = "abcdefghijk1"


def subscription(host: str, name: str) -> str:
    return f"trojan://secret-{name}@{host}:443#{name}"


@dataclass
class FakeWebCapability:
    pages: dict[str, Page]
    downloads: dict[str, str] = field(default_factory=dict)
    cancelled_urls: set[str] = field(default_factory=set)
    fetched: list[str] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)

    async def fetch_page(self, url: str, timeout_ms: int = 60000) -> Page:
        self.fetched.append(url)
        if url in self.cancelled_urls:
            raise asyncio.CancelledError
        return self.pages[url]

    async def download_file(self, url: str) -> DownloadOutcome:
        self.downloaded.append(url)
        content = self.downloads[url]
        return DownloadedText(
            url=url,
            content=content,
            byte_count=len(content.encode()),
        )


class LinkCapability(Protocol):
    async def extract_links(
        self,
        markdown: str,
        *,
        site: str = "default",
    ) -> ExtractedLinks: ...

    async def generate_pattern(
        self,
        links: list[str],
        html: str,
        *,
        site: str = "default",
    ) -> str | None: ...


@dataclass
class FakeLinkCapability:
    results: deque[ExtractedLinks] = field(default_factory=deque)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def extract_links(
        self,
        markdown: str,
        *,
        site: str = "default",
    ) -> ExtractedLinks:
        self.calls.append((site, markdown))
        return self.results.popleft() if self.results else ExtractedLinks()

    async def generate_pattern(
        self,
        links: list[str],
        html: str,
        *,
        site: str = "default",
    ) -> str | None:
        return None


@dataclass
class FakeYouTubeCapability:
    channel_videos: tuple[VideoReference, ...] = ()
    details: dict[str, DetailsOutcome] = field(default_factory=dict)
    listing_failure: YouTubeFailure | None = None
    cancel_listing: bool = False
    listed: list[str] = field(default_factory=list)
    inspected: list[str] = field(default_factory=list)

    async def list_channel_videos(
        self,
        channel_url: str,
        limit: int = 10,
    ) -> ChannelOutcome:
        self.listed.append(channel_url)
        if self.cancel_listing:
            raise asyncio.CancelledError
        if self.listing_failure is not None:
            return self.listing_failure
        videos = self.channel_videos[:limit]
        if not videos:
            return ChannelEmpty(url=channel_url)
        return ChannelVideos(channel_url=channel_url, videos=videos)

    async def get_video_details(self, video_url: str) -> DetailsOutcome:
        self.inspected.append(video_url)
        return self.details[video_url]


@dataclass
class FakeDriveCapability:
    outcomes: dict[str, DriveOutcome]
    downloaded: list[str] = field(default_factory=list)
    enter_calls: int = 0
    close_calls: int = 0

    async def __aenter__(self) -> "FakeDriveCapability":
        self.enter_calls += 1
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self.close_calls += 1

    async def download_archive(self, file_id: str) -> DriveOutcome:
        self.downloaded.append(file_id)
        return self.outcomes[file_id]


@dataclass
class FakeDriveFactory:
    outcomes: dict[str, DriveOutcome]
    calls: list[tuple[str, float]] = field(default_factory=list)
    clients: list[FakeDriveCapability] = field(default_factory=list)

    def __call__(
        self,
        *,
        proxy: str,
        timeout_s: float,
    ) -> FakeDriveCapability:
        self.calls.append((proxy, timeout_s))
        client = FakeDriveCapability(dict(self.outcomes))
        self.clients.append(client)
        return client


@dataclass
class FakeDecryptionCapability:
    page_results: dict[str, Page] = field(default_factory=dict)
    paste_results: dict[str, Page] = field(default_factory=dict)
    page_attempts: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    paste_attempts: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    enter_calls: int = 0
    close_calls: int = 0

    async def __aenter__(self) -> "FakeDecryptionCapability":
        self.enter_calls += 1
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self.close_calls += 1

    async def decrypt_page(
        self,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome:
        self.page_attempts.append((url, candidates.values))
        for attempted, password in enumerate(candidates.values, start=1):
            result = self.page_results.get(password)
            if result is not None:
                return DecryptedResource(
                    operation="password_page",
                    page=result,
                    password=password,
                    attempted=attempted,
                )
        return DecryptionRejected(
            operation="password_page",
            code="no_subscription",
            url=url,
            attempted=len(candidates.values),
        )

    async def decrypt_paste(
        self,
        url: str,
        candidates: PasswordCandidates,
    ) -> DecryptionOutcome:
        self.paste_attempts.append((url, candidates.values))
        for attempted, password in enumerate(candidates.values, start=1):
            result = self.paste_results.get(password)
            if result is not None:
                return DecryptedResource(
                    operation="paste",
                    page=result,
                    password=password,
                    attempted=attempted,
                )
        return DecryptionRejected(
            operation="paste",
            code="no_subscription",
            url=url,
            attempted=len(candidates.values),
        )


@dataclass
class FakeDecryptionFactory:
    page_results: dict[str, Page] = field(default_factory=dict)
    paste_results: dict[str, Page] = field(default_factory=dict)
    calls: list[tuple[str, float]] = field(default_factory=list)
    clients: list[FakeDecryptionCapability] = field(default_factory=list)

    def __call__(
        self,
        *,
        proxy: str,
        timeout_s: float,
    ) -> FakeDecryptionCapability:
        self.calls.append((proxy, timeout_s))
        client = FakeDecryptionCapability(
            page_results=dict(self.page_results),
            paste_results=dict(self.paste_results),
        )
        self.clients.append(client)
        return client


def page(
    url: str,
    *,
    links: tuple[PageLink, ...] = (),
    markdown: str = "",
    html: str = "",
    success: bool = True,
    error: str = "",
) -> Page:
    return Page(
        url=url,
        links=links,
        markdown=markdown,
        html=html,
        success=success,
        error=error,
    )


def article_link() -> tuple[PageLink, ...]:
    return (PageLink(href="/2026/08/29/nodes", text="8月29日 nodes"),)


def video_reference(
    url: str,
    *,
    title: str = "2026/08/29 nodes",
) -> VideoReference:
    return VideoReference(
        url=url,
        video_id=url.rsplit("=", 1)[-1],
        title=title,
        upload_date="20260829",
        channel="fixture",
    )


def video_details(
    url: str,
    *,
    title: str = "2026/08/29 nodes",
    description: str = "",
    subtitles: str = "",
) -> VideoDetails:
    subtitle_result = (
        SubtitlesAvailable(text=subtitles) if subtitles else SubtitlesEmpty(url=url)
    )
    return VideoDetails(
        url=url,
        video_id=url.rsplit("=", 1)[-1],
        title=title,
        description=description,
        upload_date="20260829",
        channel="fixture",
        subtitles=subtitle_result,
        resources=classify_video_resources(description),
    )


def youtube_failure(url: str, diagnosis: str) -> YouTubeFailure:
    return YouTubeFailure(
        operation="details",
        code="malformed_output",
        url=url,
        diagnostic=diagnosis,
    )


def config(*sites: Site, output: Path | None = None) -> Config:
    values: dict[str, object] = {
        "sites": sites,
        "crawl": CrawlConfig(max_articles=1, concurrency=3),
        "llm": LLMConfig(),
    }
    if output is not None:
        values["output"] = {"dir": output}
    return Config.model_validate(values)


def simple_site(name: str = "simple-source") -> SimpleSite:
    return SimpleSite(
        name=name,
        start_url=SIMPLE_ROOT,
        link_pattern=r"https://files\.test/simple\.txt",
    )


def password_site(name: str = "password-source") -> PasswordSite:
    return PasswordSite(
        name=name,
        start_url=PASSWORD_ROOT,
        link_pattern=r"https://never\.matches/[^\s]+",
    )


def drive_site(name: str = "drive-source") -> DriveSite:
    return DriveSite(name=name, start_url=DRIVE_CHANNEL)


@dataclass
class SuccessCapabilities:
    web: FakeWebCapability
    youtube: FakeYouTubeCapability
    drive: FakeDriveFactory
    decryption: FakeDecryptionFactory
    llm: FakeLinkCapability


def success_capabilities() -> SuccessCapabilities:
    return SuccessCapabilities(
        web=FakeWebCapability(
            pages={
                SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
                SIMPLE_ARTICLE: page(
                    SIMPLE_ARTICLE,
                    html=f'<a href="{SIMPLE_DOWNLOAD}">nodes</a>',
                ),
                PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
                PASSWORD_ARTICLE: page(
                    PASSWORD_ARTICLE,
                    markdown="protected page",
                    html=f'<input type="password"> {PASSWORD_VIDEO} ',
                ),
            },
            downloads={
                SIMPLE_DOWNLOAD: subscription("simple.example", "simple"),
                PASSWORD_DOWNLOAD: subscription("password.example", "password"),
            },
        ),
        youtube=FakeYouTubeCapability(
            channel_videos=(video_reference(DRIVE_VIDEO),),
            details={
                PASSWORD_VIDEO: video_details(
                    PASSWORD_VIDEO,
                    subtitles="密码 1122",
                ),
                DRIVE_VIDEO: video_details(
                    DRIVE_VIDEO,
                    description=(
                        f"download https://drive.google.com/file/d/{DRIVE_ID}/view"
                    ),
                ),
            },
        ),
        drive=FakeDriveFactory(
            {
                DRIVE_ID: DriveFiles(
                    file_id=DRIVE_ID,
                    files=(
                        DriveFile(
                            name=f"{DRIVE_ID}.txt",
                            content=subscription(
                                "drive.example",
                                "drive",
                            ).encode(),
                            media_type="text/plain",
                        ),
                    ),
                )
            }
        ),
        decryption=FakeDecryptionFactory(
            page_results={
                "1122": page(
                    PASSWORD_ARTICLE,
                    markdown="decrypted subscription",
                )
            }
        ),
        llm=FakeLinkCapability(
            deque((ExtractedLinks(), ExtractedLinks(txt=(PASSWORD_DOWNLOAD,))))
        ),
    )


@pytest.mark.parametrize(
    ("site", "source_url"),
    (
        (simple_site(), SIMPLE_DOWNLOAD),
        (password_site(), PASSWORD_DOWNLOAD),
        (drive_site(), f"drive://{DRIVE_ID}/{DRIVE_ID}.txt"),
    ),
)
async def test_variant_yields_source_owned_artifact(site, source_url):
    capabilities = success_capabilities()

    outcome = await SiteProcessor(
        site,
        config(site),
        capabilities.llm,
        capabilities.youtube,
        capabilities.web,
        capabilities.decryption,
        capabilities.drive,
    ).discover()

    assert outcome.kind == "success"
    assert outcome.site_name == site.name
    assert tuple(artifact.site for artifact in outcome.artifacts) == (site.name,)
    assert tuple(artifact.source_url for artifact in outcome.artifacts) == (source_url,)


async def test_simple_inline_node_is_admitted_without_a_download():
    site = SimpleSite(name="inline-source", start_url=SIMPLE_ROOT)
    inline = subscription("inline.example", "inline")
    web = FakeWebCapability(
        pages={
            SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
            SIMPLE_ARTICLE: page(SIMPLE_ARTICLE, markdown="inline payload"),
        }
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(deque((ExtractedLinks(inline=(inline,)),))),
        FakeYouTubeCapability(),
        web,
        FakeDecryptionFactory(),
    ).discover()

    assert outcome.kind == "success"
    assert tuple(artifact.content for artifact in outcome.artifacts) == (inline,)
    assert web.downloaded == []


async def test_simple_discovery_selects_newest_non_navigation_articles():
    root = "https://articles.test/blog/"
    newest = "https://articles.test/newest"
    middle = "https://articles.test/middle"
    older = "https://articles.test/older"
    newest_download = "https://files.test/newest.txt"
    middle_download = "https://files.test/middle.txt"
    site = SimpleSite(
        name="article-source",
        start_url=root,
        link_pattern=r"https://files\.test/[a-z]+\.txt",
    )
    run_config = config(site).model_copy(
        update={"crawl": CrawlConfig(max_articles=2, concurrency=1)}
    )
    web = FakeWebCapability(
        pages={
            root: page(
                root,
                links=(
                    PageLink(href="/older", text="8月27日 older"),
                    PageLink(href="/category/news", text="8月30日 navigation"),
                    PageLink(href="/middle", text="2026/8/29 middle"),
                    PageLink(href="/newest", text="8月30日 newest"),
                ),
            ),
            newest: page(newest, html=newest_download),
            middle: page(middle, html=middle_download),
        },
        downloads={
            newest_download: subscription("newest.example", "newest"),
            middle_download: subscription("middle.example", "middle"),
        },
    )

    outcome = await SiteProcessor(
        site,
        run_config,
        FakeLinkCapability(),
        FakeYouTubeCapability(),
        web,
        FakeDecryptionFactory(),
    ).discover()

    assert outcome.kind == "success"
    assert web.fetched == [root, newest, middle]
    assert older not in web.fetched
    assert tuple(artifact.source_url for artifact in outcome.artifacts) == (
        newest_download,
        middle_download,
    )


async def test_simple_discovery_uses_dated_markdown_when_links_are_absent():
    root = "https://markdown.test/blog/"
    article = "https://markdown.test/2026/08/29/nodes"
    download = "https://files.test/markdown.txt"
    site = SimpleSite(
        name="markdown-source",
        start_url=root,
        link_pattern=r"https://files\.test/markdown\.txt",
    )
    web = FakeWebCapability(
        pages={
            root: page(root, markdown=f"## [8月29日 nodes]({article})"),
            article: page(article, html=download),
        },
        downloads={download: subscription("markdown.example", "markdown")},
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        FakeYouTubeCapability(),
        web,
        FakeDecryptionFactory(),
    ).discover()

    assert outcome.kind == "success"
    assert web.fetched == [root, article]
    assert tuple(artifact.source_url for artifact in outcome.artifacts) == (download,)


async def test_simple_discovery_retries_failed_downloads(monkeypatch):
    download = "https://files.test/unavailable.txt"
    site = SimpleSite(
        name="retry-source",
        start_url=SIMPLE_ROOT,
        link_pattern=r"https://files\.test/unavailable\.txt",
    )

    @dataclass
    class FailingWeb(FakeWebCapability):
        attempts: int = 0

        async def download_file(self, url: str) -> DownloadOutcome:
            self.attempts += 1
            return DownloadFailure(
                code="http_error",
                url=url,
                diagnostic="upstream unavailable",
            )

    async def no_wait(delay: float) -> None:
        return None

    monkeypatch.setattr("src.site_processor.asyncio.sleep", no_wait)
    web = FailingWeb(
        pages={
            SIMPLE_ROOT: page(SIMPLE_ROOT, links=article_link()),
            SIMPLE_ARTICLE: page(SIMPLE_ARTICLE, html=download),
        }
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        FakeYouTubeCapability(),
        web,
        FakeDecryptionFactory(),
    ).discover()

    assert outcome.kind == "failure"
    assert web.attempts == 3
    assert any("upstream unavailable" in error for error in outcome.errors)


@pytest.mark.parametrize("variant", ("simple", "yt_pwd", "cloud_drive"))
async def test_variant_unavailable_preserves_source_identity(variant):
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    web = FakeWebCapability(pages={})
    if variant == "cloud_drive":
        site = drive_site()
    else:
        site = simple_site() if variant == "simple" else password_site()
        root = SIMPLE_ROOT if variant == "simple" else PASSWORD_ROOT
        web = FakeWebCapability(
            pages={
                root: page(
                    root,
                    success=False,
                    error="unavailable",
                )
            }
        )
    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        web,
        decryption,
    ).discover()

    assert outcome.kind == "failure"
    assert outcome.site_name == site.name
    assert outcome.artifacts == ()
    assert outcome.errors


@pytest.mark.parametrize("variant", ("simple", "yt_pwd", "cloud_drive"))
async def test_variant_empty_preserves_source_identity(variant):
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    web = FakeWebCapability(pages={})
    if variant == "cloud_drive":
        site = drive_site()
        youtube = FakeYouTubeCapability(
            channel_videos=(video_reference(DRIVE_VIDEO, title="undated video"),)
        )
        expected = "no dated videos found"
    else:
        site = simple_site() if variant == "simple" else password_site()
        root = SIMPLE_ROOT if variant == "simple" else PASSWORD_ROOT
        web = FakeWebCapability(pages={root: page(root)})
        expected = "no articles found"

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        web,
        decryption,
    ).discover()

    assert outcome.kind == "failure"
    assert outcome.site_name == site.name
    assert outcome.errors == (expected,)


@pytest.mark.parametrize(
    "variant",
    (
        "simple",
        "yt_pwd",
        "cloud_drive",
    ),
)
async def test_variant_malformed_effect_preserves_diagnosis(variant):
    diagnosis = "malformed external payload"
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    web = FakeWebCapability(pages={})
    if variant == "simple":
        site = simple_site()
        web = FakeWebCapability(
            pages={
                SIMPLE_ROOT: page(
                    SIMPLE_ROOT,
                    success=False,
                    error=diagnosis,
                )
            }
        )
    elif variant == "cloud_drive":
        site = drive_site()
        youtube = FakeYouTubeCapability(
            channel_videos=(video_reference(DRIVE_VIDEO),),
            details={
                DRIVE_VIDEO: youtube_failure(DRIVE_VIDEO, diagnosis),
            },
        )
    else:
        site = password_site()
        web = FakeWebCapability(
            pages={
                PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
                PASSWORD_ARTICLE: page(
                    PASSWORD_ARTICLE,
                    markdown="protected page",
                    html=f'<input type="password"> {PASSWORD_VIDEO} ',
                ),
            }
        )
        youtube = FakeYouTubeCapability(
            details={
                PASSWORD_VIDEO: youtube_failure(PASSWORD_VIDEO, diagnosis),
            }
        )
    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        web,
        decryption,
    ).discover()

    assert outcome.kind == "failure"
    assert outcome.site_name == site.name
    assert any(diagnosis in error for error in outcome.errors)


async def test_password_article_direct_link_bypasses_youtube_decryption_and_llm():
    site = password_site().model_copy(update={"link_pattern": None})
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                html=f'<a href="{PASSWORD_DOWNLOAD}">direct</a>',
            ),
        },
        downloads={
            PASSWORD_DOWNLOAD: subscription("password.example", "direct"),
        },
    )
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    llm = FakeLinkCapability()
    outcome = await SiteProcessor(
        site,
        config(site),
        llm,
        youtube,
        web,
        decryption,
    ).discover()

    assert outcome.kind == "success"
    assert youtube.inspected == []
    assert llm.calls == []
    assert decryption.clients[0].page_attempts == []


async def test_password_page_without_video_uses_bounded_policy():
    site = password_site()
    decrypted_url = "https://files.test/bruteforce.txt"
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                markdown="protected without video",
                html='<input type="password">',
            ),
        },
        downloads={decrypted_url: subscription("password.example", "bounded")},
    )
    decryption = FakeDecryptionFactory(
        page_results={
            "0011": page(PASSWORD_ARTICLE, markdown=decrypted_url),
        }
    )
    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        FakeYouTubeCapability(),
        web,
        decryption,
    ).discover()

    attempted = decryption.clients[0].page_attempts[0][1]
    assert outcome.kind == "success"
    assert attempted[:2] == ("0011", "0022")
    assert len(attempted) == 40
    assert len(attempted) <= site.password_policy.max_candidates


async def test_unprotected_password_article_retains_reason():
    site = password_site()
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                markdown="no resource here",
                html="<p>ordinary article</p>",
            ),
        }
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        FakeYouTubeCapability(),
        web,
        FakeDecryptionFactory(),
    ).discover()

    assert outcome.kind == "failure"
    assert any("not password-protected" in error for error in outcome.errors)


@pytest.mark.parametrize(
    ("description", "expected"),
    (
        ("https://1drv.ms/f/c/fixture", "unsupported onedrive resource"),
        ("description without a resource", "video resource missing"),
    ),
)
async def test_cloud_resource_absence_is_explicit(description, expected):
    site = drive_site()
    youtube = FakeYouTubeCapability(
        channel_videos=(video_reference(DRIVE_VIDEO),),
        details={
            DRIVE_VIDEO: video_details(DRIVE_VIDEO, description=description),
        },
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        FakeWebCapability(pages={}),
        FakeDecryptionFactory(),
        FakeDriveFactory({}),
    ).discover()

    assert outcome.kind == "failure"
    assert any(expected in error for error in outcome.errors)


async def test_cloud_paste_uses_empty_first_and_downloads_bounded_link():
    site = drive_site()
    paste_url = "https://paste.to/?fixture#secret"
    download_url = "https://files.test/paste.txt"
    youtube = FakeYouTubeCapability(
        channel_videos=(video_reference(DRIVE_VIDEO),),
        details={
            DRIVE_VIDEO: video_details(
                DRIVE_VIDEO,
                description=paste_url,
                subtitles="password 1122",
            ),
        },
    )
    decryption = FakeDecryptionFactory(
        paste_results={
            "": page(
                paste_url,
                links=(PageLink(href=download_url),),
            )
        }
    )
    web = FakeWebCapability(
        pages={},
        downloads={download_url: subscription("paste.example", "paste")},
    )
    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        web,
        decryption,
        FakeDriveFactory({}),
    ).discover()

    attempted = decryption.clients[0].paste_attempts[0][1]
    assert outcome.kind == "success"
    assert attempted[:2] == ("", "1122")
    assert web.downloaded == [download_url]


async def test_cloud_google_failure_does_not_silently_switch_to_paste():
    site = drive_site()
    paste_url = "https://paste.to/?fixture#secret"
    description = f"https://drive.google.com/file/d/{DRIVE_ID}/view\n{paste_url}"
    youtube = FakeYouTubeCapability(
        channel_videos=(video_reference(DRIVE_VIDEO),),
        details={
            DRIVE_VIDEO: video_details(DRIVE_VIDEO, description=description),
        },
    )
    decryption = FakeDecryptionFactory()
    drive = FakeDriveFactory(
        {
            DRIVE_ID: DriveFailure(
                code="http_error",
                file_id=DRIVE_ID,
                diagnostic="drive unavailable",
            )
        }
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        FakeWebCapability(pages={}),
        decryption,
        drive,
    ).discover()

    assert outcome.kind == "failure"
    assert any("drive unavailable" in error for error in outcome.errors)
    assert decryption.clients[0].paste_attempts == []


async def test_password_discovery_scans_html_and_markdown_videos_in_stable_order():
    first = "https://youtube.com/watch?v=first123"
    second = "https://youtu.be/second456"
    site = password_site()
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                html=f'<input type="password"> terminal {first}',
                markdown=f"text begins here {first} then {second}",
            ),
        },
        downloads={
            PASSWORD_DOWNLOAD: subscription("password.example", "ordered"),
        },
    )
    youtube = FakeYouTubeCapability(
        details={
            first: youtube_failure(first, "first malformed"),
            second: video_details(second, subtitles="password 1122"),
        }
    )
    decryption = FakeDecryptionFactory(
        page_results={
            "1122": page(PASSWORD_ARTICLE, markdown=PASSWORD_DOWNLOAD),
        }
    )

    outcome = await SiteProcessor(
        site,
        config(site),
        FakeLinkCapability(),
        youtube,
        web,
        decryption,
    ).discover()

    assert outcome.kind == "success"
    assert youtube.inspected == [first, second]


@pytest.mark.parametrize("variant", ("simple", "yt_pwd", "cloud_drive"))
async def test_variant_cancellation_propagates(variant):
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    web = FakeWebCapability(pages={})
    if variant == "cloud_drive":
        site = drive_site()
        youtube = FakeYouTubeCapability(cancel_listing=True)
    else:
        site = simple_site() if variant == "simple" else password_site()
        root = SIMPLE_ROOT if variant == "simple" else PASSWORD_ROOT
        web = FakeWebCapability(pages={}, cancelled_urls={root})

    with pytest.raises(asyncio.CancelledError):
        await SiteProcessor(
            site,
            config(site),
            FakeLinkCapability(),
            youtube,
            web,
            decryption,
        ).discover()


class SuccessfulProbe:
    def __init__(self) -> None:
        self.sites: set[str] = set()

    async def probe(self, nodes):
        self.sites = {
            provenance.site for node in nodes for provenance in node.provenance
        }
        return tuple(
            ProbeEvidence(
                fingerprint=node.fingerprint,
                proxy_name=node.display_name,
                coarse=DelayObservation(
                    endpoint="gstatic",
                    status="success",
                    delay_ms=50,
                ),
                confirm=DelayObservation(
                    endpoint="cloudflare",
                    status="success",
                    delay_ms=60,
                ),
            )
            for node in nodes
        )


class AcceptingValidator:
    def validate_bundle(self, root: Path) -> ConsumerValidation:
        assert (root / "nodes" / "merged.yaml").exists()
        assert (root / "nodes" / "provider.yaml").exists()
        return ConsumerValidation(
            profiles=("nodes/merged.yaml", "nodes/provider.yaml"),
            provider_profiles=("nodes/provider.yaml",),
            provider_names=(),
            group_names=("select",),
        )


@dataclass
class FakeYouTubeFactory:
    client: FakeYouTubeCapability
    calls: list[tuple[str, int]] = field(default_factory=list)

    def __call__(
        self,
        *,
        proxy: str,
        concurrency: int,
    ) -> YouTubeCapability:
        self.calls.append((proxy, concurrency))
        return self.client


@dataclass
class FakeWebFactory:
    client: FakeWebCapability
    calls: int = 0

    def __call__(self) -> FakeWebCapability:
        self.calls += 1
        return self.client


async def test_all_variants_enter_one_quality_and_publication_flow(
    tmp_path,
):
    simple = simple_site()
    password = password_site()
    drive_source = drive_site()
    capabilities = success_capabilities()
    youtube_factory = FakeYouTubeFactory(capabilities.youtube)
    web_factory = FakeWebFactory(capabilities.web)
    scheduler = Scheduler(
        config(simple, password, drive_source, output=tmp_path / "nodes"),
        youtube_factory=youtube_factory,
        web_factory=web_factory,
        decryption_factory=capabilities.decryption,
        drive_factory=capabilities.drive,
    )
    scheduler.llm = capabilities.llm
    probe = SuccessfulProbe()

    receipt = await scheduler.publish_profiles(
        repository_root=tmp_path,
        validator=AcceptingValidator(),
        probe_session=probe,
    )

    assert receipt.status == "accepted"
    assert youtube_factory.calls == [("", 3)]
    assert web_factory.calls == 1
    assert capabilities.decryption.calls == [("", 30.0), ("", 30.0)]
    assert all(client.enter_calls == 1 for client in capabilities.decryption.clients)
    assert all(client.close_calls == 1 for client in capabilities.decryption.clients)
    assert capabilities.drive.calls == [("", 30.0)]
    assert capabilities.drive.clients[0].enter_calls == 1
    assert capabilities.drive.clients[0].close_calls == 1
    assert probe.sites == {simple.name, password.name, drive_source.name}
    assert (tmp_path / "nodes" / "publication-receipt.json").exists()
