import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from freenodes.config import (
    AabbPasswordSource,
    AppConfig,
    DiscoveryLimits,
    EmptyPasswordSource,
    PasswordCandidates,
    PasswordPageSource,
    PasswordPolicy,
    Source,
    SubtitlePasswordSource,
    WebSource,
    YouTubeResourceSource,
)
from freenodes.decryption import (
    DecryptedResource,
    DecryptionOutcome,
    DecryptionRejected,
)
from freenodes.discovery import DiscoveryOutcome, DiscoveryRequest, SourceDiscovery
from freenodes.drive import DriveFile, DriveFiles, DriveOutcome
from freenodes.llm import ExtractedLinks
from freenodes.web import (
    DownloadedText,
    DownloadOutcome,
    Page,
    PageLink,
)
from freenodes.youtube import (
    ChannelEmpty,
    ChannelOutcome,
    ChannelVideos,
    DetailsOutcome,
    SubtitlesAvailable,
    SubtitlesEmpty,
    VideoDetails,
    VideoReference,
    YouTubeFailure,
    classify_video_resources,
)


class ManagedFake:
    enter_calls = 0
    close_calls = 0

    async def __aenter__(self):
        self.enter_calls += 1
        return self

    async def __aexit__(self, exception_type, exception, traceback) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self.close_calls += 1


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
NOW = datetime(2026, 8, 29, tzinfo=UTC)

PASSWORD_POLICY = PasswordPolicy(
    sources=(SubtitlePasswordSource(limit=1), AabbPasswordSource(limit=1)),
    max_candidates=2,
)
PASTE_POLICY = PasswordPolicy(
    sources=(EmptyPasswordSource(), SubtitlePasswordSource(limit=1)),
    max_candidates=2,
)


def subscription(host: str, name: str) -> str:
    return f"trojan://secret-{name}@{host}:443#{name}"


@dataclass
class FakeWebCapability:
    pages: dict[str, Page]
    downloads: dict[str, str] = field(default_factory=dict)
    cancelled_urls: set[str] = field(default_factory=set)
    fetched: list[str] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)
    factory_calls: int = 0

    def __call__(self) -> "FakeWebCapability":
        self.factory_calls += 1
        return self

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


@dataclass
class FakeLinkCapability:
    results: deque[ExtractedLinks] = field(default_factory=deque)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def extract_links(
        self,
        markdown: str,
        *,
        source: str,
    ) -> ExtractedLinks:
        self.calls.append((source, markdown))
        return self.results.popleft() if self.results else ExtractedLinks()

    async def generate_pattern(self, links: list[str], *, source: str) -> str | None:
        return None


@dataclass
class FakeYouTubeCapability:
    channel_videos: tuple[VideoReference, ...] = ()
    details: dict[str, DetailsOutcome] = field(default_factory=dict)
    listing_failure: YouTubeFailure | None = None
    cancel_listing: bool = False
    inspected: list[str] = field(default_factory=list)
    factory_calls: list[tuple[str, int]] = field(default_factory=list)

    def __call__(self, *, proxy: str, concurrency: int) -> "FakeYouTubeCapability":
        self.factory_calls.append((proxy, concurrency))
        return self

    async def list_channel_videos(
        self,
        channel_url: str,
        limit: int = 10,
    ) -> ChannelOutcome:
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
class FakeDriveCapability(ManagedFake):
    outcomes: dict[str, DriveOutcome]
    downloaded: list[str] = field(default_factory=list)

    async def download_archive(self, file_id: str) -> DriveOutcome:
        self.downloaded.append(file_id)
        return self.outcomes[file_id]


@dataclass
class FakeDriveFactory:
    outcomes: dict[str, DriveOutcome]
    calls: list[tuple[str, float]] = field(default_factory=list)
    clients: list[FakeDriveCapability] = field(default_factory=list)

    def __call__(self, *, proxy: str, timeout_s: float) -> FakeDriveCapability:
        self.calls.append((proxy, timeout_s))
        client = FakeDriveCapability(dict(self.outcomes))
        self.clients.append(client)
        return client


@dataclass
class FakeDecryptionCapability(ManagedFake):
    page_results: dict[str, Page] = field(default_factory=dict)
    paste_results: dict[str, Page] = field(default_factory=dict)
    page_attempts: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    paste_attempts: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

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

    def __call__(self, *, proxy: str, timeout_s: float) -> FakeDecryptionCapability:
        self.calls.append((proxy, timeout_s))
        client = FakeDecryptionCapability(
            page_results=dict(self.page_results),
            paste_results=dict(self.paste_results),
        )
        self.clients.append(client)
        return client


@dataclass
class DiscoveryHarness:
    llm: FakeLinkCapability = field(default_factory=FakeLinkCapability)
    youtube: FakeYouTubeCapability = field(default_factory=FakeYouTubeCapability)
    web: FakeWebCapability = field(default_factory=lambda: FakeWebCapability(pages={}))
    decryption: FakeDecryptionFactory = field(default_factory=FakeDecryptionFactory)
    drive: FakeDriveFactory = field(default_factory=lambda: FakeDriveFactory({}))

    async def run(
        self, source: Source, *, limits: DiscoveryLimits | None = None
    ) -> DiscoveryOutcome:
        return await SourceDiscovery(
            source,
            request(source, limits=limits),
            self.llm,
            self.youtube,
            self.web,
            self.decryption,
            self.drive,
        ).discover()


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


def config(*sources: Source) -> AppConfig:
    return AppConfig(
        sources=sources,
        discovery=DiscoveryLimits(article_limit=1, source_concurrency=3),
    )


def request(
    *sources: Source, limits: DiscoveryLimits | None = None
) -> DiscoveryRequest:
    return DiscoveryRequest(
        limits=limits or config(*sources).discovery,
        observed_at=NOW,
    )


def web_source(name: str = "web-source") -> WebSource:
    return WebSource(
        name=name,
        start_url=SIMPLE_ROOT,
        resource_pattern=r"https://files\.test/simple\.txt",
    )


def password_source(name: str = "password-source") -> PasswordPageSource:
    return PasswordPageSource(
        name=name,
        start_url=PASSWORD_ROOT,
        resource_pattern=r"https://never\.matches/[^\s]+",
        password_policy=PASSWORD_POLICY,
        paste_policy=PASTE_POLICY,
    )


def youtube_source(name: str = "youtube-source") -> YouTubeResourceSource:
    return YouTubeResourceSource(
        name=name,
        start_url=DRIVE_CHANNEL,
        password_policy=PASTE_POLICY,
    )


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
