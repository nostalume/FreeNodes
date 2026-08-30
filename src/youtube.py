"""Typed yt-dlp process ownership and pure YouTube text parsing."""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from collections.abc import Awaitable
from pathlib import Path
from typing import Annotated, Literal, Protocol
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import FrozenModel
from src.drive import extract_drive_id

YouTubeOperation = Literal["list", "details", "subtitles"]
YouTubeFailureCode = Literal[
    "invalid_url",
    "unavailable",
    "timeout",
    "process_failed",
    "malformed_output",
]


class VideoReference(FrozenModel):
    url: str
    video_id: str = Field(min_length=1)
    title: str = ""
    upload_date: str = ""
    channel: str = ""


class YouTubeFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    operation: YouTubeOperation
    code: YouTubeFailureCode
    url: str
    diagnostic: str = ""
    return_code: int | None = Field(default=None, strict=True)


class ChannelVideos(FrozenModel):
    kind: Literal["videos"] = "videos"
    channel_url: str
    videos: tuple[VideoReference, ...] = Field(min_length=1, strict=False)


class ChannelEmpty(FrozenModel):
    kind: Literal["empty"] = "empty"
    operation: Literal["list"] = "list"
    url: str


ChannelOutcome = Annotated[
    ChannelVideos | ChannelEmpty | YouTubeFailure,
    Field(discriminator="kind"),
]


class SubtitlesAvailable(FrozenModel):
    kind: Literal["available"] = "available"
    text: str = Field(min_length=1, repr=False)


class SubtitlesEmpty(FrozenModel):
    kind: Literal["empty"] = "empty"
    operation: Literal["subtitles"] = "subtitles"
    url: str


SubtitleOutcome = Annotated[
    SubtitlesAvailable | SubtitlesEmpty | YouTubeFailure,
    Field(discriminator="kind"),
]


class GoogleDriveResource(FrozenModel):
    kind: Literal["google_drive"] = "google_drive"
    url: str
    file_id: str = Field(min_length=10)


class PasteResource(FrozenModel):
    kind: Literal["paste"] = "paste"
    url: str


class UnsupportedResource(FrozenModel):
    kind: Literal["unsupported"] = "unsupported"
    url: str
    provider: Literal["google_drive", "paste", "onedrive", "sharepoint"]
    reason: Literal[
        "unsupported_provider",
        "invalid_identifier",
        "missing_fragment",
    ]


class MissingResource(FrozenModel):
    kind: Literal["missing"] = "missing"
    video_url: str


VideoResource = Annotated[
    GoogleDriveResource | PasteResource | UnsupportedResource,
    Field(discriminator="kind"),
]

SelectedVideoResource = VideoResource | MissingResource


class VideoDetails(FrozenModel):
    kind: Literal["details"] = "details"
    url: str
    video_id: str = Field(min_length=1)
    title: str = ""
    description: str = ""
    upload_date: str = ""
    channel: str = ""
    subtitles: SubtitleOutcome
    resources: tuple[VideoResource, ...] = Field(strict=False)

    @property
    def subtitles_text(self) -> str:
        if self.subtitles.kind == "available":
            return self.subtitles.text
        return ""


DetailsOutcome = Annotated[
    VideoDetails | YouTubeFailure,
    Field(discriminator="kind"),
]


class YouTubeCapability(Protocol):
    async def list_channel_videos(
        self,
        channel_url: str,
        limit: int = 10,
    ) -> ChannelOutcome: ...

    async def get_video_details(self, video_url: str) -> DetailsOutcome: ...


class ChildProcess(Protocol):
    @property
    def returncode(self) -> int | None: ...

    async def communicate(self) -> tuple[bytes, bytes]: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class ProcessFactory(Protocol):
    def __call__(
        self,
        *command: str,
        stdin: int,
        stdout: int,
        stderr: int,
    ) -> Awaitable[ChildProcess]: ...


async def _create_process(
    *command: str,
    stdin: int,
    stdout: int,
    stderr: int,
) -> ChildProcess:
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )


class _ExternalModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _FlatVideo(_ExternalModel):
    id: str = Field(min_length=1)
    url: str | None = None
    webpage_url: str | None = None
    title: str = ""
    upload_date: str = ""
    channel: str = ""
    uploader: str = ""

    def admit(self) -> VideoReference:
        raw_url = self.webpage_url or self.url or self.id
        url = (
            raw_url
            if raw_url.startswith(("http://", "https://"))
            else f"https://www.youtube.com/watch?v={raw_url}"
        )
        return VideoReference(
            url=url,
            video_id=self.id,
            title=self.title,
            upload_date=self.upload_date,
            channel=self.channel or self.uploader,
        )


class _DetailsPayload(_ExternalModel):
    id: str = Field(min_length=1)
    webpage_url: str | None = None
    title: str = ""
    description: str = ""
    upload_date: str = ""
    channel: str = ""
    uploader: str = ""


class _ProcessOutput(FrozenModel):
    kind: Literal["output"] = "output"
    stdout: bytes = Field(repr=False)
    stderr: bytes = Field(repr=False)
    return_code: int = Field(strict=True)


ProcessOutcome = _ProcessOutput | YouTubeFailure


class YouTubeClient:
    """Run uv-owned yt-dlp children with bounded concurrency and cleanup."""

    __slots__ = (
        "_close_timeout_s",
        "_process_factory",
        "_proxy",
        "_semaphore",
        "_temporary_root",
        "_timeout_s",
    )

    def __init__(
        self,
        *,
        proxy: str = "",
        timeout_s: float = 60.0,
        close_timeout_s: float = 5.0,
        concurrency: int = 2,
        process_factory: ProcessFactory = _create_process,
        temporary_root: Path | None = None,
    ) -> None:
        if timeout_s <= 0 or close_timeout_s <= 0 or concurrency <= 0:
            raise ValueError("YouTube process limits must be positive")
        self._proxy = proxy
        self._timeout_s = timeout_s
        self._close_timeout_s = close_timeout_s
        self._semaphore = asyncio.Semaphore(concurrency)
        self._process_factory = process_factory
        self._temporary_root = temporary_root

    async def list_channel_videos(
        self,
        channel_url: str,
        limit: int = 10,
    ) -> ChannelOutcome:
        if limit <= 0:
            raise ValueError("YouTube listing limit must be positive")
        process = await self._run(
            self._command(
                "--flat-playlist",
                "--dump-json",
                "--playlist-end",
                str(limit),
                "--",
                channel_url,
            ),
            operation="list",
            url=channel_url,
        )
        if process.kind == "failure":
            return process
        lines = tuple(line for line in process.stdout.splitlines() if line.strip())
        if not lines:
            return ChannelEmpty(url=channel_url)
        try:
            videos = tuple(
                _FlatVideo.model_validate_json(line).admit() for line in lines
            )
        except ValidationError as error:
            return self._malformed("list", channel_url, error)
        return ChannelVideos(channel_url=channel_url, videos=videos)

    async def get_video_details(self, video_url: str) -> DetailsOutcome:
        expected_id = _extract_video_id(video_url)
        if expected_id is None:
            return YouTubeFailure(
                operation="details",
                code="invalid_url",
                url=video_url,
                diagnostic="invalid YouTube video URL",
            )
        process = await self._run(
            self._command(
                "--dump-single-json",
                "--skip-download",
                "--no-playlist",
                "--",
                video_url,
            ),
            operation="details",
            url=video_url,
        )
        if process.kind == "failure":
            return process
        try:
            payload = _DetailsPayload.model_validate_json(process.stdout)
        except ValidationError as error:
            return self._malformed("details", video_url, error)
        if payload.id != expected_id:
            return YouTubeFailure(
                operation="details",
                code="malformed_output",
                url=video_url,
                diagnostic="yt-dlp returned a different video identity",
            )
        subtitles = await self._download_subtitles(video_url)
        return VideoDetails(
            url=payload.webpage_url or video_url,
            video_id=payload.id,
            title=payload.title,
            description=payload.description,
            upload_date=payload.upload_date,
            channel=payload.channel or payload.uploader,
            subtitles=subtitles,
            resources=classify_video_resources(payload.description),
        )

    def _command(self, *arguments: str) -> tuple[str, ...]:
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--no-cache-dir",
        ]
        if self._proxy:
            command.extend(("--proxy", self._proxy))
        command.extend(arguments)
        return tuple(command)

    async def _run(
        self,
        command: tuple[str, ...],
        *,
        operation: YouTubeOperation,
        url: str,
    ) -> ProcessOutcome:
        async with self._semaphore:
            try:
                process = await self._process_factory(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as error:
                return YouTubeFailure(
                    operation=operation,
                    code="unavailable",
                    url=url,
                    diagnostic=self._diagnostic(str(error)),
                )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout_s,
                )
            except asyncio.CancelledError:
                await self._stop(process)
                raise
            except TimeoutError:
                await self._stop(process)
                return YouTubeFailure(
                    operation=operation,
                    code="timeout",
                    url=url,
                    diagnostic=f"yt-dlp {operation} timed out",
                )
            except Exception as error:
                await self._stop(process)
                return YouTubeFailure(
                    operation=operation,
                    code="process_failed",
                    url=url,
                    diagnostic=self._diagnostic(str(error)),
                )

            return_code = process.returncode
            if return_code is None:
                await self._stop(process)
                return YouTubeFailure(
                    operation=operation,
                    code="process_failed",
                    url=url,
                    diagnostic="yt-dlp exited without a return code",
                )
            if return_code != 0:
                return YouTubeFailure(
                    operation=operation,
                    code="process_failed",
                    url=url,
                    diagnostic=self._diagnostic(
                        stderr.decode(encoding="utf-8", errors="replace")
                    ),
                    return_code=return_code,
                )
            return _ProcessOutput(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
            )

    async def _stop(self, process: ChildProcess) -> None:
        if process.returncode is not None:
            await process.wait()
            return
        try:
            process.terminate()
        except ProcessLookupError:
            await process.wait()
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self._close_timeout_s)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=self._close_timeout_s)

    async def _download_subtitles(self, video_url: str) -> SubtitleOutcome:
        with tempfile.TemporaryDirectory(
            prefix="freenodes-youtube-",
            dir=self._temporary_root,
        ) as temporary:
            root = Path(temporary)
            process = await self._run(
                self._command(
                    "--write-subs",
                    "--write-auto-subs",
                    "--sub-langs",
                    "zh-Hans,zh,zh-CN,en",
                    "--sub-format",
                    "srt/vtt/best",
                    "--skip-download",
                    "--output",
                    str(root / "video.%(ext)s"),
                    "--",
                    video_url,
                ),
                operation="subtitles",
                url=video_url,
            )
            if process.kind == "failure":
                return process
            for path in sorted((*root.glob("*.srt"), *root.glob("*.vtt"))):
                text = _parse_subtitle(path)
                if text:
                    return SubtitlesAvailable(text=text)
        return SubtitlesEmpty(url=video_url)

    def _malformed(
        self,
        operation: YouTubeOperation,
        url: str,
        error: ValidationError,
    ) -> YouTubeFailure:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "payload"
        return YouTubeFailure(
            operation=operation,
            code="malformed_output",
            url=url,
            diagnostic=f"invalid yt-dlp {operation} output at {location}",
        )

    def _diagnostic(self, message: str) -> str:
        value = message.strip() or "yt-dlp failed without diagnostics"
        if self._proxy:
            value = value.replace(self._proxy, "<proxy>")
        return value[:500]


def extract_password_from_text(text: str) -> list[str]:
    """Extract keyword-adjacent four-digit candidates in pattern priority."""
    candidates: list[str] = []
    for line in text.splitlines():
        if "密码" not in line and "password" not in line.lower():
            continue
        for match in re.finditer(r"\d{4}", line):
            password = match.group()
            if password not in candidates:
                candidates.append(password)

    def priority(password: str) -> int:
        a, b, c, d = password
        if a == b and c == d:
            return 0
        if a == c and b == d:
            return 1
        return 2

    candidates.sort(key=priority)
    return candidates


def extract_external_links(description: str) -> list[str]:
    """Extract external links while preserving fragments and redirects."""
    links = re.findall(r'https?://[^\s<>"\')\]]+', description)
    resolved: list[str] = []
    for link in links:
        if "youtube.com/redirect" in link:
            match = re.search(r"[&?]q=([^&]+)", link)
            if match:
                resolved.append(unquote(match.group(1)))
                continue
        resolved.append(link)
    return resolved


def classify_video_resources(description: str) -> tuple[VideoResource, ...]:
    resources: list[VideoResource] = []
    seen: set[str] = set()
    for url in extract_external_links(description):
        identity = url.casefold()
        if identity in seen:
            continue
        lowered = url.lower()
        resource: VideoResource | None = None
        if "drive.google.com" in lowered:
            file_id = extract_drive_id(url)
            resource = (
                GoogleDriveResource(url=url, file_id=file_id)
                if file_id
                else UnsupportedResource(
                    url=url,
                    provider="google_drive",
                    reason="invalid_identifier",
                )
            )
        elif "1drv.ms" in lowered or "onedrive" in lowered:
            resource = UnsupportedResource(
                url=url,
                provider="onedrive",
                reason="unsupported_provider",
            )
        elif "sharepoint" in lowered:
            resource = UnsupportedResource(
                url=url,
                provider="sharepoint",
                reason="unsupported_provider",
            )
        elif any(
            provider in lowered
            for provider in ("paste.to", "privatebin", "hastebin", "dpaste")
        ):
            resource = (
                PasteResource(url=url)
                if "#" in url
                else UnsupportedResource(
                    url=url,
                    provider="paste",
                    reason="missing_fragment",
                )
            )
        if resource is not None:
            seen.add(identity)
            resources.append(resource)
    return tuple(resources)


def select_video_resource(
    video_url: str,
    resources: tuple[VideoResource, ...],
) -> SelectedVideoResource:
    for resource in resources:
        if resource.kind == "google_drive":
            return resource
    for resource in resources:
        if resource.kind == "paste":
            return resource
    for resource in resources:
        if resource.kind == "unsupported":
            return resource
    return MissingResource(video_url=video_url)


def extract_date_from_title(title: str) -> str | None:
    """Extract an ISO date from known Chinese and English title formats."""
    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", title)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", title)
    if match:
        return f"{match.group(3)}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"

    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    match = re.search(r"(\d{1,2})月(\d{1,2})日", title)
    if match:
        from datetime import date

        today = date.today()
        parsed = date(today.year, int(match.group(1)), int(match.group(2)))
        if (parsed - today).days > 30:
            parsed = parsed.replace(year=today.year - 1)
        return parsed.isoformat()

    return None


def _extract_video_id(url: str) -> str | None:
    """Extract an eleven-character video ID from known YouTube URL forms."""
    patterns = (
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _parse_subtitle(path: Path) -> str:
    """Parse SRT or WebVTT cues into stable plain text."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line == "WEBVTT"
            or "-->" in line
            or line.isdigit()
            or line.startswith(("Kind:", "Language:"))
        ):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line and line not in lines[-1:]:
            lines.append(line)
    return "\n".join(lines)


_parse_srt = _parse_subtitle
