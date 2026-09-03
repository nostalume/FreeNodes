import asyncio
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest

from freenodes.youtube import (
    GoogleDriveResource,
    MissingResource,
    PasteResource,
    UnsupportedResource,
    YouTubeClient,
    _extract_video_id,
    classify_video_resources,
    extract_date_from_title,
    extract_external_links,
    extract_password_from_text,
    select_video_resource,
)

OBSERVED_ON = date(2026, 9, 2)


@pytest.mark.parametrize(
    ("title", "expected"),
    (
        ("2026年06月22日免费节点", "2026-06-22"),
        ("【每日更新】290个免费节点（2026/6/22）", "2026-06-22"),
        ("[Daily Update] 270 Free Nodes (06/16/2026)", "2026-06-16"),
        ("Free Nodes 2026-6-22 update", "2026-06-22"),
        ("6月22日更新免费节点", "2026-06-22"),
        ("Free Nodes Update", None),
    ),
)
def test_title_date_grammar(title, expected):
    assert extract_date_from_title(title, OBSERVED_ON) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("密码：6767\n其他内容", ["6767"]),
        ("密码：1212", ["1212"]),
        ("密码可能是5678或3344", ["3344", "5678"]),
        ("今天是2026年6月22日，有203条节点", []),
        ("Password: 5566", ["5566"]),
        ("密码可能是1234或5678或9900", ["9900", "1234", "5678"]),
    ),
)
def test_password_candidates_are_keyword_gated_and_pattern_ordered(text, expected):
    assert extract_password_from_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        (
            "下载地址：https://drive.google.com/file/d/abc123/view",
            ["https://drive.google.com/file/d/abc123/view"],
        ),
        (
            "https://paste.to/?key123#secretFragment456",
            ["https://paste.to/?key123#secretFragment456"],
        ),
        (
            "https://www.youtube.com/redirect?event=video_description&redir_token=abc&q=https%3A%2F%2Fpaste.to%2F%3Fkey123%23fragment456",
            ["https://paste.to/?key123#fragment456"],
        ),
        ("没有链接的文本", []),
    ),
)
def test_external_link_grammar_preserves_fragments_and_resolves_redirects(
    text, expected
):
    assert extract_external_links(text) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    (
        ("https://www.youtube.com/watch?v=abc123def45", "abc123def45"),
        ("https://youtu.be/abc123def45", "abc123def45"),
        ("https://www.youtube.com/embed/abc123def45", "abc123def45"),
        ("https://www.youtube.com/shorts/abc123def45", "abc123def45"),
        ("https://example.com", None),
        ("https://www.youtube.com/watch?v=abc123def45&t=120", "abc123def45"),
    ),
)
def test_video_id_url_grammar(url, expected):
    assert _extract_video_id(url) == expected


VIDEO_URL = "https://www.youtube.com/watch?v=abc123def45"
CHANNEL_URL = "https://www.youtube.com/@fixture/videos"
PROXY = "http://user:secret@proxy.test:8080"


@dataclass
class FakeProcess:
    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int | None = 0
    blocked: bool = False
    ignore_terminate: bool = False
    on_communicate: Callable[[], None] | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    terminated: bool = False
    killed: bool = False
    waited: bool = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        if self.blocked:
            await self.release.wait()
        if self.on_communicate is not None:
            self.on_communicate()
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        if self.returncode is None:
            await self.release.wait()
        return self.returncode or 0


@dataclass
class FakeProcessFactory:
    processes: deque[FakeProcess]
    on_create: Callable[[tuple[str, ...], FakeProcess], None] | None = None
    commands: list[tuple[str, ...]] = field(default_factory=list)
    options: list[dict[str, object]] = field(default_factory=list)

    async def __call__(self, *command: str, **options: object) -> FakeProcess:
        self.commands.append(command)
        self.options.append(options)
        process = self.processes.popleft()
        if self.on_create is not None:
            self.on_create(command, process)
        return process


def process_factory(*processes: FakeProcess) -> FakeProcessFactory:
    return FakeProcessFactory(deque(processes))


async def test_listing_admits_json_lines_and_uses_locked_module_command():
    process = FakeProcess(
        stdout=(
            b'{"id":"abc123def45","url":"abc123def45","title":"one",'
            b'"upload_date":"20260829","channel":"fixture"}\n'
        )
    )
    factory = process_factory(process)
    client = YouTubeClient(proxy=PROXY, process_factory=factory)

    outcome = await client.list_channel_videos(CHANNEL_URL, limit=1)

    assert outcome.kind == "videos"
    assert outcome.videos[0].video_id == "abc123def45"
    assert outcome.videos[0].url == VIDEO_URL
    command = factory.commands[0]
    assert command[:3] == (sys.executable, "-m", "yt_dlp")
    assert command[command.index("--proxy") + 1] == PROXY
    assert command[-2:] == ("--", CHANNEL_URL)
    assert factory.options[0]["stdin"] == asyncio.subprocess.DEVNULL
    assert factory.options[0]["stdout"] == asyncio.subprocess.PIPE
    assert factory.options[0]["stderr"] == asyncio.subprocess.PIPE


async def test_listing_empty_is_not_process_failure():
    client = YouTubeClient(process_factory=process_factory(FakeProcess()))

    outcome = await client.list_channel_videos(CHANNEL_URL, limit=3)

    assert outcome.kind == "empty"
    assert outcome.operation == "list"


@pytest.mark.parametrize(
    "stdout",
    (
        b"not-json\n",
        b'{"id":"abc123def45","url":"abc123def45"}\nnot-json\n',
        b'{"title":"missing identity"}\n',
    ),
)
async def test_listing_rejects_malformed_or_partial_output(stdout):
    client = YouTubeClient(process_factory=process_factory(FakeProcess(stdout=stdout)))

    outcome = await client.list_channel_videos(CHANNEL_URL, limit=3)

    assert outcome.kind == "failure"
    assert outcome.code == "malformed_output"
    assert outcome.operation == "list"


async def test_nonzero_process_failure_redacts_proxy_secret():
    process = FakeProcess(
        stderr=f"proxy refused {PROXY}".encode(),
        returncode=2,
    )
    client = YouTubeClient(
        proxy=PROXY,
        process_factory=process_factory(process),
    )

    outcome = await client.list_channel_videos(CHANNEL_URL)

    assert outcome.kind == "failure"
    assert outcome.code == "process_failed"
    assert PROXY not in outcome.diagnostic
    assert "<proxy>" in outcome.diagnostic


async def test_process_start_failure_is_explicit():
    async def missing(*command: str, **options: object):
        raise FileNotFoundError("missing executable")

    client = YouTubeClient(process_factory=missing)

    outcome = await client.list_channel_videos(CHANNEL_URL)

    assert outcome.kind == "failure"
    assert outcome.code == "unavailable"


async def test_timeout_terminates_and_reaps_child():
    process = FakeProcess(returncode=None, blocked=True)
    client = YouTubeClient(
        timeout_s=0.01,
        process_factory=process_factory(process),
    )

    outcome = await client.list_channel_videos(CHANNEL_URL)

    assert outcome.kind == "failure"
    assert outcome.code == "timeout"
    assert process.terminated is True
    assert process.waited is True


async def test_timeout_kills_child_that_ignores_termination():
    process = FakeProcess(returncode=None, blocked=True, ignore_terminate=True)
    client = YouTubeClient(
        timeout_s=0.01,
        close_timeout_s=0.01,
        process_factory=process_factory(process),
    )

    outcome = await client.list_channel_videos(CHANNEL_URL)

    assert outcome.kind == "failure"
    assert outcome.code == "timeout"
    assert process.terminated is True
    assert process.killed is True
    assert process.waited is True


async def test_cancellation_terminates_and_reaps_before_propagating():
    process = FakeProcess(returncode=None, blocked=True)
    client = YouTubeClient(process_factory=process_factory(process))
    task = asyncio.create_task(client.list_channel_videos(CHANNEL_URL))
    await process.started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated is True
    assert process.waited is True


async def test_semaphore_bounds_child_processes():
    first = FakeProcess(returncode=None, blocked=True)
    second = FakeProcess(returncode=None, blocked=True)
    factory = process_factory(first, second)
    client = YouTubeClient(concurrency=1, process_factory=factory)
    first_task = asyncio.create_task(client.list_channel_videos(CHANNEL_URL))
    await first.started.wait()
    second_task = asyncio.create_task(client.list_channel_videos(CHANNEL_URL))
    await asyncio.sleep(0)

    assert len(factory.commands) == 1
    first.returncode = 0
    first.release.set()
    await second.started.wait()
    assert len(factory.commands) == 2
    second.returncode = 0
    second.release.set()

    assert (await first_task).kind == "empty"
    assert (await second_task).kind == "empty"


async def test_details_admit_metadata_and_request_scoped_subtitles(tmp_path):
    metadata = FakeProcess(
        stdout=(
            b'{"id":"abc123def45","webpage_url":"https://www.youtube.com/'
            b'watch?v=abc123def45","title":"one","description":"resource",'
            b'"upload_date":"20260829","channel":"fixture"}'
        )
    )
    subtitles = FakeProcess()
    subtitle_parent: Path | None = None

    def create_subtitle(command: tuple[str, ...], process: FakeProcess) -> None:
        nonlocal subtitle_parent
        if "--output" not in command:
            return
        template = Path(command[command.index("--output") + 1])
        subtitle_parent = template.parent

        def write_subtitle() -> None:
            (template.parent / "video.zh-Hans.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n密码 1122\n",
                encoding="utf-8",
            )

        process.on_communicate = write_subtitle

    factory = FakeProcessFactory(
        deque((metadata, subtitles)),
        on_create=create_subtitle,
    )
    client = YouTubeClient(
        proxy=PROXY,
        process_factory=factory,
        temporary_root=tmp_path,
    )

    outcome = await client.get_video_details(VIDEO_URL)

    assert outcome.kind == "details"
    assert outcome.video_id == "abc123def45"
    assert outcome.description == "resource"
    assert outcome.subtitles.kind == "available"
    assert outcome.subtitles.text == "密码 1122"
    assert subtitle_parent is not None
    assert subtitle_parent.exists() is False
    assert len(factory.commands) == 2
    assert all(
        command[command.index("--proxy") + 1] == PROXY for command in factory.commands
    )


@pytest.mark.parametrize(
    "stdout",
    (
        b"not-json",
        b'{"id":"different01","title":"wrong identity"}',
        b'{"title":"missing identity"}',
    ),
)
async def test_details_reject_malformed_or_mismatched_output(stdout):
    factory = process_factory(FakeProcess(stdout=stdout))
    client = YouTubeClient(process_factory=factory)

    outcome = await client.get_video_details(VIDEO_URL)

    assert outcome.kind == "failure"
    assert outcome.code == "malformed_output"
    assert len(factory.commands) == 1


async def test_subtitle_failure_remains_explicit_inside_details():
    metadata = FakeProcess(
        stdout=b'{"id":"abc123def45","title":"one","description":"resource"}'
    )
    subtitles = FakeProcess(stderr=b"subtitle unavailable", returncode=1)
    client = YouTubeClient(
        process_factory=process_factory(metadata, subtitles),
    )

    outcome = await client.get_video_details(VIDEO_URL)

    assert outcome.kind == "details"
    assert outcome.subtitles.kind == "failure"
    assert outcome.subtitles.code == "process_failed"
    assert "subtitle unavailable" in outcome.subtitles.diagnostic


async def test_invalid_video_url_does_not_start_process():
    factory = process_factory()
    client = YouTubeClient(process_factory=factory)

    outcome = await client.get_video_details("https://example.test/video")

    assert outcome.kind == "failure"
    assert outcome.code == "invalid_url"
    assert factory.commands == []


DRIVE_ID = "abcdefghijk1"


def test_resource_classifier_preserves_supported_order_and_identity():
    resources = classify_video_resources(
        "\n".join(
            (
                f"https://drive.google.com/file/d/{DRIVE_ID}/view",
                "https://paste.to/?fixture#fragment",
            )
        )
    )

    assert resources == (
        GoogleDriveResource(
            url=f"https://drive.google.com/file/d/{DRIVE_ID}/view",
            file_id=DRIVE_ID,
        ),
        PasteResource(url="https://paste.to/?fixture#fragment"),
    )


def test_resource_classifier_stable_deduplicates_redirected_links():
    direct = f"https://drive.google.com/file/d/{DRIVE_ID}/view"
    redirected = (
        "https://www.youtube.com/redirect?event=video_description"
        f"&q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F{DRIVE_ID}%2Fview"
    )

    resources = classify_video_resources(f"{direct}\n{redirected}")

    assert resources == (GoogleDriveResource(url=direct, file_id=DRIVE_ID),)


def test_resource_classifier_keeps_unsupported_providers_explicit():
    resources = classify_video_resources(
        "\n".join(
            (
                "https://1drv.ms/f/c/abc123",
                "https://tenant.sharepoint.com/file.zip",
            )
        )
    )

    assert resources == (
        UnsupportedResource(
            url="https://1drv.ms/f/c/abc123",
            provider="onedrive",
            reason="unsupported_provider",
        ),
        UnsupportedResource(
            url="https://tenant.sharepoint.com/file.zip",
            provider="sharepoint",
            reason="unsupported_provider",
        ),
    )


def test_resource_classifier_preserves_malformed_supported_links():
    resources = classify_video_resources(
        "https://drive.google.com/file/d/short/view\nhttps://paste.to/?fixture"
    )

    assert resources == (
        UnsupportedResource(
            url="https://drive.google.com/file/d/short/view",
            provider="google_drive",
            reason="invalid_identifier",
        ),
        UnsupportedResource(
            url="https://paste.to/?fixture",
            provider="paste",
            reason="missing_fragment",
        ),
    )


def test_resource_classifier_ignores_unrelated_links():
    assert classify_video_resources("https://example.test/nodes.txt") == ()


def test_resource_selection_prefers_google_then_paste():
    paste = PasteResource(url="https://paste.to/?fixture#fragment")
    drive = GoogleDriveResource(
        url=f"https://drive.google.com/file/d/{DRIVE_ID}/view",
        file_id=DRIVE_ID,
    )

    assert select_video_resource("video", (paste, drive)) == drive
    assert select_video_resource("video", (paste,)) == paste


def test_resource_selection_returns_explicit_missing_value():
    assert select_video_resource("video", ()) == MissingResource(video_url="video")
