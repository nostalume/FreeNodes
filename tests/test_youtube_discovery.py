from typing import Literal

import pytest

from freenodes.config import Source
from freenodes.drive import DriveFailure
from freenodes.web import (
    PageLink,
)
from tests.discovery_support import (
    DRIVE_ID,
    DRIVE_VIDEO,
    PASSWORD_ARTICLE,
    PASSWORD_ROOT,
    PASSWORD_VIDEO,
    SIMPLE_ROOT,
    DiscoveryHarness,
    FakeDecryptionFactory,
    FakeDriveFactory,
    FakeWebCapability,
    FakeYouTubeCapability,
    article_link,
    page,
    password_source,
    subscription,
    video_details,
    video_reference,
    web_source,
    youtube_failure,
    youtube_source,
)

FailureKind = Literal["unavailable", "empty", "malformed"]


def failure_case(
    variant: str, failure: FailureKind
) -> tuple[Source, DiscoveryHarness, str]:
    diagnosis = "malformed external payload"
    if variant == "youtube_resources":
        youtube = FakeYouTubeCapability()
        if failure == "empty":
            youtube.channel_videos = (
                video_reference(DRIVE_VIDEO, title="undated video"),
            )
            return (
                youtube_source(),
                DiscoveryHarness(youtube=youtube),
                "no dated videos",
            )
        if failure == "malformed":
            youtube.channel_videos = (video_reference(DRIVE_VIDEO),)
            youtube.details[DRIVE_VIDEO] = youtube_failure(DRIVE_VIDEO, diagnosis)
            return youtube_source(), DiscoveryHarness(youtube=youtube), diagnosis
        return youtube_source(), DiscoveryHarness(youtube=youtube), "no videos found"

    source = web_source() if variant == "web" else password_source()
    root = SIMPLE_ROOT if variant == "web" else PASSWORD_ROOT
    if failure == "empty":
        return (
            source,
            DiscoveryHarness(web=FakeWebCapability({root: page(root)})),
            ("no articles found"),
        )
    if failure == "unavailable" or variant == "web":
        web = FakeWebCapability(
            {
                root: page(
                    root,
                    success=False,
                    error="unavailable" if failure == "unavailable" else diagnosis,
                )
            }
        )
        return (
            source,
            DiscoveryHarness(web=web),
            ("unavailable" if failure == "unavailable" else diagnosis),
        )

    web = FakeWebCapability(
        {
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                markdown="protected page",
                html=f'<input type="password"> {PASSWORD_VIDEO} ',
            ),
        }
    )
    youtube = FakeYouTubeCapability(
        details={PASSWORD_VIDEO: youtube_failure(PASSWORD_VIDEO, diagnosis)}
    )
    return source, DiscoveryHarness(web=web, youtube=youtube), diagnosis


@pytest.mark.parametrize("variant", ("web", "password_page", "youtube_resources"))
@pytest.mark.parametrize("failure", ("unavailable", "empty", "malformed"))
async def test_variant_failures_preserve_identity_and_diagnosis(variant, failure):
    source, harness, diagnosis = failure_case(variant, failure)

    outcome = await harness.run(source)

    assert outcome.kind == "failure"
    assert outcome.site_name == source.name
    assert outcome.artifacts == ()
    assert any(diagnosis in error for error in outcome.errors)


@pytest.mark.parametrize(
    ("description", "expected"),
    (
        ("https://1drv.ms/f/c/fixture", "unsupported onedrive resource"),
        ("description without a resource", "video resource missing"),
    ),
)
async def test_cloud_resource_absence_is_explicit(description, expected):
    site = youtube_source()
    youtube = FakeYouTubeCapability(
        channel_videos=(video_reference(DRIVE_VIDEO),),
        details={
            DRIVE_VIDEO: video_details(DRIVE_VIDEO, description=description),
        },
    )

    outcome = await DiscoveryHarness(youtube=youtube).run(site)

    assert outcome.kind == "failure"
    assert any(expected in error for error in outcome.errors)


async def test_cloud_paste_uses_empty_first_and_downloads_bounded_link():
    site = youtube_source()
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
    outcome = await DiscoveryHarness(
        youtube=youtube, web=web, decryption=decryption
    ).run(site)

    attempted = decryption.clients[0].paste_attempts[0][1]
    assert outcome.kind == "success"
    assert attempted[:2] == ("", "1122")
    assert web.downloaded == [download_url]


async def test_cloud_google_failure_does_not_silently_switch_to_paste():
    site = youtube_source()
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

    outcome = await DiscoveryHarness(
        youtube=youtube, decryption=decryption, drive=drive
    ).run(site)

    assert outcome.kind == "failure"
    assert any("drive unavailable" in error for error in outcome.errors)
    assert decryption.clients[0].paste_attempts == []
