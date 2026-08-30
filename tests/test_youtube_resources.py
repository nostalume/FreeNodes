"""Typed resource admission from YouTube video descriptions."""

from src.youtube import (
    GoogleDriveResource,
    MissingResource,
    PasteResource,
    UnsupportedResource,
    classify_video_resources,
    select_video_resource,
)

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
