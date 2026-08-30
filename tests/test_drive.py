"""Bounded Google Drive HTTP and archive admission contracts."""

import asyncio
import io
import stat
import zipfile

import httpx
import pytest

from src.drive import DriveClient, DriveLimits

FILE_ID = "abcdefghijk1"


def archive(*members: tuple[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        for name, content in members:
            package.writestr(name, content)
    return output.getvalue()


def client_for(
    handler,
    *,
    limits: DriveLimits | None = None,
) -> DriveClient:
    return DriveClient(
        transport=httpx.MockTransport(handler),
        limits=limits or DriveLimits(),
    )


async def test_drive_download_admits_subscription_files_without_writing(tmp_path):
    body = archive(
        ("nested/nodes.txt", b"vmess://fixture"),
        ("config.yaml", b"proxies: []"),
        ("image.jpg", b"ignored"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async with client_for(handler) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "files"
    assert tuple(file.name for file in outcome.files) == ("nodes.txt", "config.yaml")
    assert outcome.files[0].text == "vmess://fixture"
    assert list(tmp_path.iterdir()) == []


async def test_drive_confirmation_is_followed_once():
    body = archive(("nodes.txt", b"trojan://fixture@example.test:443"))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                text='<html>virus scan warning <a href="?confirm=t">download</a>',
                headers={"content-type": "text/html"},
            )
        return httpx.Response(200, content=body)

    async with client_for(handler) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "files"
    assert len(requests) == 2
    assert requests[1].url.params["confirm"] == "t"


async def test_drive_http_failure_is_explicit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    async with client_for(handler) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "http_error"
    assert outcome.file_id == FILE_ID


async def test_drive_download_limit_is_enforced_while_reading():
    limits = DriveLimits(
        max_download_bytes=16,
        max_entries=10,
        max_member_bytes=16,
        max_total_bytes=16,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 17)

    async with client_for(handler, limits=limits) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "download_oversize"


async def test_bad_archive_is_explicit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not a zip")

    async with client_for(handler) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "bad_archive"


async def test_archive_entry_limit_is_enforced():
    body = archive(("one.txt", b"one"), ("two.txt", b"two"))
    limits = DriveLimits(
        max_download_bytes=len(body),
        max_entries=1,
        max_member_bytes=10,
        max_total_bytes=10,
    )

    async with client_for(
        lambda request: httpx.Response(200, content=body),
        limits=limits,
    ) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "excessive_entries"


async def test_archive_member_limit_is_enforced():
    body = archive(("nodes.txt", b"12345"))
    limits = DriveLimits(
        max_download_bytes=len(body),
        max_entries=10,
        max_member_bytes=4,
        max_total_bytes=10,
    )

    async with client_for(
        lambda request: httpx.Response(200, content=body),
        limits=limits,
    ) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "member_oversize"


async def test_archive_total_limit_is_enforced():
    body = archive(("one.txt", b"123"), ("two.yaml", b"456"))
    limits = DriveLimits(
        max_download_bytes=len(body),
        max_entries=10,
        max_member_bytes=4,
        max_total_bytes=5,
    )

    async with client_for(
        lambda request: httpx.Response(200, content=body),
        limits=limits,
    ) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "total_oversize"


async def test_duplicate_flattened_names_are_rejected():
    body = archive(("one/nodes.txt", b"one"), ("two/nodes.txt", b"two"))

    async with client_for(lambda request: httpx.Response(200, content=body)) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "duplicate_name"


@pytest.mark.parametrize("name", ("../nodes.txt", "/nodes.txt", "C:/nodes.txt"))
async def test_unsafe_member_paths_are_rejected(name):
    body = archive((name, b"vmess://fixture"))

    async with client_for(lambda request: httpx.Response(200, content=body)) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "unsafe_member"


async def test_symbolic_link_member_is_rejected():
    output = io.BytesIO()
    member = zipfile.ZipInfo("nodes.txt")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(output, "w") as package:
        package.writestr(member, "target")

    async with client_for(
        lambda request: httpx.Response(200, content=output.getvalue())
    ) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "failure"
    assert outcome.code == "unsafe_member"


async def test_archive_without_subscription_files_is_explicit():
    body = archive(("readme.md", b"nothing"))

    async with client_for(lambda request: httpx.Response(200, content=body)) as client:
        outcome = await client.download_archive(FILE_ID)

    assert outcome.kind == "empty"
    assert outcome.code == "no_subscription_files"


async def test_drive_cancellation_propagates():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with client_for(handler) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.download_archive(FILE_ID)
