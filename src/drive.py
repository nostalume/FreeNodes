"""Bounded Google Drive HTTP and archive admission."""

from __future__ import annotations

import io
import re
import stat
import zipfile
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol

import httpx
from pydantic import Field

from src.config import FrozenModel

_DRIVE_DIRECT = "https://drive.google.com/uc"
_SUBSCRIPTION_EXTENSIONS = {".txt", ".yaml", ".yml", ".conf", ".json"}


class DriveLimits(FrozenModel):
    max_download_bytes: int = Field(default=16 * 1024 * 1024, gt=0)
    max_entries: int = Field(default=100, gt=0)
    max_member_bytes: int = Field(default=4 * 1024 * 1024, gt=0)
    max_total_bytes: int = Field(default=16 * 1024 * 1024, gt=0)


class DriveFile(FrozenModel):
    name: str = Field(min_length=1)
    content: bytes = Field(repr=False)
    media_type: Literal["text/plain", "application/yaml"]

    @property
    def text(self) -> str:
        return self.content.decode(encoding="utf-8", errors="replace")


class DriveFiles(FrozenModel):
    kind: Literal["files"] = "files"
    file_id: str
    files: tuple[DriveFile, ...] = Field(min_length=1, strict=False)


class DriveEmpty(FrozenModel):
    kind: Literal["empty"] = "empty"
    code: Literal["no_subscription_files"] = "no_subscription_files"
    file_id: str


DriveFailureCode = Literal[
    "not_open",
    "http_error",
    "download_oversize",
    "confirmation_failed",
    "bad_archive",
    "excessive_entries",
    "unsafe_member",
    "member_oversize",
    "total_oversize",
    "duplicate_name",
]


class DriveFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    code: DriveFailureCode
    file_id: str
    diagnostic: str


DriveOutcome = Annotated[
    DriveFiles | DriveEmpty | DriveFailure,
    Field(discriminator="kind"),
]


class DriveCapability(Protocol):
    async def download_archive(self, file_id: str) -> DriveOutcome: ...


class DriveOwner(DriveCapability, Protocol):
    async def __aenter__(self) -> DriveCapability: ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None: ...

    async def aclose(self) -> None: ...


class DriveFactory(Protocol):
    def __call__(
        self,
        *,
        proxy: str,
        timeout_s: float,
    ) -> DriveOwner: ...


class _Downloaded(FrozenModel):
    kind: Literal["downloaded"] = "downloaded"
    content: bytes = Field(repr=False)
    content_type: str = ""


HttpOutcome = _Downloaded | DriveFailure


def extract_drive_id(url: str) -> str | None:
    match = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    if match:
        return match.group(1)
    return None


def create_drive_client(
    *,
    proxy: str,
    timeout_s: float,
) -> DriveOwner:
    return DriveClient(proxy=proxy, timeout_s=timeout_s)


class DriveClient:
    """Own one HTTP session and admit bounded archive members in memory."""

    __slots__ = (
        "_client",
        "_closed",
        "_limits",
        "_proxy",
        "_timeout_s",
        "_transport",
    )

    def __init__(
        self,
        *,
        proxy: str = "",
        timeout_s: float = 120.0,
        limits: DriveLimits = DriveLimits(),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("Drive timeout must be positive")
        self._proxy = proxy
        self._timeout_s = timeout_s
        self._limits = limits
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    async def __aenter__(self) -> DriveClient:
        if self._closed:
            raise RuntimeError("a closed DriveClient cannot be reopened")
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._timeout_s,
                    connect=min(15.0, self._timeout_s),
                    read=self._timeout_s,
                ),
                follow_redirects=True,
                proxy=self._proxy or None,
                transport=self._transport,
            )
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def download_archive(self, file_id: str) -> DriveOutcome:
        if self._client is None or self._closed:
            return DriveFailure(
                code="not_open",
                file_id=file_id,
                diagnostic="DriveClient must be used as an async context manager",
            )
        first = await self._request(
            file_id,
            params={"export": "download", "id": file_id},
        )
        if first.kind == "failure":
            return first
        downloaded = first
        if self._needs_confirmation(downloaded):
            confirmed = await self._request(
                file_id,
                params={"export": "download", "confirm": "t", "id": file_id},
            )
            if confirmed.kind == "failure":
                return confirmed
            if self._needs_confirmation(confirmed):
                return DriveFailure(
                    code="confirmation_failed",
                    file_id=file_id,
                    diagnostic="Google Drive confirmation did not return a file",
                )
            downloaded = confirmed
        return self._admit_archive(file_id, downloaded.content)

    async def _request(
        self,
        file_id: str,
        *,
        params: dict[str, str],
    ) -> HttpOutcome:
        client = self._client
        if client is None:
            return DriveFailure(
                code="not_open",
                file_id=file_id,
                diagnostic="Drive HTTP session is unavailable",
            )
        try:
            async with client.stream(
                "GET",
                _DRIVE_DIRECT,
                params=params,
            ) as response:
                if response.is_error:
                    return DriveFailure(
                        code="http_error",
                        file_id=file_id,
                        diagnostic=f"Google Drive returned HTTP {response.status_code}",
                    )
                declared = response.headers.get("content-length")
                if declared:
                    try:
                        if int(declared) > self._limits.max_download_bytes:
                            return DriveFailure(
                                code="download_oversize",
                                file_id=file_id,
                                diagnostic=(
                                    "Google Drive response exceeds the download limit"
                                ),
                            )
                    except ValueError:
                        pass
                chunks: list[bytes] = []
                received = 0
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > self._limits.max_download_bytes:
                        return DriveFailure(
                            code="download_oversize",
                            file_id=file_id,
                            diagnostic="Google Drive response exceeds the download limit",
                        )
                    chunks.append(chunk)
                return _Downloaded(
                    content=b"".join(chunks),
                    content_type=response.headers.get("content-type", ""),
                )
        except httpx.HTTPError as error:
            return DriveFailure(
                code="http_error",
                file_id=file_id,
                diagnostic=str(error)[:200] or "Google Drive request failed",
            )

    @staticmethod
    def _needs_confirmation(downloaded: _Downloaded) -> bool:
        if "text/html" not in downloaded.content_type.lower():
            return False
        text = downloaded.content[:1000].decode("utf-8", errors="ignore").lower()
        return any(
            marker in text for marker in ("virus scan", "confirm=", "download_warning")
        )

    def _admit_archive(self, file_id: str, content: bytes) -> DriveOutcome:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as package:
                members = package.infolist()
                if len(members) > self._limits.max_entries:
                    return DriveFailure(
                        code="excessive_entries",
                        file_id=file_id,
                        diagnostic="Drive archive exceeds the entry limit",
                    )
                return self._admit_members(file_id, package, members)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as error:
            return DriveFailure(
                code="bad_archive",
                file_id=file_id,
                diagnostic=str(error)[:200] or "Drive payload is not a valid archive",
            )

    def _admit_members(
        self,
        file_id: str,
        package: zipfile.ZipFile,
        members: list[zipfile.ZipInfo],
    ) -> DriveOutcome:
        admitted: list[DriveFile] = []
        names: set[str] = set()
        total_bytes = 0
        for member in members:
            if not self._safe_member(member):
                return DriveFailure(
                    code="unsafe_member",
                    file_id=file_id,
                    diagnostic=f"unsafe archive member: {member.filename}",
                )
            if member.is_dir():
                continue
            name = PurePosixPath(member.filename.replace("\\", "/")).name
            extension = PurePosixPath(name).suffix.lower()
            if extension not in _SUBSCRIPTION_EXTENSIONS:
                continue
            identity = name.casefold()
            if identity in names:
                return DriveFailure(
                    code="duplicate_name",
                    file_id=file_id,
                    diagnostic=f"duplicate flattened archive name: {name}",
                )
            if member.file_size > self._limits.max_member_bytes:
                return DriveFailure(
                    code="member_oversize",
                    file_id=file_id,
                    diagnostic=f"archive member exceeds its limit: {name}",
                )
            with package.open(member, pwd=None) as source:
                body = source.read(self._limits.max_member_bytes + 1)
            if len(body) > self._limits.max_member_bytes:
                return DriveFailure(
                    code="member_oversize",
                    file_id=file_id,
                    diagnostic=f"archive member exceeds its limit: {name}",
                )
            total_bytes += len(body)
            if total_bytes > self._limits.max_total_bytes:
                return DriveFailure(
                    code="total_oversize",
                    file_id=file_id,
                    diagnostic="admitted archive members exceed the total limit",
                )
            names.add(identity)
            admitted.append(
                DriveFile(
                    name=name,
                    content=body,
                    media_type=(
                        "application/yaml"
                        if extension in {".yaml", ".yml"}
                        else "text/plain"
                    ),
                )
            )
        if not admitted:
            return DriveEmpty(file_id=file_id)
        return DriveFiles(file_id=file_id, files=tuple(admitted))

    @staticmethod
    def _safe_member(member: zipfile.ZipInfo) -> bool:
        raw = member.filename.replace("\\", "/")
        path = PurePosixPath(raw)
        if path.is_absolute() or re.match(r"^[a-zA-Z]:", raw):
            return False
        if any(part == ".." for part in path.parts):
            return False
        mode = member.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        return not file_type or file_type in {stat.S_IFREG, stat.S_IFDIR}
