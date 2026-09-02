"""Pinned Mihomo acquisition and isolated consumer validation."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import FrozenModel

LOCK_PATH = Path(__file__).resolve().parent.parent / "mihomo-lock.json"


class _ExternalModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class DelayPayload(_ExternalModel):
    delay: int = Field(ge=0, strict=True)


class HttpProvider(_ExternalModel):
    type: Literal["http"]
    url: str


class ProxyGroup(_ExternalModel):
    name: str


class ProviderProfile(_ExternalModel):
    proxy_providers: dict[str, HttpProvider] = Field(alias="proxy-providers")
    proxy_groups: tuple[ProxyGroup, ...] = Field(default=(), alias="proxy-groups")
    rules: tuple[str, ...] = ()

    def yaml_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, mode="python")

    def smoke_payload(self) -> dict[str, Any]:
        return {**self.yaml_payload(), "rules": self.rules[-1:]}


class StandaloneProfile(_ExternalModel):
    proxies: tuple[dict[str, Any], ...] = Field(min_length=1)
    rules: tuple[str, ...] = ()

    def smoke_payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(by_alias=True, exclude_none=True, mode="python"),
            "rules": self.rules[-1:],
        }


class ProviderProxy(_ExternalModel):
    name: str


class ProviderRecord(_ExternalModel):
    vehicle_type: str = Field(alias="vehicleType")
    proxies: tuple[ProviderProxy, ...] = ()


class _ProviderInventory(_ExternalModel):
    providers: dict[str, ProviderRecord]


class _ProxyInventory(_ExternalModel):
    proxies: dict[str, Any]


_JSON_OBJECT = TypeAdapter(dict[str, Any])


class MihomoAcquisitionError(RuntimeError):
    pass


class MihomoValidationError(RuntimeError):
    pass


class LockedAsset(FrozenModel):
    name: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("invalid SHA-256 digest")
        return value


class _ReleaseLock(FrozenModel):
    version: str
    source: str
    assets: dict[str, LockedAsset]


class PlatformAsset(FrozenModel):
    platform: str
    name: str
    sha256: str


class ReleaseAsset(PlatformAsset):
    url: str


class PinnedRelease(FrozenModel):
    version: str
    source: str
    assets: tuple[PlatformAsset, ...] = Field(strict=False)

    @classmethod
    def load(cls, path: Path = LOCK_PATH) -> "PinnedRelease":
        try:
            locked = _ReleaseLock.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise MihomoAcquisitionError("invalid Mihomo release lock") from error
        return cls(
            version=locked.version,
            source=locked.source,
            assets=tuple(
                PlatformAsset(platform=platform_name, **asset.model_dump())
                for platform_name, asset in sorted(locked.assets.items())
            ),
        )

    def resolve(
        self, system: str | None = None, machine: str | None = None
    ) -> ReleaseAsset:
        os_name = (system or platform.system()).casefold()
        architecture = (machine or platform.machine()).casefold()
        system_key = {"windows": "windows", "linux": "linux", "darwin": "darwin"}.get(
            os_name
        )
        arch_key = {
            "amd64": "amd64",
            "x86_64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(architecture)
        key = f"{system_key}-{arch_key}"
        locked = next((asset for asset in self.assets if asset.platform == key), None)
        if system_key is None or arch_key is None or locked is None:
            raise MihomoAcquisitionError(
                f"unsupported Mihomo platform: {os_name}/{architecture}"
            )
        return ReleaseAsset(
            platform=locked.platform,
            name=locked.name,
            sha256=locked.sha256,
            url=(
                "https://github.com/MetaCubeX/mihomo/releases/download/"
                f"{self.version}/{locked.name}"
            ),
        )


class AcquiredMihomo(FrozenModel):
    executable: Path
    version: str
    executable_sha256: str
    asset_sha256: str


class ConsumerValidation(FrozenModel):
    profiles: tuple[str, ...] = Field(strict=False)
    provider_profiles: tuple[str, ...] = Field(strict=False)
    provider_names: tuple[str, ...] = Field(strict=False)
    group_names: tuple[str, ...] = Field(strict=False)


class ProviderLoadReceipt(FrozenModel):
    counts: tuple[tuple[str, PositiveInt], ...] = Field(strict=False)
    total_nodes: int = Field(gt=0, strict=True)
    groups: tuple[str, ...] = Field(strict=False)

    @model_validator(mode="after")
    def reconcile_total(self) -> "ProviderLoadReceipt":
        if sum(count for _name, count in self.counts) != self.total_nodes:
            raise ValueError("provider counts do not reconcile")
        return self


def verify_sha256(path: Path, expected: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.casefold() != expected.casefold():
        raise MihomoAcquisitionError("Mihomo release asset checksum mismatch")
    return digest


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix.casefold() == ".zip":
        with zipfile.ZipFile(archive) as package:
            members = [
                item
                for item in package.infolist()
                if not item.is_dir()
                and Path(item.filename).name.casefold().startswith("mihomo")
                and Path(item.filename).suffix.casefold() == ".exe"
            ]
            if len(members) != 1:
                raise MihomoAcquisitionError("unexpected Mihomo zip layout")
            with package.open(members[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    elif archive.suffix.casefold() == ".gz":
        with gzip.open(archive, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        raise MihomoAcquisitionError(f"unsupported Mihomo archive: {archive.suffix}")


def acquire_pinned_mihomo(cache_dir: Path) -> AcquiredMihomo:
    """Download, verify, and cache exactly the release selected by the lock."""
    release = PinnedRelease.load()
    asset = release.resolve()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    executable = cache_dir / (
        "mihomo.exe" if platform.system() == "Windows" else "mihomo"
    )
    receipt_path = cache_dir / "acquisition.json"

    if executable.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        if (
            receipt.get("version") == release.version
            and receipt.get("asset_sha256") == asset.sha256
            and receipt.get("executable_sha256") == executable_digest
        ):
            return AcquiredMihomo(
                executable=executable,
                version=release.version,
                executable_sha256=executable_digest,
                asset_sha256=asset.sha256,
            )

    with tempfile.TemporaryDirectory(prefix="freenodes-mihomo-") as temp_dir:
        archive = Path(temp_dir) / asset.name
        request = urllib.request.Request(
            asset.url,
            headers={"User-Agent": "FreeNodes-Profile-Validation"},
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        except OSError as error:
            raise MihomoAcquisitionError(
                "failed to download pinned Mihomo release"
            ) from error
        verify_sha256(archive, asset.sha256)
        extracted = Path(temp_dir) / executable.name
        _extract_archive(archive, extracted)
        executable_digest = hashlib.sha256(extracted.read_bytes()).hexdigest()
        shutil.copy2(extracted, executable)

    if os.name != "nt":
        executable.chmod(0o755)
    receipt_path.write_text(
        json.dumps(
            {
                "version": release.version,
                "asset": asset.name,
                "asset_sha256": asset.sha256,
                "executable_sha256": executable_digest,
                "source": release.source,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return AcquiredMihomo(
        executable=executable,
        version=release.version,
        executable_sha256=executable_digest,
        asset_sha256=asset.sha256,
    )


def rewrite_provider_urls(config_path: Path, origin: str) -> dict[str, Any]:
    """Point staged HTTP providers at a loopback server for consumer smoke."""
    try:
        profile = ProviderProfile.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
    except Exception as error:
        raise MihomoValidationError("provider profile is invalid") from error
    providers: dict[str, HttpProvider] = {}
    for name, provider in profile.proxy_providers.items():
        source_path = urlsplit(provider.url).path
        filename = Path(source_path).name
        if not filename or filename in {".", ".."}:
            raise MihomoValidationError("provider URL has no safe filename")
        providers[name] = provider.model_copy(
            update={"url": f"{origin.rstrip('/')}/nodes/{filename}"}
        )
    return profile.model_copy(update={"proxy_providers": providers}).yaml_payload()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class _QuietHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        return


def loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class MihomoValidator:
    """Validate rendered profiles with one exact, isolated Mihomo process."""

    def __init__(self, executable: Path, *, timeout: float = 20.0):
        self.executable = executable.resolve()
        self.timeout = timeout

    def validate_config(self, config_path: Path, home_dir: Path | None = None) -> None:
        config_path = config_path.resolve()
        with tempfile.TemporaryDirectory(
            prefix="freenodes-mihomo-home-"
        ) as temporary_home:
            home = (home_dir or Path(temporary_home)).resolve()
            home.mkdir(parents=True, exist_ok=True)
            try:
                result = subprocess.run(
                    [
                        str(self.executable),
                        "-t",
                        "-d",
                        str(home),
                        "-f",
                        str(config_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise MihomoValidationError(
                    "Mihomo configuration check timed out"
                ) from error
        if result.returncode != 0:
            raise MihomoValidationError(
                f"Mihomo configuration rejected: {config_path.name}"
            )

    def validate_bundle(self, output_dir: Path) -> ConsumerValidation:
        output_dir = output_dir.resolve()
        standalone = output_dir / "nodes" / "merged.yaml"
        provider_files = [
            output_dir / "nodes" / "provider.yaml",
            output_dir / "nodes" / "provider-cdn.yaml",
        ]
        provider_names: set[str] = set()
        group_names: set[str] = set()
        with tempfile.TemporaryDirectory(
            prefix="freenodes-mihomo-bundle-"
        ) as temporary:
            home = Path(temporary)
            self.validate_standalone(standalone, home)
            for profile in provider_files:
                loaded = self._smoke_provider_profile(output_dir, profile, home)
                provider_names.update(name for name, _count in loaded.counts)
                group_names.update(loaded.groups)
        return ConsumerValidation(
            profiles=(
                "nodes/merged.yaml",
                "nodes/provider.yaml",
                "nodes/provider-cdn.yaml",
            ),
            provider_profiles=("nodes/provider.yaml", "nodes/provider-cdn.yaml"),
            provider_names=tuple(sorted(provider_names)),
            group_names=tuple(sorted(group_names)),
        )

    def validate_standalone(self, profile: Path, home: Path | None = None) -> None:
        try:
            admitted = StandaloneProfile.model_validate(
                yaml.safe_load(profile.read_text(encoding="utf-8"))
            )
        except Exception as error:
            raise MihomoValidationError("standalone profile is invalid") from error
        with tempfile.TemporaryDirectory(prefix="freenodes-standalone-") as temporary:
            consumer_home = home or Path(temporary)
            smoke = consumer_home / "standalone-smoke.yaml"
            smoke.write_text(yaml.safe_dump(admitted.smoke_payload()), encoding="utf-8")
            self.validate_config(smoke, consumer_home)

    def _smoke_provider_profile(
        self,
        bundle_dir: Path,
        profile: Path,
        home: Path,
    ) -> ProviderLoadReceipt:
        handler = partial(_QuietHandler, directory=str(bundle_dir))
        server = _QuietHTTPServer(("127.0.0.1", 0), handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            origin = f"http://127.0.0.1:{server.server_port}"
            config = rewrite_provider_urls(profile, origin)
            return self._boot_provider_config(config, profile.name, home)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

    def smoke_remote_provider(self, profile: Path) -> ProviderLoadReceipt:
        """Boot a provider profile without rewriting its remote nested URLs."""
        try:
            config = ProviderProfile.model_validate(
                yaml.safe_load(profile.read_text(encoding="utf-8"))
            )
        except Exception as error:
            raise MihomoValidationError("remote profile is invalid") from error
        if not config.proxy_providers:
            raise MihomoValidationError("remote profile has no proxy-providers")
        return self._boot_provider_config(config.yaml_payload(), profile.name)

    def _boot_provider_config(
        self,
        config: dict[str, Any],
        profile_name: str,
        home: Path | None = None,
    ) -> ProviderLoadReceipt:
        process: subprocess.Popen[str] | None = None
        with tempfile.TemporaryDirectory(prefix="freenodes-mihomo-smoke-") as temporary:
            temp_path = Path(temporary)
            controller_port = loopback_port()
            profile = ProviderProfile.model_validate(config)
            config = profile.smoke_payload()
            config["external-controller"] = f"127.0.0.1:{controller_port}"
            smoke_config = temp_path / profile_name
            smoke_config.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            consumer_home = home or temp_path / "home"
            self.validate_config(smoke_config, consumer_home)
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    [
                        str(self.executable),
                        "-d",
                        str(consumer_home),
                        "-f",
                        str(smoke_config),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    creationflags=creation_flags,
                )
                proxies = self._await_json(controller_port, "/proxies")
                proxy_map = _ProxyInventory.model_validate(proxies).proxies
                expected_providers = set(profile.proxy_providers)
                expected_groups = {item.name for item in profile.proxy_groups}
                if not expected_groups.issubset(proxy_map):
                    raise MihomoValidationError("Mihomo did not expose every group")
                return self._await_provider_load(
                    controller_port, expected_providers, expected_groups
                )
            finally:
                if process is not None:
                    self._terminate(process)

    def _await_provider_load(
        self,
        port: int,
        expected: set[str],
        groups: set[str],
    ) -> ProviderLoadReceipt:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                payload = self._await_json(port, "/providers/proxies")
                providers = _ProviderInventory.model_validate(payload).providers
            except ValidationError as error:
                raise MihomoValidationError(
                    "Mihomo provider inventory is invalid"
                ) from error
            missing = expected - providers.keys()
            empty = {name for name in expected - missing if not providers[name].proxies}
            invalid = {
                name
                for name in expected - missing
                if providers[name].vehicle_type != "HTTP"
            }
            if not (missing or empty or invalid):
                counts = tuple(
                    (name, len(providers[name].proxies)) for name in sorted(expected)
                )
                return ProviderLoadReceipt(
                    counts=counts,
                    total_nodes=sum(count for _name, count in counts),
                    groups=sorted(groups),
                )
            if time.monotonic() >= deadline:
                details = f"missing={sorted(missing)}, empty={sorted(empty)}, invalid={sorted(invalid)}"
                raise MihomoValidationError(f"Mihomo provider load failed: {details}")
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _await_json(self, port: int, path: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        url = f"http://127.0.0.1:{port}{path}"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(url, timeout=1) as response:
                    value = json.loads(response.read().decode("utf-8"))
                return _JSON_OBJECT.validate_python(value)
            except Exception as error:
                last_error = error
                time.sleep(0.1)
        raise MihomoValidationError(
            "Mihomo controller did not become ready"
        ) from last_error
