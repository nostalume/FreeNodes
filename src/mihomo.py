"""Pinned Mihomo acquisition and isolated consumer validation."""

from __future__ import annotations

import asyncio
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
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode, urlsplit

import httpx
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import FrozenModel
from src.nodes import ProbeableNode

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

    def yaml_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, mode="python")


class StandaloneProfile(_ExternalModel):
    proxies: tuple[dict[str, Any], ...] = Field(min_length=1)


class _ProviderInventory(_ExternalModel):
    providers: dict[str, Any]


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


ProbeStatus = Literal["success", "timeout", "api_error", "process_error", "cancelled"]


class ProbeCandidate(FrozenModel):
    fingerprint: str
    proxy_name: str = Field(repr=False)

    @model_validator(mode="after")
    def validate_identity(self) -> "ProbeCandidate":
        if not self.fingerprint or not self.proxy_name:
            raise ValueError("probe candidate fields must not be empty")
        return self


class DelayObservation(FrozenModel):
    endpoint: str
    status: ProbeStatus
    delay_ms: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_measurement(self) -> "DelayObservation":
        if (self.status == "success") != (self.delay_ms is not None):
            raise ValueError("only successful observations carry delay evidence")
        return self


class ProbeEvidence(FrozenModel):
    fingerprint: str
    proxy_name: str = Field(repr=False)
    coarse: DelayObservation
    confirm: DelayObservation | None = None

    @property
    def status(self) -> ProbeStatus:
        if self.coarse.status != "success":
            return self.coarse.status
        if self.confirm is None:
            return "cancelled"
        return self.confirm.status


JsonRequest = Callable[[str, float], Awaitable[dict[str, Any]]]


class MihomoDelayProbe:
    """Collect bounded two-endpoint delay evidence from a Mihomo controller."""

    COARSE_ENDPOINT = "https://www.gstatic.com/generate_204"
    CONFIRM_ENDPOINT = "https://cp.cloudflare.com/generate_204"

    def __init__(
        self,
        *,
        request_json: JsonRequest | None = None,
        timeout_ms: int = 2500,
        concurrency: int = 64,
        deadline: float = 300.0,
        max_candidates: int = 4000,
    ):
        if timeout_ms <= 0 or concurrency <= 0 or deadline <= 0 or max_candidates <= 0:
            raise ValueError("probe limits must be positive")
        self.request_json = request_json or self._request_json
        self.timeout_ms = timeout_ms
        self.concurrency = concurrency
        self.deadline = deadline
        self.max_candidates = max_candidates

    async def _request_json(self, url: str, timeout: float) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            return _JSON_OBJECT.validate_python(response.json())

    async def probe_controller(
        self,
        controller: str,
        candidates: Sequence[ProbeCandidate],
    ) -> tuple[ProbeEvidence, ...]:
        if len(candidates) > self.max_candidates:
            raise ValueError("probe candidate budget exceeded")
        if not candidates:
            return ()

        semaphore = asyncio.Semaphore(self.concurrency)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.deadline
        coarse = await self._probe_wave(
            controller,
            candidates,
            self.COARSE_ENDPOINT,
            semaphore,
            deadline,
        )
        confirm_candidates = [
            candidate
            for candidate in candidates
            if coarse[candidate.fingerprint].status == "success"
        ]
        confirm = await self._probe_wave(
            controller,
            confirm_candidates,
            self.CONFIRM_ENDPOINT,
            semaphore,
            deadline,
        )
        return tuple(
            ProbeEvidence(
                fingerprint=candidate.fingerprint,
                proxy_name=candidate.proxy_name,
                coarse=coarse[candidate.fingerprint],
                confirm=confirm.get(candidate.fingerprint),
            )
            for candidate in candidates
        )

    async def _probe_wave(
        self,
        controller: str,
        candidates: Sequence[ProbeCandidate],
        endpoint: str,
        semaphore: asyncio.Semaphore,
        deadline: float,
    ) -> dict[str, DelayObservation]:
        if not candidates:
            return {}
        tasks = {
            asyncio.create_task(
                self._probe_one(controller, candidate, endpoint, semaphore)
            ): candidate
            for candidate in candidates
        }
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        observations: dict[str, DelayObservation] = {}
        for task, candidate in tasks.items():
            if task in pending or task.cancelled():
                observation = DelayObservation(endpoint=endpoint, status="cancelled")
            else:
                observation = task.result()
            observations[candidate.fingerprint] = observation
        return observations

    async def _probe_one(
        self,
        controller: str,
        candidate: ProbeCandidate,
        endpoint: str,
        semaphore: asyncio.Semaphore,
    ) -> DelayObservation:
        query = urlencode(
            {"url": endpoint, "timeout": str(self.timeout_ms), "expected": "204"}
        )
        name = quote(candidate.proxy_name, safe="")
        url = f"{controller.rstrip('/')}/proxies/{name}/delay?{query}"
        try:
            async with semaphore:
                value = await self.request_json(url, self.timeout_ms / 1000 + 1)
            delay = DelayPayload.model_validate(value)
            return DelayObservation(
                endpoint=endpoint,
                status="success",
                delay_ms=delay.delay,
            )
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, httpx.TimeoutException):
            return DelayObservation(endpoint=endpoint, status="timeout")
        except Exception:
            return DelayObservation(endpoint=endpoint, status="api_error")


ProcessFactory = Callable[..., subprocess.Popen[str]]
ConfigValidator = Callable[[Path, Path], None]


class MihomoProbeSession:
    """Own one isolated Mihomo child for a complete bounded probe run."""

    def __init__(
        self,
        executable: Path,
        *,
        delay_probe: MihomoDelayProbe | None = None,
        process_factory: ProcessFactory = subprocess.Popen,
        validate_config: ConfigValidator | None = None,
        startup_timeout: float = 20.0,
    ):
        self.executable = executable.resolve()
        self.delay_probe = delay_probe or MihomoDelayProbe()
        self.process_factory = process_factory
        self.validate_config = validate_config or self._validate_config
        self.startup_timeout = startup_timeout

    def _validate_config(self, config_path: Path, home_dir: Path) -> None:
        MihomoValidator(self.executable, timeout=self.startup_timeout).validate_config(
            config_path,
            home_dir,
        )

    async def probe(self, nodes: Sequence[ProbeableNode]) -> tuple[ProbeEvidence, ...]:
        candidates = tuple(
            ProbeCandidate(
                fingerprint=node.fingerprint,
                proxy_name=node.display_name,
            )
            for node in nodes
        )
        if not candidates:
            return ()
        if len(candidates) > self.delay_probe.max_candidates:
            raise ValueError("probe candidate budget exceeded")
        if len({item.proxy_name for item in candidates}) != len(candidates):
            raise ValueError("Mihomo probe names must be unique")

        process: subprocess.Popen[str] | None = None
        with tempfile.TemporaryDirectory(prefix="freenodes-mihomo-probe-") as temporary:
            root = Path(temporary).resolve()
            check_home = root / "check-home"
            run_home = root / "run-home"
            check_home.mkdir()
            run_home.mkdir()
            controller_port = _free_port()
            config_path = root / "probe.yaml"
            proxies = []
            for node in nodes:
                proxy = node.proxy.mihomo_payload()
                proxy["name"] = node.display_name
                proxies.append(proxy)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "allow-lan": False,
                        "ipv6": False,
                        "log-level": "silent",
                        "external-controller": f"127.0.0.1:{controller_port}",
                        "proxies": proxies,
                    },
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            try:
                await asyncio.to_thread(self.validate_config, config_path, check_home)
                creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                process = self.process_factory(
                    [
                        str(self.executable),
                        "-d",
                        str(run_home),
                        "-f",
                        str(config_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    creationflags=creation_flags,
                )
                controller = f"http://127.0.0.1:{controller_port}"
                await self._await_ready(process, controller)
                return await self.delay_probe.probe_controller(controller, candidates)
            except asyncio.CancelledError:
                raise
            except Exception:
                return tuple(
                    ProbeEvidence(
                        fingerprint=candidate.fingerprint,
                        proxy_name=candidate.proxy_name,
                        coarse=DelayObservation(
                            endpoint="mihomo-process",
                            status="process_error",
                        ),
                    )
                    for candidate in candidates
                )
            finally:
                if process is not None:
                    await asyncio.to_thread(self._terminate, process)

    async def _await_ready(
        self, process: subprocess.Popen[str], controller: str
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if process.poll() is not None:
                raise MihomoValidationError(
                    "Mihomo probe process exited before readiness"
                )
            try:
                value = await self.delay_probe.request_json(
                    f"{controller}/version", 1.0
                )
                if value.get("version"):
                    return
            except Exception as error:
                last_error = error
            await asyncio.sleep(0.05)
        raise MihomoValidationError(
            "Mihomo probe controller did not become ready"
        ) from last_error

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


def _free_port() -> int:
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
            result = subprocess.run(
                [str(self.executable), "-t", "-d", str(home), "-f", str(config_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        if result.returncode != 0:
            raise MihomoValidationError(
                f"Mihomo configuration rejected: {config_path.name}"
            )

    def validate_bundle(self, output_dir: Path) -> ConsumerValidation:
        output_dir = output_dir.resolve()
        standalone = output_dir / "nodes" / "merged.yaml"
        self.validate_config(standalone)
        provider_files = [
            output_dir / "nodes" / "provider.yaml",
            output_dir / "nodes" / "provider-cdn.yaml",
        ]
        provider_names: set[str] = set()
        group_names: set[str] = set()
        for profile in provider_files:
            providers, groups = self._smoke_provider_profile(output_dir, profile)
            provider_names.update(providers)
            group_names.update(groups)
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

    def _smoke_provider_profile(
        self,
        bundle_dir: Path,
        profile: Path,
    ) -> tuple[set[str], set[str]]:
        handler = partial(_QuietHandler, directory=str(bundle_dir))
        server = _QuietHTTPServer(("127.0.0.1", 0), handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            origin = f"http://127.0.0.1:{server.server_port}"
            config = rewrite_provider_urls(profile, origin)
            return self._boot_provider_config(config, profile.name)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

    def smoke_remote_provider(self, profile: Path) -> tuple[set[str], set[str]]:
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
    ) -> tuple[set[str], set[str]]:
        process: subprocess.Popen[str] | None = None
        with tempfile.TemporaryDirectory(prefix="freenodes-mihomo-smoke-") as temporary:
            temp_path = Path(temporary)
            controller_port = _free_port()
            profile = ProviderProfile.model_validate(config)
            config = profile.yaml_payload()
            config["external-controller"] = f"127.0.0.1:{controller_port}"
            smoke_config = temp_path / profile_name
            smoke_config.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.validate_config(smoke_config, temp_path / "check-home")
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    [
                        str(self.executable),
                        "-d",
                        str(temp_path / "run-home"),
                        "-f",
                        str(smoke_config),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    creationflags=creation_flags,
                )
                providers = self._await_json(controller_port, "/providers/proxies")
                proxies = self._await_json(controller_port, "/proxies")
                provider_map = _ProviderInventory.model_validate(providers).providers
                proxy_map = _ProxyInventory.model_validate(proxies).proxies
                expected_providers = set(profile.proxy_providers)
                if not expected_providers.issubset(provider_map):
                    raise MihomoValidationError("Mihomo did not load every provider")
                expected_groups = {item.name for item in profile.proxy_groups}
                if not expected_groups.issubset(proxy_map):
                    raise MihomoValidationError("Mihomo did not expose every group")
                return expected_providers, expected_groups
            finally:
                if process is not None:
                    MihomoProbeSession._terminate(process)

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
