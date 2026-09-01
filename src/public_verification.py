"""Verify remotely published URLs as their declared client operations."""

from __future__ import annotations

import asyncio
import base64
import binascii
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import AwareDatetime, TypeAdapter, model_validator

from src.config import FrozenModel
from src.mihomo import MihomoValidator, ProviderProfile, StandaloneProfile
from src.profiles import PublicEntryRegistry
from src.quality_manifest import admit_quality_manifest_json


class PublicVerificationError(RuntimeError):
    pass


FetchBytes = Callable[[str], Awaitable[bytes]]
ContentCheck = Callable[[bytes], None]


class PublicVerificationReceipt(FrozenModel):
    direct: Literal["current"]
    cdn: Literal["current", "lagging", "degraded"]
    direct_generation: str
    cdn_generation: str | None

    @model_validator(mode="after")
    def validate_generations(self) -> "PublicVerificationReceipt":
        TypeAdapter(AwareDatetime).validate_python(self.direct_generation)
        if self.cdn == "degraded":
            if self.cdn_generation is not None:
                raise ValueError("degraded CDN cannot claim a generation")
            return self
        if self.cdn_generation is None:
            raise ValueError("verified CDN requires a generation")
        TypeAdapter(AwareDatetime).validate_python(self.cdn_generation)
        if (self.cdn == "current") != (self.cdn_generation == self.direct_generation):
            raise ValueError("CDN state and generation disagree")
        return self


class PublicEntryVerifier:
    def __init__(
        self,
        registry: PublicEntryRegistry,
        *,
        fetch: FetchBytes,
        validate_standalone: ContentCheck,
        smoke_provider: ContentCheck,
    ):
        self.registry = registry
        self.fetch = fetch
        self.validate_standalone = validate_standalone
        self.smoke_provider = smoke_provider

    async def verify(self) -> PublicVerificationReceipt:
        try:
            direct_generation = await self._verify_channel(cdn=False)
        except Exception as error:
            raise PublicVerificationError(
                "direct public entries failed verification"
            ) from error
        try:
            cdn_generation = await self._verify_channel(cdn=True)
            cdn_status = "current" if cdn_generation == direct_generation else "lagging"
        except Exception:
            cdn_generation = None
            cdn_status = "degraded"
        return PublicVerificationReceipt(
            direct="current",
            cdn=cdn_status,
            direct_generation=direct_generation,
            cdn_generation=cdn_generation,
        )

    async def _verify_channel(self, *, cdn: bool) -> str:
        urls = {
            "encoded": self.registry.v2ray.for_channel(cdn=cdn),
            "plain": self.registry.legacy.for_channel(cdn=cdn),
            "standalone": self.registry.clash.for_channel(cdn=cdn),
            "provider": self.registry.provider.for_channel(cdn=cdn),
            "quality": self.registry.quality.for_channel(cdn=cdn),
        }
        content = {name: await self.fetch(url) for name, url in urls.items()}
        if any(not body for body in content.values()):
            raise ValueError("empty public entry")
        try:
            decoded = base64.b64decode(content["encoded"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("invalid V2Ray base64") from error
        if decoded != content["plain"] or not decoded.strip():
            raise ValueError("V2Ray base64 and plain URI entries differ")

        StandaloneProfile.model_validate(yaml.safe_load(content["standalone"]))
        provider = ProviderProfile.model_validate(yaml.safe_load(content["provider"]))
        if not provider.proxy_providers:
            raise ValueError("provider Clash profile has no providers")
        expected_host = "cdn.jsdelivr.net" if cdn else "raw.githubusercontent.com"
        nested_hosts = {
            urlsplit(item.url).hostname for item in provider.proxy_providers.values()
        }
        if nested_hosts != {expected_host}:
            raise ValueError("provider profile mixes publication channels")

        manifest = admit_quality_manifest_json(content["quality"])

        self.validate_standalone(content["standalone"])
        self.smoke_provider(content["provider"])
        return manifest.generated_at


async def verify_remote_entries(
    registry: PublicEntryRegistry,
    executable: Path,
    *,
    attempts: int = 3,
    retry_delay: float = 10.0,
) -> PublicVerificationReceipt:
    """Fetch and consume direct/CDN entries with bounded propagation retries."""
    if attempts <= 0 or retry_delay < 0:
        raise ValueError("verification retry limits are invalid")
    validator = MihomoValidator(executable, timeout=60)

    def validate_standalone(content: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="freenodes-remote-clash-") as temporary:
            profile = Path(temporary) / "standalone.yaml"
            profile.write_bytes(content)
            validator.validate_config(profile)

    def smoke_provider(content: bytes) -> None:
        with tempfile.TemporaryDirectory(
            prefix="freenodes-remote-provider-"
        ) as temporary:
            profile = Path(temporary) / "provider.yaml"
            profile.write_bytes(content)
            validator.smoke_remote_provider(profile)

    last_error: PublicVerificationError | None = None
    last_receipt: PublicVerificationReceipt | None = None
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:

        async def fetch(url: str) -> bytes:
            response = await client.get(url, headers={"Cache-Control": "no-cache"})
            response.raise_for_status()
            return response.content

        verifier = PublicEntryVerifier(
            registry,
            fetch=fetch,
            validate_standalone=validate_standalone,
            smoke_provider=smoke_provider,
        )
        for attempt in range(attempts):
            try:
                last_receipt = await verifier.verify()
                if last_receipt.cdn == "current":
                    return last_receipt
            except PublicVerificationError as error:
                last_error = error
            if attempt < attempts - 1:
                await asyncio.sleep(retry_delay)
    if last_receipt is not None:
        return last_receipt
    assert last_error is not None
    raise last_error
