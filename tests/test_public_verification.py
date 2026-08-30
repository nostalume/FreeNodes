"""Remote import-entry verification contracts."""

import base64
import json

import pytest
import yaml

from src.config import RepositoryIdentity
from src.profiles import PublicEntryRegistry
from src.public_verification import PublicEntryVerifier, PublicVerificationError


def bodies(
    registry: PublicEntryRegistry, *, cdn_generation: str = "2026-08-29T00:00:00+00:00"
):
    uri = b"trojan://opaque@example.test:443\n"
    standalone = yaml.safe_dump(
        {
            "proxies": [{"name": "one", "type": "direct"}],
            "proxy-groups": [],
            "rules": [],
        }
    ).encode()
    direct_provider = yaml.safe_dump(
        {
            "proxy-providers": {
                "source": {
                    "type": "http",
                    "url": registry.site_provider("source", cdn=False),
                }
            },
            "proxy-groups": [],
        }
    ).encode()
    cdn_provider = yaml.safe_dump(
        {
            "proxy-providers": {
                "source": {
                    "type": "http",
                    "url": registry.site_provider("source", cdn=True),
                }
            },
            "proxy-groups": [],
        }
    ).encode()

    def manifest(generated):
        return json.dumps(
            {
                "schema": 1,
                "status": "quality_verified",
                "generated_at": generated,
                "tool": {"name": "freenodespider", "version": "0.1.0"},
                "runner_vantage": "test",
                "policy": {
                    "max_candidates": 1,
                    "max_published": 1,
                    "max_per_source": 1,
                    "max_delay_ms": 2500,
                    "history_days": 7,
                    "required_endpoints": 2,
                },
                "counts": {
                    "admitted": 1,
                    "selected_for_probe": 1,
                    "probe_success": 1,
                    "published": 1,
                    "excluded": 0,
                },
                "exclusions": {},
                "sources": [],
                "published": [
                    {"id": "0" * 24, "worst_delay_ms": 50, "reliability": None}
                ],
                "history": [],
            }
        ).encode()

    return {
        registry.v2ray.direct: base64.b64encode(uri),
        registry.v2ray.cdn: base64.b64encode(uri),
        registry.legacy.direct: uri,
        registry.legacy.cdn: uri,
        registry.clash.direct: standalone,
        registry.clash.cdn: standalone,
        registry.provider.direct: direct_provider,
        registry.provider.cdn: cdn_provider,
        registry.quality.direct: manifest("2026-08-29T00:00:00+00:00"),
        registry.quality.cdn: manifest(cdn_generation),
    }


async def test_direct_and_current_cdn_entries_are_admitted_as_user_operations():
    registry = PublicEntryRegistry.from_identity(
        RepositoryIdentity(owner="owner", repository="repo")
    )
    content = bodies(registry)
    standalone_checks = []
    provider_checks = []

    async def fetch(url):
        return content[url]

    verifier = PublicEntryVerifier(
        registry,
        fetch=fetch,
        validate_standalone=lambda body: standalone_checks.append(body),
        smoke_provider=lambda body: provider_checks.append(body),
    )
    receipt = await verifier.verify()

    assert receipt.direct == "current"
    assert receipt.cdn == "current"
    assert len(standalone_checks) == 2
    assert len(provider_checks) == 2


async def test_valid_older_cdn_is_reported_as_lagging_not_current():
    registry = PublicEntryRegistry.from_identity(
        RepositoryIdentity(owner="owner", repository="repo")
    )
    content = bodies(registry, cdn_generation="2026-08-28T00:00:00+00:00")

    async def fetch(url):
        return content[url]

    receipt = await PublicEntryVerifier(
        registry,
        fetch=fetch,
        validate_standalone=lambda body: None,
        smoke_provider=lambda body: None,
    ).verify()

    assert receipt.direct == "current"
    assert receipt.cdn == "lagging"


async def test_direct_format_failure_is_a_release_error():
    registry = PublicEntryRegistry.from_identity(
        RepositoryIdentity(owner="owner", repository="repo")
    )
    content = bodies(registry)
    content[registry.v2ray.direct] = b"not matching base64"

    async def fetch(url):
        return content[url]

    with pytest.raises(PublicVerificationError, match="direct"):
        await PublicEntryVerifier(
            registry,
            fetch=fetch,
            validate_standalone=lambda body: None,
            smoke_provider=lambda body: None,
        ).verify()


async def test_provider_channel_cannot_mix_direct_and_cdn_nested_urls():
    registry = PublicEntryRegistry.from_identity(
        RepositoryIdentity(owner="owner", repository="repo")
    )
    content = bodies(registry)
    content[registry.provider.cdn] = content[registry.provider.direct]

    async def fetch(url):
        return content[url]

    receipt = await PublicEntryVerifier(
        registry,
        fetch=fetch,
        validate_standalone=lambda body: None,
        smoke_provider=lambda body: None,
    ).verify()

    assert receipt.direct == "current"
    assert receipt.cdn == "degraded"
