"""Remote import-entry verification contracts."""

import base64
import hashlib
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

    files = {
        "nodes/v2ray.txt": base64.b64encode(uri),
        "nodes/merged.txt": uri,
        "nodes/merged.yaml": standalone,
        "nodes/provider.yaml": direct_provider,
        "nodes/provider-cdn.yaml": cdn_provider,
    }

    def manifest(generated):
        return json.dumps(
            {
                "schema": 2,
                "status": "accepted",
                "created_at": generated,
                "base_revision": None,
                "selection_limit": 500,
                "admission": {
                    "attempted_sources": 1,
                    "failed_sources": 0,
                    "empty_sources": 0,
                    "sources_with_artifacts": 1,
                    "discovered_artifacts": 1,
                    "rejected_artifacts": 0,
                    "decoded_artifacts": 1,
                    "candidate_records": 1,
                    "rejected_records": 0,
                    "eligible_occurrences": 1,
                    "unique_eligible": 1,
                    "duplicate_occurrences": 0,
                },
                "counts": {
                    "published": 1,
                    "clash": 1,
                    "uri": 1,
                },
                "rejection_codes": [],
                "sources": [],
                "files": {
                    path: hashlib.sha256(body).hexdigest()
                    for path, body in files.items()
                },
                "managed_files": [
                    *sorted(files),
                    "nodes/publication-receipt.json",
                ],
                "removed_files": [],
            }
        ).encode()

    return {
        registry.v2ray.direct: files["nodes/v2ray.txt"],
        registry.v2ray.cdn: files["nodes/v2ray.txt"],
        registry.legacy.direct: uri,
        registry.legacy.cdn: uri,
        registry.clash.direct: standalone,
        registry.clash.cdn: standalone,
        registry.provider.direct: direct_provider,
        registry.provider.cdn: cdn_provider,
        registry.receipt.direct: manifest("2026-08-29T00:00:00+00:00"),
        registry.receipt.cdn: manifest(cdn_generation),
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


async def test_direct_profile_digest_mismatch_is_a_release_error():
    registry = PublicEntryRegistry.from_identity(
        RepositoryIdentity(owner="owner", repository="repo")
    )
    content = bodies(registry)
    content[registry.clash.direct] = yaml.safe_dump(
        {
            "proxies": [{"name": "changed", "type": "direct"}],
            "proxy-groups": [],
            "rules": [],
        }
    ).encode()

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
