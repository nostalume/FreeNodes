import asyncio
import base64
import hashlib

import httpx
import pytest
import yaml

from freenodes.config import RepositoryIdentity
from freenodes.mihomo import ProviderLoadReceipt
from freenodes.profiles import SubscriptionURLs
from freenodes.publication import (
    PublicationCounts,
    PublicationManifestV2,
)
from freenodes.verification import (
    PublicEntryVerifier,
    PublicVerificationError,
    PublicVerificationReceipt,
)
from tests.support import publication_admission_summary

LOADED_PROVIDER = ProviderLoadReceipt(
    counts=(("source", 1),), total_nodes=1, groups=("🚀 Auto",)
)


def bodies(
    registry: SubscriptionURLs, *, cdn_generation: str = "2026-08-29T00:00:00+00:00"
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
        return (
            PublicationManifestV2(
                schema=2,
                status="accepted",
                created_at=generated,
                selection_limit=500,
                admission=publication_admission_summary().counts,
                counts=PublicationCounts(published=1, clash=1, uri=1),
                files={
                    path: hashlib.sha256(body).hexdigest()
                    for path, body in files.items()
                },
                managed_files=(*sorted(files), "nodes/publication-receipt.json"),
                removed_files=(),
            )
            .model_dump_json(by_alias=True)
            .encode()
        )

    return {
        registry.v2ray.direct: files["nodes/v2ray.txt"],
        registry.v2ray.cdn: files["nodes/v2ray.txt"],
        registry.plain.direct: uri,
        registry.plain.cdn: uri,
        registry.clash.direct: standalone,
        registry.clash.cdn: standalone,
        registry.provider.direct: direct_provider,
        registry.provider.cdn: cdn_provider,
        registry.receipt.direct: manifest("2026-08-29T00:00:00+00:00"),
        registry.receipt.cdn: manifest(cdn_generation),
    }


async def test_direct_and_current_cdn_entries_are_admitted_as_user_operations():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
    )
    content = bodies(registry)
    standalone_checks = []
    provider_checks = []

    async def fetch(url):
        return content[url]

    def smoke_provider(body):
        provider_checks.append(body)
        return LOADED_PROVIDER

    verifier = PublicEntryVerifier(
        registry,
        fetch=fetch,
        validate_standalone=lambda body: standalone_checks.append(body),
        smoke_provider=smoke_provider,
    )
    receipt = await verifier.verify()

    assert receipt.direct == "current"
    assert receipt.cdn == "current"
    assert len(standalone_checks) == 2
    assert len(provider_checks) == 2


async def test_valid_older_cdn_is_reported_as_lagging_not_current():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
    )
    content = bodies(registry, cdn_generation="2026-08-28T00:00:00+00:00")

    async def fetch(url):
        return content[url]

    receipt = await PublicEntryVerifier(
        registry,
        fetch=fetch,
        validate_standalone=lambda body: None,
        smoke_provider=lambda body: LOADED_PROVIDER,
    ).verify()

    assert receipt.direct == "current"
    assert receipt.cdn == "lagging"


async def test_direct_format_failure_is_a_release_error():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
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
            smoke_provider=lambda body: LOADED_PROVIDER,
        ).verify()


async def test_direct_profile_digest_mismatch_is_a_release_error():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
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
            smoke_provider=lambda body: LOADED_PROVIDER,
        ).verify()


async def test_provider_channel_cannot_mix_direct_and_cdn_nested_urls():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
    )
    content = bodies(registry)
    content[registry.provider.cdn] = content[registry.provider.direct]

    async def fetch(url):
        return content[url]

    receipt = await PublicEntryVerifier(
        registry,
        fetch=fetch,
        validate_standalone=lambda body: None,
        smoke_provider=lambda body: LOADED_PROVIDER,
    ).verify()

    assert receipt.direct == "current"
    assert receipt.cdn == "degraded"


async def test_provider_name_only_smoke_is_not_complete_entry_validation():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
    )
    content = bodies(registry)

    async def fetch(url):
        return content[url]

    with pytest.raises(PublicVerificationError, match="direct"):
        await PublicEntryVerifier(
            registry,
            fetch=fetch,
            validate_standalone=lambda body: None,
            smoke_provider=lambda body: None,
        ).verify()


@pytest.mark.parametrize(
    "failure, stage",
    (
        ("fetch", "fetch"),
        ("digest", "artifact"),
        ("standalone", "standalone"),
        ("provider", "provider"),
    ),
)
async def test_cdn_failures_report_only_redacted_boundary(failure, stage):
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
    )
    content = bodies(registry)
    if failure == "digest":
        content[registry.plain.cdn] += b"credential-secret"
    seen = {"standalone": 0, "provider": 0}

    async def fetch(url):
        if failure == "fetch" and url == registry.receipt.cdn:
            raise httpx.HTTPStatusError(
                "credential-secret " + url,
                request=httpx.Request("GET", url),
                response=httpx.Response(403),
            )
        return content[url]

    def standalone(body):
        seen["standalone"] += 1
        if failure == "standalone" and seen["standalone"] == 2:
            raise ValueError("credential-secret")

    def provider(body):
        seen["provider"] += 1
        if failure == "provider" and seen["provider"] == 2:
            raise ValueError("credential-secret")
        return LOADED_PROVIDER

    receipt = await PublicEntryVerifier(
        registry, fetch=fetch, validate_standalone=standalone, smoke_provider=provider
    ).verify()
    assert receipt.direct == "current" and receipt.cdn == "degraded"
    assert receipt.cdn_diagnostic.stage == stage
    assert receipt.cdn_diagnostic.http_status == (403 if failure == "fetch" else None)
    assert receipt.cdn_diagnostic.entry == (
        "receipt"
        if failure == "fetch"
        else failure
        if failure in ("standalone", "provider")
        else None
    )
    assert "credential-secret" not in receipt.model_dump_json()
    assert "https://" not in receipt.model_dump_json()


def test_legacy_public_receipt_without_diagnostic_remains_valid():
    receipt = PublicVerificationReceipt.model_validate(
        {
            "direct": "current",
            "cdn": "degraded",
            "direct_generation": "2026-08-29T00:00:00+00:00",
            "cdn_generation": None,
        }
    )
    assert receipt.cdn_diagnostic is None
    with pytest.raises(ValueError, match="failure diagnostic"):
        PublicVerificationReceipt.model_validate(
            {
                "direct": "current",
                "cdn": "current",
                "direct_generation": receipt.direct_generation,
                "cdn_generation": receipt.direct_generation,
                "cdn_diagnostic": {"stage": "fetch", "error_type": "TimeoutError"},
            }
        )


async def test_public_cancellation_is_not_relabelled_as_degraded():
    registry = SubscriptionURLs.from_identity(
        RepositoryIdentity(owner="owner", name="repo")
    )
    content = bodies(registry)

    async def fetch(url):
        if url == registry.receipt.cdn:
            raise asyncio.CancelledError
        return content[url]

    with pytest.raises(asyncio.CancelledError):
        await PublicEntryVerifier(
            registry,
            fetch=fetch,
            validate_standalone=lambda body: None,
            smoke_provider=lambda body: LOADED_PROVIDER,
        ).verify()
