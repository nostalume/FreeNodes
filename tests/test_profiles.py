"""Contracts for deterministic V2Ray and Clash profile projections."""

import base64
import importlib
import json
from datetime import UTC, datetime
from urllib.parse import quote, unquote, urlsplit

import pytest
import yaml

from src.config import RepositoryIdentity
from src.nodes import SourceArtifact, admit_artifacts, semantic_fingerprint

NOW = datetime(2026, 8, 29, tzinfo=UTC)


def profile_module():
    return importlib.import_module("src.profiles")


def sample_catalog():
    uri = "ss://YWVzLTEyOC1nY206c2VjcmV0@uri.example:443#URI"
    provider = """proxies:
  - &shared {name: Clash, type: trojan, server: clash.example, port: 443, password: clash-password}
"""
    return admit_artifacts(
        [
            SourceArtifact.inline(site="uri-source", content=uri, observed_at=NOW),
            SourceArtifact(
                site="yaml source",
                source_url="https://example.test/provider.yaml",
                content=provider,
                observed_at=NOW,
                media_type="application/yaml",
            ),
        ],
        now=NOW,
    )


def uri_catalog(uri, site="source"):
    return admit_artifacts(
        [SourceArtifact.inline(site=site, content=uri, observed_at=NOW)], now=NOW
    )


def test_registry_is_the_single_explicit_import_url_contract():
    profiles = profile_module()
    registry = profiles.PublicEntryRegistry()

    assert (
        registry.v2ray.direct
        == "https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/v2ray.txt"
    )
    assert (
        registry.v2ray.cdn
        == "https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/v2ray.txt"
    )
    assert registry.clash.direct.endswith("/HEAD/nodes/merged.yaml")
    assert registry.clash.cdn.endswith("/nodes/merged.yaml")
    assert registry.provider.direct.endswith("/HEAD/nodes/provider.yaml")
    assert registry.provider.cdn.endswith("/nodes/provider-cdn.yaml")
    assert registry.legacy.direct.endswith("/HEAD/nodes/merged.txt")


def test_registry_is_constructed_from_the_admitted_repository_identity():
    profiles = profile_module()
    registry = profiles.PublicEntryRegistry.from_identity(
        RepositoryIdentity(owner="owner", repository="repo")
    )

    assert registry.clash.direct.startswith(
        "https://raw.githubusercontent.com/owner/repo/"
    )


def test_rendered_bundle_has_exact_aggregate_contract_and_safe_site_provider():
    profiles = profile_module()
    bundle = profiles.render_profiles(sample_catalog(), profiles.PublicEntryRegistry())

    assert bundle.aggregate_files == (
        "nodes/merged.txt",
        "nodes/v2ray.txt",
        "nodes/merged.yaml",
        "nodes/provider.yaml",
        "nodes/provider-cdn.yaml",
    )
    assert "nodes/yaml-source.yaml" in bundle.files
    assert all(
        ".." not in name and not name.startswith(("/", "\\")) for name in bundle.files
    )


def test_plain_and_base64_v2ray_profiles_are_byte_equivalent():
    profiles = profile_module()
    bundle = profiles.render_profiles(sample_catalog(), profiles.PublicEntryRegistry())

    plain = bundle.files["nodes/merged.txt"]
    encoded = bundle.files["nodes/v2ray.txt"]

    assert base64.b64decode(encoded) == plain
    assert len(plain.decode().splitlines()) == 1
    assert plain.startswith(b"ss://YWVzLTEyOC1nY206c2VjcmV0@uri.example:443#")


def test_standalone_and_provider_profiles_describe_same_clash_nodes():
    profiles = profile_module()
    bundle = profiles.render_profiles(sample_catalog(), profiles.PublicEntryRegistry())
    standalone = yaml.safe_load(bundle.files["nodes/merged.yaml"])
    site_providers = [
        yaml.safe_load(content)
        for name, content in bundle.files.items()
        if name in {"nodes/uri-source.yaml", "nodes/yaml-source.yaml"}
    ]

    standalone_ids = {semantic_fingerprint(proxy) for proxy in standalone["proxies"]}
    provider_ids = {
        semantic_fingerprint(proxy)
        for provider in site_providers
        for proxy in provider["proxies"]
    }

    assert standalone_ids == provider_ids
    assert standalone["proxy-groups"]
    assert standalone["rules"]


def test_provider_profiles_use_http_sources_and_never_yaml_aliases():
    profiles = profile_module()
    registry = profiles.PublicEntryRegistry()
    bundle = profiles.render_profiles(sample_catalog(), registry)
    direct_bytes = bundle.files["nodes/provider.yaml"]
    cdn_bytes = bundle.files["nodes/provider-cdn.yaml"]
    direct = yaml.safe_load(direct_bytes)
    cdn = yaml.safe_load(cdn_bytes)

    direct_provider = direct["proxy-providers"]["yaml-source"]
    cdn_provider = cdn["proxy-providers"]["yaml-source"]
    assert set(direct["proxy-providers"]) == {"yaml-source"}
    assert direct_provider["type"] == "http"
    assert direct_provider["url"] == registry.site_provider("yaml-source", cdn=False)
    assert cdn_provider["url"] == registry.site_provider("yaml-source", cdn=True)
    assert direct_provider["path"] == "./proxy_providers/yaml-source.yaml"
    assert b"&id" not in direct_bytes + cdn_bytes
    assert b"*id" not in direct_bytes + cdn_bytes


def test_yaml_quotes_scientific_looking_reality_short_id():
    profiles = profile_module()
    catalog = admit_artifacts(
        [
            SourceArtifact(
                site="reality",
                source_url="https://example.test/reality.yaml",
                observed_at=NOW,
                media_type="application/yaml",
                content="""proxies:
  - name: Reality
    type: vless
    server: reality.example
    port: 443
    uuid: 11111111-1111-1111-1111-111111111111
    tls: true
    reality-opts:
      public-key: CGR1XzsRlvVmgeAiqLiA9SwEnxgtkbmANavMhmn6IVc
      short-id: "062898e8"
""",
            )
        ],
        now=NOW,
    )

    rendered = profiles.render_profiles(catalog).files["nodes/merged.yaml"].decode()

    assert 'short-id: "062898e8"' in rendered


def test_consumer_outputs_share_one_canonical_name_and_semantic_identity():
    profiles = profile_module()
    original = (
        "trojan://secret@one.example:443?security=tls#"
        "%F0%9F%87%BA%F0%9F%87%B8US_2%7C711KB%2Fs"
    )
    artifacts = [
        SourceArtifact.inline(site=site, content=original, observed_at=NOW)
        for site in ("z-source", "a-source")
    ]
    catalog = admit_artifacts(artifacts, now=NOW)

    bundle = profiles.render_profiles(catalog)
    standalone = yaml.safe_load(bundle.files["nodes/merged.yaml"])
    rendered_uri = bundle.files["nodes/merged.txt"].decode().strip()
    expected = f"US · TROJAN · a-source · {catalog.nodes[0].fingerprint[:8].upper()}"

    assert standalone["proxies"][0]["name"] == expected
    assert unquote(urlsplit(rendered_uri).fragment) == expected
    assert (
        uri_catalog(rendered_uri).nodes[0].fingerprint == catalog.nodes[0].fingerprint
    )


def test_vmess_ps_uses_the_same_canonical_name_without_changing_identity():
    profiles = profile_module()
    payload = {
        "ps": "🇨🇳_CN_中国->🇫🇷_FR_法国",
        "add": "vmess.example",
        "port": 443,
        "id": "11111111-1111-1111-1111-111111111111",
    }
    original = "vmess://" + base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    catalog = uri_catalog(original, "vmess")

    rendered = profiles.render_profiles(catalog).files["nodes/merged.txt"].decode()
    encoded = rendered.strip().removeprefix("vmess://")
    decoded = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))

    assert decoded["ps"] == (
        f"CN→FR · VMESS · vmess · {catalog.nodes[0].fingerprint[:8].upper()}"
    )
    assert uri_catalog(rendered).nodes[0].fingerprint == catalog.nodes[0].fingerprint


@pytest.mark.parametrize("label", ("promotion-no-region", "_XX_invalid"))
def test_unverified_country_text_remains_unknown(label):
    catalog = uri_catalog(f"trojan://secret@unknown.example:443#{quote(label)}")
    profile = yaml.safe_load(
        profile_module().render_profiles(catalog).files["nodes/merged.yaml"]
    )

    assert profile["proxies"][0]["name"].startswith("ZZ · TROJAN ·")


def test_optional_region_group_partitions_only_explicit_terminal_hints():
    catalog = uri_catalog(
        "\n".join(
            f"trojan://secret@us{index}.example:443#🇺🇸US-{index}" for index in range(2)
        )
        + "\ntrojan://secret@fr.example:443#🇫🇷FR"
    )
    profile = yaml.safe_load(
        profile_module().render_profiles(catalog).files["nodes/merged.yaml"]
    )
    groups = {group["name"]: group for group in profile["proxy-groups"]}

    assert len(groups["🌐 US"]["proxies"]) == 2
    assert all(name.startswith("US ·") for name in groups["🌐 US"]["proxies"])
    assert sum(group["type"] == "url-test" for group in groups.values()) == 2


def test_profiles_apply_ordered_user_overridable_routing_without_repeat_probes():
    profiles = profile_module()
    bundle = profiles.render_profiles(sample_catalog())
    standalone = yaml.safe_load(bundle.files["nodes/merged.yaml"])
    groups = {group["name"]: group for group in standalone["proxy-groups"]}

    assert list(groups) == [
        "🚀 Auto",
        "🌍 Proxy",
        "🇨🇳 Mainland China",
        "🤖 AI",
        "🎬 Media",
        "💬 Messaging",
    ]
    assert {
        "type": "url-test",
        "interval": 600,
        "tolerance": 150,
        "lazy": True,
        "timeout": 5000,
        "expected-status": 204,
    }.items() <= groups["🚀 Auto"].items()
    assert groups["🇨🇳 Mainland China"]["proxies"] == ["DIRECT", "🌍 Proxy"]
    assert all(
        "url" not in group for name, group in groups.items() if name != "🚀 Auto"
    )
    assert standalone["profile"]["store-selected"] is True
    assert standalone["geodata-loader"] == "memconservative"
    assert standalone["rules"] == [
        "GEOSITE,private,DIRECT",
        "GEOIP,private,DIRECT,no-resolve",
        "GEOSITE,category-ai-!cn,🤖 AI",
        "GEOSITE,youtube,🎬 Media",
        "GEOSITE,netflix,🎬 Media",
        "GEOSITE,telegram,💬 Messaging",
        "GEOSITE,cn,🇨🇳 Mainland China",
        "GEOIP,CN,🇨🇳 Mainland China,no-resolve",
        "MATCH,🌍 Proxy",
    ]
