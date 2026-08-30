"""Contracts for deterministic V2Ray and Clash profile projections."""

import base64
import importlib
from datetime import UTC, datetime

import yaml

from src.config import RepositoryIdentity
from src.nodes import SourceArtifact, admit_artifacts, semantic_fingerprint

NOW = datetime(2026, 8, 29, tzinfo=UTC)


def profile_module():
    return importlib.import_module("src.profiles")


def sample_catalog():
    uri = "trojan://uri-password@uri.example:443?security=tls#URI"
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
    assert plain.decode().splitlines() == [
        "trojan://uri-password@uri.example:443?security=tls#URI"
    ]


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
