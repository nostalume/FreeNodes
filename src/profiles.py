"""Pure, deterministic projections from an admitted catalog to profiles."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import Field, model_validator

from src.config import FrozenModel, RepositoryIdentity
from src.nodes import NodeCatalog, ProbeableNode


class EntryPair(FrozenModel):
    direct: str
    cdn: str

    def for_channel(self, *, cdn: bool) -> str:
        return self.cdn if cdn else self.direct


class PublicEntryRegistry(FrozenModel):
    """Single source of truth for every user-facing import URL."""

    identity: RepositoryIdentity = RepositoryIdentity()

    @classmethod
    def from_identity(cls, identity: RepositoryIdentity) -> "PublicEntryRegistry":
        return cls(identity=identity)

    def _pair(self, direct_path: str, cdn_path: str | None = None) -> EntryPair:
        cdn_file = cdn_path or direct_path
        return EntryPair(
            direct=(
                f"https://raw.githubusercontent.com/{self.identity.owner}/"
                f"{self.identity.repository}"
                f"/HEAD/{direct_path}"
            ),
            cdn=(
                f"https://cdn.jsdelivr.net/gh/{self.identity.owner}/"
                f"{self.identity.repository}"
                f"/{cdn_file}"
            ),
        )

    @property
    def v2ray(self) -> EntryPair:
        return self._pair("nodes/v2ray.txt")

    @property
    def legacy(self) -> EntryPair:
        return self._pair("nodes/merged.txt")

    @property
    def clash(self) -> EntryPair:
        return self._pair("nodes/merged.yaml")

    @property
    def provider(self) -> EntryPair:
        return self._pair("nodes/provider.yaml", "nodes/provider-cdn.yaml")

    @property
    def quality(self) -> EntryPair:
        return self._pair("nodes/quality-manifest.json")

    def site_provider(self, slug: str, *, cdn: bool) -> str:
        pair = self._pair(f"nodes/{site_slug(slug)}.yaml")
        return pair.cdn if cdn else pair.direct


class BundleFile(FrozenModel):
    path: str = Field(min_length=1)
    content: bytes = Field(repr=False)


class OutputBundle(FrozenModel):
    entries: tuple[BundleFile, ...] = Field(strict=False, repr=False)
    accepted_count: int = Field(ge=0, strict=True)
    clash_count: int = Field(ge=0, strict=True)
    uri_count: int = Field(ge=0, strict=True)
    aggregate_files: tuple[str, ...] = (
        "nodes/merged.txt",
        "nodes/v2ray.txt",
        "nodes/merged.yaml",
        "nodes/provider.yaml",
        "nodes/provider-cdn.yaml",
    )

    @classmethod
    def from_files(
        cls,
        files: Mapping[str, bytes],
        *,
        accepted_count: int,
        clash_count: int,
        uri_count: int,
        aggregate_files: tuple[str, ...] | None = None,
    ) -> "OutputBundle":
        values: dict[str, object] = {
            "entries": tuple(
                BundleFile(path=path, content=content)
                for path, content in files.items()
            ),
            "accepted_count": accepted_count,
            "clash_count": clash_count,
            "uri_count": uri_count,
        }
        if aggregate_files is not None:
            values["aggregate_files"] = aggregate_files
        return cls.model_validate(values)

    @model_validator(mode="after")
    def validate_bundle(self) -> "OutputBundle":
        paths = tuple(entry.path for entry in self.entries)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate bundle file path")
        if not set(self.aggregate_files).issubset(paths):
            raise ValueError("aggregate file is absent from the bundle")
        if max(self.clash_count, self.uri_count) > self.accepted_count:
            raise ValueError("projected profile count exceeds accepted nodes")
        return self

    @property
    def files(self) -> Mapping[str, bytes]:
        return MappingProxyType({entry.path: entry.content for entry in self.entries})


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


_AMBIGUOUS_YAML_STRING = re.compile(
    r"^(?:"
    r"[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+|"
    r"0[0-9_]+|0[xX][0-9a-fA-F_]+|"
    r"true|false|yes|no|on|off|null|~"
    r")$",
    flags=re.IGNORECASE,
)


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = '"' if _AMBIGUOUS_YAML_STRING.fullmatch(value) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_NoAliasDumper.add_representer(str, _represent_string)


def _yaml_bytes(value: object) -> bytes:
    return yaml.dump(
        value,
        Dumper=_NoAliasDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    ).encode("utf-8")


def site_slug(value: str) -> str:
    """Return the stable public filename component for a source name."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "source"


def _site_slugs(catalog: NodeCatalog) -> dict[str, str]:
    names = sorted({item.site for node in catalog.nodes for item in node.provenance})
    used: set[str] = set()
    result: dict[str, str] = {}
    for site in names:
        base = site_slug(site)
        slug = base
        suffix = 2
        while slug in used:
            slug = f"{base}-{suffix}"
            suffix += 1
        used.add(slug)
        result[site] = slug
    return result


def _render_proxy(node: ProbeableNode) -> dict[str, Any]:
    return {**node.proxy.mihomo_payload(), "name": node.display_name}


def _standalone_config(proxies: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(proxy["name"]) for proxy in proxies]
    if not names:
        return {
            "mixed-port": 7890,
            "allow-lan": False,
            "mode": "rule",
            "log-level": "info",
            "proxies": [],
            "proxy-groups": [
                {"name": "🌍 手动选择", "type": "select", "proxies": ["DIRECT"]}
            ],
            "rules": ["MATCH,🌍 手动选择"],
        }
    return {
        "mixed-port": 7890,
        "allow-lan": False,
        "mode": "rule",
        "log-level": "info",
        "ipv6": True,
        "proxies": proxies,
        "proxy-groups": [
            {
                "name": "🚀 自动选择",
                "type": "url-test",
                "proxies": names,
                "url": "https://www.gstatic.com/generate_204",
                "interval": 300,
                "tolerance": 50,
            },
            {
                "name": "🌍 手动选择",
                "type": "select",
                "proxies": ["🚀 自动选择", "DIRECT", *names],
            },
        ],
        "rules": ["MATCH,🌍 手动选择"],
    }


def _provider_config(
    slugs: list[str],
    registry: PublicEntryRegistry,
    *,
    cdn: bool,
) -> dict[str, Any]:
    providers = {
        slug: {
            "type": "http",
            "url": registry.site_provider(slug, cdn=cdn),
            "path": f"./proxy_providers/{slug}.yaml",
            "interval": 3600,
            "health-check": {
                "enable": True,
                "url": "https://www.gstatic.com/generate_204",
                "interval": 300,
                "timeout": 5000,
                "lazy": True,
                "expected-status": 204,
            },
        }
        for slug in slugs
    }
    if not providers:
        return _standalone_config([])
    return {
        "mixed-port": 7890,
        "allow-lan": False,
        "mode": "rule",
        "log-level": "info",
        "ipv6": True,
        "proxy-providers": providers,
        "proxy-groups": [
            {
                "name": "🚀 自动选择",
                "type": "url-test",
                "use": slugs,
                "url": "https://www.gstatic.com/generate_204",
                "interval": 300,
                "tolerance": 50,
            },
            {
                "name": "🌍 手动选择",
                "type": "select",
                "proxies": ["🚀 自动选择", "DIRECT"],
                "use": slugs,
            },
        ],
        "rules": ["MATCH,🌍 手动选择"],
    }


def render_profiles(
    catalog: NodeCatalog,
    registry: PublicEntryRegistry | None = None,
) -> OutputBundle:
    """Project one catalog into every supported subscription format."""
    entries = registry or PublicEntryRegistry()
    uri_lines = [node.uri for node in catalog.uri_nodes]
    plain = (("\n".join(uri_lines) + "\n") if uri_lines else "").encode("utf-8")
    clash_nodes = catalog.clash_nodes
    proxies = [_render_proxy(node) for node in clash_nodes]
    slugs_by_site = _site_slugs(catalog)

    site_nodes: dict[str, list[dict[str, Any]]] = {
        slug: [] for slug in slugs_by_site.values()
    }
    for node in clash_nodes:
        proxy = _render_proxy(node)
        sites = {item.site for item in node.provenance}
        for site in sites:
            site_nodes[slugs_by_site[site]].append(dict(proxy))

    files: dict[str, bytes] = {
        "nodes/merged.txt": plain,
        "nodes/v2ray.txt": base64.b64encode(plain),
        "nodes/merged.yaml": _yaml_bytes(_standalone_config(proxies)),
        "nodes/provider.yaml": _yaml_bytes(
            _provider_config(sorted(site_nodes), entries, cdn=False)
        ),
        "nodes/provider-cdn.yaml": _yaml_bytes(
            _provider_config(sorted(site_nodes), entries, cdn=True)
        ),
    }
    for slug, site_proxies in sorted(site_nodes.items()):
        files[f"nodes/{slug}.yaml"] = _yaml_bytes({"proxies": site_proxies})

    return OutputBundle.from_files(
        files,
        accepted_count=catalog.accepted_count,
        clash_count=len(proxies),
        uri_count=len(uri_lines),
    )
