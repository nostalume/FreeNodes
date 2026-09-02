"""Pure, deterministic projections from an admitted catalog to profiles."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import yaml
from pydantic import Field, TypeAdapter, ValidationError, model_validator
from pydantic_extra_types.country import CountryAlpha2

from src.config import FrozenModel, RepositoryIdentity
from src.nodes import Node, NodeCatalog, VmessUriPayload

_COUNTRY = TypeAdapter(CountryAlpha2)
_FLAG = re.compile(r"[\U0001F1E6-\U0001F1FF]{2}")
_ALPHA2 = re.compile(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])")
_ROUTE = re.compile(r"(?:->|→|➡️?)")
_REGION_GROUPS = ("HK", "TW", "JP", "SG", "US")
_PROBE_URL = "https://www.gstatic.com/generate_204"


def _admit_country(value: str) -> CountryAlpha2 | None:
    try:
        return _COUNTRY.validate_python(value)
    except ValidationError:
        return None


def _country_evidence(value: str) -> tuple[CountryAlpha2, ...]:
    flags = (
        "".join(chr(ord(char) - 0x1F1E6 + ord("A")) for char in match.group())
        for match in _FLAG.finditer(value)
    )
    tokens = (match.group(1) for match in _ALPHA2.finditer(value))
    return tuple(code for raw in (*flags, *tokens) if (code := _admit_country(raw)))


class RegionHint(FrozenModel):
    route: tuple[CountryAlpha2, ...] = Field(default=(), max_length=2, strict=False)

    @classmethod
    def from_label(cls, label: str) -> "RegionHint":
        bounded = label[:256]
        sides = _ROUTE.split(bounded, maxsplit=1)
        if len(sides) == 2:
            origin, destination = map(_country_evidence, sides)
            if origin and destination:
                return cls(route=(origin[0], destination[0]))
        found = _country_evidence(bounded)
        return cls(route=found[:1])

    @property
    def text(self) -> str:
        return "→".join(self.route) if self.route else "ZZ"


class CanonicalLabel(FrozenModel):
    region: RegionHint
    protocol: str
    source: str
    identity: str = Field(pattern=r"^[0-9A-F]{8}$")

    @classmethod
    def from_node(cls, node: Node) -> "CanonicalLabel":
        if not node.provenance:
            raise ValueError("canonical node has no provenance")
        provenance = min(
            node.provenance,
            key=lambda item: (
                item.authority,
                item.site,
                item.artifact_digest,
                item.item_index,
            ),
        )
        source = re.sub(r"[^a-z0-9]+", "-", provenance.site.casefold()).strip("-")
        if not source:
            raise ValueError("canonical node source has no ASCII slug")
        match node.kind:
            case "clash" | "dual":
                protocol = node.proxy.type
            case "uri":
                protocol = urlsplit(node.uri).scheme
        return cls(
            region=RegionHint.from_label(node.display_name),
            protocol=protocol.upper(),
            source=source[:24],
            identity=node.fingerprint[:8].upper(),
        )

    @property
    def text(self) -> str:
        return f"{self.region.text} · {self.protocol} · {self.source} · {self.identity}"


class FragmentShare(FrozenModel):
    base_uri: str = Field(repr=False)

    @classmethod
    def from_uri(cls, uri: str) -> "FragmentShare":
        parsed = urlsplit(uri)
        return cls(base_uri=urlunsplit((*parsed[:4], "")))

    def render(self, label: str) -> str:
        return f"{self.base_uri}#{quote(label)}"


class ProxyGroupBase(FrozenModel):
    name: str
    proxies: tuple[str, ...] = ()
    use: tuple[str, ...] = ()

    def _payload(self, kind: str) -> dict[str, Any]:
        group: dict[str, Any] = {"name": self.name, "type": kind}
        if self.proxies:
            group["proxies"] = list(self.proxies)
        if self.use:
            group["use"] = list(self.use)
        return group


class UrlTestGroup(ProxyGroupBase):
    def payload(self) -> dict[str, Any]:
        return {
            **self._payload("url-test"),
            "url": _PROBE_URL,
            "interval": 600,
            "tolerance": 150,
            "lazy": True,
            "timeout": 5000,
            "expected-status": 204,
        }


class SelectGroup(ProxyGroupBase):
    def payload(self) -> dict[str, Any]:
        return self._payload("select")


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
    def receipt(self) -> EntryPair:
        return self._pair("nodes/publication-receipt.json")

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


def _policy_config(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mixed-port": 7890,
        "allow-lan": False,
        "mode": "rule",
        "log-level": "info",
        "ipv6": True,
        "profile": {"store-selected": True},
        "geodata-mode": True,
        "geodata-loader": "memconservative",
        "geo-auto-update": True,
        "geo-update-interval": 24,
        "proxy-groups": groups,
        "rules": [
            "GEOSITE,private,DIRECT",
            "GEOIP,private,DIRECT,no-resolve",
            "GEOSITE,category-ai-!cn,🤖 AI",
            "GEOSITE,youtube,🎬 Media",
            "GEOSITE,netflix,🎬 Media",
            "GEOSITE,telegram,💬 Messaging",
            "GEOSITE,cn,🇨🇳 Mainland China",
            "GEOIP,CN,🇨🇳 Mainland China,no-resolve",
            "MATCH,🌍 Proxy",
        ],
    }


def _policy_groups(
    auto: UrlTestGroup,
    *,
    node_names: tuple[str, ...] = (),
    provider_names: tuple[str, ...] = (),
    regions: tuple[UrlTestGroup, ...] = (),
) -> list[dict[str, Any]]:
    region_names = tuple(group.name for group in regions)
    proxy = SelectGroup(
        name="🌍 Proxy",
        proxies=("🚀 Auto", *region_names, "DIRECT", *node_names),
        use=provider_names,
    )
    services = ("🤖 AI", "🎬 Media", "💬 Messaging")
    return [
        auto.payload(),
        *(group.payload() for group in regions),
        proxy.payload(),
        SelectGroup(name="🇨🇳 Mainland China", proxies=("DIRECT", "🌍 Proxy")).payload(),
        *(
            SelectGroup(
                name=name, proxies=("🌍 Proxy", *region_names, "DIRECT")
            ).payload()
            for name in services
        ),
    ]


def _standalone_config(
    proxies: list[dict[str, Any]], regions: Mapping[str, str]
) -> dict[str, Any]:
    names = [str(proxy["name"]) for proxy in proxies]
    if not names:
        return {**_policy_config([]), "proxies": []}
    regional = tuple(
        UrlTestGroup(
            name=f"🌐 {region}",
            proxies=tuple(name for name in names if regions.get(name) == region),
        )
        for region in _REGION_GROUPS
        if sum(regions.get(name) == region for name in names) >= 2
    )
    groups = _policy_groups(
        UrlTestGroup(name="🚀 Auto", proxies=tuple(names)),
        node_names=tuple(names),
        regions=regional,
    )
    return {**_policy_config(groups), "proxies": proxies}


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
        }
        for slug in slugs
    }
    if not providers:
        return _standalone_config([], {})
    groups = _policy_groups(
        UrlTestGroup(name="🚀 Auto", use=tuple(slugs)),
        provider_names=tuple(slugs),
    )
    return {
        **_policy_config(groups),
        "proxy-providers": providers,
    }


def render_profiles(
    catalog: NodeCatalog,
    registry: PublicEntryRegistry | None = None,
) -> OutputBundle:
    """Project one catalog into every supported subscription format."""
    entries = registry or PublicEntryRegistry()
    labels = {
        node.fingerprint: CanonicalLabel.from_node(node) for node in catalog.nodes
    }
    rendered_labels = [label.text for label in labels.values()]
    if len(rendered_labels) != len(set(rendered_labels)):
        raise ValueError("canonical node identity prefix collision")
    uri_lines = []
    for node in catalog.uri_nodes:
        share = (
            VmessUriPayload.from_uri(node.uri)
            if urlsplit(node.uri).scheme.casefold() == "vmess"
            else FragmentShare.from_uri(node.uri)
        )
        uri_lines.append(share.render(labels[node.fingerprint].text))
    plain = (("\n".join(uri_lines) + "\n") if uri_lines else "").encode("utf-8")
    clash_nodes = catalog.clash_nodes
    proxies_by_fingerprint = {
        node.fingerprint: {
            **node.proxy.mihomo_payload(),
            "name": labels[node.fingerprint].text,
        }
        for node in clash_nodes
    }
    proxies = list(proxies_by_fingerprint.values())
    regions = {
        labels[node.fingerprint].text: labels[node.fingerprint].region.route[-1]
        for node in clash_nodes
        if labels[node.fingerprint].region.route
    }
    slugs_by_site = _site_slugs(catalog)

    site_nodes: dict[str, list[dict[str, Any]]] = {}
    for node in clash_nodes:
        proxy = proxies_by_fingerprint[node.fingerprint]
        sites = {item.site for item in node.provenance}
        for site in sites:
            site_nodes.setdefault(slugs_by_site[site], []).append(proxy)

    files: dict[str, bytes] = {
        "nodes/merged.txt": plain,
        "nodes/v2ray.txt": base64.b64encode(plain),
        "nodes/merged.yaml": _yaml_bytes(_standalone_config(proxies, regions)),
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
