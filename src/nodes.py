"""Typed proxy admission, semantic identity, provenance, and catalog ownership."""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import parse_qs, parse_qsl, unquote, urlsplit
from uuid import UUID

import yaml
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import FrozenModel

_URI_SCHEMES = frozenset(
    {
        "vmess",
        "vless",
        "trojan",
        "ss",
        "ssr",
        "socks",
        "socks5",
        "hysteria",
        "hysteria2",
        "hy2",
        "tuic",
    }
)
_URI_LINE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://\S+$")


def _valid_host(value: str) -> str:
    host = value.strip().rstrip(".")
    if not host or any(char.isspace() for char in host):
        raise ValueError("invalid server")
    try:
        ipaddress.ip_address(host)
        return value
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) <= 1 or not all(
        label
        and len(label) <= 63
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
        for label in labels
    ):
        raise ValueError("invalid server")
    return value


class RealityOptions(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        strict=True,
        populate_by_name=True,
    )

    public_key: str = Field(alias="public-key")

    @field_validator("public_key")
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", value):
            raise ValueError("invalid reality public key")
        return value


class ProxyBase(BaseModel):
    """Typed Mihomo proxy boundary with forward-compatible transport options."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        strict=True,
        populate_by_name=True,
    )

    name: str
    type: str
    server: str
    port: int = Field(ge=1, le=65535, strict=False)
    reality_opts: RealityOptions | None = Field(default=None, alias="reality-opts")

    _secret_names: ClassVar[tuple[str, ...]] = ()

    @field_validator("server")
    @classmethod
    def validate_server(cls, value: str) -> str:
        return _valid_host(value)

    def _secrets(self) -> dict[str, str]:
        return {}

    def mihomo_payload(self) -> dict[str, Any]:
        payload = self.model_dump(
            by_alias=True,
            exclude_none=True,
            exclude=set(self._secret_names),
            mode="json",
        )
        payload.update(self._secrets())
        return payload

    def identity_payload(self) -> dict[str, Any]:
        payload = self.mihomo_payload()
        payload.pop("name", None)
        return payload


class VmessProxy(ProxyBase):
    type: Literal["vmess"]
    uuid: UUID = Field(strict=False)
    alter_id: int = Field(default=0, alias="alterId", strict=False)
    cipher: str = "auto"


class VlessProxy(ProxyBase):
    type: Literal["vless"]
    uuid: UUID = Field(strict=False)


class TrojanProxy(ProxyBase):
    type: Literal["trojan"]
    password: SecretStr = Field(repr=False)
    _secret_names = ("password",)

    def _secrets(self) -> dict[str, str]:
        return {"password": self.password.get_secret_value()}


class ShadowsocksProxy(ProxyBase):
    type: Literal["ss"]
    cipher: str
    password: SecretStr = Field(repr=False)
    _secret_names = ("password",)

    def _secrets(self) -> dict[str, str]:
        return {"password": self.password.get_secret_value()}


class ShadowsocksRProxy(ProxyBase):
    type: Literal["ssr"]
    password: SecretStr = Field(repr=False)
    _secret_names = ("password",)

    def _secrets(self) -> dict[str, str]:
        return {"password": self.password.get_secret_value()}


class HttpProxy(ProxyBase):
    type: Literal["http"]
    username: SecretStr | None = Field(default=None, repr=False)
    password: SecretStr | None = Field(default=None, repr=False)
    _secret_names = ("username", "password")

    def _secrets(self) -> dict[str, str]:
        return {
            name: value.get_secret_value()
            for name, value in (
                ("username", self.username),
                ("password", self.password),
            )
            if value is not None
        }


class Socks5Proxy(HttpProxy):
    type: Literal["socks5"]


class HysteriaProxy(ProxyBase):
    type: Literal["hysteria"]
    password: SecretStr | None = Field(default=None, repr=False)
    auth: SecretStr | None = Field(default=None, repr=False)
    auth_str: SecretStr | None = Field(default=None, alias="auth-str", repr=False)
    _secret_names = ("password", "auth", "auth_str")

    @model_validator(mode="after")
    def require_auth(self) -> HysteriaProxy:
        if not any((self.password, self.auth, self.auth_str)):
            raise ValueError("missing auth")
        return self

    def _secrets(self) -> dict[str, str]:
        return {
            name: value.get_secret_value()
            for name, value in (
                ("password", self.password),
                ("auth", self.auth),
                ("auth-str", self.auth_str),
            )
            if value is not None
        }


class Hysteria2Proxy(HysteriaProxy):
    type: Literal["hysteria2"]


class TuicProxy(ProxyBase):
    type: Literal["tuic"]
    uuid: UUID = Field(strict=False)
    password: SecretStr | None = Field(default=None, repr=False)
    _secret_names = ("password",)

    def _secrets(self) -> dict[str, str]:
        return (
            {"password": self.password.get_secret_value()}
            if self.password is not None
            else {}
        )


class AnyTlsProxy(ProxyBase):
    type: Literal["anytls"]
    password: SecretStr = Field(repr=False)
    _secret_names = ("password",)

    def _secrets(self) -> dict[str, str]:
        return {"password": self.password.get_secret_value()}


class DirectProxy(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    name: str
    type: Literal["direct"]
    udp: bool = True

    def mihomo_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, mode="json")

    def identity_payload(self) -> dict[str, Any]:
        payload = self.mihomo_payload()
        payload.pop("name", None)
        return payload


Proxy = Annotated[
    VmessProxy
    | VlessProxy
    | TrojanProxy
    | ShadowsocksProxy
    | ShadowsocksRProxy
    | HttpProxy
    | Socks5Proxy
    | HysteriaProxy
    | Hysteria2Proxy
    | TuicProxy
    | AnyTlsProxy
    | DirectProxy,
    Field(discriminator="type"),
]
PROXY_ADAPTER = TypeAdapter(Proxy)


def admit_proxy(value: object) -> Proxy:
    """Construct one supported proxy variant at an explicit boundary."""
    return PROXY_ADAPTER.validate_python(value)


class SourceArtifact(FrozenModel):
    site: str
    source_url: str
    content: str = Field(repr=False)
    observed_at: AwareDatetime
    media_type: str | None = None

    @field_validator("site", "source_url")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @classmethod
    def inline(
        cls,
        *,
        site: str,
        content: str,
        observed_at: datetime,
        media_type: str = "text/plain",
    ) -> SourceArtifact:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return cls(
            site=site,
            source_url=f"inline://{site}/{digest}",
            content=content,
            observed_at=observed_at,
            media_type=media_type,
        )


class NodeProvenance(FrozenModel):
    site: str
    source_url: str
    observed_at: AwareDatetime
    artifact_digest: str
    item_index: int = Field(ge=0)


class NodeBase(FrozenModel):
    fingerprint: str
    display_name: str
    provenance: tuple[NodeProvenance, ...] = Field(default=(), strict=False)


class ClashNode(NodeBase):
    kind: Literal["clash"] = "clash"
    proxy: Proxy = Field(repr=False)


class UriNode(NodeBase):
    kind: Literal["uri"] = "uri"
    uri: str = Field(repr=False)


class DualNode(NodeBase):
    kind: Literal["dual"] = "dual"
    proxy: Proxy = Field(repr=False)
    uri: str = Field(repr=False)


Node = Annotated[ClashNode | UriNode | DualNode, Field(discriminator="kind")]
ProbeableNode = ClashNode | DualNode
UriCapableNode = UriNode | DualNode


class Rejection(FrozenModel):
    code: str
    site: str
    source_url: str
    item_index: int | None = None
    message: str = ""


class SourceReceipt(FrozenModel):
    site: str
    source_url: str
    artifact_digest: str
    observed_at: AwareDatetime
    freshness: Literal["current", "stale", "expired", "future"]


class NodeCatalog(FrozenModel):
    nodes: tuple[Node, ...] = Field(default=(), strict=False)
    rejections: tuple[Rejection, ...] = Field(default=(), strict=False)
    receipts: tuple[SourceReceipt, ...] = Field(default=(), strict=False)

    @property
    def accepted_count(self) -> int:
        return len(self.nodes)

    @property
    def rejected_count(self) -> int:
        return len(self.rejections)

    @property
    def clash_nodes(self) -> tuple[ProbeableNode, ...]:
        selected: list[ProbeableNode] = []
        for node in self.nodes:
            match node.kind:
                case "clash" | "dual":
                    selected.append(node)
                case "uri":
                    continue
        return tuple(selected)

    @property
    def uri_nodes(self) -> tuple[UriCapableNode, ...]:
        selected: list[UriCapableNode] = []
        for node in self.nodes:
            match node.kind:
                case "uri" | "dual":
                    selected.append(node)
                case "clash":
                    continue
        return tuple(selected)

    @property
    def uri_count(self) -> int:
        return len(self.uri_nodes)

    @property
    def clash_count(self) -> int:
        return len(self.clash_nodes)


def _identity_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_fingerprint(proxy: Proxy | Mapping[str, Any]) -> str:
    admitted = PROXY_ADAPTER.validate_python(proxy)
    return _identity_digest(admitted.identity_payload())


def _keyword_matches(name: str, keyword: str) -> bool:
    if keyword.isascii():
        return (
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])",
                name,
                flags=re.IGNORECASE,
            )
            is not None
        )
    return keyword.casefold() in name.casefold()


def detect_regions(
    names: Sequence[str],
    region_keywords: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    regions: dict[str, list[str]] = {}
    for name in names:
        label = next(
            (
                region
                for region, keywords in region_keywords.items()
                if any(_keyword_matches(name, keyword) for keyword in keywords)
            ),
            "🌍 其他",
        )
        regions.setdefault(label, []).append(name)
    return {key: regions[key] for key in sorted(regions)}


class _ProxyEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    proxies: tuple[object, ...] = Field(default=(), strict=False)

    @property
    def candidates(self) -> tuple[object, ...]:
        return self.proxies


class _ProxyList(RootModel[tuple[object, ...]]):
    root: tuple[object, ...] = Field(strict=False)

    @property
    def candidates(self) -> tuple[object, ...]:
        return self.root


_DOCUMENT_ADAPTER = TypeAdapter(_ProxyEnvelope | _ProxyList)


def _proxy_error_code(error: ValidationError) -> str:
    first = error.errors(include_url=False)[0]
    location = {str(part) for part in first["loc"]}
    message = str(first["msg"]).casefold()
    if str(first["type"]).startswith("union_tag"):
        return "unsupported_proxy_type"
    if "server" in location:
        return "invalid_server"
    if "port" in location:
        return "invalid_port"
    if "uuid" in location:
        return "invalid_uuid"
    if location & {"public_key", "public-key", "reality_opts", "reality-opts"}:
        return "invalid_reality_public_key"
    if "cipher" in location:
        return "missing_cipher"
    if "password" in location:
        return "missing_password"
    if "auth" in message:
        return "missing_auth"
    return "invalid_proxy"


def _decode_base64_container(content: str) -> str | None:
    compact = "".join(content.split())
    if (
        not compact
        or len(compact) < 16
        or not re.fullmatch(r"[A-Za-z0-9_+/=-]+", compact)
    ):
        return None
    padded = compact + "=" * (-len(compact) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return (
        decoded if any(f"{scheme}://" in decoded for scheme in _URI_SCHEMES) else None
    )


class _VmessUriPayload(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)
    name: str = Field(default="vmess", alias="ps")
    server: str = Field(alias="add")
    port: int = Field(strict=False)
    uuid: UUID = Field(alias="id", strict=False)
    alter_id: int = Field(default=0, alias="aid", strict=False)
    cipher: str = Field(default="auto", alias="scy")
    network: str = Field(default="tcp", alias="net")
    tls: str = ""
    servername: str | None = Field(default=None, alias="sni")
    path: str = "/"
    host: str | None = None


def _vmess_from_uri(uri: str) -> tuple[Proxy, str, str] | None:
    payload = uri.removeprefix("vmess://").split("#", 1)[0]
    padded = payload + "=" * (-len(payload) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        raw = _VmessUriPayload.model_validate(decoded)
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ):
        return None
    values: dict[str, Any] = {
        "name": raw.name,
        "type": "vmess",
        "server": raw.server,
        "port": raw.port,
        "uuid": raw.uuid,
        "alterId": raw.alter_id,
        "cipher": raw.cipher,
        "network": raw.network,
    }
    if raw.tls.casefold() in {"tls", "true", "1"}:
        values["tls"] = True
    if raw.servername:
        values["servername"] = raw.servername
    if raw.network == "ws":
        ws_options: dict[str, Any] = {"path": raw.path}
        if raw.host:
            ws_options["headers"] = {"Host": raw.host}
        values["ws-opts"] = ws_options
    try:
        proxy = PROXY_ADAPTER.validate_python(values)
    except ValidationError:
        return None
    identity = raw.model_dump(by_alias=True, exclude={"name"}, mode="json")
    return proxy, raw.name, _identity_digest(identity)


def _proxy_from_uri(uri: str) -> tuple[Proxy | None, str, str] | None:
    match = _URI_LINE.fullmatch(uri.strip())
    if not match:
        return None
    scheme = match.group("scheme").casefold()
    if scheme not in _URI_SCHEMES:
        return None
    if scheme == "vmess":
        return _vmess_from_uri(uri.strip())
    try:
        parsed = urlsplit(uri.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    name = unquote(parsed.fragment) or f"{scheme}-{host or 'unknown'}"
    query = {
        key: values[-1] for key, values in parse_qs(parsed.query).items() if values
    }
    fingerprint = _identity_digest(
        {
            "scheme": scheme,
            "username": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "host": (host or "").casefold(),
            "port": port,
            "path": unquote(parsed.path),
            "query": sorted(parse_qsl(parsed.query, keep_blank_values=True)),
        }
    )
    if scheme not in {"vless", "trojan"}:
        try:
            _valid_host(host or "")
        except ValueError:
            return None
        if not port:
            return None
        return None, name, fingerprint

    values: dict[str, Any] = {
        "name": name,
        "type": scheme,
        "server": host,
        "port": port,
        "uuid" if scheme == "vless" else "password": unquote(parsed.username or ""),
    }
    network = query.get("type") or query.get("network")
    if network:
        values["network"] = network
    if query.get("security") in {"tls", "reality"}:
        values["tls"] = True
    for source_key, target_key in (
        ("sni", "servername"),
        ("flow", "flow"),
        ("fp", "client-fingerprint"),
    ):
        if query.get(source_key):
            values[target_key] = query[source_key]
    if network == "ws":
        ws_options: dict[str, Any] = {"path": query.get("path", "/")}
        if query.get("host"):
            ws_options["headers"] = {"Host": query["host"]}
        values["ws-opts"] = ws_options
    try:
        return PROXY_ADAPTER.validate_python(values), name, fingerprint
    except ValidationError:
        return None


def _looks_like_yaml(artifact: SourceArtifact) -> bool:
    media_type = (artifact.media_type or "").casefold()
    return "yaml" in media_type or bool(
        re.search(r"(?m)^\s*proxies\s*:", artifact.content)
    )


def _yaml_candidates(
    artifact: SourceArtifact,
) -> tuple[list[tuple[Proxy, str, str]], list[Rejection]]:
    try:
        documents = tuple(yaml.safe_load_all(artifact.content))
    except yaml.YAMLError:
        return [], [
            Rejection(
                code="malformed_yaml",
                site=artifact.site,
                source_url=artifact.source_url,
            )
        ]
    accepted: list[tuple[Proxy, str, str]] = []
    rejected: list[Rejection] = []
    item_index = 0
    for raw_document in documents:
        try:
            document = _DOCUMENT_ADAPTER.validate_python(raw_document)
        except ValidationError:
            continue
        for raw_proxy in document.candidates:
            try:
                proxy = PROXY_ADAPTER.validate_python(raw_proxy)
            except ValidationError as error:
                rejected.append(
                    Rejection(
                        code=_proxy_error_code(error),
                        site=artifact.site,
                        source_url=artifact.source_url,
                        item_index=item_index,
                    )
                )
            else:
                accepted.append((proxy, proxy.name, semantic_fingerprint(proxy)))
            item_index += 1
    if not accepted and not rejected:
        rejected.append(
            Rejection(
                code="unsupported_content",
                site=artifact.site,
                source_url=artifact.source_url,
            )
        )
    return accepted, rejected


def _artifact_candidates(
    artifact: SourceArtifact,
) -> tuple[list[tuple[Proxy | None, str | None, str, str]], list[Rejection]]:
    if _looks_like_yaml(artifact):
        accepted, rejected = _yaml_candidates(artifact)
        return [
            (proxy, None, name, fingerprint) for proxy, name, fingerprint in accepted
        ], rejected

    decoded = _decode_base64_container(artifact.content)
    content = decoded if decoded is not None else artifact.content
    accepted: list[tuple[Proxy | None, str | None, str, str]] = []
    for line in (line.strip() for line in content.splitlines() if line.strip()):
        parsed = _proxy_from_uri(line)
        if parsed is not None:
            proxy, name, fingerprint = parsed
            accepted.append((proxy, line, name, fingerprint))
    if accepted:
        return accepted, []
    return [], [
        Rejection(
            code="unsupported_content",
            site=artifact.site,
            source_url=artifact.source_url,
        )
    ]


def _new_node(
    *,
    fingerprint: str,
    display_name: str,
    proxy: Proxy | None,
    uri: str | None,
    provenance: tuple[NodeProvenance, ...],
) -> Node:
    if proxy is not None and uri is not None:
        return DualNode(
            fingerprint=fingerprint,
            display_name=display_name,
            proxy=proxy,
            uri=uri,
            provenance=provenance,
        )
    if proxy is not None:
        return ClashNode(
            fingerprint=fingerprint,
            display_name=display_name,
            proxy=proxy,
            provenance=provenance,
        )
    assert uri is not None
    return UriNode(
        fingerprint=fingerprint,
        display_name=display_name,
        uri=uri,
        provenance=provenance,
    )


def _merge_node(existing: Node, incoming: Node, provenance: NodeProvenance) -> Node:
    combined = existing.provenance + (provenance,)
    if existing.kind == "clash":
        if incoming.kind == "uri":
            return DualNode(
                fingerprint=existing.fingerprint,
                display_name=existing.display_name,
                proxy=existing.proxy,
                uri=incoming.uri,
                provenance=combined,
            )
    elif existing.kind == "uri":
        if incoming.kind == "clash":
            return DualNode(
                fingerprint=existing.fingerprint,
                display_name=existing.display_name,
                proxy=incoming.proxy,
                uri=existing.uri,
                provenance=combined,
            )
    return existing.model_copy(update={"provenance": combined})


def admit_artifacts(
    artifacts: Sequence[SourceArtifact],
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(hours=24),
    expires_after: timedelta = timedelta(hours=48),
) -> NodeCatalog:
    observed_now = now or datetime.now(UTC)
    if observed_now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    records: dict[str, Node] = {}
    rejections: list[Rejection] = []
    receipts: list[SourceReceipt] = []
    for artifact in artifacts:
        age = observed_now - artifact.observed_at
        if age.total_seconds() < 0:
            freshness = "future"
            rejection_code = "clock_inversion"
        elif age > expires_after:
            freshness = "expired"
            rejection_code = "source_expired"
        else:
            freshness = "stale" if age > stale_after else "current"
            rejection_code = None
        receipts.append(
            SourceReceipt(
                site=artifact.site,
                source_url=artifact.source_url,
                artifact_digest=artifact.digest,
                observed_at=artifact.observed_at,
                freshness=freshness,
            )
        )
        if rejection_code:
            rejections.append(
                Rejection(
                    code=rejection_code,
                    site=artifact.site,
                    source_url=artifact.source_url,
                )
            )
            continue

        candidates, artifact_rejections = _artifact_candidates(artifact)
        rejections.extend(artifact_rejections)
        for index, (proxy, uri, name, fingerprint) in enumerate(candidates):
            provenance = NodeProvenance(
                site=artifact.site,
                source_url=artifact.source_url,
                observed_at=artifact.observed_at,
                artifact_digest=artifact.digest,
                item_index=index,
            )
            incoming = _new_node(
                fingerprint=fingerprint,
                display_name=name,
                proxy=proxy,
                uri=uri,
                provenance=(provenance,),
            )
            existing = records.get(fingerprint)
            records[fingerprint] = (
                _merge_node(existing, incoming, provenance) if existing else incoming
            )

    used_names: set[str] = set()
    suffixes: dict[str, int] = {}
    named_records: list[Node] = []
    for record in records.values():
        base_name = record.display_name or "unknown"
        display_name = base_name
        if display_name in used_names:
            suffix = suffixes.get(base_name, 2)
            while f"{base_name}_{suffix}" in used_names:
                suffix += 1
            display_name = f"{base_name}_{suffix}"
            suffixes[base_name] = suffix + 1
        used_names.add(display_name)
        named_records.append(record.model_copy(update={"display_name": display_name}))

    return NodeCatalog(
        nodes=tuple(named_records),
        rejections=tuple(rejections),
        receipts=tuple(receipts),
    )
