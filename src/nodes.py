"""Typed proxy admission, semantic identity, provenance, and catalog ownership."""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import re
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from itertools import chain
from typing import Annotated, Any, ClassVar, Literal, TypeVar
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
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        return value
    labels = host.split(".")
    if len(labels) <= 1 or not all(
        label
        and len(label) <= 63
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
        for label in labels
    ):
        raise ValueError("invalid server")
    return value


def _globally_scoped_endpoint(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return True


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

    def endpoint(self) -> str:
        return self.server


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

    def endpoint(self) -> None:
        return None


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


class PublishedInstant(FrozenModel):
    kind: Literal["instant"] = "instant"
    at: AwareDatetime


class PublishedDate(FrozenModel):
    kind: Literal["date"] = "date"
    on: date


class UnknownPublicationTime(FrozenModel):
    kind: Literal["unknown"] = "unknown"


PublicationTime = Annotated[
    PublishedInstant | PublishedDate | UnknownPublicationTime,
    Field(discriminator="kind"),
]


def _publication_date(value: PublicationTime) -> date | None:
    match value:
        case PublishedInstant(at=published_at):
            return published_at.date()
        case PublishedDate(on=published_on):
            return published_on
        case UnknownPublicationTime():
            return None


class SourceArtifact(FrozenModel):
    authority: str = ""
    site: str
    source_url: str
    content: bytes = Field(repr=False)
    observed_at: AwareDatetime
    publication_time: PublicationTime = UnknownPublicationTime()
    media_type: str | None = None

    @field_validator("site", "source_url")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("content", mode="before")
    @classmethod
    def encode_content(cls, value: bytes | str) -> bytes:
        return value.encode("utf-8") if isinstance(value, str) else value

    @property
    def authority_id(self) -> str:
        return self.authority or self.site

    @property
    def published_on(self) -> date | None:
        return _publication_date(self.publication_time)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @classmethod
    def inline(
        cls,
        *,
        site: str,
        content: str,
        observed_at: datetime,
        published_on: date | None = None,
        published_at: datetime | None = None,
        authority: str = "",
        media_type: str = "text/plain",
    ) -> SourceArtifact:
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        if published_at is not None:
            publication_time: PublicationTime = PublishedInstant(at=published_at)
        elif published_on is not None:
            publication_time = PublishedDate(on=published_on)
        else:
            publication_time = UnknownPublicationTime()
        return cls(
            authority=authority,
            site=site,
            source_url=f"inline://{site}/{digest}",
            content=encoded,
            observed_at=observed_at,
            publication_time=publication_time,
            media_type=media_type,
        )


class NodeProvenance(FrozenModel):
    authority: str
    site: str
    source_url: str
    observed_at: AwareDatetime
    publication_time: PublicationTime = UnknownPublicationTime()
    artifact_digest: str
    item_index: int = Field(ge=0)

    @property
    def published_on(self) -> date | None:
        return _publication_date(self.publication_time)


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
NodeT = TypeVar("NodeT", bound=ClashNode | UriNode | DualNode)


class Rejection(FrozenModel):
    code: str
    site: str
    source_url: str
    item_index: int | None = None
    message: str = ""


class SourceReceipt(FrozenModel):
    authority: str
    site: str
    source_url: str
    artifact_digest: str
    observed_at: AwareDatetime
    publication_time: PublicationTime = UnknownPublicationTime()
    freshness: Literal["current", "stale", "expired", "future", "unknown"]

    @property
    def published_on(self) -> date | None:
        return _publication_date(self.publication_time)


class CodeCount(FrozenModel):
    code: str
    count: int = Field(ge=0, strict=True)


class SourceAdmissionSummary(FrozenModel):
    source: str
    authority: str
    status: Literal["available", "empty", "failed"]
    artifacts: int = Field(ge=0, strict=True)
    candidate_records: int = Field(ge=0, strict=True)
    unique_eligible: int = Field(ge=0, strict=True)
    rejection_codes: tuple[CodeCount, ...] = Field(default=(), strict=False)


class AdmissionCounts(FrozenModel):
    attempted_sources: int = Field(ge=0, strict=True)
    failed_sources: int = Field(ge=0, strict=True)
    empty_sources: int = Field(ge=0, strict=True)
    sources_with_artifacts: int = Field(ge=0, strict=True)
    discovered_artifacts: int = Field(ge=0, strict=True)
    rejected_artifacts: int = Field(ge=0, strict=True)
    decoded_artifacts: int = Field(ge=0, strict=True)
    candidate_records: int = Field(ge=0, strict=True)
    rejected_records: int = Field(ge=0, strict=True)
    eligible_occurrences: int = Field(ge=0, strict=True)
    unique_eligible: int = Field(ge=0, strict=True)
    duplicate_occurrences: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_accounting(self) -> "AdmissionCounts":
        if self.attempted_sources != (
            self.failed_sources + self.empty_sources + self.sources_with_artifacts
        ):
            raise ValueError("source accounting does not balance")
        if self.discovered_artifacts != (
            self.rejected_artifacts + self.decoded_artifacts
        ):
            raise ValueError("artifact accounting does not balance")
        if self.candidate_records != (
            self.rejected_records + self.eligible_occurrences
        ):
            raise ValueError("candidate accounting does not balance")
        if self.eligible_occurrences != (
            self.unique_eligible + self.duplicate_occurrences
        ):
            raise ValueError("deduplication accounting does not balance")
        return self


class AdmissionSummary(FrozenModel):
    counts: AdmissionCounts
    rejection_codes: tuple[CodeCount, ...] = Field(default=(), strict=False)
    sources: tuple[SourceAdmissionSummary, ...] = Field(default=(), strict=False)


class NodeCatalog(FrozenModel):
    nodes: tuple[Node, ...] = Field(default=(), strict=False)
    rejections: tuple[Rejection, ...] = Field(default=(), strict=False)
    receipts: tuple[SourceReceipt, ...] = Field(default=(), strict=False)
    summary: AdmissionSummary | None = None

    @property
    def accepted_count(self) -> int:
        return len(self.nodes)

    @property
    def rejected_count(self) -> int:
        if self.summary is None:
            return len(self.rejections)
        return (
            self.summary.counts.rejected_artifacts
            + self.summary.counts.rejected_records
        )

    def latest_source_dates(self) -> dict[str, date | None]:
        receipts = ((item.site, item.published_on) for item in self.receipts)
        provenance = (
            (item.site, item.published_on)
            for node in self.nodes
            for item in node.provenance
        )
        latest: dict[str, date | None] = {}
        for source, published_on in chain(receipts, provenance):
            previous = latest.setdefault(source, None)
            if published_on is not None and (
                previous is None or published_on > previous
            ):
                latest[source] = published_on
        return latest

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


class VmessUriPayload(BaseModel):
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

    @classmethod
    def from_uri(cls, uri: str) -> "VmessUriPayload":
        payload = uri.removeprefix("vmess://").split("#", 1)[0]
        padded = payload + "=" * (-len(payload) % 4)
        return cls.model_validate_json(base64.urlsafe_b64decode(padded))

    def render(self, name: str) -> str:
        payload = self.model_copy(update={"name": name}).model_dump(
            by_alias=True, exclude_none=True, mode="json"
        )
        serialized = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        encoded = base64.urlsafe_b64encode(serialized.encode()).decode().rstrip("=")
        return f"vmess://{encoded}"


def _vmess_from_uri(uri: str) -> tuple[Proxy, str, str] | None:
    try:
        raw = VmessUriPayload.from_uri(uri)
    except (
        binascii.Error,
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


def _looks_like_yaml(artifact: SourceArtifact, content: str) -> bool:
    media_type = (artifact.media_type or "").casefold()
    return "yaml" in media_type or bool(re.search(r"(?m)^\s*proxies\s*:", content))


def _yaml_candidates(
    artifact: SourceArtifact,
    content: str,
) -> tuple[list[tuple[Proxy, str, str]], list[Rejection]]:
    try:
        documents = yaml.safe_load_all(content)
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
    while True:
        try:
            raw_document = next(documents)
        except StopIteration:
            break
        except yaml.YAMLError:
            return [], [
                Rejection(
                    code="malformed_yaml",
                    site=artifact.site,
                    source_url=artifact.source_url,
                )
            ]
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
    content = artifact.content.decode("utf-8", errors="replace")
    if _looks_like_yaml(artifact, content):
        accepted, rejected = _yaml_candidates(artifact, content)
        return [
            (proxy, None, name, fingerprint) for proxy, name, fingerprint in accepted
        ], rejected

    decoded = _decode_base64_container(content)
    content = decoded if decoded is not None else content
    accepted: list[tuple[Proxy | None, str | None, str, str]] = []
    rejected: list[Rejection] = []
    for index, line in enumerate(
        line.strip() for line in content.splitlines() if line.strip()
    ):
        parsed = _proxy_from_uri(line)
        if parsed is not None:
            proxy, name, fingerprint = parsed
            accepted.append((proxy, line, name, fingerprint))
            continue
        match = _URI_LINE.match(line)
        rejected.append(
            Rejection(
                code=(
                    "unsupported_protocol"
                    if match and match.group("scheme").casefold() not in _URI_SCHEMES
                    else "malformed_node"
                ),
                site=artifact.site,
                source_url=artifact.source_url,
                item_index=index,
            )
        )
    if accepted or rejected:
        return accepted, rejected
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


class _NodeBuilder:
    """Accumulate duplicate provenance and freeze one admitted node once."""

    __slots__ = (
        "display_name",
        "fingerprint",
        "provenance",
        "provenance_keys",
        "proxy",
        "uri",
    )

    def __init__(
        self,
        *,
        fingerprint: str,
        display_name: str,
        proxy: Proxy | None,
        uri: str | None,
    ) -> None:
        self.fingerprint = fingerprint
        self.display_name = display_name
        self.proxy = proxy
        self.uri = uri
        self.provenance: list[NodeProvenance] = []
        self.provenance_keys: set[tuple[str, str, str]] = set()

    def absorb(
        self,
        proxy: Proxy | None,
        uri: str | None,
        provenance: NodeProvenance,
    ) -> None:
        self.proxy = self.proxy or proxy
        self.uri = self.uri or uri
        key = (provenance.authority, provenance.site, provenance.artifact_digest)
        if key in self.provenance_keys:
            return
        self.provenance_keys.add(key)
        self.provenance.append(provenance)

    def freeze(self) -> Node:
        return _new_node(
            fingerprint=self.fingerprint,
            display_name=self.display_name,
            proxy=self.proxy,
            uri=self.uri,
            provenance=tuple(self.provenance),
        )


def _freshness(
    publication_time: PublicationTime,
    observed_at: datetime,
    *,
    stale_after: timedelta,
    expires_after: timedelta,
) -> tuple[Literal["current", "stale", "expired", "future", "unknown"], str | None]:
    match publication_time:
        case PublishedInstant(at=published_at):
            age = observed_at - published_at
            if age.total_seconds() < 0:
                return "future", "clock_inversion"
            if age <= stale_after:
                return "current", None
            if age <= expires_after:
                return "stale", None
            return "expired", "source_expired"
        case PublishedDate(on=published_on):
            days = (observed_at.date() - published_on).days
            if days < 0:
                return "future", "clock_inversion"
            if days == 0:
                return "current", None
            if days <= 2:
                return "stale", None
            return "expired", "source_expired"
        case UnknownPublicationTime():
            return "unknown", None


def _code_counts(values: Counter[str]) -> tuple[CodeCount, ...]:
    return tuple(
        CodeCount(code=code, count=count) for code, count in sorted(values.items())
    )


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

    records: dict[str, _NodeBuilder] = {}
    rejections: list[Rejection] = []
    rejection_samples: Counter[tuple[str, str]] = Counter()
    receipts: list[SourceReceipt] = []
    code_counts: Counter[str] = Counter()
    source_candidates: Counter[str] = Counter()
    source_unique: dict[str, set[str]] = defaultdict(set)
    source_codes: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_artifacts = 0
    decoded_artifacts = 0
    candidate_records = 0
    rejected_records = 0
    eligible_occurrences = 0

    def retain_sample(rejection: Rejection) -> None:
        key = (rejection.site, rejection.code)
        if rejection_samples[key] < 3:
            rejections.append(rejection)
            rejection_samples[key] += 1

    for artifact in artifacts:
        artifact_digest = artifact.digest
        freshness, rejection_code = _freshness(
            artifact.publication_time,
            observed_now,
            stale_after=stale_after,
            expires_after=expires_after,
        )
        receipts.append(
            SourceReceipt(
                authority=artifact.authority_id,
                site=artifact.site,
                source_url=artifact.source_url,
                artifact_digest=artifact_digest,
                observed_at=artifact.observed_at,
                publication_time=artifact.publication_time,
                freshness=freshness,
            )
        )
        if rejection_code:
            rejected_artifacts += 1
            code_counts[rejection_code] += 1
            source_codes[artifact.site][rejection_code] += 1
            retain_sample(
                Rejection(
                    code=rejection_code,
                    site=artifact.site,
                    source_url=artifact.source_url,
                )
            )
            continue

        decoded_artifacts += 1
        candidates, artifact_rejections = _artifact_candidates(artifact)
        rejected_records += len(artifact_rejections)
        candidate_records += len(candidates) + len(artifact_rejections)
        source_candidates[artifact.site] += len(candidates) + len(artifact_rejections)
        for rejection in artifact_rejections:
            retain_sample(rejection)
            code_counts[rejection.code] += 1
            source_codes[artifact.site][rejection.code] += 1
        for index, (proxy, uri, name, fingerprint) in enumerate(candidates):
            server = (
                proxy.endpoint() if proxy is not None else urlsplit(uri or "").hostname
            )
            if server is not None and not _globally_scoped_endpoint(server):
                rejected_records += 1
                rejection = Rejection(
                    code="endpoint_scope",
                    site=artifact.site,
                    source_url=artifact.source_url,
                    item_index=index,
                )
                retain_sample(rejection)
                code_counts[rejection.code] += 1
                source_codes[artifact.site][rejection.code] += 1
                continue
            eligible_occurrences += 1
            provenance = NodeProvenance(
                authority=artifact.authority_id,
                site=artifact.site,
                source_url=artifact.source_url,
                observed_at=artifact.observed_at,
                publication_time=artifact.publication_time,
                artifact_digest=artifact_digest,
                item_index=index,
            )
            if fingerprint in records:
                code_counts["duplicate"] += 1
                source_codes[artifact.site]["duplicate"] += 1
            builder = records.setdefault(
                fingerprint,
                _NodeBuilder(
                    fingerprint=fingerprint,
                    display_name=name,
                    proxy=proxy,
                    uri=uri,
                ),
            )
            builder.absorb(proxy, uri, provenance)
            source_unique[artifact.site].add(fingerprint)

    source_names = tuple(dict.fromkeys(artifact.site for artifact in artifacts))
    counts = AdmissionCounts(
        attempted_sources=len(source_names),
        failed_sources=0,
        empty_sources=0,
        sources_with_artifacts=len(source_names),
        discovered_artifacts=len(artifacts),
        rejected_artifacts=rejected_artifacts,
        decoded_artifacts=decoded_artifacts,
        candidate_records=candidate_records,
        rejected_records=rejected_records,
        eligible_occurrences=eligible_occurrences,
        unique_eligible=len(records),
        duplicate_occurrences=eligible_occurrences - len(records),
    )
    summary = AdmissionSummary(
        counts=counts,
        rejection_codes=_code_counts(code_counts),
        sources=tuple(
            SourceAdmissionSummary(
                source=source,
                authority=next(
                    artifact.authority_id
                    for artifact in artifacts
                    if artifact.site == source
                ),
                status="available",
                artifacts=sum(artifact.site == source for artifact in artifacts),
                candidate_records=source_candidates[source],
                unique_eligible=len(source_unique[source]),
                rejection_codes=_code_counts(source_codes[source]),
            )
            for source in source_names
        ),
    )
    return NodeCatalog(
        nodes=tuple(builder.freeze() for builder in records.values()),
        rejections=tuple(rejections),
        receipts=tuple(receipts),
        summary=summary,
    )


def select_source_fair(
    catalog: NodeCatalog,
    authority_order: Sequence[str],
    *,
    limit: int,
) -> NodeCatalog:
    """Select a deterministic bounded catalog without network evidence."""
    if limit <= 0:
        raise ValueError("selection limit must be positive")
    nodes = {node.fingerprint: node for node in catalog.nodes}
    memberships: dict[str, set[str]] = defaultdict(set)
    for node in catalog.nodes:
        for authority in {item.authority for item in node.provenance}:
            memberships[authority].add(node.fingerprint)
    ordered = list(dict.fromkeys(authority_order))
    ordered.extend(sorted(set(memberships) - set(ordered)))
    buckets = {authority: sorted(memberships[authority]) for authority in ordered}
    selected: list[str] = []
    emitted: set[str] = set()

    def reserve(kind: Literal["clash", "uri"]) -> None:
        if len(selected) >= limit:
            return
        for authority in ordered:
            for fingerprint in buckets[authority]:
                node = nodes[fingerprint]
                compatible = (
                    node.kind in {"clash", "dual"}
                    if kind == "clash"
                    else node.kind in {"uri", "dual"}
                )
                if compatible and fingerprint not in emitted:
                    selected.append(fingerprint)
                    emitted.add(fingerprint)
                    return

    reserve("clash")
    reserve("uri")
    cursors = {authority: 0 for authority in ordered}
    active = deque(authority for authority in ordered if buckets[authority])
    while active and len(selected) < limit:
        authority = active.popleft()
        bucket = buckets[authority]
        cursor = cursors[authority]
        while cursor < len(bucket) and bucket[cursor] in emitted:
            cursor += 1
        if cursor < len(bucket):
            fingerprint = bucket[cursor]
            selected.append(fingerprint)
            emitted.add(fingerprint)
            cursor += 1
        cursors[authority] = cursor
        if cursor < len(bucket):
            active.append(authority)
    return NodeCatalog(
        nodes=tuple(nodes[fingerprint] for fingerprint in selected),
        rejections=catalog.rejections,
        receipts=catalog.receipts,
        summary=catalog.summary,
    )
