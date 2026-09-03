from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import parse_qs, parse_qsl, unquote, urlsplit
from uuid import UUID

import yaml
from pydantic import (
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

from freenodes.config import FrozenModel


class UriScheme(StrEnum):
    VMESS = "vmess"
    VLESS = "vless"
    TROJAN = "trojan"
    SS = "ss"
    SSR = "ssr"
    SOCKS = "socks"
    SOCKS5 = "socks5"
    HYSTERIA = "hysteria"
    HYSTERIA2 = "hysteria2"
    HY2 = "hy2"
    TUIC = "tuic"


URI_SCHEME_ADAPTER = TypeAdapter(UriScheme)
_URI_SCHEMES = frozenset(UriScheme)
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


def is_global_endpoint(value: str) -> bool:
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
    return PROXY_ADAPTER.validate_python(value)


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
    def from_uri(cls, uri: str) -> VmessUriPayload:
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


class DecodedProxy(FrozenModel):
    proxy: Proxy | None = Field(default=None, repr=False)
    uri: str | None = Field(default=None, repr=False)
    name: str
    fingerprint: str


class ParsedProxyUri(FrozenModel):
    value: str = Field(repr=False)
    scheme: UriScheme
    host: str | None
    port: int | None
    name: str
    username: str = Field(repr=False)
    password: str = Field(repr=False)
    path: str
    query: dict[str, str]
    fingerprint: str

    @classmethod
    def admit(cls, value: str) -> ParsedProxyUri | None:
        try:
            parsed = urlsplit(value)
            scheme = URI_SCHEME_ADAPTER.validate_python(parsed.scheme.casefold())
            host = parsed.hostname
            port = parsed.port
        except (ValidationError, ValueError):
            return None
        username = unquote(parsed.username or "")
        password = unquote(parsed.password or "")
        path = unquote(parsed.path)
        query = {
            key: values[-1] for key, values in parse_qs(parsed.query).items() if values
        }
        return cls(
            value=value,
            scheme=scheme,
            host=host,
            port=port,
            name=unquote(parsed.fragment) or f"{scheme}-{host or 'unknown'}",
            username=username,
            password=password,
            path=path,
            query=query,
            fingerprint=_identity_digest(
                {
                    "scheme": scheme,
                    "username": username,
                    "password": password,
                    "host": (host or "").casefold(),
                    "port": port,
                    "path": path,
                    "query": sorted(parse_qsl(parsed.query, keep_blank_values=True)),
                }
            ),
        )

    def decode(self) -> DecodedProxy | None:
        if self.scheme in {"vless", "trojan"}:
            return self._clash_proxy()
        try:
            _valid_host(self.host or "")
        except ValueError:
            return None
        if not self.port:
            return None
        return DecodedProxy(
            uri=self.value,
            name=self.name,
            fingerprint=self.fingerprint,
        )

    def _clash_proxy(self) -> DecodedProxy | None:
        values: dict[str, Any] = {
            "name": self.name,
            "type": self.scheme,
            "server": self.host,
            "port": self.port,
            "uuid" if self.scheme == "vless" else "password": self.username,
        }
        network = self.query.get("type") or self.query.get("network")
        if network:
            values["network"] = network
        if self.query.get("security") in {"tls", "reality"}:
            values["tls"] = True
        for source, target in (
            ("sni", "servername"),
            ("flow", "flow"),
            ("fp", "client-fingerprint"),
        ):
            if self.query.get(source):
                values[target] = self.query[source]
        if network == "ws":
            options: dict[str, Any] = {"path": self.query.get("path", "/")}
            if self.query.get("host"):
                options["headers"] = {"Host": self.query["host"]}
            values["ws-opts"] = options
        try:
            proxy = PROXY_ADAPTER.validate_python(values)
        except ValidationError:
            return None
        return DecodedProxy(
            proxy=proxy,
            uri=self.value,
            name=self.name,
            fingerprint=self.fingerprint,
        )


class ProxyDecodeIssue(FrozenModel):
    code: str
    item_index: int | None = Field(default=None, ge=0)


class ProxyDocument(FrozenModel):
    candidates: tuple[DecodedProxy, ...] = Field(default=(), strict=False)
    issues: tuple[ProxyDecodeIssue, ...] = Field(default=(), strict=False)


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


def _vmess_from_uri(uri: str) -> DecodedProxy | None:
    try:
        raw = VmessUriPayload.from_uri(uri)
    except (binascii.Error, ValidationError, ValueError):
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
    return DecodedProxy(
        proxy=proxy,
        uri=uri,
        name=raw.name,
        fingerprint=_identity_digest(identity),
    )


def _proxy_from_uri(uri: str) -> DecodedProxy | None:
    value = uri.strip()
    match = _URI_LINE.fullmatch(value)
    if not match:
        return None
    scheme = match.group("scheme").casefold()
    if scheme == "vmess":
        return _vmess_from_uri(value)
    parsed = ParsedProxyUri.admit(value)
    if parsed is None:
        return None
    return parsed.decode()


def _yaml_document(content: str) -> ProxyDocument:
    try:
        documents = yaml.safe_load_all(content)
    except yaml.YAMLError:
        return ProxyDocument(issues=(ProxyDecodeIssue(code="malformed_yaml"),))
    accepted: list[DecodedProxy] = []
    rejected: list[ProxyDecodeIssue] = []
    item_index = 0
    while True:
        try:
            raw_document = next(documents)
        except StopIteration:
            break
        except yaml.YAMLError:
            return ProxyDocument(issues=(ProxyDecodeIssue(code="malformed_yaml"),))
        try:
            document = _DOCUMENT_ADAPTER.validate_python(raw_document)
        except ValidationError:
            continue
        for raw_proxy in document.candidates:
            try:
                proxy = PROXY_ADAPTER.validate_python(raw_proxy)
            except ValidationError as error:
                rejected.append(
                    ProxyDecodeIssue(
                        code=_proxy_error_code(error),
                        item_index=item_index,
                    )
                )
            else:
                accepted.append(
                    DecodedProxy(
                        proxy=proxy,
                        name=proxy.name,
                        fingerprint=semantic_fingerprint(proxy),
                    )
                )
            item_index += 1
    if not accepted and not rejected:
        rejected.append(ProxyDecodeIssue(code="unsupported_content"))
    return ProxyDocument(candidates=tuple(accepted), issues=tuple(rejected))


def decode_proxies(content: str, *, yaml_hint: bool) -> ProxyDocument:
    if yaml_hint:
        return _yaml_document(content)
    decoded = _decode_base64_container(content)
    text = decoded if decoded is not None else content
    accepted: list[DecodedProxy] = []
    rejected: list[ProxyDecodeIssue] = []
    for index, line in enumerate(
        line.strip() for line in text.splitlines() if line.strip()
    ):
        parsed = _proxy_from_uri(line)
        if parsed is not None:
            accepted.append(parsed)
            continue
        match = _URI_LINE.match(line)
        rejected.append(
            ProxyDecodeIssue(
                code=(
                    "unsupported_protocol"
                    if match and match.group("scheme").casefold() not in _URI_SCHEMES
                    else "malformed_node"
                ),
                item_index=index,
            )
        )
    if not accepted and not rejected:
        rejected.append(ProxyDecodeIssue(code="unsupported_content"))
    return ProxyDocument(candidates=tuple(accepted), issues=tuple(rejected))
