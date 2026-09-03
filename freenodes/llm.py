import re
from typing import Literal
from urllib.parse import urlsplit

from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from freenodes.config import FrozenModel, OpenRouterLimits

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_CREDENTIAL_ENV = "OPENROUTER_API_KEY"


class ExtractedLinks(FrozenModel):
    txt: tuple[str, ...] = Field(default=(), strict=False)
    yaml: tuple[str, ...] = Field(default=(), strict=False)
    other: tuple[str, ...] = Field(default=(), strict=False)
    inline: tuple[str, ...] = Field(default=(), strict=False)

    @property
    def downloads(self) -> tuple[str, ...]:
        return self.txt + self.yaml + self.other

    @classmethod
    def from_text(cls, text: str) -> "ExtractedLinks":
        txt: list[str] = []
        yaml: list[str] = []
        inline: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r'https?://[^\s<>"\')\]]+', text):
            url = match.group().rstrip(".,;)")
            if url in seen:
                continue
            path = urlsplit(url).path.lower()
            if path.endswith(".txt"):
                txt.append(url)
            elif path.endswith((".yaml", ".yml")):
                yaml.append(url)
            else:
                continue
            seen.add(url)
        for match in re.finditer(
            r"(?:vmess|vless|trojan|ss|ssr)://[^\s<>\"']+",
            text,
            re.IGNORECASE,
        ):
            url = match.group().rstrip(".,;)")
            if url in seen:
                continue
            seen.add(url)
            inline.append(url)
        return cls(txt=tuple(txt), yaml=tuple(yaml), inline=tuple(inline))


class FallbackText(FrozenModel):
    kind: Literal["text"] = "text"
    value: str


class FallbackUnavailable(FrozenModel):
    kind: Literal["unavailable"] = "unavailable"
    reason: Literal[
        "no_credential",
        "budget_exhausted",
        "rate_limited",
        "empty_response",
    ]


class FallbackFailure(FrozenModel):
    kind: Literal["failure"] = "failure"
    diagnostic: str


FallbackResult = FallbackText | FallbackUnavailable | FallbackFailure


class _ResponseMessage(BaseModel):
    content: str | None = None
    reasoning: str | None = None


class _ResponseChoice(BaseModel):
    message: _ResponseMessage


class _ProviderResponse(BaseModel):
    choices: tuple[_ResponseChoice, ...]


class OpenRouterFallback:
    def __init__(self, limits: OpenRouterLimits, *, credential: str | None):
        self._remaining = limits.request_limit_per_run
        self._source_limit = limits.request_limit_per_source
        self._used_by_source: dict[str, int] = {}
        self._client = (
            AsyncOpenAI(
                base_url=OPENROUTER_ENDPOINT,
                api_key=credential,
                timeout=limits.request_timeout_seconds,
            )
            if credential
            else None
        )

    async def ask(
        self,
        prompt: str,
        *,
        source: str,
        max_tokens: int = 1024,
    ) -> FallbackResult:
        if self._client is None:
            return FallbackUnavailable(reason="no_credential")
        used = self._used_by_source.get(source, 0)
        if self._remaining == 0 or used >= self._source_limit:
            return FallbackUnavailable(reason="budget_exhausted")
        self._remaining -= 1
        self._used_by_source[source] = used + 1
        try:
            response = await self._client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except RateLimitError:
            return FallbackUnavailable(reason="rate_limited")
        except Exception as error:
            return FallbackFailure(diagnostic=type(error).__name__)
        try:
            admitted = _ProviderResponse.model_validate(response, from_attributes=True)
        except ValidationError:
            return FallbackFailure(diagnostic="invalid_response")
        if not admitted.choices:
            return FallbackUnavailable(reason="empty_response")
        message = admitted.choices[0].message
        text = message.content or message.reasoning
        return (
            FallbackText(value=text)
            if text
            else FallbackUnavailable(reason="empty_response")
        )

    async def extract_links(
        self,
        markdown: str,
        *,
        source: str,
    ) -> ExtractedLinks:
        prompt = (
            "Extract all subscription URLs and proxy protocol links from this page. "
            "Return JSON with optional txt, yaml, other, and inline arrays only.\n\n"
            f"Content:\n{markdown[:8000]}"
        )
        result = await self.ask(prompt, source=source, max_tokens=1024)
        return (
            self._parse_json(result.value)
            if result.kind == "text"
            else ExtractedLinks()
        )

    async def generate_pattern(
        self,
        known_links: list[str],
        *,
        source: str,
    ) -> str | None:
        prompt = (
            f"Links found on the page:\n{chr(10).join(known_links[:10])}\n\n"
            "Write one Python regex matching these and similar URLs on the same site. "
            "Return the raw regex only."
        )
        result = await self.ask(prompt, source=source, max_tokens=256)
        return self._extract_regex(result.value) if result.kind == "text" else None

    @staticmethod
    def verify_pattern(pattern: str, known_links: list[str], html: str) -> bool:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return False
        matches = tuple(
            match.group(1) if compiled.groups else match.group(0)
            for match in compiled.finditer(html)
        )
        trailing = re.compile(r'[),;.\'"]+$')
        match_urls = {trailing.sub("", match) for match in matches if match}
        known_urls = {trailing.sub("", link) for link in known_links}
        false_count = sum(
            bool(match)
            and (
                not match.startswith("http")
                or any(
                    marker in match.lower()
                    for marker in ("javascript:", "#", "xmlrpc", "favicon")
                )
            )
            for match in matches
        )
        return known_urls <= match_urls and (
            not matches or false_count / len(matches) <= 0.2
        )

    @staticmethod
    def _extract_regex(text: str) -> str | None:
        fenced = re.search(r"```(?:regex)?\s*\n?(.*?)```", text, re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
            if candidate:
                return candidate
        lines = tuple(line.strip() for line in text.strip().splitlines())
        return next(
            (
                line
                for line in reversed(lines)
                if line
                and "/" in line
                and any(marker in line for marker in ("\\", ".", "*", "+"))
            ),
            None,
        )

    @staticmethod
    def _parse_json(raw: str) -> ExtractedLinks:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return ExtractedLinks.model_validate_json(raw)
        except ValidationError:
            return ExtractedLinks()
