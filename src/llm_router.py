"""LLM Router — multi-provider weighted routing with fallback and health tracking.

Priority: weighted random (not sequential), task-aware, auto-recovery.

Provider chain:
  OpenRouter (weight 50) → Cerebras (weight 30) → Opencode (weight 20)
"""

import logging
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import Config, FrozenModel, ProviderConfig

load_dotenv()

logger = logging.getLogger(__name__)


class ExtractedLinks(FrozenModel):
    """Validated link categories returned by an LLM provider."""

    txt: tuple[str, ...] = Field(default=(), strict=False)
    yaml: tuple[str, ...] = Field(default=(), strict=False)
    other: tuple[str, ...] = Field(default=(), strict=False)
    inline: tuple[str, ...] = Field(default=(), strict=False)

    @property
    def downloads(self) -> tuple[str, ...]:
        return self.txt + self.yaml + self.other

    @classmethod
    def from_text(cls, text: str) -> "ExtractedLinks":
        """Admit exact subscription URLs without an external inference call."""
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


class _ExternalResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, from_attributes=True)


class _ResponseMessage(_ExternalResponseModel):
    content: str | None = None
    reasoning: str | None = None


class _ResponseChoice(_ExternalResponseModel):
    message: _ResponseMessage


class _ProviderResponse(_ExternalResponseModel):
    choices: tuple[_ResponseChoice, ...]


@dataclass
class RequestBudget:
    """Charge LLM attempts against one run and one source site."""

    run_limit: int = 30
    site_limit: int = 3
    used_total: int = 0
    used_by_site: dict[str, int] = field(default_factory=dict)

    def charge(self, site: str) -> bool:
        if self.used_total >= self.run_limit:
            return False
        used_by_site = self.used_by_site.get(site, 0)
        if used_by_site >= self.site_limit:
            return False
        self.used_total += 1
        self.used_by_site[site] = used_by_site + 1
        return True


class WeightedSelector:
    """Pick a provider by weighted random selection, excluding unhealthy ones."""

    @staticmethod
    def pick(
        weights: dict[str, int],
        is_healthy: Callable[[str], bool],
        exclude: set[str] | None = None,
    ) -> str | None:
        exclude = exclude or set()
        candidates = {
            name: w
            for name, w in weights.items()
            if is_healthy(name) and name not in exclude
        }
        if not candidates:
            return None
        total = sum(candidates.values())
        if total == 0:
            return None
        r = random.uniform(0, total)
        cumulative = 0
        for name, w in candidates.items():
            cumulative += w
            if r <= cumulative:
                return name
        return None


class HealthTracker:
    """Track provider health; disable after consecutive failures, auto-recover."""

    def __init__(self, max_failures: int = 5, cooldown_seconds: int = 1800):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._failures: dict[str, int] = {}
        self._disabled_until: dict[str, float] = {}

    def record_success(self, provider: str):
        self._failures[provider] = 0
        self._disabled_until.pop(provider, None)

    def record_failure(self, provider: str):
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= self.max_failures:
            self._disabled_until[provider] = time.time() + self.cooldown_seconds

    def is_healthy(self, provider: str) -> bool:
        if provider not in self._disabled_until:
            return True
        if time.time() >= self._disabled_until[provider]:
            del self._disabled_until[provider]
            return True
        return False

    def health_report(self) -> dict[str, dict]:
        return {
            name: {"failures": count, "healthy": self.is_healthy(name)}
            for name, count in self._failures.items()
        }


class LLMRouter:
    """Unified LLM interface with weighted routing, fallback, and health tracking.

    Uses ``AsyncOpenAI`` to avoid blocking the asyncio event loop during LLM calls.
    """

    def __init__(self, config: Config, timeout_s: int = 20):
        self._timeout_s = timeout_s
        self._health = HealthTracker()
        self._selector = WeightedSelector()
        self._providers: dict[str, ProviderConfig] = {
            provider.name: provider
            for provider in config.llm.providers
            if os.getenv(provider.api_key_env, "").strip()
        }
        self._default_weights: dict[str, int] = {
            name: provider.default_weight for name, provider in self._providers.items()
        }
        self._task_routing: dict[str, dict[str, int]] = {
            task: {
                name: weight
                for name, weight in weights.items()
                if name in self._providers
            }
            for task, weights in config.llm.task_routing.items()
        }
        self._budget = RequestBudget(
            run_limit=config.llm.max_requests_per_run,
            site_limit=config.llm.max_requests_per_site,
        )
        self._validate_routing()

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def _validate_routing(self):
        """Warn if task_routing references provider names not in the provider list."""
        known = set(self._providers.keys())
        for task, weights in self._task_routing.items():
            for name in weights:
                if name not in known:
                    logger.warning(
                        "LLM routing config: provider '%s' in task_routing['%s'] "
                        "is not defined in providers list. Available: %s",
                        name,
                        task,
                        sorted(known),
                    )

    # ── Public API (all async) ──

    async def ask(
        self,
        prompt: str,
        task_type: str = "default",
        max_tokens: int = 1024,
        site: str = "default",
    ) -> str:
        """Send *prompt* through weighted routing. Returns empty when all fail."""
        weights = self._task_routing.get(task_type, self._default_weights)
        tried: set[str] = set()
        for _ in range(len(self._providers)):
            name = self._selector.pick(weights, self._health.is_healthy, tried)
            if name is None:
                break
            tried.add(name)
            provider = self._providers.get(name)
            if provider is None:
                continue
            if not self._budget.charge(site):
                logger.warning("LLM request budget exhausted for site '%s'", site)
                break
            result = await self._try_provider(provider, prompt, max_tokens)
            if result is not None:
                self._health.record_success(name)
                return result
            self._health.record_failure(name)
        logger.error(
            "ALL LLM providers failed for task_type='%s'. Providers tried: %s. "
            "Check provider API keys and model names in config.yaml.",
            task_type,
            sorted(tried) if tried else "none",
        )
        return ""

    async def extract_links(
        self,
        markdown: str,
        *,
        site: str = "default",
    ) -> ExtractedLinks:
        """Extract subscription links from page markdown.

        Looks for downloadable subscription files (.txt/.yaml URLs),
        subscription service URLs, and inline protocol links (ss://, vmess://).
        """
        prompt = (
            "Extract ALL subscription links and node links from the following "
            "web page content.\n"
            "Categories:\n"
            '  "txt": URLs ending in .txt (V2Ray subscription files)\n'
            '  "yaml": URLs ending in .yaml (Clash subscription files)\n'
            '  "other": other subscription URLs (nodebuf.com, '
            "custom paths, etc.)\n"
            '  "inline": protocol links like vmess://, vless://, ss://, '
            "trojan://, ssr:// (one per line)\n"
            "Return JSON with ONLY the arrays that have items. "
            "Omit empty arrays.\n"
            "Return nothing except the JSON.\n\n"
            f"Content:\n{markdown[:8000]}"
        )
        text = await self.ask(
            prompt,
            task_type="extract_links",
            max_tokens=1024,
            site=site,
        )
        return self._parse_json(text)

    async def generate_pattern(
        self,
        known_links: list[str],
        html: str,
        *,
        site: str = "default",
    ) -> str | None:
        """Ask LLM to produce a regex matching the site's subscription links.

        Returns a regex string or None. Post-processes the response to
        extract the regex even when reasoning models add analysis text.
        """
        prompt = (
            f"Links found on the page:\n"
            f"{chr(10).join(known_links[:10])}\n\n"
            "Write a Python regex that matches these and similar URLs "
            "on the same site.\n"
            "RULES (obey strictly):\n"
            "- Output ONE LINE: the raw regex only\n"
            "- No numbers, no bullets, no analysis, no code fences\n"
            "- No explanations — not even a single word outside the regex\n"
            "Regex:"
        )
        text = await self.ask(
            prompt,
            task_type="generate_pattern",
            max_tokens=256,
            site=site,
        )
        if not text:
            return None
        return self._extract_regex(text)

    @staticmethod
    def verify_pattern(pattern: str, known_links: list[str], html: str) -> bool:
        """Three-layer verification: syntax, recall, precision."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return False
        matches = tuple(
            match.group(1) if compiled.groups else match.group(0)
            for match in compiled.finditer(html)
        )
        match_urls: set[str] = set()
        for match in matches:
            if match:
                match_urls.add(re.sub(r'[),;.\'"]+$', "", match))
        for link in known_links:
            clean = re.sub(r'[),;.\'"]+$', "", link)
            if clean not in match_urls:
                return False
        false_count = 0
        for match in matches:
            if not match or not match.startswith("http"):
                false_count += 1
            elif any(
                marker in match.lower()
                for marker in ("javascript:", "#", "xmlrpc", "favicon")
            ):
                false_count += 1
        if matches and false_count / len(matches) > 0.2:
            return False
        return True

    # ── Internals ──

    async def _try_provider(
        self, cfg: ProviderConfig, prompt: str, max_tokens: int
    ) -> str | None:
        api_key = os.getenv(cfg.api_key_env, "")
        if not api_key:
            logger.warning("No API key for %s (env: %s)", cfg.name, cfg.api_key_env)
            return None
        client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=api_key, timeout=self._timeout_s
        )
        for model in cfg.models:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                text = self._response_text(resp, cfg.is_reasoning_model)
                if text:
                    return text
            except RateLimitError:
                logger.warning(
                    "LLM rate limited [%s/%s]; not retrying", cfg.name, model
                )
                return None
            except Exception as error:
                logger.warning("LLM call failed [%s/%s]: %s", cfg.name, model, error)
                continue
        return None

    @staticmethod
    def _response_text(response: object, is_reasoning: bool) -> str:
        admitted = _ProviderResponse.model_validate(response)
        if not admitted.choices:
            return ""
        message = admitted.choices[0].message
        if message.content:
            return message.content
        if is_reasoning and message.reasoning:
            return message.reasoning
        return ""

    @staticmethod
    def _extract_regex(text: str) -> str | None:
        """Extract a regex pattern from LLM output that may contain analysis text.

        Tries:
          1. Content inside backtick code fences
          2. Last line containing regex metacharacters
        """
        # Strategy 1: content inside backtick fences
        m = re.search(r"```(?:regex)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return candidate

        # Strategy 2: last line containing regex metacharacters
        lines = text.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            has_meta = "\\" in line or "." in line or "*" in line or "+" in line
            has_dot_slash = "/" in line
            if has_meta and has_dot_slash:
                return line

        return None

    @staticmethod
    def _parse_json(raw: str) -> ExtractedLinks:
        """Parse LLM JSON response. Strips markdown fences if present."""
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return ExtractedLinks.model_validate_json(raw)
        except ValidationError:
            return ExtractedLinks()
