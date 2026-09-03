# Goal

Publish bounded, actively tested V2Ray and Clash Verge subscriptions through one typed dataflow, deterministic quality policy, consumer validation, and rollback-capable publication.

## Technical stack

- Python 3.12 with asyncio
- uv dependency and lock management
- Pydantic v2 boundary and domain models
- Crawl4AI, Playwright, yt-dlp, and httpx discovery adapters
- OpenRouter `openrouter/free` for bounded LLM fallback
- Pinned Mihomo for delay probing and Clash consumer validation
- PyYAML profile rendering
- Ruff formatting/lint, pytest, pytest-asyncio, and ty verification
