# FreeNodes

FreeNodes discovers public proxy subscriptions, admits structurally valid nodes, tests reachability and HTTP delay with a pinned Mihomo core, and publishes only the bounded accepted snapshot.

## Simple import URLs

Use the direct URL first. jsDelivr is a fallback and may temporarily serve an older generation.

| Client / format | Direct URL | CDN fallback |
| --- | --- | --- |
| V2Ray base64 subscription | [v2ray.txt](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/v2ray.txt) | [v2ray.txt](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/v2ray.txt) |
| Clash Verge standalone profile (recommended) | [merged.yaml](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/merged.yaml) | [merged.yaml](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/merged.yaml) |
| Clash Verge provider profile (advanced) | [provider.yaml](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/provider.yaml) | [provider-cdn.yaml](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/provider-cdn.yaml) |
| Plain proxy URI list | [merged.txt](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/merged.txt) | [merged.txt](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/merged.txt) |

The standalone Clash profile embeds the accepted nodes and needs only one download. The provider profile is smaller, keeps sources separate, and refreshes nested provider files independently; those extra requests make it more sensitive to origin or CDN availability.

The current non-secret quality summary is [quality-manifest.json](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/quality-manifest.json). Delay reflects the GitHub runner’s network vantage, not guaranteed performance from every user location.

## Data flow

```text
configured sources
  -> immutable source artifacts
  -> typed, deduplicated node catalog
  -> bounded two-endpoint Mihomo probes
  -> deterministic delay and source quotas
  -> V2Ray and Clash profiles from one accepted catalog
  -> Mihomo consumer validation
  -> rollback-capable publication with receipt written last
```

Discovery uses OpenRouter’s `openrouter/free` route only when `OPENROUTER_API_KEY` is present. It is bounded to 30 requests per run and 3 per source. Missing credentials, rate limits, failed sources, empty quality results, and consumer rejection do not replace the previous accepted snapshot.

## Development

```bash
uv sync --locked --extra youtube
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked ty check
uv run --locked pytest -q
uv run --locked --extra youtube python main.py --validate-profiles .private/profile-validation
uv run --locked python main.py --verify-public
```

The `youtube` extra installs `yt-dlp`, which is required by configured YouTube-backed sources. Google Drive discovery uses the core HTTP dependency. The normal `uv run --locked --extra youtube python main.py` command performs discovery, quality probing, validation, and local publication. Supplying a source name performs discovery only and does not change public files. `--verify-public` reads the published direct and CDN URLs, checks their schemas and generation, and asks pinned Mihomo to consume both Clash forms without changing repository files.

Pushes and pull requests run the same locked formatting, lint, type, and test sequence used before scheduled publication. Publication preparation has no repository write permission; a separate job admits only receipt-owned paths and commits them, then a read-only job observes the public URLs. Failed checks, discovery, quality admission, consumer validation, receipt admission, commit, or direct remote observation stop that run. CDN lag or temporary CDN failure is reported without invalidating a current direct publication.

## Disclaimer

Public nodes are collected from the internet for learning and research. Availability, privacy, legality, and security are not guaranteed. Follow applicable law and do not send sensitive traffic through untrusted proxies.
