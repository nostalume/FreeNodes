# FreeNodes

FreeNodes discovers public proxy subscriptions and publishes a bounded,
deduplicated catalog that passes deterministic freshness, syntax, endpoint-scope,
and Mihomo consumer checks. Live node choice runs in Clash Verge from the user's
network; runner-relative network measurements remain an optional audit.

## Simple import URLs

Use the direct URL first. jsDelivr is a fallback and may temporarily serve an older generation.

| Client / format | Direct URL | CDN fallback |
| --- | --- | --- |
| V2Ray base64 subscription | [v2ray.txt](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/v2ray.txt) | [v2ray.txt](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/v2ray.txt) |
| Clash Verge standalone profile (recommended) | [merged.yaml](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/merged.yaml) | [merged.yaml](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/merged.yaml) |
| Clash Verge provider profile (advanced) | [provider.yaml](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/provider.yaml) | [provider-cdn.yaml](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/provider-cdn.yaml) |
| Plain proxy URI list | [merged.txt](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/merged.txt) | [merged.txt](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/merged.txt) |

The standalone Clash profile embeds the accepted nodes and needs only one download. The provider profile is smaller, keeps sources separate, and refreshes nested provider files independently; those extra requests make it more sensitive to origin or CDN availability.

## Data flow

```text
configured sources
  -> immutable source artifacts
  -> deterministic freshness and typed node admission
  -> semantic deduplication and source-fair bounded selection
  -> V2Ray and Clash profiles from one selected catalog
  -> Mihomo consumer validation
  -> rollback-capable publication with receipt written last
```

Discovery uses OpenRouter’s `openrouter/free` route only when `OPENROUTER_API_KEY` is present. It is bounded to 30 requests per run and 3 per source. Missing credentials, rate limits, zero eligible results, and consumer rejection do not replace the previous accepted snapshot.
One source failure does not discard productive peers. `python main.py --audit-sources`
runs the retained Mihomo delay and bounded-transfer measurement as a read-only
diagnostic; its runner-relative results never authorize publication.

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

The `youtube` extra installs `yt-dlp`, which is required by configured YouTube-backed sources. Google Drive discovery uses the core HTTP dependency. The normal `uv run --locked --extra youtube python main.py` command performs deterministic discovery, admission, consumer validation, and local publication. Supplying a source name performs discovery and typed-admission diagnostics only; it does not change public files. `--verify-public` reads the published direct and CDN URLs, checks receipt digests, counts, schemas, and generation, and asks pinned Mihomo to consume both Clash forms without changing repository files.

Pushes and pull requests run the same locked formatting, lint, type, and test sequence used before scheduled publication. Publication preparation has no repository write permission; a separate job admits only receipt-owned paths and commits them, then a read-only job observes the public URLs. Failed checks, empty deterministic admission, consumer validation, receipt admission, commit, or direct remote observation stop that run. CDN lag or temporary CDN failure is reported without invalidating a current direct publication.

## Disclaimer

Public nodes are collected from the internet for learning and research. Availability, privacy, legality, and security are not guaranteed. Follow applicable law and do not send sensitive traffic through untrusted proxies.
