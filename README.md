# FreeNodes

FreeNodes discovers public proxy subscriptions and publishes a bounded,
deduplicated catalog that passes deterministic freshness and syntax admission,
then proves real HTTPS service capability through Mihomo before publication.
Live node choice still runs in Clash Verge from the user's network.

## Simple import URLs

Use the direct URL first. jsDelivr is a best-effort fallback: independently cached files may be older or belong to different generations. Prefer the direct standalone profile; CDN freshness or consistency is not guaranteed.

| Client / format | Direct URL | CDN fallback |
| --- | --- | --- |
| V2Ray base64 subscription | [v2ray.txt](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/v2ray.txt) | [v2ray.txt](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/v2ray.txt) |
| Clash Verge standalone profile (recommended) | [merged.yaml](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/merged.yaml) | [merged.yaml](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/merged.yaml) |
| Clash Verge provider profile (advanced) | [provider.yaml](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/provider.yaml) | [provider-cdn.yaml](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/provider-cdn.yaml) |
| Plain proxy URI list | [merged.txt](https://raw.githubusercontent.com/nostalume/FreeNodes/HEAD/nodes/merged.txt) | [merged.txt](https://cdn.jsdelivr.net/gh/nostalume/FreeNodes/nodes/merged.txt) |

The standalone Clash profile embeds the accepted nodes and needs only one download. The provider profile is smaller, keeps sources separate, and refreshes nested provider files independently; those extra requests make it more sensitive to origin or CDN availability.

Publication runs are scheduled by GitHub Actions every four hours. This publication cadence is independent from client-side profile refresh: standalone profiles currently use a 600-second lazy `url-test`, while provider profiles refresh their nested files every 3600 seconds. A failed run preserves the last accepted snapshot; snapshots older than 12 hours are reported stale and older than 24 hours are reported expired.

The publication receipt also retains up to three completed, bounded runner-side quality generations. Previously capable nodes are preferred during bounded probing, while two consecutive completed failures quarantine a node. Inconclusive or failed publication runs do not replace that history, and the history is evidence for deterministic selection—not a claim of end-user uptime.

Node remarks use `region hint · protocol · source · identity`, for example
`US · VLESS · wzmwayne · A1B2C3D4`. Region hints come only from explicit source
labels and are not verified exit locations. Clash and URI/VMess subscriptions use
the same remark for the same semantic node.

For Clash Verge, import the standalone URL, choose **Rule** mode, then enable
System Proxy or TUN. `🚀 Auto` lazily chooses by delay; `🌍 Proxy` also permits
manual regions, individual nodes, and `DIRECT`. Private traffic is direct. The
Mainland China policy defaults to `DIRECT`, named AI/media/messaging traffic and
unmatched traffic default to `🌍 Proxy`, and each selector can be changed on the
Proxies page. Choices persist across subscription updates. China classification
uses the GEO data currently loaded by Clash Verge/Mihomo.

## Data flow

```text
configured sources
  -> immutable source artifacts
  -> deterministic freshness and typed node admission
  -> semantic deduplication and source-fair bounded capability planning
  -> controlled 2-of-3 HTTPS requests through Mihomo
  -> canonical V2Ray and Clash profiles from the capable catalog
  -> Mihomo syntax, group, and non-empty provider validation
  -> rollback-capable publication with receipt written last
```

Discovery uses OpenRouter’s `openrouter/free` route only when `OPENROUTER_API_KEY` is present. It is bounded to 30 requests per run and 3 per source. Missing credentials, rate limits, zero eligible results, inconclusive target controls, and consumer rejection do not replace the previous accepted snapshot.
GitHub commit metadata uses optional `GH_TOKEN`, supplied by the publishing job's read-only GitHub token. It is sent only to `api.github.com`, not to raw artifact downloads or other adapters. Without it, local discovery remains unauthenticated and can hit shared-IP rate limits; authenticated failures do not trigger an anonymous retry.
One source failure does not discard productive peers. Normal publication requires
runner-relative capability evidence; `python main.py --audit-sources` performs the
same bounded measurement without publishing.

## Configuration

| YAML key | Meaning and units | Authority and effect |
| --- | --- | --- |
| `sources` | Active web, password-page, YouTube-resource, or revision-pinned GitHub inputs. Web sources declare article exclusions and an optional resource regex; password policies declare ordered, bounded evidence sources. | Discovery, admission, capability testing, and publication. Source names and GitHub paths must be unique. |
| `audit_sources` | Reserve GitHub inputs with the same typed identity as active GitHub sources. | Read only by `--audit-sources`; never activated by normal discovery or publication. |
| `discovery` | Source concurrency and article/artifact counts; request timeout in seconds; byte ceilings per source/run. `proxy_url: null` means direct adapter traffic. | Bounds discovery effects. A non-null URL is the outbound proxy for web, YouTube, decryption, and Drive adapters—not a published node or capability result. |
| `openrouter` | Per-run/per-source request counts and timeout in seconds for `openrouter/free`. | Optional fallback only. `OPENROUTER_API_KEY` is the sole secret and remains in the environment/GitHub secret, never YAML. |
| `publication` | Stale and expiry windows in hours; maximum published node count; strict `rank_latency` boolean (default `false`, enabled in this repository). Expiry must exceed staleness. | Deterministic admission and output bound; incapable or inconclusive nodes cannot be published. Set `rank_latency: false` to restore first-capable selection without changing other gates. |
| `repository` | GitHub owner and repository name. | Constructs every direct/CDN import URL and publication identity. |

## Development

```bash
uv sync --locked --extra youtube
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked ty check
uv run --locked pytest -q
uv run --locked --extra youtube python main.py --validate-profiles .private/profile-validation
uv run --locked --extra youtube python main.py --validate-profiles .private/latency-pilot --rank-latency
uv run --locked --extra youtube python -m freenodes.latency_pilot
uv run --locked python main.py --verify-public
```

The `youtube` extra installs `yt-dlp`, which is required by configured YouTube-backed sources. Google Drive discovery uses the core HTTP dependency. The normal `uv run --locked --extra youtube python main.py` command performs deterministic discovery, admission, capability measurement, consumer validation, and local publication. Supplying a source name performs discovery and typed-admission diagnostics only; it does not change public files. `--verify-public` reads the published direct and CDN URLs, checks receipt digests, counts, schemas, and generation, and asks pinned Mihomo to consume both Clash forms without changing repository files.

Bounded latency ranking is enabled for publication by `publication.rank_latency: true`. It first probes until the normal capable-node limit is reached at a completed block, then tests at most one publication limit of additional candidates (currently 500). The existing 4000-candidate ceiling and 300-second probe budget remain unchanged. A completed extension selects up to 500 capable nodes using historical success and median successful-target delay within source/protocol round-robin, but only if coverage and source/protocol presence are preserved and median latency strictly improves. Otherwise, or on a deadline, it retains the exact first-capable selection; inconclusive controls reject the run. Preparation logs record selection/fallback reason, accepted/attempted counts, medians and elapsed measurement time; the public receipt retains selected capability evidence and bounded history. Consumer validation and rollback are unchanged. Checks still stop once a node passes the unchanged 2-of-3 quorum, so delay samples can cover different targets; these runner-relative timings are not throughput or universal service-access guarantees.

The CLI `--rank-latency` flag still applies only to private `--validate-profiles` runs, not publication. Its private receipt records shared observations and source/protocol counts as `[before, after]` without changing public files or history.

`python -m freenodes.latency_pilot` requires authenticated `gh` access and compares first-capable and ranked selections from one measurement of active GitHub sources; it does not activate reserves or publish subscriptions. Both reported durations and attempt counts describe that shared measurement, not an early-stop speed comparison. It consumer-validates both private bundles and requires coverage, latency, source/protocol presence and fresh verified direct entries, then writes only a redacted report to `.cache/latency-comparison/report.json` (which must not already exist). CDN lag or failure remains visible but does not block a valid direct-primary comparison. Degraded CDN observations identify the fetch/artifact/consumer boundary and error type without exposing profiles or raw exceptions. The runner uploads only that report. A failed required gate exits nonzero; a single runner comparison is not a general performance guarantee.

Pushes and pull requests run the same locked formatting, lint, type, and test sequence used before scheduled publication. Publication preparation has no repository write permission; a separate job admits only receipt-owned paths and commits them, then a read-only job observes the public URLs. Failed checks, empty deterministic admission, consumer validation, receipt admission, commit, or direct remote observation stop that run. CDN lag or temporary CDN failure is reported without invalidating a current direct publication.

## Disclaimer

Public nodes are collected from the internet for learning and research. Availability, privacy, legality, and security are not guaranteed. Follow applicable law and do not send sensitive traffic through untrusted proxies.
