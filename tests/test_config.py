from pathlib import Path

import pytest
from pydantic import ValidationError

from freenodes.config import PasswordEvidence, load_config


def write_config(path: Path, sources: str, audit_sources: str = "[]") -> None:
    path.write_text(
        f"""
discovery: {{source_concurrency: 2, article_limit: 3, request_timeout_seconds: 30, proxy_url: null, artifact_limit_per_source: 12, byte_limit_per_source: 16777216, byte_limit_per_run: 67108864}}
openrouter: {{request_limit_per_run: 30, request_limit_per_source: 3, request_timeout_seconds: 20}}
publication: {{stale_after_hours: 24, expires_after_hours: 48, node_limit: 500}}
repository: {{owner: nostalume, name: FreeNodes}}
sources:
{sources}
audit_sources: {audit_sources}
""",
        encoding="utf-8",
    )


def test_complete_functional_schema_admits_and_old_key_rejects(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(
        path,
        '  - {kind: web, name: active, start_url: "https://example.test", resource_pattern: null, article_exclusions: [category, page-]}',
        "[{kind: github_file, name: reserve, owner: upstream, repository: subscriptions, branch: main, path: output/mihomo.yaml}]",
    )

    load_config(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("sources:", "sites:", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="sites"):
        load_config(path)


@pytest.mark.parametrize(
    ("source", "error"),
    (
        (
            '  - {kind: web, name: source, start_url: "https://example.test", unexpected: value}',
            "unexpected",
        ),
        (
            '  - {kind: unknown, name: source, start_url: "https://example.test"}',
            "kind",
        ),
    ),
)
def test_unknown_source_shape_rejects(tmp_path: Path, source: str, error: str):
    path = tmp_path / "config.yaml"
    write_config(path, source)
    with pytest.raises(ValidationError, match=error):
        load_config(path)


def test_password_policy_resolves_in_declared_order_and_bound(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(
        path,
        """  - kind: password_page
    name: source
    start_url: https://example.test
    password_policy:
      max_candidates: 3
      sources:
        - {type: subtitles, limit: 1}
        - {type: aabb, limit: 2}
    paste_policy:
      max_candidates: 1
      sources:
        - {type: empty}""",
    )

    candidates = (
        load_config(path)
        .sources[0]
        .password_policy.resolve(PasswordEvidence(subtitles="密码可能是3344或5566"))
    )

    assert candidates.values == ("3344", "0011", "0022")


@pytest.mark.parametrize("path_value", ("../nodes.yaml", "/nodes.yaml", "a\\b.yaml"))
def test_github_file_rejects_unsafe_path(tmp_path: Path, path_value: str):
    path = tmp_path / "config.yaml"
    write_config(
        path,
        "  []",
        f"[{{kind: github_file, name: reserve, owner: upstream, repository: subscriptions, branch: main, path: {path_value}}}]",
    )
    with pytest.raises(ValidationError, match="path"):
        load_config(path)


def test_source_identity_must_be_unique_across_active_and_audit(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(
        path,
        '  - {kind: web, name: duplicate, start_url: "https://example.test"}',
        "[{kind: github_file, name: duplicate, owner: upstream, repository: subscriptions, branch: main, path: nodes.yaml}]",
    )
    with pytest.raises(ValidationError, match="source names must be unique"):
        load_config(path)
