from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import load_config


def write_config(path: Path, site: str) -> None:
    path.write_text(
        f"""
crawl: {{}}
output:
  dir: nodes
llm: {{}}
sites:
  - name: source
    start_url: https://example.test
{site}
""",
        encoding="utf-8",
    )


def test_config_rejects_unknown_site_field(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(path, "    type: simple\n    unexpected: value")

    with pytest.raises(ValidationError, match="unexpected"):
        load_config(path)


def test_config_rejects_unknown_site_variant(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(path, "    type: unknown")

    with pytest.raises(ValidationError, match="type"):
        load_config(path)


def test_config_decodes_ordered_password_policy(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(
        path,
        """    type: yt_pwd
    password_policy:
      max_candidates: 3
      sources:
        - type: subtitles
          limit: 1
        - type: aabb
          limit: 2""",
    )

    config = load_config(path)
    site = config.sites[0]

    assert site.type == "yt_pwd"
    assert tuple(source.type for source in site.password_policy.sources) == (
        "subtitles",
        "aabb",
    )


def test_config_keeps_github_candidates_outside_active_sites(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
crawl: {}
output:
  dir: nodes
llm: {}
sites:
  - name: active
    start_url: https://example.test
    type: simple
source_candidates:
  - name: reserve
    owner: upstream
    repository: subscriptions
    branch: main
    path: output/mihomo.yaml
    type: github
""",
        encoding="utf-8",
    )

    config = load_config(path)
    candidate = config.source_candidates[0]

    assert tuple(site.name for site in config.sites) == ("active",)
    assert candidate.raw_url == (
        "https://raw.githubusercontent.com/upstream/subscriptions/"
        "main/output/mihomo.yaml"
    )
    assert candidate.commits_api_url == (
        "https://api.github.com/repos/upstream/subscriptions/commits"
    )


@pytest.mark.parametrize("path_value", ("../nodes.yaml", "/nodes.yaml", "a\\b.yaml"))
def test_config_rejects_unsafe_github_candidate_path(
    tmp_path: Path,
    path_value: str,
):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
crawl: {{}}
output:
  dir: nodes
llm: {{}}
sites: []
source_candidates:
  - name: reserve
    owner: upstream
    repository: subscriptions
    branch: main
    path: {path_value}
    type: github
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="path"):
        load_config(path)
