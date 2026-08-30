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
