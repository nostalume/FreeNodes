"""CLI grammar and process-exit contracts."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "main.py", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def test_help_exposes_publish_discover_validate_and_verify_operations():
    result = run_cli("--help")

    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "publish all configured sites" in help_text
    assert "discover one site without publishing" in help_text
    assert "--validate-profiles" in help_text
    assert "--verify-public" in help_text


def test_validation_and_public_verification_are_mutually_exclusive():
    result = run_cli(
        "--verify-public",
        "--validate-profiles",
        ".private/validation",
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_public_verification_rejects_a_source_target():
    result = run_cli("source", "--verify-public")

    assert result.returncode == 2
    assert "--verify-public does not accept a target" in result.stderr
