import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import main as cli
from freenodes.capability import CapabilityRunReceipt, ProbeDiagnostic
from freenodes.config import AppConfig
from freenodes.mihomo import AcquiredMihomo
from freenodes.publication import PublicationCounts, PublicationManifestV1

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


def test_help_exposes_publish_discover_validate_verify_and_audit_operations():
    result = run_cli("--help")

    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "publish all configured sites" in help_text
    assert "discover one site without publishing" in help_text
    assert "--validate-profiles" in help_text
    assert "--verify-public" in help_text
    assert "--audit-sources" in help_text


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ("--verify-public", "--validate-profiles", ".private/validation"),
            "not allowed with argument",
        ),
        (("source", "--verify-public"), "--verify-public does not accept a target"),
        (("source", "--audit-sources"), "--audit-sources does not accept a target"),
    ),
)
def test_incompatible_operations_are_rejected(arguments, message):
    result = run_cli(*arguments)
    assert result.returncode == 2
    assert message in result.stderr


async def test_publication_handoff_runs_without_discovery_configuration(
    monkeypatch, tmp_path, capsys
):
    repository = tmp_path / "repository"
    managed = repository / "nodes" / "merged.yaml"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"proxies: []\n")
    receipt = PublicationManifestV1(
        schema=1,
        status="accepted",
        created_at="2026-09-01T00:00:00+00:00",
        counts=PublicationCounts(published=1, clash=1, uri=1),
        files={"nodes/merged.yaml": hashlib.sha256(managed.read_bytes()).hexdigest()},
        managed_files=(
            "nodes/merged.yaml",
            "nodes/publication-receipt.json",
        ),
        removed_files=(),
    )
    receipt_path = repository / "nodes" / "publication-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    payload = tmp_path / "payload"
    pathspec = repository / ".git" / "publication-paths"
    monkeypatch.chdir(repository)

    def reject_config_load():
        raise AssertionError("publication handoff loaded discovery configuration")

    monkeypatch.setattr(cli, "load_config", reject_config_load)

    assert await cli.run(cli.PreparePublicationCommand(payload=payload)) == 0
    assert capsys.readouterr().out.strip() == receipt_sha
    assert (
        await cli.run(
            cli.ApplyPublicationCommand(
                payload=payload,
                receipt_sha256=receipt_sha,
                pathspec_output=pathspec,
            )
        )
        == 0
    )
    assert "Publication artifact no_change" in capsys.readouterr().out
    assert pathspec.with_name(f"{pathspec.name}-managed").read_text(
        encoding="utf-8"
    ).splitlines() == list(receipt.managed_files)
    assert (
        pathspec.with_name(f"{pathspec.name}-removed").read_text(encoding="utf-8") == ""
    )


async def test_source_audit_returns_nonzero_for_no_qualified_nodes(
    monkeypatch,
    capsys,
):
    receipt = CapabilityRunReceipt(
        status="inconclusive",
        diagnostic=ProbeDiagnostic(code="control_unavailable"),
    )

    async def audit_sources(application, **options):
        return receipt

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: AppConfig(sources=[]),
    )
    monkeypatch.setattr(cli.Application, "audit_sources", audit_sources)
    monkeypatch.setattr(
        cli,
        "acquire_pinned_mihomo",
        lambda path: AcquiredMihomo(
            executable=path / "mihomo",
            version="test",
            executable_sha256="0" * 64,
            asset_sha256="0" * 64,
        ),
    )

    assert await cli.run(cli.AuditSourcesCommand()) == 2
    assert '"status":"inconclusive"' in capsys.readouterr().out.replace(" ", "")
