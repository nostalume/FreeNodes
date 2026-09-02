"""Consumer-validation contracts for the pinned Mihomo adapter."""

import hashlib
import importlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml


def mihomo_module():
    return importlib.import_module("src.mihomo")


def test_release_lock_resolves_supported_ci_platforms():
    mihomo = mihomo_module()
    release = mihomo.PinnedRelease.load()

    windows = release.resolve("Windows", "AMD64")
    linux = release.resolve("Linux", "x86_64")

    assert release.version == "v1.19.30"
    assert windows.name == "mihomo-windows-amd64-v1.19.30.zip"
    assert (
        windows.sha256
        == "22c09fd67673895ef7cd6b1820563918275c3d316f2462b306208675118db3c0"
    )
    assert linux.name == "mihomo-linux-amd64-v1.19.30.gz"
    assert len(linux.sha256) == 64


def test_checksum_mismatch_is_a_hard_failure(tmp_path):
    mihomo = mihomo_module()
    archive = tmp_path / "mihomo.zip"
    archive.write_bytes(b"tampered")

    with pytest.raises(mihomo.MihomoAcquisitionError, match="checksum"):
        mihomo.verify_sha256(archive, "0" * 64)

    assert hashlib.sha256(archive.read_bytes()).hexdigest() != "0" * 64


def test_validation_invokes_isolated_home_and_rejects_core_failure(
    monkeypatch, tmp_path
):
    mihomo = mihomo_module()
    config = tmp_path / "broken.yaml"
    config.write_text("proxies: [", encoding="utf-8")
    executable = tmp_path / "mihomo.exe"
    executable.write_bytes(b"fake")
    calls = []

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "parse error"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Failed()

    monkeypatch.setattr("src.mihomo.subprocess.run", fake_run)
    validator = mihomo.MihomoValidator(executable)

    with pytest.raises(mihomo.MihomoValidationError, match="configuration rejected"):
        validator.validate_config(config)

    command = calls[0][0]
    assert command[0] == str(executable)
    assert "-t" in command and "-d" in command and "-f" in command
    assert command[command.index("-f") + 1] == str(config.resolve())


def test_provider_rewrite_targets_loopback_without_file_providers(tmp_path):
    mihomo = mihomo_module()
    source = tmp_path / "provider.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "proxy-providers": {
                    "site": {
                        "type": "http",
                        "url": "https://raw.githubusercontent.com/o/r/HEAD/nodes/site.yaml",
                        "path": "./proxy_providers/site.yaml",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rewritten = mihomo.rewrite_provider_urls(source, "http://127.0.0.1:12345")
    provider = rewritten["proxy-providers"]["site"]

    assert provider["type"] == "http"
    assert provider["url"] == "http://127.0.0.1:12345/nodes/site.yaml"
    assert provider["path"] == "./proxy_providers/site.yaml"


def provider_profile(tmp_path):
    profile = tmp_path / "provider.yaml"
    profile.write_text(
        "proxy-providers:\n"
        "  site: {type: http, url: https://example.test/site}\n"
        "proxy-groups:\n"
        "  - {name: 🚀 Auto, type: url-test, use: [site]}\n",
        encoding="utf-8",
    )
    return profile


def provider_validator(monkeypatch, tmp_path, proxies, *, timeout=20):
    mihomo = mihomo_module()
    validator = mihomo.MihomoValidator(tmp_path / "mihomo", timeout=timeout)
    monkeypatch.setattr(validator, "validate_config", lambda *args: None)
    monkeypatch.setattr(validator, "_terminate", lambda process: None)
    monkeypatch.setattr(mihomo.subprocess, "Popen", lambda *args, **kwargs: object())

    def controller(_port, path):
        if path == "/providers/proxies":
            return {
                "providers": {
                    "site": {"vehicleType": "HTTP", "proxies": proxies},
                    "default": {"vehicleType": "Compatible", "proxies": []},
                }
            }
        return {"proxies": {"🚀 Auto": {}}}

    monkeypatch.setattr(validator, "_await_json", controller)
    return mihomo, validator


def test_remote_provider_rejects_name_only_inventory(monkeypatch, tmp_path):
    mihomo, validator = provider_validator(monkeypatch, tmp_path, [], timeout=0.01)

    with pytest.raises(mihomo.MihomoValidationError, match="empty.*site"):
        validator.smoke_remote_provider(provider_profile(tmp_path))


def test_remote_provider_returns_loaded_counts_and_ignores_builtins(
    monkeypatch, tmp_path
):
    _, validator = provider_validator(
        monkeypatch, tmp_path, [{"name": "one"}, {"name": "two"}]
    )

    receipt = validator.smoke_remote_provider(provider_profile(tmp_path))

    assert receipt.counts == (("site", 2),)
    assert receipt.total_nodes == 2
    assert receipt.groups == ("🚀 Auto",)


@pytest.mark.skipif(
    os.getenv("FREENODES_REAL_MIHOMO") != "1"
    or not Path(".cache/mihomo/mihomo.exe").is_file(),
    reason="real pinned Mihomo integration is opt-in",
)
def test_real_mihomo_loads_every_rendered_provider_node(tmp_path):
    from src.nodes import SourceArtifact, admit_artifacts
    from src.profiles import render_profiles

    now = datetime(2026, 9, 1, tzinfo=UTC)
    uri = "trojan://secret@real.example:443#US"
    artifact = SourceArtifact.inline(site="real", content=uri, observed_at=now)
    catalog = admit_artifacts([artifact], now=now)
    for name, content in render_profiles(catalog).files.items():
        output = tmp_path / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)

    receipt = (
        mihomo_module()
        .MihomoValidator(Path(".cache/mihomo/mihomo.exe"))
        .validate_bundle(tmp_path)
    )

    assert receipt.provider_names == ("real",)
