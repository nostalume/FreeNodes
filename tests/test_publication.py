"""Atomic public snapshot promotion contracts."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.mihomo import ConsumerValidation
from src.nodes import SourceArtifact, admit_artifacts
from src.profiles import OutputBundle, render_profiles
from src.publication import (
    PublicationError,
    publish_bundle,
    validate_bundle_output_parent,
    write_bundle,
    write_validated_bundle,
)

NOW = datetime(2026, 8, 29, 6, 0, tzinfo=UTC)


class AcceptingValidator:
    def validate_bundle(self, output_dir):
        assert (output_dir / "nodes" / "merged.yaml").is_file()
        return ConsumerValidation(
            profiles=(
                "nodes/merged.yaml",
                "nodes/provider.yaml",
                "nodes/provider-cdn.yaml",
            ),
            provider_profiles=("nodes/provider.yaml", "nodes/provider-cdn.yaml"),
            provider_names=("source",),
            group_names=("auto", "manual"),
        )


def sample_catalog():
    return admit_artifacts(
        [
            SourceArtifact(
                site="source",
                source_url="https://secret.example/sub?token=must-not-leak",
                content=(
                    "proxies:\n  - {name: One, type: ss, server: one.example, "
                    "port: 8388, cipher: aes-128-gcm, password: hidden}\n"
                ),
                observed_at=NOW,
                media_type="application/yaml",
            )
        ],
        now=NOW,
    )


class Validator:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[Path] = []

    def validate_bundle(self, root: Path) -> ConsumerValidation:
        self.calls.append(root)
        assert (root / "nodes" / "merged.yaml").read_bytes()
        if self.fail:
            raise RuntimeError("consumer rejected staging")
        return ConsumerValidation(
            profiles=("nodes/merged.yaml",),
            provider_profiles=(),
            provider_names=(),
            group_names=("select",),
        )


def bundle(version: str = "new") -> OutputBundle:
    return OutputBundle.from_files(
        {
            "nodes/merged.yaml": f"proxies: [{version}]\n".encode(),
            "nodes/merged.txt": f"uri://{version}\n".encode(),
            "nodes/v2ray.txt": b"encoded",
            "nodes/provider.yaml": b"proxy-providers: {}\n",
            "nodes/provider-cdn.yaml": b"proxy-providers: {}\n",
            "nodes/quality-manifest.json": b'{"status":"quality_verified"}\n',
            "nodes/source.yaml": b"proxies: []\n",
        },
        accepted_count=1,
        clash_count=1,
        uri_count=1,
    )


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".freenodes-")
    }


def test_private_validation_is_exclusive_and_receipted_without_secrets(tmp_path):
    catalog = sample_catalog()
    profiles = render_profiles(catalog)
    public = tmp_path / "nodes"
    public.mkdir()
    sentinel = public / "merged.yaml"
    sentinel.write_text("current public profile", encoding="utf-8")

    receipt = write_validated_bundle(
        catalog=catalog,
        bundle=profiles,
        output_parent=tmp_path / "validation-output",
        validator=AcceptingValidator(),
        now=NOW,
    )

    assert sentinel.read_text(encoding="utf-8") == "current public profile"
    assert receipt.status == "consumer_validated"
    assert receipt.accepted_count == catalog.accepted_count
    raw_receipt = (receipt.output_dir / "validation-receipt.json").read_text(
        encoding="utf-8"
    )
    parsed = json.loads(raw_receipt)
    assert "must-not-leak" not in raw_receipt
    assert "hidden" not in raw_receipt
    assert (
        parsed["files"]["nodes/merged.yaml"]
        == hashlib.sha256(profiles.files["nodes/merged.yaml"]).hexdigest()
    )


def test_private_bundle_rejects_path_traversal_before_writing(tmp_path):
    unsafe = OutputBundle.from_files(
        {"../escape.yaml": b"bad"},
        accepted_count=0,
        clash_count=0,
        uri_count=0,
        aggregate_files=(),
    )

    with pytest.raises(PublicationError, match="unsafe"):
        write_bundle(unsafe, tmp_path)

    assert not (tmp_path.parent / "escape.yaml").exists()


def test_validation_output_cannot_be_public_or_nested_under_it(tmp_path):
    public = tmp_path / "nodes"

    with pytest.raises(PublicationError, match="outside"):
        validate_bundle_output_parent(public, public)
    with pytest.raises(PublicationError, match="outside"):
        validate_bundle_output_parent(public / ".validation", public)

    assert (
        validate_bundle_output_parent(tmp_path / "validation", public)
        == (tmp_path / "validation").resolve()
    )


def test_promotion_validates_staging_and_writes_receipt_last(tmp_path):
    validator = Validator()
    replacements: list[str] = []

    receipt = publish_bundle(
        bundle(),
        tmp_path,
        validator=validator,
        now=NOW,
        before_replace=lambda relative, index: replacements.append(relative),
    )

    assert receipt.status == "accepted"
    assert validator.calls and validator.calls[0] != tmp_path
    assert (tmp_path / "nodes" / "merged.yaml").read_text() == "proxies: [new]\n"
    assert replacements[-1] == "nodes/publication-receipt.json"
    persisted = json.loads((tmp_path / replacements[-1]).read_text())
    assert persisted["status"] == "accepted"
    assert persisted["managed_files"][-1] == "nodes/publication-receipt.json"


def test_failure_before_commit_restores_every_live_byte(tmp_path):
    publish_bundle(bundle("old"), tmp_path, validator=Validator(), now=NOW)
    before = snapshot(tmp_path)

    def fail_midway(relative: str, index: int):
        if index == 3:
            raise RuntimeError("injected replacement failure")

    with pytest.raises(PublicationError, match="promotion failed"):
        publish_bundle(
            bundle("new"),
            tmp_path,
            validator=Validator(),
            now=NOW,
            before_replace=fail_midway,
        )

    assert snapshot(tmp_path) == before


def test_only_explicit_previous_managed_files_are_removed(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "obsolete.yaml").write_text("old", encoding="utf-8")
    (nodes / "user-file.yaml").write_text("keep", encoding="utf-8")

    publish_bundle(
        bundle(),
        tmp_path,
        validator=Validator(),
        now=NOW,
        previous_managed=("nodes/obsolete.yaml",),
    )

    assert not (nodes / "obsolete.yaml").exists()
    assert (nodes / "user-file.yaml").read_text() == "keep"


def test_validation_failure_and_empty_bundle_leave_public_tree_unchanged(tmp_path):
    (tmp_path / "sentinel").write_bytes(b"unchanged")
    before = snapshot(tmp_path)

    with pytest.raises(RuntimeError, match="consumer rejected"):
        publish_bundle(bundle(), tmp_path, validator=Validator(fail=True), now=NOW)
    with pytest.raises(PublicationError, match="non-empty"):
        publish_bundle(
            OutputBundle.from_files(
                {}, accepted_count=0, clash_count=0, uri_count=0, aggregate_files=()
            ),
            tmp_path,
            validator=Validator(),
            now=NOW,
        )

    assert snapshot(tmp_path) == before


def test_unsafe_bundle_path_is_rejected_before_writing(tmp_path):
    unsafe = OutputBundle.from_files(
        {"../escape": b"bad"},
        accepted_count=1,
        clash_count=1,
        uri_count=1,
        aggregate_files=(),
    )

    with pytest.raises(PublicationError, match="unsafe"):
        publish_bundle(unsafe, tmp_path, validator=Validator(), now=NOW)

    assert not (tmp_path.parent / "escape").exists()


def test_partial_existing_receipt_cannot_authorize_deletion(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    obsolete = nodes / "obsolete.yaml"
    obsolete.write_bytes(b"keep")
    (nodes / "publication-receipt.json").write_text(
        json.dumps({"managed_files": ["nodes/obsolete.yaml"]}),
        encoding="utf-8",
    )

    with pytest.raises(PublicationError, match="existing publication receipt"):
        publish_bundle(bundle(), tmp_path, validator=Validator(), now=NOW)

    assert obsolete.read_bytes() == b"keep"
