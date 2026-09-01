"""Validated, rollback-capable promotion of one complete public snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol

from pydantic import (
    AwareDatetime,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config import FrozenModel
from src.mihomo import ConsumerValidation
from src.nodes import NodeCatalog
from src.profiles import OutputBundle
from src.quality_manifest import QualityManifestV3

RECEIPT_PATH = "nodes/publication-receipt.json"


class PublicationError(RuntimeError):
    pass


class PublicationCounts(FrozenModel):
    published: int = Field(gt=0, strict=True)
    clash: int = Field(gt=0, strict=True)
    uri: int = Field(gt=0, strict=True)


class PublicationManifestV1(FrozenModel):
    schema_version: Literal[1] = Field(alias="schema")
    status: Literal["accepted"]
    created_at: str = Field(min_length=1)
    counts: PublicationCounts
    files: dict[str, str]
    managed_files: tuple[str, ...] = Field(strict=False)
    removed_files: tuple[str, ...] = Field(strict=False)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        TypeAdapter(AwareDatetime).validate_python(value)
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> "PublicationManifestV1":
        if not self.managed_files or self.managed_files[-1] != RECEIPT_PATH:
            raise ValueError("publication receipt must be the final managed file")
        if len(set(self.managed_files)) != len(self.managed_files):
            raise ValueError("duplicate managed file")
        if set(self.files) != set(self.managed_files) - {RECEIPT_PATH}:
            raise ValueError("file digests do not match managed files")
        if set(self.removed_files) & set(self.managed_files):
            raise ValueError("removed and managed files overlap")
        hexadecimal = set("0123456789abcdef")
        if any(
            len(digest) != 64 or not set(digest).issubset(hexadecimal)
            for digest in self.files.values()
        ):
            raise ValueError("invalid file digest")
        return self


class PublishedFile(FrozenModel):
    path: str
    digest: str = Field(min_length=64, max_length=64, repr=False)


class PublicationReceipt(FrozenModel):
    status: Literal["accepted", "no_change"]
    created_at: AwareDatetime
    managed_files: tuple[str, ...] = Field(strict=False)
    file_digests: tuple[PublishedFile, ...] = Field(strict=False, repr=False)
    removed_files: tuple[str, ...] = Field(default=(), strict=False)

    @property
    def files(self):
        return MappingProxyType({item.path: item.digest for item in self.file_digests})


class PreparedPublication(FrozenModel):
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    managed_files: tuple[str, ...] = Field(strict=False)
    removed_files: tuple[str, ...] = Field(default=(), strict=False)


class PublicationPathspecs(FrozenModel):
    managed: Path = Field(strict=False)
    removed: Path = Field(strict=False)

    @classmethod
    def contained(
        cls,
        repository: Path,
        output: Path | None,
    ) -> "PublicationPathspecs | None":
        if output is None:
            return None
        base = output.resolve()
        try:
            relative = base.relative_to(repository)
        except ValueError as error:
            raise PublicationError(
                "publication pathspec must stay in the repository"
            ) from error
        if len(relative.parts) < 2 or relative.parts[0] != ".git":
            raise PublicationError("publication pathspec must stay in .git")
        base.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            managed=base.with_name(f"{base.name}-managed"),
            removed=base.with_name(f"{base.name}-removed"),
        )

    def temporary(self) -> "PublicationPathspecs":
        nonce = uuid.uuid4().hex
        return PublicationPathspecs(
            managed=self.managed.with_name(f".{self.managed.name}-{nonce}.tmp"),
            removed=self.removed.with_name(f".{self.removed.name}-{nonce}.tmp"),
        )

    def write(self, receipt: PublicationManifestV1) -> None:
        for path, values in (
            (self.managed, receipt.managed_files),
            (self.removed, receipt.removed_files),
        ):
            with path.open("x", encoding="utf-8", newline="\n") as output:
                output.writelines(f"{value}\n" for value in sorted(values))

    def promote_to(self, destination: "PublicationPathspecs") -> None:
        os.replace(self.managed, destination.managed)
        os.replace(self.removed, destination.removed)

    def cleanup(self) -> None:
        self.managed.unlink(missing_ok=True)
        self.removed.unlink(missing_ok=True)


class AppliedPublication(FrozenModel):
    status: Literal["applied", "no_change"]
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    managed_files: tuple[str, ...] = Field(strict=False)
    removed_files: tuple[str, ...] = Field(default=(), strict=False)
    pathspecs: PublicationPathspecs | None = None


class PublicationArtifact(FrozenModel):
    root: Path = Field(strict=False)
    receipt: PublicationManifestV1
    receipt_bytes: bytes = Field(repr=False)

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_bytes).hexdigest()

    @classmethod
    def admit(
        cls,
        root: Path,
        *,
        exact_inventory: bool = False,
        expected_receipt_sha256: str | None = None,
    ) -> "PublicationArtifact":
        admitted_root = root.resolve()
        receipt_path = _local(admitted_root, _publication_relative(RECEIPT_PATH))
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = PublicationManifestV1.model_validate_json(receipt_bytes)
            for value in (*receipt.managed_files, *receipt.removed_files):
                _publication_relative(value)
            for value in receipt.managed_files:
                relative = _publication_relative(value)
                source = _local(admitted_root, relative)
                if not source.is_file() or source.is_symlink():
                    raise PublicationError(f"publication file is missing: {value}")
                if value != RECEIPT_PATH:
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    if digest != receipt.files[value]:
                        raise PublicationError(f"publication digest mismatch: {value}")
            for value in receipt.removed_files:
                if _local(admitted_root, _publication_relative(value)).exists():
                    raise PublicationError(
                        f"removed publication file is still present: {value}"
                    )
        except PublicationError:
            raise
        except (OSError, ValueError, ValidationError) as error:
            raise PublicationError("publication artifact is invalid") from error
        artifact = cls(
            root=admitted_root,
            receipt=receipt,
            receipt_bytes=receipt_bytes,
        )
        if (
            expected_receipt_sha256 is not None
            and artifact.receipt_sha256 != expected_receipt_sha256
        ):
            raise PublicationError("publication receipt digest mismatch")
        if exact_inventory and _file_inventory(admitted_root) != set(
            receipt.managed_files
        ):
            raise PublicationError("publication artifact inventory mismatch")
        return artifact

    def stage_to(self, destination: Path) -> PreparedPublication:
        payload = destination.resolve()
        try:
            payload.mkdir(parents=True, exist_ok=False)
            for value in self.receipt.managed_files:
                relative = _publication_relative(value)
                target = _local(payload, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_local(self.root, relative), target)
            PublicationArtifact.admit(
                payload,
                exact_inventory=True,
                expected_receipt_sha256=self.receipt_sha256,
            )
        except Exception:
            if payload.is_dir():
                shutil.rmtree(payload)
            raise
        return PreparedPublication(
            receipt_sha256=self.receipt_sha256,
            managed_files=self.receipt.managed_files,
            removed_files=self.receipt.removed_files,
        )

    def apply_to(
        self,
        repository_root: Path,
        *,
        pathspec_output: Path | None = None,
        before_replace: BeforeReplace | None = None,
    ) -> AppliedPublication:
        repository = repository_root.resolve()
        repository.mkdir(parents=True, exist_ok=True)
        pathspecs = PublicationPathspecs.contained(repository, pathspec_output)
        temporary_pathspecs = pathspecs.temporary() if pathspecs else None
        if temporary_pathspecs:
            temporary_pathspecs.write(self.receipt)
        try:
            if self._matches(repository):
                if pathspecs and temporary_pathspecs:
                    temporary_pathspecs.promote_to(pathspecs)
                return self._applied("no_change", pathspecs)
            self._replace(
                repository,
                pathspecs,
                temporary_pathspecs,
                before_replace,
            )
            return self._applied("applied", pathspecs)
        finally:
            if temporary_pathspecs:
                temporary_pathspecs.cleanup()

    def _matches(self, repository: Path) -> bool:
        for value in self.receipt.managed_files:
            relative = _publication_relative(value)
            current = _local(repository, relative)
            if not current.is_file() or current.is_symlink():
                return False
            if value == RECEIPT_PATH:
                if current.read_bytes() != self.receipt_bytes:
                    return False
            elif (
                hashlib.sha256(current.read_bytes()).hexdigest()
                != self.receipt.files[value]
            ):
                return False
        return all(
            not _local(repository, _publication_relative(value)).exists()
            for value in self.receipt.removed_files
        )

    def content_matches(self, repository: Path) -> bool:
        return not self.receipt.removed_files and all(
            (current := _local(repository, _publication_relative(value))).is_file()
            and not current.is_symlink()
            and hashlib.sha256(current.read_bytes()).hexdigest()
            == self.receipt.files[value]
            for value in self.receipt.managed_files[:-1]
        )

    def _replace(
        self,
        repository: Path,
        pathspecs: PublicationPathspecs | None,
        temporary_pathspecs: PublicationPathspecs | None,
        before_replace: BeforeReplace | None,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".freenodes-apply-", dir=repository
        ) as temporary:
            staging = Path(temporary).resolve()
            backup = staging / ".backup"
            backup.mkdir()
            for value in self.receipt.managed_files:
                relative = _publication_relative(value)
                staged = _local(staging, relative)
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_local(self.root, relative), staged)
            targets = (*self.receipt.managed_files, *self.receipt.removed_files)
            for value in dict.fromkeys(targets):
                relative = _publication_relative(value)
                current = _local(repository, relative)
                if current.exists() and (not current.is_file() or current.is_symlink()):
                    raise PublicationError(f"publication target is not a file: {value}")
                if current.is_file():
                    saved = _local(backup, relative)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(current, saved)
            touched: list[str] = []
            try:
                replacement_index = 0
                for value in self.receipt.managed_files[:-1]:
                    relative = _publication_relative(value)
                    destination = _local(repository, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if before_replace:
                        before_replace(value, replacement_index)
                    os.replace(_local(staging, relative), destination)
                    touched.append(value)
                    replacement_index += 1
                for value in self.receipt.removed_files:
                    destination = _local(repository, _publication_relative(value))
                    if destination.is_file():
                        if before_replace:
                            before_replace(value, replacement_index)
                        destination.unlink()
                        touched.append(value)
                        replacement_index += 1
                receipt_relative = _publication_relative(RECEIPT_PATH)
                receipt_destination = _local(repository, receipt_relative)
                receipt_destination.parent.mkdir(parents=True, exist_ok=True)
                if before_replace:
                    before_replace(RECEIPT_PATH, replacement_index)
                os.replace(_local(staging, receipt_relative), receipt_destination)
                touched.append(RECEIPT_PATH)
                if pathspecs and temporary_pathspecs:
                    temporary_pathspecs.promote_to(pathspecs)
            except Exception as error:
                rollback_errors = self._restore(repository, backup, touched)
                if rollback_errors:
                    raise ExceptionGroup(
                        "publication apply failed and rollback was incomplete",
                        [error, *rollback_errors],
                    )
                raise PublicationError(
                    "publication apply failed; previous snapshot restored"
                ) from error

    @staticmethod
    def _restore(
        repository: Path,
        backup: Path,
        touched: Sequence[str],
    ) -> list[Exception]:
        errors: list[Exception] = []
        for value in reversed(touched):
            relative = _publication_relative(value)
            destination = _local(repository, relative)
            saved = _local(backup, relative)
            try:
                if saved.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(saved, destination)
                elif destination.is_file():
                    destination.unlink()
            except Exception as error:
                errors.append(error)
        return errors

    def _applied(
        self,
        status: Literal["applied", "no_change"],
        pathspecs: PublicationPathspecs | None,
    ) -> AppliedPublication:
        return AppliedPublication(
            status=status,
            receipt_sha256=self.receipt_sha256,
            managed_files=self.receipt.managed_files,
            removed_files=self.receipt.removed_files,
            pathspecs=pathspecs,
        )


class BundleValidator(Protocol):
    def validate_bundle(self, output_dir: Path, /) -> ConsumerValidation: ...


class WrittenBundle(FrozenModel):
    output_dir: Path
    file_digests: tuple[PublishedFile, ...] = Field(strict=False, repr=False)

    @property
    def files(self):
        return MappingProxyType({item.path: item.digest for item in self.file_digests})


class ValidationReceipt(FrozenModel):
    status: Literal["consumer_validated"]
    output_dir: Path
    created_at: AwareDatetime
    accepted_count: int = Field(ge=0, strict=True)
    clash_count: int = Field(ge=0, strict=True)
    uri_count: int = Field(ge=0, strict=True)
    rejected_count: int = Field(ge=0, strict=True)
    file_digests: tuple[PublishedFile, ...] = Field(strict=False, repr=False)
    validated_profiles: tuple[str, ...] = Field(default=(), strict=False)
    provider_names: tuple[str, ...] = Field(default=(), strict=False)
    group_names: tuple[str, ...] = Field(default=(), strict=False)

    @property
    def files(self):
        return MappingProxyType({item.path: item.digest for item in self.file_digests})


class ValidationCounts(FrozenModel):
    accepted: int = Field(ge=0, strict=True)
    rejected: int = Field(ge=0, strict=True)
    clash: int = Field(ge=0, strict=True)
    uri: int = Field(ge=0, strict=True)


class ValidationConsumer(FrozenModel):
    profiles: tuple[str, ...] = Field(strict=False)
    providers: tuple[str, ...] = Field(strict=False)
    groups: tuple[str, ...] = Field(strict=False)


class ValidationSource(FrozenModel):
    site: str
    source_url_sha256: str = Field(min_length=64, max_length=64)
    artifact_sha256: str = Field(min_length=64, max_length=64)
    observed_at: str
    published_on: date | None = None
    freshness: Literal["current", "stale", "expired", "future", "unknown"]


class ValidationManifestV1(FrozenModel):
    schema_version: Literal[1] = Field(alias="schema")
    status: Literal["consumer_validated"]
    created_at: str
    counts: ValidationCounts
    files: dict[str, str]
    consumer_validation: ValidationConsumer
    sources: tuple[ValidationSource, ...] = Field(strict=False)


BeforeReplace = Callable[[str, int], None]


def _safe_relative(value: str) -> PurePosixPath:
    if "\\" in value:
        raise PublicationError(f"unsafe publication path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicationError(f"unsafe publication path: {value!r}")
    return path


def _publication_relative(value: str) -> PurePosixPath:
    if any(character in value for character in ("\0", "\r", "\n")):
        raise PublicationError(f"unsafe publication path: {value!r}")
    path = _safe_relative(value)
    if value != "IMPORT.md" and (len(path.parts) < 2 or path.parts[0] != "nodes"):
        raise PublicationError(f"unsafe publication path: {value!r}")
    return path


def _local(root: Path, relative: PurePosixPath) -> Path:
    destination = root.joinpath(*relative.parts).resolve()
    if root != destination and root not in destination.parents:
        raise PublicationError(f"unsafe publication path: {relative.as_posix()!r}")
    return destination


def _file_inventory(root: Path) -> set[str]:
    inventory: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PublicationError("publication artifact contains a symbolic link")
        if path.is_file():
            inventory.add(path.relative_to(root).as_posix())
    return inventory


def prepare_publication(
    repository_root: Path,
    payload_root: Path,
) -> PreparedPublication:
    """Copy exactly one admitted receipt-owned snapshot into a new payload."""
    return PublicationArtifact.admit(repository_root).stage_to(payload_root)


def apply_publication(
    payload_root: Path,
    repository_root: Path,
    *,
    expected_receipt_sha256: str,
    pathspec_output: Path,
    before_replace: BeforeReplace | None = None,
) -> AppliedPublication:
    """Admit and apply one exact artifact, then emit bounded Git pathspecs."""
    artifact = PublicationArtifact.admit(
        payload_root,
        exact_inventory=True,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return artifact.apply_to(
        repository_root,
        pathspec_output=pathspec_output,
        before_replace=before_replace,
    )


def render_publication_report(repository_root: Path) -> str:
    """Render one admitted, redacted quality summary for workflow output."""
    manifest_path = repository_root.resolve() / "nodes" / "quality-manifest.json"
    if not manifest_path.is_file():
        return "## Publication preparation\n- Rejected before a quality manifest was produced\n"
    try:
        manifest = QualityManifestV3.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValueError, ValidationError):
        return "## Publication preparation\n- Quality manifest is invalid\n"
    lines = [
        "## Publication preparation",
        (
            f"- Decision: {manifest.decision}; published "
            f"{manifest.counts.published} of {manifest.counts.admitted} admitted nodes"
        ),
        (
            f"- Node evidence: passed {manifest.counts.probe_success}, "
            f"failed {manifest.counts.failed}, "
            f"inconclusive {manifest.counts.inconclusive}, "
            f"not probed {manifest.counts.not_probed}"
        ),
        "- Contributing authorities: " + ", ".join(manifest.contributing_authorities),
    ]
    lines.extend(
        f"- Transfer target `{target.authority}`: controls "
        f"{target.controls_passed}/{target.controls_attempted}, "
        f"candidate attempts {target.candidate_attempts}"
        for target in manifest.transfer_targets
    )
    lines.extend(
        f"- Probe failure `{code}`: {count}"
        for code, count in manifest.probe_failures.items()
    )
    lines.extend(
        f"- Source `{source.name}`: {source.status} ({source.reason}); "
        f"qualified {source.observation.qualified}/{source.observation.sampled}"
        for source in manifest.sources
    )
    return "\n".join(lines) + "\n"


def validate_bundle_output_parent(output_parent: Path, public_dir: Path) -> Path:
    """Reject validation output located inside the public profile tree."""
    private = output_parent.resolve()
    public = public_dir.resolve()
    if private == public or public in private.parents:
        raise PublicationError(
            "validation output must be outside the public nodes directory"
        )
    return private


def write_bundle(
    bundle: OutputBundle,
    output_parent: Path,
    *,
    now: datetime | None = None,
) -> WrittenBundle:
    """Write a new exclusive private bundle; never overwrite an old run."""
    safe_files = [
        (name, _safe_relative(name), content) for name, content in bundle.files.items()
    ]
    observed_now = now or datetime.now(UTC)
    timestamp = observed_now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    parent = output_parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output_dir = parent / f"validation-{timestamp}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(exist_ok=False)

    digests: dict[str, str] = {}
    for name, relative, content in safe_files:
        destination = _local(output_dir, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            output.write(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    return WrittenBundle(
        output_dir=output_dir,
        file_digests=tuple(
            PublishedFile(path=name, digest=digest)
            for name, digest in sorted(digests.items())
        ),
    )


def _source_summary(catalog: NodeCatalog) -> tuple[ValidationSource, ...]:
    return tuple(
        ValidationSource(
            site=receipt.site,
            source_url_sha256=hashlib.sha256(
                receipt.source_url.encode("utf-8")
            ).hexdigest(),
            artifact_sha256=receipt.artifact_digest,
            observed_at=receipt.observed_at.astimezone(UTC).isoformat(),
            published_on=receipt.published_on,
            freshness=receipt.freshness,
        )
        for receipt in catalog.receipts
    )


def write_validated_bundle(
    *,
    catalog: NodeCatalog,
    bundle: OutputBundle,
    output_parent: Path,
    validator: BundleValidator,
    now: datetime | None = None,
) -> ValidationReceipt:
    """Write, validate with a consumer, then seal a redacted receipt."""
    observed_now = now or datetime.now(UTC)
    written = write_bundle(bundle, output_parent, now=observed_now)
    validation = validator.validate_bundle(written.output_dir)
    receipt = ValidationReceipt(
        status="consumer_validated",
        output_dir=written.output_dir,
        created_at=observed_now,
        accepted_count=catalog.accepted_count,
        clash_count=bundle.clash_count,
        uri_count=bundle.uri_count,
        rejected_count=catalog.rejected_count,
        file_digests=written.file_digests,
        validated_profiles=validation.profiles,
        provider_names=validation.provider_names,
        group_names=validation.group_names,
    )
    manifest = ValidationManifestV1(
        schema=1,
        status=receipt.status,
        created_at=observed_now.astimezone(UTC).isoformat(),
        counts=ValidationCounts(
            accepted=receipt.accepted_count,
            rejected=receipt.rejected_count,
            clash=receipt.clash_count,
            uri=receipt.uri_count,
        ),
        files=dict(receipt.files),
        consumer_validation=ValidationConsumer(
            profiles=receipt.validated_profiles,
            providers=receipt.provider_names,
            groups=receipt.group_names,
        ),
        sources=_source_summary(catalog),
    )
    receipt_path = written.output_dir / "validation-receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(
            manifest.model_dump(mode="json", by_alias=True),
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")
    return receipt


def _existing_managed(receipt_path: Path) -> tuple[str, ...]:
    if not receipt_path.is_file():
        return ()
    try:
        receipt = PublicationManifestV1.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
        for value in receipt.managed_files:
            _publication_relative(value)
        return receipt.managed_files
    except (OSError, ValueError, ValidationError) as error:
        raise PublicationError("existing publication receipt is invalid") from error


def publish_bundle(
    bundle: OutputBundle,
    repository_root: Path,
    *,
    validator: BundleValidator,
    now: datetime | None = None,
    previous_managed: Sequence[str] = (),
    before_replace: BeforeReplace | None = None,
) -> PublicationReceipt:
    """Validate staging, replace managed files, and publish the receipt last."""
    if min(bundle.accepted_count, bundle.clash_count, bundle.uri_count) <= 0:
        raise PublicationError(
            "publication requires non-empty V2Ray and Clash profiles"
        )
    relative_files = {name: _publication_relative(name) for name in bundle.files}
    if "nodes/quality-manifest.json" not in bundle.files:
        raise PublicationError("publication requires a quality manifest")

    root = repository_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_relative = _publication_relative(RECEIPT_PATH)
    receipt_destination = _local(root, receipt_relative)
    old_managed = set(_existing_managed(receipt_destination))
    for value in previous_managed:
        old_managed.add(_publication_relative(value).as_posix())
    old_managed.discard(RECEIPT_PATH)
    new_managed = set(relative_files)
    obsolete = sorted(old_managed - new_managed)
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    digests = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(bundle.files.items())
    }
    managed_files = tuple([*sorted(new_managed), RECEIPT_PATH])
    receipt_manifest = PublicationManifestV1(
        schema=1,
        status="accepted",
        created_at=observed_at.astimezone(UTC).isoformat(),
        counts=PublicationCounts(
            published=bundle.accepted_count,
            clash=bundle.clash_count,
            uri=bundle.uri_count,
        ),
        files=digests,
        managed_files=managed_files,
        removed_files=tuple(obsolete),
    )
    receipt_bytes = (
        json.dumps(
            receipt_manifest.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    with tempfile.TemporaryDirectory(
        prefix=".freenodes-publish-", dir=root
    ) as temporary:
        staging = Path(temporary).resolve()
        for name, relative in relative_files.items():
            destination = _local(staging, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(bundle.files[name])
        staged_receipt = _local(staging, receipt_relative)
        staged_receipt.parent.mkdir(parents=True, exist_ok=True)
        staged_receipt.write_bytes(receipt_bytes)

        validator.validate_bundle(staging)
        artifact = PublicationArtifact.admit(staging, exact_inventory=True)
        if artifact.content_matches(root):
            return PublicationReceipt(
                status="no_change",
                created_at=observed_at,
                managed_files=managed_files,
                removed_files=(),
                file_digests=tuple(
                    PublishedFile(path=name, digest=digest)
                    for name, digest in sorted(digests.items())
                ),
            )
        artifact.apply_to(root, before_replace=before_replace)

    return PublicationReceipt(
        status="accepted",
        created_at=observed_at,
        managed_files=managed_files,
        removed_files=tuple(obsolete),
        file_digests=tuple(
            PublishedFile(path=name, digest=digest)
            for name, digest in sorted(digests.items())
        ),
    )
