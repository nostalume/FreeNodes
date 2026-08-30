"""Validated, rollback-capable promotion of one complete public snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
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
    freshness: Literal["current", "stale", "expired", "future"]


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


def _local(root: Path, relative: PurePosixPath) -> Path:
    destination = root.joinpath(*relative.parts).resolve()
    if root != destination and root not in destination.parents:
        raise PublicationError(f"unsafe publication path: {relative.as_posix()!r}")
    return destination


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
            _safe_relative(value)
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
    relative_files = {name: _safe_relative(name) for name in bundle.files}
    if "nodes/quality-manifest.json" not in bundle.files:
        raise PublicationError("publication requires a quality manifest")

    root = repository_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_relative = _safe_relative(RECEIPT_PATH)
    receipt_destination = _local(root, receipt_relative)
    old_managed = set(_existing_managed(receipt_destination))
    for value in previous_managed:
        old_managed.add(_safe_relative(value).as_posix())
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
        backup = staging / ".backup"
        backup.mkdir()
        for name, relative in relative_files.items():
            destination = _local(staging, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(bundle.files[name])
        staged_receipt = _local(staging, receipt_relative)
        staged_receipt.parent.mkdir(parents=True, exist_ok=True)
        staged_receipt.write_bytes(receipt_bytes)

        validator.validate_bundle(staging)
        unchanged = (
            not obsolete
            and receipt_destination.is_file()
            and all(
                _local(root, relative).is_file()
                and hashlib.sha256(_local(root, relative).read_bytes()).hexdigest()
                == digests[name]
                for name, relative in relative_files.items()
            )
        )
        if unchanged:
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

        targets = [*sorted(new_managed), *obsolete, RECEIPT_PATH]
        for name in dict.fromkeys(targets):
            relative = _safe_relative(name)
            current = _local(root, relative)
            if current.is_file():
                saved = _local(backup, relative)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, saved)

        touched: list[str] = []
        replacement_index = 0
        try:
            for name in sorted(new_managed):
                relative = relative_files[name]
                destination = _local(root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if before_replace:
                    before_replace(name, replacement_index)
                os.replace(_local(staging, relative), destination)
                touched.append(name)
                replacement_index += 1
            for name in obsolete:
                destination = _local(root, _safe_relative(name))
                if destination.is_file():
                    if before_replace:
                        before_replace(name, replacement_index)
                    destination.unlink()
                    touched.append(name)
                    replacement_index += 1
            if before_replace:
                before_replace(RECEIPT_PATH, replacement_index)
            receipt_destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_receipt, receipt_destination)
            touched.append(RECEIPT_PATH)
        except Exception as error:
            rollback_errors: list[Exception] = []
            for name in reversed(touched):
                relative = _safe_relative(name)
                destination = _local(root, relative)
                saved = _local(backup, relative)
                try:
                    if saved.is_file():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(saved, destination)
                    elif destination.is_file():
                        destination.unlink()
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise ExceptionGroup(
                    "publication failed and rollback was incomplete",
                    [error, *rollback_errors],
                )
            raise PublicationError(
                "publication promotion failed; previous snapshot restored"
            ) from error

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
