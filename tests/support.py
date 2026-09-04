import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from freenodes.application import Application
from freenodes.capability import (
    CapabilityRunReceipt,
    CapableCatalog,
    NodeCapabilityDecision,
)
from freenodes.config import AppConfig, DiscoveryLimits, WebSource
from freenodes.mihomo import ConsumerValidation, acquire_pinned_mihomo
from freenodes.nodes import (
    AdmissionSummary,
    PublishedDate,
    SourceArtifact,
    admit_artifacts,
)
from freenodes.profiles import OutputBundle
from freenodes.publication import publish_bundle as _publish_bundle


@pytest.fixture(scope="session")
def real_mihomo_path() -> Path:
    if os.getenv("FREENODES_REAL_MIHOMO") != "1":
        pytest.skip("real pinned Mihomo integration is opt-in")
    return acquire_pinned_mihomo(Path(".cache") / "mihomo").executable


def make_application(config: AppConfig, **dependencies: Any) -> Application:
    return Application(
        config.sources,
        config.discovery,
        candidate_sources=config.audit_sources,
        openrouter=config.openrouter,
        publication=config.publication,
        repository=config.repository,
        **dependencies,
    )


def capability_receipt(fingerprints):
    decisions = tuple(
        NodeCapabilityDecision(
            fingerprint=fingerprint,
            status="capable",
            successful_targets=("github", "google"),
            reason="quorum",
        )
        for fingerprint in fingerprints
    )
    return CapabilityRunReceipt(
        status="complete",
        planned=len(decisions),
        termination="candidates_exhausted",
        decisions=decisions,
        accepted_fingerprints=tuple(item.fingerprint for item in decisions),
    )


def capable_catalog(admitted):
    receipt = capability_receipt(node.fingerprint for node in admitted.clash_nodes)
    return CapableCatalog.from_measurement(admitted, receipt)


class CapableProbe:
    async def probe_capabilities(self, plan, targets, policy):
        return capability_receipt(entry.node.fingerprint for entry in plan.entries)


@dataclass
class ConsumerValidator:
    required_files: tuple[str, ...] = ("nodes/merged.yaml",)
    forbidden_files: tuple[str, ...] = ()
    fail: bool = False
    calls: list[Path] = field(default_factory=list)

    def validate_bundle(self, root: Path) -> ConsumerValidation:
        self.calls.append(root)
        assert all((root / name).read_bytes() for name in self.required_files)
        assert all(not (root / name).exists() for name in self.forbidden_files)
        if self.fail:
            raise RuntimeError("consumer rejected staging")
        return ConsumerValidation(
            profiles=("nodes/merged.yaml",),
            provider_profiles=(),
            provider_names=(),
            group_names=("select",),
        )


def web_sources_config(sites: tuple[str, ...] = ("a", "b")) -> AppConfig:
    return AppConfig(
        sources=[
            WebSource(name=name, start_url=f"https://{name}.test") for name in sites
        ],
        discovery=DiscoveryLimits(source_concurrency=2),
    )


def vless_artifact(site: str, observed_at: datetime) -> SourceArtifact:
    return SourceArtifact.inline(
        site=site,
        content=(
            "vless://11111111-1111-1111-1111-111111111111"
            f"@{site}.example:443?security=tls&type=tcp#{site}"
        ),
        observed_at=observed_at,
        published_on=observed_at.date(),
    )


def publication_admission_summary() -> AdmissionSummary:
    summary = sample_catalog(datetime(2026, 8, 29, tzinfo=UTC)).admitted.summary
    assert summary is not None
    return summary


def publish_bundle(*args: Any, **kwargs: Any):
    return _publish_bundle(
        *args,
        admission_summary=publication_admission_summary(),
        selection_limit=500,
        **kwargs,
    )


def sample_catalog(now: datetime):
    admitted = admit_artifacts(
        [
            SourceArtifact(
                site="source",
                source_url="https://secret.example/sub?token=must-not-leak",
                content="trojan://hidden@one.example:443#One",
                observed_at=now,
                publication_time=PublishedDate(on=now.date()),
                media_type="text/plain",
            )
        ],
        now=now,
    )
    return capable_catalog(admitted)


def bundle(version: str = "new") -> OutputBundle:
    return OutputBundle.from_files(
        {
            "nodes/merged.yaml": f"proxies: [{version}]\n".encode(),
            "nodes/merged.txt": f"uri://{version}\n".encode(),
            "nodes/v2ray.txt": b"encoded",
            "nodes/provider.yaml": b"proxy-providers: {}\n",
            "nodes/provider-cdn.yaml": b"proxy-providers: {}\n",
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
