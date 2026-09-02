"""Service-capability publication contract.

These tests intentionally lead the implementation.  They use only controlled,
in-memory observations; no public proxy or target traffic occurs here.
"""

from datetime import UTC, datetime

import pytest

import src.quality as quality
from src.config import Config, LLMConfig, SimpleSite
from src.mihomo import ConsumerValidation
from src.nodes import (
    ClashNode,
    DualNode,
    NodeCatalog,
    NodeProvenance,
    PublishedDate,
    SourceArtifact,
    UriNode,
    admit_proxy,
)
from src.publication import PublicationError
from src.quality import CapabilityRunReceipt, ProbeDiagnostic, QualityPolicy
from src.scheduler import Scheduler
from src.site_processor import DiscoverySuccess

NOW = datetime(2026, 9, 2, tzinfo=UTC)
FINGERPRINT = "a" * 64
TARGET_IDS = ("github", "google", "cloudflare")


def control_windows(*, invalid: tuple[str, ...] = (), closing: bool = False):
    return tuple(
        quality.TargetControlWindow(
            target_id=target_id,
            opening="failure" if target_id in invalid and not closing else "success",
            closing="failure" if target_id in invalid and closing else "success",
        )
        for target_id in TARGET_IDS
    )


def observations(*values: tuple[str, str]):
    items = []
    for target_id, status in values:
        diagnostic = None
        delay_ms = 80 if status == "success" else None
        if status != "success":
            diagnostic = ProbeDiagnostic(
                code="request_timeout" if status == "timeout" else "controller_error"
            )
        items.append(
            quality.TargetObservation(
                target_id=target_id,
                status=status,
                delay_ms=delay_ms,
                diagnostic=diagnostic,
            )
        )
    return tuple(items)


@pytest.mark.parametrize(
    ("measured", "expected"),
    (
        ((("github", "success"), ("google", "success")), "capable"),
        ((("github", "success"), ("cloudflare", "success")), "capable"),
        (
            (
                ("github", "success"),
                ("google", "request_error"),
                ("cloudflare", "request_error"),
            ),
            "failed",
        ),
        ((("github", "success"), ("google", "timeout")), "inconclusive"),
    ),
)
def test_capability_is_two_distinct_controlled_target_successes(measured, expected):
    decision = quality.classify_capability(
        FINGERPRINT,
        observations(*measured),
        control_windows(),
        quorum=2,
    )

    assert decision.status == expected
    assert decision.fingerprint == FINGERPRINT


@pytest.mark.parametrize("closing", (False, True))
def test_invalid_direct_control_window_makes_quorum_inconclusive(closing):
    decision = quality.classify_capability(
        FINGERPRINT,
        observations(
            ("github", "success"),
            ("google", "success"),
            ("cloudflare", "success"),
        ),
        control_windows(invalid=("google", "cloudflare"), closing=closing),
        quorum=2,
    )

    assert decision.status == "inconclusive"


def provenance(source: str, index: int) -> NodeProvenance:
    return NodeProvenance(
        authority=source,
        site=source,
        source_url=f"https://{source}.test/nodes",
        observed_at=NOW,
        publication_time=PublishedDate(on=NOW.date()),
        artifact_digest=f"{index + 1:064x}",
        item_index=index,
    )


def test_capability_decision_has_no_source_identity_input():
    measured = observations(("github", "success"), ("google", "success"))
    windows = control_windows()
    before = quality.classify_capability(FINGERPRINT, measured, windows, quorum=2)
    after = quality.classify_capability(FINGERPRINT, measured, windows, quorum=2)
    assert before == after


def test_source_renaming_cannot_change_accepted_identity():
    decision = quality.classify_capability(
        FINGERPRINT,
        observations(("github", "success"), ("google", "success")),
        control_windows(),
        quorum=2,
    )
    proxy = admit_proxy(
        {
            "name": "same endpoint",
            "type": "ss",
            "server": "example.test",
            "port": 443,
            "cipher": "aes-128-gcm",
            "password": "secret",
        }
    )
    old = DualNode(
        fingerprint=FINGERPRINT,
        display_name="same endpoint",
        provenance=(provenance("old-source", 0),),
        proxy=proxy,
        uri="ss://same",
    )
    renamed = old.model_copy(update={"provenance": (provenance("renamed-source", 0),)})

    assert (
        quality.select_capable(NodeCatalog(nodes=(old,)), (decision,))
        .nodes[0]
        .fingerprint
        == quality.select_capable(NodeCatalog(nodes=(renamed,)), (decision,))
        .nodes[0]
        .fingerprint
    )


def test_target_registry_requires_https_distinct_authorities_and_feasible_quorum():
    target = quality.CapabilityTarget
    admitted = target.admit_registry(
        (
            target(
                id="github",
                url="https://raw.githubusercontent.com/x",
                expected_status=200,
            ),
            target(
                id="google",
                url="https://www.gstatic.com/generate_204",
                expected_status=204,
            ),
        ),
        quorum=2,
    )

    assert len(admitted) == 2
    with pytest.raises(ValueError, match="distinct authorities"):
        target.admit_registry(
            (
                target(id="a", url="https://example.test/a", expected_status=204),
                target(id="b", url="https://example.test/b", expected_status=204),
            ),
            quorum=2,
        )
    with pytest.raises(ValueError, match="quorum"):
        target.admit_registry(admitted[:1], quorum=2)
    with pytest.raises(ValueError, match="HTTPS"):
        target(id="plain", url="http://example.test", expected_status=204)


def test_tested_default_excludes_uri_only_and_requires_both_projections():
    proxy = admit_proxy({"name": "direct", "type": "direct"})
    clash = ClashNode(
        fingerprint="1" * 64,
        display_name="clash",
        provenance=(provenance("a", 1),),
        proxy=proxy,
    )
    uri = UriNode(
        fingerprint="2" * 64,
        display_name="uri",
        provenance=(provenance("b", 2),),
        uri="ssr://opaque",
    )
    dual = DualNode(
        fingerprint="3" * 64,
        display_name="dual",
        provenance=(provenance("c", 3),),
        proxy=proxy,
        uri="ss://dual",
    )
    decisions = tuple(
        quality.classify_capability(
            node.fingerprint,
            observations(("github", "success"), ("google", "success")),
            control_windows(),
            quorum=2,
        )
        for node in (clash, uri, dual)
    )

    selected = quality.select_capable(NodeCatalog(nodes=(clash, uri, dual)), decisions)

    assert tuple(node.fingerprint for node in selected.nodes) == (
        clash.fingerprint,
        dual.fingerprint,
    )
    with pytest.raises(quality.QualityError, match="Clash and URI"):
        quality.select_capable(NodeCatalog(nodes=(clash,)), decisions[:1])


class StructuralValidator:
    def validate_bundle(self, root):
        return ConsumerValidation(
            profiles=("nodes/merged.yaml",),
            provider_profiles=("nodes/provider.yaml",),
            provider_names=("a",),
            group_names=("select",),
        )


class InconclusiveProbe:
    async def probe_capabilities(self, plan, targets, policy):
        return CapabilityRunReceipt(
            status="inconclusive",
            diagnostic=ProbeDiagnostic(code="control_unavailable"),
        )


async def test_inconclusive_measurement_preserves_the_previous_generation(
    monkeypatch, tmp_path
):
    output = tmp_path / "nodes"
    output.mkdir()
    previous = {
        "merged.yaml": b"previous-clash",
        "v2ray.txt": b"previous-v2ray",
        "publication-receipt.json": b"previous-receipt",
    }
    for name, content in previous.items():
        (output / name).write_bytes(content)

    async def discover(self):
        return DiscoverySuccess(
            site_name="a",
            artifacts=(
                SourceArtifact.inline(
                    site="a",
                    content="proxies:\n- {name: direct, type: direct}\n",
                    observed_at=NOW,
                    published_on=NOW.date(),
                ),
            ),
        )

    monkeypatch.setattr("src.scheduler.SiteProcessor.discover", discover)
    config = Config(
        sites=[SimpleSite(name="a", start_url="https://a.test")],
        output={"dir": str(output)},
        llm=LLMConfig(),
    )

    with pytest.raises(PublicationError, match="inconclusive"):
        await Scheduler(config).publish_profiles(
            repository_root=tmp_path,
            validator=StructuralValidator(),
            probe_session=InconclusiveProbe(),
            policy=QualityPolicy(),
            now=NOW,
        )

    assert {name: (output / name).read_bytes() for name in previous} == previous
