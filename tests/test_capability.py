from datetime import UTC, datetime

import pytest

import freenodes.capability as capability
from freenodes.config import AppConfig, GitHubFileSource, WebSource
from freenodes.discovery import DiscoveryFailure, DiscoverySuccess
from freenodes.nodes import (
    AdmittedCatalog,
    ClashNode,
    DualNode,
    NodeProvenance,
    PublishedDate,
    SourceArtifact,
    UriNode,
)
from freenodes.proxies import admit_proxy
from freenodes.publication import PublicationError
from tests.support import ConsumerValidator, make_application

NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


def artifact(source: str) -> SourceArtifact:
    identity = f"{source * 8}-{source * 4}-{source * 4}-{source * 4}-{source * 12}"
    return SourceArtifact.inline(
        site=source,
        content=f"vless://{identity}@{source}.example:443?security=tls&type=tcp#{source}",
        observed_at=NOW,
        published_on=NOW.date(),
    )


def config() -> AppConfig:
    return AppConfig(
        sources=[WebSource(name="a", start_url="https://a.test")],
        audit_sources=[
            GitHubFileSource(
                name="b",
                owner="upstream",
                repository="subscriptions",
                branch="main",
                path="mihomo.yaml",
            )
        ],
    )


class CapableProbe:
    def __init__(self):
        self.sources: tuple[tuple[str, ...], ...] = ()
        self.targets: tuple[str, ...] = ()

    async def probe_capabilities(self, plan, targets, policy):
        self.sources = tuple(entry.sources for entry in plan.entries)
        self.targets = tuple(target.id for target in targets)
        decisions = tuple(
            capability.NodeCapabilityDecision(
                fingerprint=entry.node.fingerprint,
                status="capable",
                successful_targets=("github", "google"),
                reason="quorum",
            )
            for entry in plan.entries
        )
        return capability.CapabilityRunReceipt(
            status="complete",
            decisions=decisions,
            accepted_fingerprints=tuple(item.fingerprint for item in decisions),
        )


async def test_audit_measures_every_available_source_without_writing(
    monkeypatch, tmp_path
):
    audit_config = config()
    sentinel = tmp_path / "unchanged"
    sentinel.write_text("keep", encoding="utf-8")

    async def discover_active(processor):
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    async def discover_candidate(client, site, *, observed_at):
        return DiscoverySuccess(site_name="b", artifacts=(artifact("b"),))

    monkeypatch.setattr(
        "freenodes.application.SourceDiscovery.discover", discover_active
    )
    monkeypatch.setattr(
        "freenodes.application.GitHubSourceClient.discover", discover_candidate
    )
    probe = CapableProbe()

    receipt = await make_application(audit_config).audit_sources(
        probe_session=probe,
        now=NOW,
    )

    assert receipt.status == "complete"
    assert probe.sources == (("a",), ("b",))
    assert probe.targets == ("github", "google", "cloudflare")
    assert tuple(path.name for path in tmp_path.iterdir()) == ("unchanged",)


async def test_audit_is_inconclusive_when_controls_are_unusable(monkeypatch, tmp_path):
    async def discover_active(processor):
        return DiscoverySuccess(site_name="a", artifacts=(artifact("a"),))

    async def discover_candidate(client, site, *, observed_at):
        return DiscoveryFailure(site_name="b", errors=("unavailable",))

    class InconclusiveProbe:
        async def probe_capabilities(self, plan, targets, policy):
            return capability.CapabilityRunReceipt(
                status="inconclusive",
                diagnostic=capability.ProbeDiagnostic(code="control_unavailable"),
            )

    monkeypatch.setattr(
        "freenodes.application.SourceDiscovery.discover", discover_active
    )
    monkeypatch.setattr(
        "freenodes.application.GitHubSourceClient.discover", discover_candidate
    )

    receipt = await make_application(config()).audit_sources(
        probe_session=InconclusiveProbe(),
        now=NOW,
    )

    assert receipt.status == "inconclusive"
    assert receipt.diagnostic.code == "control_unavailable"


NOW = datetime(2026, 9, 2, tzinfo=UTC)
FINGERPRINT = "a" * 64
TARGET_IDS = ("github", "google", "cloudflare")


def control_windows(*, invalid: tuple[str, ...] = (), closing: bool = False):
    return tuple(
        capability.TargetControlWindow(
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
            diagnostic = capability.ProbeDiagnostic(
                code="request_timeout" if status == "timeout" else "controller_error"
            )
        items.append(
            capability.TargetObservation(
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
    decision = capability.classify_capability(
        FINGERPRINT,
        observations(*measured),
        control_windows(),
        quorum=2,
    )

    assert decision.status == expected
    assert decision.fingerprint == FINGERPRINT


@pytest.mark.parametrize("closing", (False, True))
def test_invalid_direct_control_window_makes_quorum_inconclusive(closing):
    decision = capability.classify_capability(
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


def dual_node(
    fingerprint: str = FINGERPRINT, *, source: str = "source", index: int = 0
) -> DualNode:
    return DualNode(
        fingerprint=fingerprint,
        display_name="dual",
        provenance=(provenance(source, index),),
        proxy=admit_proxy({"name": "direct", "type": "direct"}),
        uri="ss://dual",
    )


def test_source_renaming_cannot_change_accepted_identity():
    decision = capability.classify_capability(
        FINGERPRINT,
        observations(("github", "success"), ("google", "success")),
        control_windows(),
        quorum=2,
    )
    old = dual_node(source="old-source")
    renamed = old.model_copy(update={"provenance": (provenance("renamed-source", 0),)})
    receipt = capability.CapabilityRunReceipt(
        status="complete",
        decisions=(decision,),
        accepted_fingerprints=(FINGERPRINT,),
    )

    old_selected = capability.CapableCatalog.from_measurement(
        AdmittedCatalog(nodes=(old,)), receipt
    )
    renamed_selected = capability.CapableCatalog.from_measurement(
        AdmittedCatalog(nodes=(renamed,)), receipt
    )
    assert old_selected.nodes[0].fingerprint == renamed_selected.nodes[0].fingerprint


def test_target_registry_requires_https_distinct_authorities_and_feasible_quorum():
    target = capability.CapabilityTarget
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


def test_measurement_order_excludes_unaccepted_and_requires_both_projections():
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
    dual = dual_node("3" * 64, source="c", index=3)
    decisions = tuple(
        capability.classify_capability(
            node.fingerprint,
            observations(("github", "success"), ("google", "success")),
            control_windows(),
            quorum=2,
        )
        for node in (clash, uri, dual)
    )

    receipt = capability.CapabilityRunReceipt(
        status="complete",
        decisions=decisions,
        accepted_fingerprints=(dual.fingerprint, clash.fingerprint),
    )
    selected = capability.CapableCatalog.from_measurement(
        AdmittedCatalog(nodes=(clash, uri, dual)), receipt
    )

    assert tuple(node.fingerprint for node in selected.nodes) == (
        dual.fingerprint,
        clash.fingerprint,
    )
    clash_only = receipt.model_copy(
        update={"accepted_fingerprints": (clash.fingerprint,)}
    )
    with pytest.raises(capability.CapabilityError, match="Clash and URI"):
        capability.CapableCatalog.from_measurement(
            AdmittedCatalog(nodes=(clash, uri, dual)), clash_only
        )


def test_capable_catalog_rejects_foreign_or_duplicate_acceptance():
    node = dual_node()
    decision = capability.NodeCapabilityDecision(
        fingerprint=FINGERPRINT,
        status="capable",
        successful_targets=("github", "google"),
        reason="quorum",
    )
    admitted = AdmittedCatalog(nodes=(node,))
    for accepted in ((FINGERPRINT, FINGERPRINT), ("foreign",)):
        measured = decision.model_copy(update={"fingerprint": accepted[0]})
        receipt = capability.CapabilityRunReceipt(
            status="complete",
            decisions=(measured,),
            accepted_fingerprints=accepted,
        )
        with pytest.raises(capability.CapabilityError, match="does not belong"):
            capability.CapableCatalog.from_measurement(admitted, receipt)


class InconclusiveProbe:
    async def probe_capabilities(self, plan, targets, policy):
        return capability.CapabilityRunReceipt(
            status="inconclusive",
            diagnostic=capability.ProbeDiagnostic(code="control_unavailable"),
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

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)
    config = AppConfig(
        sources=[WebSource(name="a", start_url="https://a.test")],
    )

    with pytest.raises(PublicationError, match="inconclusive"):
        await make_application(config).publish(
            repository_root=tmp_path,
            validator=ConsumerValidator(),
            probe_session=InconclusiveProbe(),
            policy=capability.CapabilityPolicy(),
            now=NOW,
        )

    assert {name: (output / name).read_bytes() for name in previous} == previous
