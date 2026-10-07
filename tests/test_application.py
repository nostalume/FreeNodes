import asyncio
import json
from datetime import UTC, datetime

import pytest
import yaml

from freenodes.capability import (
    CapabilityPolicy,
    CapabilityRunReceipt,
    NodeCapabilityDecision,
    ProbeDiagnostic,
)
from freenodes.config import (
    AppConfig,
    DiscoveryLimits,
    GitHubFileSource,
    PublicationPolicy,
    WebSource,
)
from freenodes.discovery import DiscoveryFailure, DiscoverySuccess
from freenodes.nodes import SourceArtifact
from freenodes.publication import PublicationError
from tests.support import (
    CapableProbe,
    ConsumerValidator,
    make_application,
    vless_artifact,
    web_sources_config,
)


@pytest.fixture
def three_site_config():
    return web_sources_config(("site-a", "site-b", "site-c"))


class TestRun:
    async def test_run_does_not_activate_audit_sources(self, monkeypatch):
        candidate = GitHubFileSource(
            name="candidate",
            owner="upstream",
            repository="subscriptions",
            branch="main",
            path="mihomo.yaml",
        )
        config = AppConfig(
            sources=[WebSource(name="active", start_url="https://active.test")],
            audit_sources=[candidate],
        )

        async def active_discovery(processor):
            return DiscoverySuccess(site_name=processor.site.name)

        async def invalid_candidate_discovery(client, site):
            raise AssertionError("candidate was activated")

        monkeypatch.setattr(
            "freenodes.application.SourceDiscovery.discover", active_discovery
        )
        monkeypatch.setattr(
            "freenodes.application.GitHubSourceClient.discover",
            invalid_candidate_discovery,
        )

        assert await make_application(config).run() == [
            DiscoverySuccess(site_name="active")
        ]

    async def test_run_dispatches_github_without_blog_processor(self, monkeypatch):
        source = GitHubFileSource(
            name="direct",
            owner="upstream",
            repository="subscriptions",
            branch="main",
            path="mihomo.yaml",
        )
        config = AppConfig(
            sources=[source],
        )

        async def direct_discovery(client, site, *, observed_at):
            return DiscoverySuccess(site_name=site.name)

        async def invalid_blog_dispatch(processor):
            raise AssertionError("GitHub source entered the blog processor")

        monkeypatch.setattr(
            "freenodes.application.GitHubSourceClient.discover",
            direct_discovery,
        )
        monkeypatch.setattr(
            "freenodes.application.SourceDiscovery.discover",
            invalid_blog_dispatch,
        )

        assert await make_application(config).run() == [
            DiscoverySuccess(site_name="direct")
        ]

    async def test_run_rejects_source_and_run_artifact_budget_overflow(
        self, monkeypatch
    ):
        config = AppConfig(
            sources=[
                WebSource(name=name, start_url=f"http://{name}.test/")
                for name in ("site-a", "site-b", "site-c", "site-d")
            ],
            discovery=DiscoveryLimits(
                source_concurrency=1,
                artifact_limit_per_source=1,
                byte_limit_per_source=5,
                byte_limit_per_run=8,
            ),
        )

        async def fake_run(self):
            contents = {
                "site-a": ("aaaa",),
                "site-b": ("bb", "bb"),
                "site-c": ("cccccc",),
                "site-d": ("dddd",),
            }[self.site.name]
            artifacts = tuple(
                SourceArtifact.inline(
                    site=self.site.name,
                    content=content,
                    observed_at=datetime(2026, 8, 30, tzinfo=UTC),
                )
                for content in contents
            )
            return DiscoverySuccess(
                site_name=self.site.name,
                total_bytes=sum(len(content.encode()) for content in contents),
                artifacts=artifacts,
            )

        monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", fake_run)

        results = await make_application(config).run()

        assert [result.kind for result in results] == [
            "success",
            "failure",
            "failure",
            "success",
        ]
        assert "artifact count limit" in results[1].errors[0]
        assert "source byte limit" in results[2].errors[0]
        assert sum(result.total_bytes for result in results) == 8

    async def test_run_budget_is_independent_of_network_completion_order(
        self, monkeypatch
    ):
        config = AppConfig(
            sources=[
                WebSource(name=name, start_url=f"http://{name}.test/")
                for name in ("a", "b")
            ],
            discovery=DiscoveryLimits(
                source_concurrency=2,
                artifact_limit_per_source=1,
                byte_limit_per_source=4,
                byte_limit_per_run=4,
            ),
        )
        reverse = False
        peer_finished = asyncio.Event()

        async def discover(self):
            if (self.site.name == "a") == reverse:
                await peer_finished.wait()
            else:
                peer_finished.set()
            return DiscoverySuccess(
                site_name=self.site.name,
                total_bytes=4,
                artifacts=(
                    SourceArtifact.inline(
                        site=self.site.name,
                        content="aaaa",
                        observed_at=datetime(2026, 8, 30, tzinfo=UTC),
                    ),
                ),
            )

        monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)
        first = await make_application(config).run()
        reverse = True
        peer_finished = asyncio.Event()
        second = await make_application(config).run()

        assert [result.kind for result in first] == ["success", "failure"]
        assert [result.kind for result in second] == ["success", "failure"]
        assert first[1].errors == second[1].errors

    @pytest.mark.parametrize(
        ("target", "expected"),
        (("site-b", ["site-b"]), ("nonexistent", []), ("SITE-A", [])),
    )
    async def test_target_selection_is_exact(
        self, target, expected, three_site_config, monkeypatch
    ):
        async def fake_run(self):
            return DiscoverySuccess(site_name=self.site.name)

        monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", fake_run)

        results = await make_application(three_site_config).run(target=target)
        assert [result.site_name for result in results] == expected

    async def test_handles_site_crash(self, three_site_config, monkeypatch, capsys):
        async def fake_run(self):
            if self.site.name == "site-b":
                raise RuntimeError("crash!")
            return DiscoverySuccess(site_name=self.site.name, articles_processed=1)

        monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", fake_run)

        results = await make_application(three_site_config).run()
        failures = [result for result in results if result.kind == "failure"]
        assert failures == [
            DiscoveryFailure(
                site_name="site-b",
                errors=("unhandled exception: crash!",),
            )
        ]
        assert [result.site_name for result in results] == [
            "site-a",
            "site-b",
            "site-c",
        ]
        assert capsys.readouterr().out == ""

    async def test_respects_concurrency_limit(self, three_site_config, monkeypatch):
        running = 0
        max_concurrent = 0

        async def fake_run(self):
            nonlocal running, max_concurrent
            running += 1
            max_concurrent = max(max_concurrent, running)
            await asyncio.sleep(0.05)
            running -= 1
            return DiscoverySuccess(site_name=self.site.name)

        monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", fake_run)

        application = make_application(three_site_config)
        await application.run()
        assert max_concurrent <= 2, f"concurrency exceeded limit: {max_concurrent}"


async def test_application_validates_profiles_without_public_cutover(
    monkeypatch, tmp_path
):
    public = tmp_path / "public-nodes"
    public.mkdir()
    sentinel = public / "merged.yaml"
    sentinel.write_text("unchanged", encoding="utf-8")
    config = AppConfig(
        sources=[WebSource(name="source", start_url="https://example.test")],
        discovery=DiscoveryLimits(
            article_limit=1,
            request_timeout_seconds=30,
            source_concurrency=1,
        ),
    )

    async def fake_discover(self):
        artifact = SourceArtifact(
            site="source",
            source_url="inline://source/fixture",
            content="trojan://test@one.example:443#One",
            observed_at=datetime.now(UTC),
            media_type="text/plain",
        )
        return DiscoverySuccess(site_name="source", yaml_count=1, artifacts=(artifact,))

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", fake_discover)

    receipt = await make_application(config).validate_profiles(
        output_parent=tmp_path / "validation-output",
        validator=ConsumerValidator(),
        probe_session=CapableProbe(),
    )

    assert receipt.status == "consumer_validated"
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert receipt.output_dir.is_relative_to((tmp_path / "validation-output").resolve())


@pytest.mark.parametrize("real_consumer", (False, True))
async def test_private_latency_pilot_ranks_and_receipts_without_public_mutation(
    monkeypatch,
    tmp_path,
    request,
    real_consumer,
):
    from freenodes.mihomo import MihomoValidator
    from freenodes.publication import ValidationManifestV1

    consumer = (
        MihomoValidator(request.getfixturevalue("real_mihomo_path"))
        if real_consumer
        else ConsumerValidator()
    )
    monkeypatch.chdir(tmp_path)
    public = tmp_path / "nodes"
    public.mkdir()
    receipt_path = public / "publication-receipt.json"
    receipt_path.write_bytes(b"previous receipt untouched")
    config = web_sources_config(("a",))
    discovery_calls = 0

    async def discover(self):
        nonlocal discovery_calls
        discovery_calls += 1
        now = datetime.now(UTC)
        return DiscoverySuccess(
            site_name="a",
            artifacts=tuple(
                SourceArtifact.inline(
                    site="a",
                    content=f"trojan://secret@node{index}.example:443#Node{index}",
                    observed_at=now,
                )
                for index in range(3)
            ),
        )

    class MeasuredProbe:
        async def probe_capabilities(self, plan, targets, policy):
            assert len(plan.entries) == 3
            assert policy.rank_latency is True
            decisions = tuple(
                NodeCapabilityDecision(
                    fingerprint=entry.node.fingerprint,
                    status="capable",
                    reason="quorum",
                    successful_targets=("github", "google"),
                    target_delays=(("github", delay), ("google", delay)),
                )
                for entry, delay in zip(plan.entries, (900, 20, 30), strict=True)
            )
            return CapabilityRunReceipt(
                status="complete",
                planned=3,
                termination="candidates_exhausted",
                decisions=decisions,
                accepted_fingerprints=(decisions[0].fingerprint,),
            )

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)
    application = make_application(config)
    receipt = await application.validate_profiles(
        output_parent=tmp_path / "private",
        validator=consumer,
        probe_session=MeasuredProbe(),
        policy=CapabilityPolicy(max_published=1, rank_latency=True),
    )
    assert receipt.accepted_count == receipt.uri_count == receipt.clash_count == 1
    assert receipt.latency_ranking.selected_median_delay_ms == 20
    assert receipt.latency_ranking.baseline_median_delay_ms == 900
    manifest = ValidationManifestV1.model_validate_json(
        (receipt.output_dir / "validation-receipt.json").read_bytes()
    )
    assert manifest.latency_ranking == receipt.latency_ranking
    assert receipt_path.read_bytes() == b"previous receipt untouched"
    assert set(public.iterdir()) == {receipt_path}

    with pytest.raises(PublicationError, match="not enabled for publication"):
        await application.publish(
            repository_root=tmp_path,
            validator=consumer,
            probe_session=MeasuredProbe(),
            policy=CapabilityPolicy(max_published=1, rank_latency=True),
        )
    assert receipt_path.read_bytes() == b"previous receipt untouched"

    assert discovery_calls == 1


NOW = datetime(2026, 8, 29, tzinfo=UTC)


@pytest.mark.parametrize(
    ("mode", "real_consumer"),
    (
        ("disabled", False),
        ("ranked", False),
        ("ranked", True),
        ("deadline", False),
        ("no_improvement", False),
        ("override", False),
        ("consumer", False),
        ("inconclusive", False),
    ),
)
async def test_publication_ranking_rollout_and_recovery(
    monkeypatch, tmp_path, request, caplog, mode, real_consumer
):
    from freenodes.mihomo import MihomoValidator
    from freenodes.publication import PublicationArtifact
    from tests.support import snapshot

    async def discover(self):
        return DiscoverySuccess(
            site_name="a",
            artifacts=(
                SourceArtifact.inline(
                    site="a",
                    content="\n".join(
                        f"trojan://secret@node{index}.example:443#Node{index}"
                        for index in range(3)
                    ),
                    observed_at=NOW,
                ),
            ),
        )

    class MeasuredProbe:
        stage = "disabled"
        expected_ranking = False
        expected_limit = 3
        first = faster = ""

        async def probe_capabilities(self, plan, targets, policy):
            assert policy.rank_latency == self.expected_ranking
            assert policy.max_published == self.expected_limit
            if self.stage == "inconclusive":
                return CapabilityRunReceipt(
                    status="inconclusive",
                    diagnostic=ProbeDiagnostic(code="control_unavailable"),
                )
            decisions = tuple(
                NodeCapabilityDecision(
                    fingerprint=entry.node.fingerprint,
                    status="capable",
                    reason="quorum",
                    successful_targets=("github", "google"),
                    target_delays=(("github", delay), ("google", delay)),
                )
                for index, entry in enumerate(plan.entries)
                for delay in (
                    900
                    if index == 0 or self.stage == "no_improvement"
                    else 10 + index * 10,
                )
            )
            self.first, self.faster = decisions[0].fingerprint, decisions[1].fingerprint
            self.servers = {
                entry.node.fingerprint: entry.node.proxy.server
                for entry in plan.entries
            }
            deadline = self.stage == "deadline"
            return CapabilityRunReceipt(
                status="complete",
                planned=3 + int(deadline),
                termination="time_budget" if deadline else "candidates_exhausted",
                deadline_reached=deadline,
                decisions=decisions,
                accepted_fingerprints=(self.first,),
            )

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)
    probe = MeasuredProbe()
    config = web_sources_config(("a",)).model_copy(
        update={"publication": PublicationPolicy(node_limit=3)}
    )
    # A valid previous generation/history must survive every rejected rollout.
    await make_application(config).publish(
        repository_root=tmp_path,
        validator=ConsumerValidator(),
        probe_session=probe,
        now=NOW,
    )
    before = snapshot(tmp_path)
    config = config.model_copy(
        update={
            "publication": PublicationPolicy(
                node_limit=1,
                rank_latency=mode != "disabled",
            )
        }
    )
    probe.stage = mode
    probe.expected_limit = 1
    probe.expected_ranking = mode not in ("disabled", "override")
    consumer = (
        MihomoValidator(request.getfixturevalue("real_mihomo_path"))
        if real_consumer
        else ConsumerValidator(fail=mode == "consumer")
    )
    caplog.set_level("INFO", logger="freenodes.application")
    arguments = {
        "repository_root": tmp_path,
        "validator": consumer,
        "probe_session": probe,
        "now": NOW,
        "policy": CapabilityPolicy(max_published=1) if mode == "override" else None,
    }
    if mode in ("consumer", "inconclusive"):
        with pytest.raises(RuntimeError if mode == "consumer" else PublicationError):
            await make_application(config).publish(**arguments)
        assert snapshot(tmp_path) == before
        return
    await make_application(config).publish(**arguments)
    artifact = PublicationArtifact.admit(tmp_path)
    selected = probe.faster if mode == "ranked" else probe.first
    assert artifact.receipt.capability.accepted == 1
    profile = yaml.safe_load(
        (tmp_path / "nodes/merged.yaml").read_text(encoding="utf-8")
    )
    assert [proxy["server"] for proxy in profile["proxies"]] == [
        probe.servers[selected]
    ]
    assert (
        artifact.receipt.quality_history.generations[0].entries[0].fingerprint
        == selected
    )
    if probe.expected_ranking:
        reason = "incomplete_pool" if mode == "deadline" else mode
        assert f"Latency selection: {reason}; accepted=1" in caplog.text
    else:
        assert "Latency selection:" not in caplog.text


class FirstOnlyProbe:
    async def probe_capabilities(self, plan, targets, policy):
        decisions = tuple(
            NodeCapabilityDecision(
                fingerprint=entry.node.fingerprint,
                status="capable" if index == 0 else "failed",
                successful_targets=("github", "google") if index == 0 else (),
                failed_targets=() if index == 0 else ("github", "google"),
                reason="quorum" if index == 0 else "target_failures",
            )
            for index, entry in enumerate(plan.entries)
        )
        return CapabilityRunReceipt(
            status="complete",
            planned=len(decisions),
            termination="candidates_exhausted",
            decisions=decisions,
            accepted_fingerprints=(decisions[0].fingerprint,),
        )


def validator() -> ConsumerValidator:
    return ConsumerValidator(
        required_files=("nodes/merged.yaml", "nodes/v2ray.txt"),
        forbidden_files=("nodes/quality-manifest.json",),
    )


async def test_publication_requires_capability_after_deterministic_admission(
    monkeypatch, tmp_path, capsys
):
    async def discover(self):
        if self.site.name == "b":
            return DiscoveryFailure(site_name="b", errors=("unavailable",))
        return DiscoverySuccess(site_name="a", artifacts=(vless_artifact("a", NOW),))

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)

    receipt = await make_application(web_sources_config()).publish(
        repository_root=tmp_path,
        validator=validator(),
        probe_session=CapableProbe(),
        now=NOW,
    )

    assert receipt.status == "accepted"
    manifest = json.loads(
        (tmp_path / "nodes" / "publication-receipt.json").read_bytes()
    )
    assert manifest["schema"] == 5
    assert len(manifest["quality_history"]["generations"]) == 1
    assert manifest["capability"]["planned"] == 1
    assert manifest["capability"]["termination"] == "candidates_exhausted"
    assert manifest["capability"]["accepted"] == 1
    assert manifest["admission"]["attempted_sources"] == 2
    assert manifest["admission"]["failed_sources"] == 1
    assert capsys.readouterr().out == ""


async def test_publish_freshness_uses_explicit_run_time_not_wall_clock(
    monkeypatch, tmp_path
):
    as_of = datetime(2000, 1, 1, tzinfo=UTC)

    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name,
            artifacts=(vless_artifact(self.site.name, as_of),),
        )

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)

    receipt = await make_application(web_sources_config()).publish(
        repository_root=tmp_path,
        validator=validator(),
        probe_session=CapableProbe(),
        now=as_of,
    )

    assert receipt.status == "accepted"


async def test_publication_projects_only_the_accepted_capable_identity(
    monkeypatch, tmp_path
):
    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name, artifacts=(vless_artifact(self.site.name, NOW),)
        )

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)

    receipt = await make_application(web_sources_config()).publish(
        repository_root=tmp_path,
        validator=validator(),
        probe_session=FirstOnlyProbe(),
        now=NOW,
    )

    manifest = json.loads(
        (tmp_path / "nodes" / "publication-receipt.json").read_bytes()
    )
    profile = (tmp_path / "nodes" / "merged.yaml").read_text(encoding="utf-8")
    assert receipt.status == "accepted"
    assert manifest["admission"]["unique_eligible"] == 2
    assert tuple(
        manifest["capability"][key]
        for key in ("attempted", "capable", "failed", "inconclusive", "accepted")
    ) == (2, 1, 1, 0, 1)
    assert "a.example" in profile and "b.example" not in profile


async def test_publish_discovers_each_source_once_and_removes_legacy_site_text(
    monkeypatch, tmp_path
):
    calls: list[str] = []

    async def discover(self):
        calls.append(self.site.name)
        return DiscoverySuccess(
            site_name=self.site.name, artifacts=(vless_artifact(self.site.name, NOW),)
        )

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)
    legacy = tmp_path / "nodes" / "a.txt"
    legacy.parent.mkdir()
    legacy.write_text("legacy", encoding="utf-8")

    await make_application(web_sources_config()).publish(
        repository_root=tmp_path,
        validator=validator(),
        probe_session=CapableProbe(),
        now=NOW,
    )

    assert calls == ["a", "b"]
    assert not legacy.exists()


async def test_zero_deterministically_eligible_nodes_preserves_snapshot(
    monkeypatch, tmp_path
):
    sentinel = tmp_path / "nodes" / "merged.yaml"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"previous")

    async def discover(self):
        return DiscoverySuccess(
            site_name=self.site.name,
            artifacts=(
                SourceArtifact.inline(
                    site=self.site.name,
                    content="not a proxy",
                    observed_at=NOW,
                    published_on=NOW.date(),
                ),
            ),
        )

    monkeypatch.setattr("freenodes.application.SourceDiscovery.discover", discover)

    with pytest.raises(PublicationError, match="admitted no nodes"):
        await make_application(web_sources_config(("a",))).publish(
            repository_root=tmp_path,
            validator=validator(),
            probe_session=CapableProbe(),
            now=NOW,
        )

    assert sentinel.read_bytes() == b"previous"
