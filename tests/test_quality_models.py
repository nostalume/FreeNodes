import pytest
from pydantic import ValidationError

from src.quality import (
    DelayObservation,
    ProbeDiagnostic,
    ProbeEvidence,
    QualityPolicy,
    TransferObservation,
    admit_assessment,
)


def test_successful_delay_requires_a_non_boolean_measurement():
    with pytest.raises(ValidationError):
        DelayObservation(endpoint="coarse", status="success")
    with pytest.raises(ValidationError):
        DelayObservation(endpoint="coarse", status="success", delay_ms=True)


def test_probe_failure_cannot_carry_success_delay_evidence():
    with pytest.raises(ValidationError):
        DelayObservation(endpoint="coarse", status="timeout", delay_ms=20)


def test_quality_policy_and_history_counts_are_admitted_fields():
    with pytest.raises(ValidationError):
        QualityPolicy(max_candidates=0)
    with pytest.raises(ValidationError):
        QualityPolicy(source_history_size=2, source_history_successes=3)


def test_assessment_variants_reject_unknown_or_incomplete_states():
    with pytest.raises(ValidationError):
        admit_assessment({"fingerprint": "a" * 64, "status": "mystery"})
    with pytest.raises(ValidationError):
        admit_assessment({"fingerprint": "a" * 64, "status": "slow"})


def test_probe_status_is_derived_from_both_observations():
    evidence = ProbeEvidence(
        fingerprint="a" * 64,
        proxy_name="one",
        coarse=DelayObservation(
            endpoint="coarse",
            status="success",
            delay_ms=20,
        ),
        confirm=DelayObservation(
            endpoint="confirm",
            status="timeout",
            diagnostic=ProbeDiagnostic(code="request_timeout"),
        ),
    )

    assert evidence.status == "timeout"


def test_probe_evidence_rejects_a_transfer_for_another_node():
    with pytest.raises(ValidationError, match="fingerprint"):
        ProbeEvidence(
            fingerprint="a" * 64,
            proxy_name="one",
            coarse=DelayObservation(endpoint="coarse", status="success", delay_ms=20),
            confirm=DelayObservation(endpoint="confirm", status="success", delay_ms=20),
            transfer=TransferObservation(
                fingerprint="b" * 64,
                target="test",
                status="success",
                bytes_received=1024 * 1024,
                elapsed_ms=100,
                bytes_per_second=10_000_000,
            ),
        )


def test_transfer_observation_requires_target_identity_and_admits_uncertainty():
    with pytest.raises(ValidationError, match="target"):
        TransferObservation(
            fingerprint="a" * 64,
            status="success",
            bytes_received=1024,
            elapsed_ms=10,
            bytes_per_second=102_400,
        )

    uncertain = TransferObservation(
        fingerprint="a" * 64,
        target="cloudflare+hetzner",
        status="inconclusive",
        bytes_received=0,
        elapsed_ms=10,
        diagnostic=ProbeDiagnostic(code="control_transfer"),
    )

    assert uncertain.status == "inconclusive"
