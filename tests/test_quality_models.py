from datetime import date

import pytest
from pydantic import ValidationError

from src.mihomo import DelayObservation, ProbeEvidence
from src.quality import DailyReliability, QualityPolicy, admit_assessment


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
        DailyReliability(day=date(2026, 8, 29), successes=2, attempts=1)


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
        confirm=DelayObservation(endpoint="confirm", status="timeout"),
    )

    assert evidence.status == "timeout"
