"""
Tests for core/models.py
Validates schema constraints, boundary limits, and sanitization behavior.
"""

import pytest
from pydantic import ValidationError
from core.models import EvidenceSource, VerificationRequest, VerificationVerdict


def test_evidence_source_defaults_and_validation():
    src = EvidenceSource(title="PIB Fact Check", url="https://pib.gov.in/factcheck")
    assert src.title == "PIB Fact Check"
    assert src.url == "https://pib.gov.in/factcheck"
    assert src.authority_tier == 4  # Default before provenance classification
    assert src.is_official is False


def test_evidence_source_clean_url():
    src = EvidenceSource(url="   https://rbi.org.in/press/123   \n")
    assert src.url == "https://rbi.org.in/press/123"


def test_verification_request_valid():
    req = VerificationRequest(claim="मेरठ में नमक की कमी की अफवाह", city="Meerut")
    assert req.claim == "मेरठ में नमक की कमी की अफवाह"
    assert req.city == "Meerut"
    assert req.source_medium == "text"
    assert req.correlation_id.startswith("sach-")


def test_verification_request_empty_claim_raises():
    with pytest.raises(ValidationError):
        VerificationRequest(claim="   ")


def test_verification_request_too_short_raises():
    with pytest.raises(ValidationError):
        VerificationRequest(claim="ab")


def test_verification_verdict_confidence_range():
    v = VerificationVerdict(verdict="Likely False", confidence=95)
    assert v.confidence == 95

    with pytest.raises(ValidationError):
        VerificationVerdict(verdict="Likely False", confidence=150)

    with pytest.raises(ValidationError):
        VerificationVerdict(verdict="Likely False", confidence=-5)


def test_verification_verdict_to_legacy_dict():
    v = VerificationVerdict(
        verdict="Likely True",
        confidence=90,
        explanation_hindi="सच है",
        explanation_english="Is true",
        sources=[EvidenceSource(title="RBI", url="https://rbi.org.in", authority_tier=1, is_official=True)],
    )
    d = v.to_legacy_dict()
    assert isinstance(d, dict)
    assert d["verdict"] == "Likely True"
    assert d["confidence"] == 90
    assert len(d["sources"]) == 1
    assert d["sources"][0]["authority_tier"] == 1
