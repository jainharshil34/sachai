"""
Tests for core/evaluator.py
Validates deterministic evaluation metrics, citation scoring, and consistency checks.
"""

from core.evaluator import (
    compute_citation_support_score,
    evaluate_batch_verdicts,
    is_verdict_consistent,
)
from core.models import EvidenceSource, VerificationVerdict


def test_citation_support_score():
    empty_score = compute_citation_support_score([])
    assert empty_score == 0.0

    tier1_src = [EvidenceSource(authority_tier=1, is_official=True)]
    score1 = compute_citation_support_score(tier1_src)
    assert score1 > 0.3

    tier1_and_tier2 = [
        EvidenceSource(authority_tier=1, is_official=True),
        EvidenceSource(authority_tier=1, is_official=True),
        EvidenceSource(authority_tier=2),
    ]
    max_score = compute_citation_support_score(tier1_and_tier2)
    assert max_score == 1.0


def test_verdict_consistency():
    assert is_verdict_consistent("Likely True", "Low") is True
    assert is_verdict_consistent("Likely False", "High Viral Threat") is True
    # Violation: Claim is declared True but marked as Severe Scam
    assert is_verdict_consistent("Likely True", "Severe Scam") is False
    # Violation: Claim is declared a Scam but marked as Low risk
    assert is_verdict_consistent("Cyber Fraud / Scam", "Low") is False


def test_evaluate_batch_verdicts():
    verdicts = [
        VerificationVerdict(
            verdict="Likely False",
            confidence=95,
            risk_level="High Viral Threat",
            sources=[EvidenceSource(authority_tier=1, is_official=True)],
        ),
        VerificationVerdict(
            verdict="Unclear / Verification Needed",
            confidence=50,
            risk_level="Moderate",
            sources=[],
        ),
        VerificationVerdict(
            verdict="Likely True",
            confidence=90,
            risk_level="Low",
            sources=[EvidenceSource(authority_tier=3)],
        ),
    ]

    report = evaluate_batch_verdicts(verdicts)
    assert report.total_evaluated == 3
    assert report.abstention_rate == 0.333  # 1 out of 3 is Unclear
    assert report.official_citation_rate == 0.333  # 1 out of 3 has Tier 1
    assert report.consistency_violations == 0
