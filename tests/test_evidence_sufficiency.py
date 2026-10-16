"""
Tests for Part D: Evidence Sufficiency & Verdict Pipeline
Validates outcome categories: supported, contradicted, misleading/missing context,
conflicting evidence, and insufficient evidence/abstention.
Validates independence discounting and separation of model confidence from sufficiency.
"""

from core.models import EvidenceRecord, EvidenceRelationship, RelationshipType, StanceType, SufficiencyOutcome
from core.sufficiency import evaluate_evidence_sufficiency


def test_sufficiency_outcome_supported():
    rec = EvidenceRecord(
        evidence_id="ev-1",
        source_url="https://pib.gov.in/notice",
        domain="pib.gov.in",
        passage="Cabinet approved scheme.",
        content_hash="h1",
        authority_tier=1,
        stance=StanceType.SUPPORTS,
    )
    sufficiency, report, display_verdict, conf = evaluate_evidence_sufficiency(
        correlation_id="test-cid-1",
        claim="Cabinet approved scheme",
        records=[rec],
        relationships=[],
        explicit_conflicts=[],
    )

    assert sufficiency.outcome == SufficiencyOutcome.SUPPORTED
    assert display_verdict == "Likely True"
    assert conf >= 90
    assert report.abstention_reason is None


def test_sufficiency_outcome_contradicted():
    rec = EvidenceRecord(
        evidence_id="ev-2",
        source_url="https://pib.gov.in/factcheck",
        domain="pib.gov.in",
        passage="Fact check confirms claim is fake.",
        content_hash="h2",
        authority_tier=1,
        stance=StanceType.CONTRADICTS,
    )
    sufficiency, report, display_verdict, conf = evaluate_evidence_sufficiency(
        correlation_id="test-cid-2",
        claim="Claim is real",
        records=[rec],
        relationships=[],
        explicit_conflicts=[],
    )

    assert sufficiency.outcome == SufficiencyOutcome.CONTRADICTED
    assert display_verdict == "Likely False"
    assert conf >= 90


def test_sufficiency_outcome_conflicting_evidence():
    rec1 = EvidenceRecord(
        evidence_id="ev-a",
        source_url="https://source-a.gov.in",
        domain="source-a.gov.in",
        passage="Official says project confirmed.",
        content_hash="ha",
        authority_tier=1,
        stance=StanceType.SUPPORTS,
    )
    rec2 = EvidenceRecord(
        evidence_id="ev-b",
        source_url="https://source-b.gov.in",
        domain="source-b.gov.in",
        passage="Ministry denies project approval.",
        content_hash="hb",
        authority_tier=1,
        stance=StanceType.CONTRADICTS,
    )
    conflicts = [{
        "source_a_tier": 1,
        "source_b_tier": 1,
        "explanation": "Direct conflict between Tier-1 sources",
    }]

    sufficiency, report, display_verdict, conf = evaluate_evidence_sufficiency(
        correlation_id="test-cid-3",
        claim="Project approved",
        records=[rec1, rec2],
        relationships=[],
        explicit_conflicts=conflicts,
    )

    assert sufficiency.outcome == SufficiencyOutcome.CONFLICTING_EVIDENCE
    assert display_verdict == "Conflicting Evidence"
    assert report.abstention_reason is not None
    assert "disagree" in report.abstention_reason.lower()


def test_sufficiency_outcome_insufficient_evidence_and_abstention():
    # Only unrelated or neutral passages
    rec = EvidenceRecord(
        evidence_id="ev-neutral",
        source_url="https://weather.com",
        domain="weather.com",
        passage="Weather will be sunny.",
        content_hash="hn",
        authority_tier=4,
        stance=StanceType.NEUTRAL_OR_UNRELATED,
    )

    sufficiency, report, display_verdict, conf = evaluate_evidence_sufficiency(
        correlation_id="test-cid-4",
        claim="Government announced holiday",
        records=[rec],
        relationships=[],
        explicit_conflicts=[],
    )

    assert sufficiency.outcome == SufficiencyOutcome.INSUFFICIENT_EVIDENCE
    assert display_verdict == "Unclear / Insufficient Evidence"
    assert report.abstention_reason is not None
    assert "Abstaining" in report.abstention_reason


def test_syndication_independence_discounting():
    # 5 articles that all belong to the SAME syndication cluster
    records = []
    for i in range(5):
        records.append(EvidenceRecord(
            evidence_id=f"ev-rep-{i}",
            source_url=f"https://newspaper{i}.com/story",
            domain=f"newspaper{i}.com",
            passage="Shared wire reprint text.",
            content_hash="shared_hash",
            syndication_cluster_id="cluster-pti-123",
            authority_tier=3,
            stance=StanceType.SUPPORTS,
        ))

    sufficiency, _, _, _ = evaluate_evidence_sufficiency(
        correlation_id="test-cid-5",
        claim="Shared wire claim",
        records=records,
        relationships=[],
        explicit_conflicts=[],
    )

    # 5 outlets in 1 cluster should count as 1 independent unit -> independence = 1/5 = 0.20
    assert sufficiency.source_independence == 0.20
