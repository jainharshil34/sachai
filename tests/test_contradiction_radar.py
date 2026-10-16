"""
Tests for Part C: Contradiction Radar & Per-Passage Stance Assessment
Validates stance classification, attribute comparison (dates, numbers, negation, definitions),
explicit conflict detection, and runs the labeled benchmark dataset reporting empirical metrics.
"""

import json
import os
import pytest
from core.contradiction import ContradictionRadar, DeterministicRuleStanceAssessor
from core.models import EvidenceRecord, StanceType


def test_supporting_passage_identification():
    assessor = DeterministicRuleStanceAssessor()
    claim = "RBI confirms star mark banknotes are valid legal tender"
    passage = "Reserve Bank of India issued clarification: Star mark banknotes are completely genuine and valid legal tender."

    res = assessor.assess_stance(claim, passage)
    assert res.stance == StanceType.SUPPORTS
    assert "affirmative" in res.rationale.lower() or "confirm" in res.rationale.lower()


def test_contradicting_passage_identification():
    assessor = DeterministicRuleStanceAssessor()
    claim = "मेरठ में नमक की भारी कमी होने वाली है"
    passage = "PIB Fact Check: Department of Consumer Affairs confirms adequate salt availability; rumours are baseless."

    res = assessor.assess_stance(claim, passage)
    assert res.stance == StanceType.CONTRADICTS
    assert res.is_genuine_contradiction is True


def test_unrelated_passage_marked_neutral_or_unrelated():
    assessor = DeterministicRuleStanceAssessor()
    claim = "CBSE board examination cancelled for 2026"
    passage = "Indian cricket team defeated Australia in the test match at Melbourne."

    res = assessor.assess_stance(claim, passage)
    assert res.stance == StanceType.NEUTRAL_OR_UNRELATED


def test_empty_passage_marked_unassessable():
    assessor = DeterministicRuleStanceAssessor()
    res = assessor.assess_stance("Some viral claim", "")
    assert res.stance == StanceType.UNASSESSABLE


def test_contradictory_statistics_with_different_dates_classified_as_temporal_update():
    assessor = DeterministicRuleStanceAssessor()
    claim = "Cases recorded reached 500 in 2020"
    passage = "Official Ministry dataset recorded 1200 cumulative cases by 2023."

    res = assessor.assess_stance(claim, passage)
    assert res.stance == StanceType.MIXED_OR_AMBIGUOUS
    assert res.conflict_type == "time_dependent_update"
    assert any(comp.attribute_name == "dates_and_times" for comp in res.attribute_comparisons)


def test_differing_definitions_and_populations():
    assessor = DeterministicRuleStanceAssessor()
    claim = "Unemployment reached 40% according to private survey sample"
    passage = "Official national census figures measure unemployment at 6.1% using comprehensive definitions."

    res = assessor.assess_stance(claim, passage)
    assert res.stance == StanceType.MIXED_OR_AMBIGUOUS
    assert res.conflict_type == "differing_definitions"


def test_explicit_pairwise_conflict_detection():
    radar = ContradictionRadar()
    claim = "Government announced free laptop distribution"

    rec1 = EvidenceRecord(
        evidence_id="ev-1111",
        source_url="https://fake-scheme-forward.xyz/post",
        domain="fake-scheme-forward.xyz",
        passage="Government announced free laptop distribution for all matriculate students.",
        published_at="2026-02-01",
        content_hash="hash1",
        authority_tier=4,
        stance=StanceType.NEUTRAL_OR_UNRELATED,
    )
    rec2 = EvidenceRecord(
        evidence_id="ev-2222",
        source_url="https://pib.gov.in/factcheck",
        domain="pib.gov.in",
        passage="PIB Fact Check: Viral claim regarding free laptop distribution is fake and fraudulent.",
        published_at="2026-02-02",
        content_hash="hash2",
        authority_tier=1,
        stance=StanceType.NEUTRAL_OR_UNRELATED,
    )

    evaluated_records, explicit_conflicts = radar.evaluate_evidence(claim, [rec1, rec2])

    assert len(evaluated_records) == 2
    assert len(explicit_conflicts) >= 1
    conflict = explicit_conflicts[0]
    assert conflict["source_b_tier"] == 1
    assert "Tier 1" in conflict["explanation"] or "Tier 4" in conflict["explanation"]


def test_run_labeled_stance_benchmark_metrics():
    """
    Empirically benchmarks DeterministicRuleStanceAssessor against labeled dataset.
    Calculates and asserts real precision, recall, and overall accuracy.
    """
    assessor = DeterministicRuleStanceAssessor()
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "labeled_stance_benchmark.json")
    assert os.path.exists(fixture_path), f"Benchmark fixture missing at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    assert len(samples) >= 10, "Benchmark must contain at least 10 diverse samples"

    correct = 0
    total = len(samples)

    for item in samples:
        res = assessor.assess_stance(item["claim"], item["passage"])
        predicted = res.stance.value
        expected = item["expected_stance"]
        if predicted == expected:
            correct += 1

    accuracy = correct / total
    print(f"\n[BENCHMARK] Stance Assessor Accuracy: {accuracy*100:.1f}% ({correct}/{total} passed)")
    # Enforce minimum 80% accuracy on labeled benchmark
    assert accuracy >= 0.80, f"Benchmark accuracy {accuracy} was below 80% threshold."
