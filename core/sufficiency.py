"""
SachAI Evidence Sufficiency & Investigation Report Engine
Evaluates whether collected evidence is sufficient, independent, and authoritative enough
to justify a factual verdict. Distinguishes confirmed facts, inferred relationships,
model assessments, and unresolved questions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from core.models import (
    EvidenceRecord,
    EvidenceRelationship,
    InvestigationReport,
    RelationshipType,
    StanceType,
    SufficiencyAssessment,
    SufficiencyOutcome,
)

logger = logging.getLogger("SachAI.Sufficiency")


def evaluate_evidence_sufficiency(
    correlation_id: str,
    claim: str,
    records: List[EvidenceRecord],
    relationships: List[EvidenceRelationship],
    explicit_conflicts: List[Dict[str, Any]],
    explanation_hindi: str = "",
    explanation_english: str = "",
    grounded_verdict: Optional[str] = None,
    grounded_confidence: Optional[int] = None,
) -> Tuple[SufficiencyAssessment, InvestigationReport, str, int]:
    """
    Synthesize factual outcome based strictly on evidence sufficiency, source independence,
    and contradiction resolution.
    Returns: (SufficiencyAssessment, InvestigationReport, display_verdict, calibrated_confidence)
    """
    supporting = [r for r in records if r.stance == StanceType.SUPPORTS]
    contradicting = [r for r in records if r.stance == StanceType.CONTRADICTS]
    neutral = [r for r in records if r.stance in [StanceType.NEUTRAL_OR_UNRELATED, StanceType.MIXED_OR_AMBIGUOUS, StanceType.UNASSESSABLE]]

    has_tier1 = any(r.authority_tier == 1 for r in records)
    has_primary = any(r.is_primary_source for r in records)
    has_tier2 = any(r.authority_tier == 2 for r in records)

    # 1. Source Independence & Deduplication Discounting
    # If 10 sources are just wire reprints of 1 agency, count as 1 independent source
    sup_clusters = set()
    for s in supporting:
        cluster_id = s.syndication_cluster_id or s.domain
        sup_clusters.add(cluster_id)
    independent_support_count = len(sup_clusters)

    con_clusters = set()
    for c in contradicting:
        cluster_id = c.syndication_cluster_id or c.domain
        con_clusters.add(cluster_id)
    independent_contradict_count = len(con_clusters)

    total_evidence_units = independent_support_count + independent_contradict_count
    source_independence = (
        round(total_evidence_units / max(1, len(supporting) + len(contradicting)), 2)
        if (supporting or contradicting) else 0.0
    )

    # 2. Relevance & Passage Support
    relevant_count = len(supporting) + len(contradicting)
    claim_relevance = min(1.0, round(relevant_count / max(1, len(records)), 2)) if records else (1.0 if grounded_verdict else 0.0)
    passage_level_support = 1.0 if (supporting or contradicting or grounded_verdict) else 0.0

    # Missing Information Tracking
    missing_info: List[str] = []
    if not has_tier1 and not has_tier2 and not grounded_verdict:
        missing_info.append("No official government registry (.gov.in / PIB) or IFCN-accredited fact-checker record found.")
    if not any(r.published_at for r in records) and not grounded_verdict:
        missing_info.append("Precise publication timestamps unavailable for retrieved web passages.")

    # 3. Determine Outcome Category
    outcome: SufficiencyOutcome
    display_verdict: str
    reason: str
    abstention_reason: Optional[str] = None
    calibrated_score: float = 0.5

    # Check for direct unresolved conflicts between authoritative sources
    high_tier_conflict = False
    for conf in explicit_conflicts:
        if conf.get("source_a_tier") in [1, 2] and conf.get("source_b_tier") in [1, 2]:
            high_tier_conflict = True
            break

    # If grounded in official verified archive (e.g. PIB match)
    if grounded_verdict:
        display_verdict = grounded_verdict
        calibrated_score = (grounded_confidence or 95) / 100.0
        if "False" in grounded_verdict or "Scam" in grounded_verdict:
            outcome = SufficiencyOutcome.CONTRADICTED
            reason = f"Ground truth PIB / Law Enforcement verification confirmed: {grounded_verdict}."
        elif "True" in grounded_verdict:
            outcome = SufficiencyOutcome.SUPPORTED
            reason = f"Ground truth PIB / Law Enforcement verification confirmed: {grounded_verdict}."
        elif "Misleading" in grounded_verdict:
            outcome = SufficiencyOutcome.MISLEADING_OR_MISSING_CONTEXT
            reason = f"Ground truth PIB advisory indicates missing context: {grounded_verdict}."
        else:
            outcome = SufficiencyOutcome.INSUFFICIENT_EVIDENCE
            reason = f"Official advisory indicates unverified status: {grounded_verdict}."
    elif not records or (not supporting and not contradicting):
        outcome = SufficiencyOutcome.INSUFFICIENT_EVIDENCE
        display_verdict = "Unclear / Insufficient Evidence"
        reason = "No verifiable evidence passages take a definitive stance on the claim."
        abstention_reason = "Abstaining: Evidence passages are missing, purely speculative, or unrelated to core factual assertion."
        calibrated_score = 0.35

    elif high_tier_conflict:
        outcome = SufficiencyOutcome.CONFLICTING_EVIDENCE
        display_verdict = "Conflicting Evidence"
        reason = "Authoritative sources hold unresolved, contradictory stances on this claim."
        abstention_reason = "Abstaining from definitive true/false: Authoritative Tier-1/2 sources disagree directly on factual details."
        calibrated_score = 0.50

    elif independent_contradict_count > 0 and independent_support_count == 0:
        outcome = SufficiencyOutcome.CONTRADICTED
        display_verdict = "Likely False"
        reason = f"Refuted by {independent_contradict_count} independent source cluster(s)."
        calibrated_score = 0.95 if (has_tier1 or has_tier2) else 0.75

    elif independent_support_count > 0 and independent_contradict_count == 0:
        outcome = SufficiencyOutcome.SUPPORTED
        display_verdict = "Likely True"
        reason = f"Confirmed by {independent_support_count} independent source cluster(s)."
        calibrated_score = 0.95 if (has_tier1 or has_tier2) else 0.75

    elif independent_contradict_count > independent_support_count * 2:
        outcome = SufficiencyOutcome.CONTRADICTED
        display_verdict = "Likely False"
        reason = "Refuting evidence significantly outweighs affirmative claims."
        calibrated_score = 0.85

    elif independent_support_count > independent_contradict_count * 2:
        outcome = SufficiencyOutcome.SUPPORTED
        display_verdict = "Likely True"
        reason = "Supporting evidence significantly outweighs claims of error."
        calibrated_score = 0.85

    else:
        outcome = SufficiencyOutcome.MISLEADING_OR_MISSING_CONTEXT
        display_verdict = "Misleading / Missing Context"
        reason = "Evidence demonstrates factual ambiguity, incomplete reporting, or conflicting minor attributes."
        abstention_reason = "Caution advised: Elements of the claim are reported inconsistently across outlets."
        calibrated_score = 0.60

    # Build Sufficiency Assessment
    sufficiency = SufficiencyAssessment(
        outcome=outcome,
        score=calibrated_score,
        claim_relevance=claim_relevance,
        passage_level_support=passage_level_support,
        source_independence=source_independence,
        has_primary_source=has_primary,
        has_tier1_source=has_tier1,
        has_unresolved_contradictions=high_tier_conflict or bool(explicit_conflicts),
        missing_information=missing_info,
        reason_for_verdict=reason,
    )

    # 4. Partition Structured Facts & Claims
    confirmed_facts: List[str] = []
    inferred_relationships: List[str] = []
    model_assessments: List[str] = []
    unresolved_questions: List[str] = []

    for r in records:
        if r.authority_tier == 1:
            confirmed_facts.append(f"Official record from {r.domain}: \"{r.passage[:140]}...\"")
        elif r.authority_tier == 2:
            confirmed_facts.append(f"Accredited fact-check by {r.domain}: \"{r.passage[:140]}...\"")

    for rel in relationships:
        if rel.relationship_type == RelationshipType.CONFIRMED_DEPENDENCY:
            inferred_relationships.append(f"Confirmed wire syndication between {rel.source_id} and {rel.target_id}: {rel.rationale}")
        elif rel.relationship_type == RelationshipType.SUSPECTED_DEPENDENCY:
            inferred_relationships.append(f"Suspected syndication / reprinted copy: {rel.rationale}")

    model_assessments.append(f"Passage stance classification determined {len(supporting)} supporting and {len(contradicting)} contradicting passages.")
    model_assessments.append(f"Source independence factor calculated at {int(source_independence * 100)}% to penalize duplicate wire syndication.")

    if missing_info:
        unresolved_questions.extend(missing_info)
    if explicit_conflicts:
        unresolved_questions.append(f"{len(explicit_conflicts)} pairwise passage conflict(s) observed between retrieved sources.")

    # Format Investigation Report
    report = InvestigationReport(
        correlation_id=correlation_id,
        claim=claim,
        verdict=display_verdict,
        sufficiency=sufficiency,
        explanation_hindi=explanation_hindi or "दावे की पुष्टि प्रक्रिया पूरी हुई।",
        explanation_english=explanation_english or reason,
        supporting_passages=supporting,
        contradicting_passages=contradicting,
        neutral_passages=neutral,
        evidence_relationships=relationships,
        confirmed_facts=confirmed_facts,
        inferred_relationships=inferred_relationships,
        model_assessments=model_assessments,
        unresolved_questions=unresolved_questions,
        abstention_reason=abstention_reason,
    )

    confidence_percentage = int(calibrated_score * 100)
    return sufficiency, report, display_verdict, confidence_percentage
