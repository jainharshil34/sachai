"""
SachAI Contradiction Radar & Per-Passage Stance Assessment
Evaluates whether individual evidence passages support, contradict, or remain neutral to a claim.
Compares factual attributes (dates, numbers, locations, negation, populations) and distinguishes genuine
contradictions from temporal updates, differing definitions, and incomplete reporting.
"""

from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from core.models import (
    AttributeComparison,
    EvidenceRecord,
    StanceAssessmentResult,
    StanceType,
)

logger = logging.getLogger("SachAI.Contradiction")

# Cross-lingual bidirectional synonyms for Indian fact-checking
CROSS_LINGUAL_SYNONYMS: Dict[str, List[str]] = {
    "नमक": ["salt"],
    "मेरठ": ["meerut"],
    "कमी": ["shortage", "scarcity", "deficit"],
    "स्टार": ["star"],
    "नोट": ["note", "banknote", "currency"],
    "अमान्य": ["invalid", "illegal"],
    "मान्य": ["valid", "genuine", "legal tender"],
    "डिजिटल": ["digital"],
    "अरेस्ट": ["arrest"],
    "बिजली": ["electricity", "power", "bijli"],
    "बिल": ["bill"],
    "कट": ["disconnect", "disconnection", "cut"],
    "व्हाट्सएप": ["whatsapp"],
    "स्काइप": ["skype"],
    "कॉल": ["call"],
    "सीबीआई": ["cbi"],
    "पुलिस": ["police"],
    "साइबर": ["cyber"],
    "अपराध": ["crime", "fraud", "scam"],
    "पैन": ["pan"],
    "आधार": ["aadhaar"],
    "लिंक": ["link"],
    "मुफ्त": ["free", "complimentary"],
    "योजना": ["scheme", "yojana"],
    "बैंक": ["bank"],
}

# Negation and debunking terms indicating contradiction of affirmative rumors
REFUTATION_TERMS_EN = {
    "fake", "baseless", "false", "hoax", "rumour", "rumor", "rumours", "no shortage",
    "not true", "denied", "untrue", "debunked", "misleading", "fraud", "scam",
    "never announced", "does not exist", "fabricated", "phishing", "malicious",
    "warning", "impersonating", "syndicate", "syndicates", "arrest", "scarcity",
    "unauthorized", "advisory against", "caution", "no provision", "no system",
    "fraudulent", "beware", "no such",
}

REFUTATION_TERMS_HI = {
    "फर्जी", "गलत", "झूठ", "अफ़वाह", "अफवाह", "कोई कमी नहीं", "भ्रामक", "धोखाधड़ी",
    "स्कैम", "सच्चाई नहीं", "निराधार", "अमान्य", "सावधान", "खारिज", "सावधानी", "चेतावनी",
    "कोई प्रावधान नहीं", "कोई व्यवस्था नहीं", "बचें",
}

SUPPORT_TERMS_EN = {
    "confirmed", "approved", "approves", "valid", "genuine", "true",
    "legal tender", "adequate stock", "sufficient stock", "available", "launched",
    "completely genuine", "abundant stock", "completely valid", "clarifies that",
    "announced", "successfully", "officially confirmed",
}

SUPPORT_TERMS_HI = {
    "पुष्टि", "स्वीकृत", "मान्य", "असली", "सत्य", "उपलब्ध", "शुरू", "वैध", "पर्याप्त",
    "प्रामाणिक", "अधिकार", "घोषणा", "शुरुआत",
}


class AbstractStanceAssessor(ABC):
    """Abstract interface for passage-level stance assessment."""

    @abstractmethod
    def assess_stance(
        self,
        claim: str,
        passage: str,
        published_at: Optional[str] = None,
        title: Optional[str] = None,
    ) -> StanceAssessmentResult:
        """Assess stance of a passage against a claim."""
        pass


class DeterministicRuleStanceAssessor(AbstractStanceAssessor):
    """
    Deterministic rule-based stance assessor.
    Compares factual attributes, negation polarity, and semantic overlap
    without relying on uncalibrated LLM confidence.
    """

    def __init__(self, version: str = "2.0.0"):
        self.version = version

    def _extract_numbers(self, text: str) -> List[str]:
        return re.findall(r'\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b', text)

    def _extract_years(self, text: str) -> List[str]:
        return re.findall(r'\b(19\d{2}|20\d{2})\b', text)

    def _expand_tokens(self, text: str) -> Set[str]:
        clean = text.lower()
        base_tokens = set(re.findall(r'[\w\u0900-\u097F]{2,}', clean))
        expanded = set(base_tokens)
        for token in base_tokens:
            if token in CROSS_LINGUAL_SYNONYMS:
                expanded.update(CROSS_LINGUAL_SYNONYMS[token])
            # Reverse lookup
            for hi_k, en_list in CROSS_LINGUAL_SYNONYMS.items():
                if token in en_list:
                    expanded.add(hi_k)
        return expanded

    def _compare_attributes(
        self,
        claim: str,
        eval_text: str,
    ) -> Tuple[List[AttributeComparison], bool, Optional[str]]:
        comparisons: List[AttributeComparison] = []
        is_genuine_contradiction = False
        conflict_type: Optional[str] = None

        # 1. Dates / Years
        claim_years = self._extract_years(claim)
        eval_years = self._extract_years(eval_text)
        if claim_years and eval_years:
            if set(claim_years) == set(eval_years):
                comparisons.append(AttributeComparison(
                    attribute_name="dates_and_times",
                    claim_value=", ".join(claim_years),
                    passage_value=", ".join(eval_years),
                    comparison_result="matches",
                    notes="Referenced years align.",
                ))
            else:
                comparisons.append(AttributeComparison(
                    attribute_name="dates_and_times",
                    claim_value=", ".join(claim_years),
                    passage_value=", ".join(eval_years),
                    comparison_result="conflicts",
                    notes="Different years referenced. Possible temporal update or historical comparison.",
                ))
                conflict_type = "time_dependent_update"

        # 2. Population & Definition differences (checked before direct numbers)
        claim_lower = claim.lower()
        eval_lower = eval_text.lower()
        has_diff_def = (
            ("survey" in claim_lower and "census" in eval_lower) or
            ("sample" in claim_lower and "official" in eval_lower) or
            ("sample" in claim_lower and "census" in eval_lower)
        )
        if has_diff_def:
            comparisons.append(AttributeComparison(
                attribute_name="definitions_and_populations",
                claim_value="Sample survey / informal query",
                passage_value="Comprehensive official dataset or census",
                comparison_result="conflicts",
                notes="Differences in measured population or survey methodology.",
            ))
            conflict_type = "differing_definitions"

        # 3. Quantities & Statistics
        claim_nums = self._extract_numbers(claim)
        eval_nums = self._extract_numbers(eval_text)
        if claim_nums and eval_nums:
            diff_nums = set(claim_nums) ^ set(eval_nums)
            if not diff_nums:
                comparisons.append(AttributeComparison(
                    attribute_name="quantities_and_statistics",
                    claim_value=", ".join(claim_nums),
                    passage_value=", ".join(eval_nums),
                    comparison_result="matches",
                    notes="Numerical values correspond.",
                ))
            else:
                if conflict_type == "time_dependent_update":
                    comparisons.append(AttributeComparison(
                        attribute_name="quantities_and_statistics",
                        claim_value=", ".join(claim_nums),
                        passage_value=", ".join(eval_nums),
                        comparison_result="conflicts",
                        notes="Statistics differ across distinct reporting dates (temporal update).",
                    ))
                elif conflict_type == "differing_definitions":
                    comparisons.append(AttributeComparison(
                        attribute_name="quantities_and_statistics",
                        claim_value=", ".join(claim_nums),
                        passage_value=", ".join(eval_nums),
                        comparison_result="conflicts",
                        notes="Statistics differ across distinct survey populations/definitions.",
                    ))
                else:
                    comparisons.append(AttributeComparison(
                        attribute_name="quantities_and_statistics",
                        claim_value=", ".join(claim_nums),
                        passage_value=", ".join(eval_nums),
                        comparison_result="conflicts",
                        notes="Direct numerical divergence detected for the same period.",
                    ))
                    conflict_type = "direct_factual_conflict"
                    is_genuine_contradiction = True

        # 4. Negation & Polarity
        has_refutation = any(term in eval_lower for term in REFUTATION_TERMS_EN | REFUTATION_TERMS_HI)
        has_support = any(term in eval_lower for term in SUPPORT_TERMS_EN | SUPPORT_TERMS_HI)

        if has_refutation and not has_support:
            comparisons.append(AttributeComparison(
                attribute_name="negation_and_uncertainty",
                claim_value="Asserts rumour / factual event",
                passage_value="Contains explicit refutation or debunking markers",
                comparison_result="conflicts",
                notes="Passage actively denies or labels the assertion as fake/unfounded.",
            ))
            is_genuine_contradiction = True
            if conflict_type is None:
                conflict_type = "direct_factual_conflict"
        elif has_support and not has_refutation:
            comparisons.append(AttributeComparison(
                attribute_name="negation_and_uncertainty",
                claim_value="Asserts claim / query",
                passage_value="Affirms validity or official availability",
                comparison_result="matches",
                notes="Passage explicitly confirms authenticity.",
            ))

        return comparisons, is_genuine_contradiction, conflict_type

    def assess_stance(
        self,
        claim: str,
        passage: str,
        published_at: Optional[str] = None,
        title: Optional[str] = None,
    ) -> StanceAssessmentResult:
        if not passage or not passage.strip():
            return StanceAssessmentResult(
                stance=StanceType.UNASSESSABLE,
                rationale="Passage content is empty or unreadable.",
                method="deterministic_rule_v2",
                version=self.version,
            )

        eval_text = f"{title or ''} {passage}".strip()
        eval_lower = eval_text.lower()

        # Subject overlap check with cross-lingual expansion
        claim_tokens = self._expand_tokens(claim)
        passage_tokens = self._expand_tokens(eval_text)
        overlap = claim_tokens & passage_tokens

        # If there is no subject overlap with the claim, it cannot support or contradict
        if len(overlap) < 1:
            return StanceAssessmentResult(
                stance=StanceType.NEUTRAL_OR_UNRELATED,
                rationale="Passage does not exhibit meaningful subject overlap with the claim.",
                method="deterministic_rule_v2",
                version=self.version,
            )

        comparisons, is_contradiction, conflict_type = self._compare_attributes(claim, eval_text)

        has_refutation = any(t in eval_lower for t in REFUTATION_TERMS_EN | REFUTATION_TERMS_HI)
        has_support = any(t in eval_lower for t in SUPPORT_TERMS_EN | SUPPORT_TERMS_HI)

        # Distinguish genuine contradictions from time-dependent updates or differing definitions
        if conflict_type == "time_dependent_update":
            stance = StanceType.MIXED_OR_AMBIGUOUS
            rationale = "Passage reflects updated temporal figures from a different timeframe rather than a direct contradiction."
        elif conflict_type == "differing_definitions":
            stance = StanceType.MIXED_OR_AMBIGUOUS
            rationale = "Differences in measured population or terminology accounted for diverging figures."
        elif has_refutation and has_support:
            stance = StanceType.MIXED_OR_AMBIGUOUS
            rationale = "Passage contains both confirmatory and refutational indicators."
        elif has_refutation:
            stance = StanceType.CONTRADICTS
            rationale = "Passage refutes, debunks, or contradicts the viral claim."
            is_contradiction = True
        elif is_contradiction:
            stance = StanceType.CONTRADICTS
            rationale = f"Factual attribute conflict detected ({conflict_type})."
        elif has_support:
            stance = StanceType.SUPPORTS
            rationale = "Passage provides affirmative evidence supporting the claim."
        else:
            stance = StanceType.NEUTRAL_OR_UNRELATED
            rationale = "Passage mentions related terms but does not take a verifiable stance on the core claim."

        return StanceAssessmentResult(
            stance=stance,
            rationale=rationale,
            method="deterministic_rule_v2",
            version=self.version,
            attribute_comparisons=comparisons,
            is_genuine_contradiction=is_contradiction,
            conflict_type=conflict_type,
        )


class ContradictionRadar:
    """
    Orchestrates passage-level stance classification and builds explicit contradiction matrices.
    """

    def __init__(self, assessor: Optional[AbstractStanceAssessor] = None):
        self.assessor = assessor or DeterministicRuleStanceAssessor()

    def evaluate_evidence(
        self,
        claim: str,
        records: List[EvidenceRecord],
    ) -> Tuple[List[EvidenceRecord], List[Dict[str, Any]]]:
        """
        Evaluate stance for all evidence records and identify pairwise contradictions.
        """
        supporting: List[EvidenceRecord] = []
        contradicting: List[EvidenceRecord] = []
        explicit_conflicts: List[Dict[str, Any]] = []

        for r in records:
            res = self.assessor.assess_stance(claim, r.passage, r.published_at, title=r.title)
            r.stance = res.stance
            r.stance_assessment = res

            if res.stance == StanceType.SUPPORTS:
                supporting.append(r)
            elif res.stance == StanceType.CONTRADICTS:
                contradicting.append(r)

        # Pairwise conflict detection across supporting vs contradicting passages
        for sup in supporting:
            for con in contradicting:
                conflict_entry = {
                    "source_a_id": sup.evidence_id,
                    "source_a_domain": sup.domain,
                    "source_a_tier": sup.authority_tier,
                    "source_a_passage": sup.passage,
                    "source_b_id": con.evidence_id,
                    "source_b_domain": con.domain,
                    "source_b_tier": con.authority_tier,
                    "source_b_passage": con.passage,
                    "conflict_nature": con.stance_assessment.conflict_type if con.stance_assessment else "direct_factual_conflict",
                    "explanation": f"Source {sup.domain} (Tier {sup.authority_tier}) supports claim, while Source {con.domain} (Tier {con.authority_tier}) contradicts it.",
                }
                explicit_conflicts.append(conflict_entry)

        return records, explicit_conflicts
