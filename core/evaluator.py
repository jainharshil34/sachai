"""
SachAI Evaluation Framework
Provides deterministic metrics and evaluation interfaces for citation quality,
verdict consistency, abstention rates, and latency tracking.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from core.models import EvidenceSource, VerificationVerdict


class EvaluationReport(BaseModel):
    """Structured evaluation report containing deterministic metrics."""
    total_evaluated: int
    abstention_rate: float = Field(..., description="Fraction of claims marked as 'Unclear' due to insufficient evidence")
    official_citation_rate: float = Field(..., description="Fraction of evaluations backed by at least one Tier-1 official source")
    avg_sources_per_claim: float
    verdict_distribution: Dict[str, int]
    consistency_violations: int = Field(default=0, description="Count of logically conflicting verdict/risk combinations")


def compute_citation_support_score(sources: List[EvidenceSource]) -> float:
    """
    Deterministic score [0.0 to 1.0] reflecting evidence density and authority.
    Weights:
      Tier 1 (Official Gov/PIB): 1.0
      Tier 2 (Accredited Fact Checker): 0.8
      Tier 3 (Mainstream Wire/News): 0.6
      Tier 4 (General Web): 0.3
    Max score 1.0 reached with multiple high-authority independent citations.
    """
    if not sources:
        return 0.0

    weight_map = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3}
    total_weight = sum(weight_map.get(s.authority_tier, 0.3) for s in sources)
    # Normalize with diminishing returns (3+ tier-1/2 sources = 1.0)
    return min(1.0, round(total_weight / 2.5, 3))


def is_verdict_consistent(verdict: str, risk_level: str) -> bool:
    """
    Validate that verdict and risk level do not hold contradictory stances.
    e.g., 'Likely True' paired with 'Severe Scam' is a consistency violation.
    """
    v_lower = verdict.lower()
    r_lower = risk_level.lower()

    if ("true" in v_lower or "valid" in v_lower) and ("scam" in r_lower or "fraud" in r_lower):
        return False
    if ("scam" in v_lower or "fraud" in v_lower) and ("low" in r_lower):
        return False
    return True


def evaluate_batch_verdicts(verdicts: List[VerificationVerdict]) -> EvaluationReport:
    """
    Compute aggregate deterministic metrics across a batch of verification outputs.
    """
    total = len(verdicts)
    if total == 0:
        return EvaluationReport(
            total_evaluated=0,
            abstention_rate=0.0,
            official_citation_rate=0.0,
            avg_sources_per_claim=0.0,
            verdict_distribution={},
            consistency_violations=0,
        )

    unclear_count = 0
    official_backed_count = 0
    total_sources = 0
    violations = 0
    dist: Dict[str, int] = {}

    for v in verdicts:
        dist[v.verdict] = dist.get(v.verdict, 0) + 1
        if "unclear" in v.verdict.lower() or "verification needed" in v.verdict.lower():
            unclear_count += 1

        has_official = any(s.authority_tier == 1 for s in v.sources)
        if has_official:
            official_backed_count += 1

        total_sources += len(v.sources)

        if not is_verdict_consistent(v.verdict, v.risk_level):
            violations += 1

    return EvaluationReport(
        total_evaluated=total,
        abstention_rate=round(unclear_count / total, 3),
        official_citation_rate=round(official_backed_count / total, 3),
        avg_sources_per_claim=round(total_sources / total, 2),
        verdict_distribution=dist,
        consistency_violations=violations,
    )
