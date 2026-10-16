"""
SachAI Domain Models & Schemas
Validated domain contracts for requests, evidence sources, evidence records,
stance assessments, sufficiency ratings, and persistent records.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from core.tracing import generate_correlation_id


class StanceType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL_OR_UNRELATED = "neutral_or_unrelated"
    MIXED_OR_AMBIGUOUS = "mixed_or_ambiguous"
    UNASSESSABLE = "unassessable"


class RelationshipType(str, Enum):
    CONFIRMED_DEPENDENCY = "confirmed_dependency"  # Explicit attribution (e.g. "Reported by PTI")
    SUSPECTED_DEPENDENCY = "suspected_dependency"  # High text similarity (>0.80) without attribution
    INDEPENDENT = "independent"                    # Distinct primary/independent reporting
    UNKNOWN = "unknown"


class SufficiencyOutcome(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MISLEADING_OR_MISSING_CONTEXT = "misleading_or_missing_context"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AttributeComparison(BaseModel):
    """Factual attribute comparison between claim and evidence passage."""
    attribute_name: str  # 'dates_and_times', 'quantities_and_statistics', 'locations', 'names_and_attributed_statements', 'negation_and_uncertainty', 'definitions_and_populations'
    claim_value: Optional[str] = None
    passage_value: Optional[str] = None
    comparison_result: Literal["matches", "conflicts", "different_scope", "unspecified"] = "unspecified"
    notes: str = ""


class StanceAssessmentResult(BaseModel):
    """Detailed stance evaluation of a single evidence passage against a claim."""
    stance: StanceType
    rationale: str
    method: str = "deterministic_rule_v2"
    version: str = "2.0.0"
    attribute_comparisons: List[AttributeComparison] = Field(default_factory=list)
    is_genuine_contradiction: bool = False
    conflict_type: Optional[str] = None  # 'direct_factual_conflict', 'time_dependent_update', 'differing_definition', 'unrelated_population', 'incomplete_reporting', None


class EvidenceRelationship(BaseModel):
    """Relationship between two evidence documents (e.g. syndication, wire copy)."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class EvidenceRecord(BaseModel):
    """
    Comprehensive, normalized evidence model.
    Distinguishes missing metadata from verified metadata without fabricating dates or quotations.
    """
    evidence_id: str = Field(..., description="Stable SHA-256 derived ID from canonical URL and passage")
    source_url: str = Field(..., description="Original raw URL")
    canonical_url: Optional[str] = None
    publisher: Optional[str] = None
    domain: str
    title: Optional[str] = None
    passage: str = Field(..., min_length=1, description="Exact retrieved passage")
    published_at: Optional[str] = None  # None if unknown; never fabricate
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_type: str = "web"  # 'official_portal', 'fact_checker', 'mainstream_wire', 'news_outlet', 'social_media', 'unverified_web'
    authority_tier: int = Field(default=4, ge=1, le=4)
    is_primary_source: bool = False  # e.g. official gazette, court ruling, direct RBI/PIB announcement
    content_hash: str
    upstream_source_refs: List[str] = Field(default_factory=list)  # e.g. ["PTI", "ANI"]
    syndication_cluster_id: Optional[str] = None
    stance: StanceType = StanceType.NEUTRAL_OR_UNRELATED
    stance_assessment: Optional[StanceAssessmentResult] = None
    verification_status: Literal["verified", "unverified", "disputed"] = "unverified"

    @classmethod
    def generate_id(cls, canonical_url: str, passage: str) -> str:
        """Deterministic, stable evidence ID."""
        key = f"{canonical_url.strip().lower()}|{passage.strip()}"
        return f"ev-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


class SufficiencyAssessment(BaseModel):
    """
    Assessment of whether collected evidence is sufficient to justify a factual conclusion.
    Keeps factual support, credibility, and model confidence strictly separated.
    """
    outcome: SufficiencyOutcome
    score: float = Field(..., ge=0.0, le=1.0, description="Evidence sufficiency index")
    claim_relevance: float = Field(..., ge=0.0, le=1.0)
    passage_level_support: float = Field(..., ge=0.0, le=1.0)
    source_independence: float = Field(..., ge=0.0, le=1.0)
    has_primary_source: bool = False
    has_tier1_source: bool = False
    has_unresolved_contradictions: bool = False
    missing_information: List[str] = Field(default_factory=list)
    reason_for_verdict: str


class InvestigationReport(BaseModel):
    """
    Comprehensive investigative report exposing provenance, contradictions, and evidence sufficiency.
    """
    correlation_id: str
    claim: str
    verdict: str
    sufficiency: SufficiencyAssessment
    explanation_hindi: str
    explanation_english: str
    supporting_passages: List[EvidenceRecord] = Field(default_factory=list)
    contradicting_passages: List[EvidenceRecord] = Field(default_factory=list)
    neutral_passages: List[EvidenceRecord] = Field(default_factory=list)
    evidence_relationships: List[EvidenceRelationship] = Field(default_factory=list)
    syndication_clusters: Dict[str, List[str]] = Field(default_factory=dict)
    confirmed_facts: List[str] = Field(default_factory=list)
    inferred_relationships: List[str] = Field(default_factory=list)
    model_assessments: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    abstention_reason: Optional[str] = None


class EvidenceSource(BaseModel):
    """Legacy-compatible evidence source citation with authority tiering."""
    title: str = Field(default="Official Source")
    url: str = Field(default="")
    source_type: str = Field(default="Web")
    snippet: str = Field(default="")
    authority_tier: int = Field(default=4, ge=1, le=4)
    is_official: bool = Field(default=False)
    stance: Optional[str] = None

    @field_validator("url", mode="before")
    @classmethod
    def clean_url(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return ""
        return v.strip()


class VerificationRequest(BaseModel):
    """Validated input request for claim fact-checking."""
    claim: str = Field(..., min_length=3, max_length=3000, description="Claim text to verify")
    city: Optional[str] = Field(default=None, max_length=100)
    category: Optional[str] = Field(default=None, max_length=100)
    language_hint: str = Field(default="mixed", max_length=50)
    source_medium: Literal["text", "screenshot", "audio_note", "sms"] = "text"
    correlation_id: str = Field(default_factory=generate_correlation_id)

    @field_validator("claim")
    @classmethod
    def validate_claim(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Claim text cannot be empty or whitespace only.")
        return cleaned


class VerificationVerdict(BaseModel):
    """Structured and normalized fact-check verdict with backward-compatible schema."""
    verdict: str = Field(default="Unclear")
    confidence: int = Field(default=60, ge=0, le=100)
    explanation_hindi: str = Field(default="इस दावे की पुष्टि की जा रही है।")
    explanation_english: str = Field(default="Verification underway with official sources.")
    short_whatsapp_hindi: str = Field(default="⚠️ कृपया बिना आधिकारिक पुष्टि के इस संदेश को आगे न भेजें।")
    short_whatsapp_english: str = Field(default="⚠️ Avoid forwarding without official confirmation.")
    category: str = Field(default="General")
    risk_level: str = Field(default="Moderate")
    scam_indicators: List[str] = Field(default_factory=list)
    sources: List[EvidenceSource] = Field(default_factory=list)
    engine_used: str = Field(default="SachAI Engine")
    is_grounded: bool = Field(default=False)
    correlation_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Phase 2 Investigation Report & Sufficiency
    sufficiency_outcome: Optional[SufficiencyOutcome] = None
    investigation_report: Optional[InvestigationReport] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Backward-compatible dictionary representation for legacy consumers."""
        d = {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "category": self.category,
            "risk_level": self.risk_level,
            "explanation_hindi": self.explanation_hindi,
            "explanation_english": self.explanation_english,
            "short_whatsapp_hindi": self.short_whatsapp_hindi,
            "short_whatsapp_english": self.short_whatsapp_english,
            "scam_indicators": self.scam_indicators,
            "sources": [s.model_dump() for s in self.sources],
            "engine_used": self.engine_used,
            "is_grounded": self.is_grounded,
            "correlation_id": self.correlation_id,
        }
        if self.investigation_report:
            d["investigation_report"] = self.investigation_report.model_dump()
        if self.sufficiency_outcome:
            d["sufficiency_outcome"] = self.sufficiency_outcome.value
        return d


class ScamAuditResult(BaseModel):
    """Structured assessment from daily scam and SMS heuristic / AI analyzer."""
    verdict: str = Field(...)
    scam_score: int = Field(..., ge=0, le=100)
    risk_level: str = Field(...)
    red_flags: List[str] = Field(default_factory=list)
    explanation_hindi: str = Field(...)
    explanation_english: str = Field(...)
    action_checklist: List[str] = Field(default_factory=list)
    extracted_links: List[str] = Field(default_factory=list)
    sources: List[EvidenceSource] = Field(default_factory=list)
    correlation_id: Optional[str] = None


class VerificationRecord(BaseModel):
    """Snapshot model for persistent storage in SQLite."""
    id: Optional[int] = None
    correlation_id: str
    claim: str
    verdict: str
    confidence: int
    category: str
    risk_level: str
    engine_used: str
    created_at: str
    payload_json: str
