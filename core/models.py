"""
SachAI Domain Models & Schemas
Validated domain contracts for requests, evidence sources, verdicts, and persistent records.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from core.tracing import generate_correlation_id


class EvidenceSource(BaseModel):
    """Normalized evidence source citation with authority tiering."""
    title: str = Field(default="Official Source")
    url: str = Field(default="")
    source_type: str = Field(default="Web")
    snippet: str = Field(default="")
    authority_tier: int = Field(default=4, ge=1, le=4)  # 1=Official/PIB/RBI, 2=FactChecker, 3=Wire/News, 4=General Web
    is_official: bool = Field(default=False)

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
    """Structured and normalized fact-check verdict."""
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

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Backward-compatible dictionary representation for legacy consumers."""
        return {
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
