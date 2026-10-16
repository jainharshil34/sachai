"""
SachAI Verification Pipeline Orchestrator
Coordinates multimodal ingestion, prompt injection sanitization, evidence retrieval,
provenance enrichment, contradiction radar, evidence sufficiency, and SQLite storage.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from core.contradiction import ContradictionRadar
from core.models import (
    EvidenceRecord,
    EvidenceSource,
    InvestigationReport,
    SufficiencyOutcome,
    VerificationRequest,
    VerificationVerdict,
)
from core.provenance import analyze_evidence_provenance
from core.security import sanitize_untrusted_passage
from core.storage import AbstractStorage, SQLiteStorage
from core.sufficiency import evaluate_evidence_sufficiency
from core.tracing import StageTimer, sanitize_error

logger = logging.getLogger("SachAI.Pipeline")


def sanitize_untrusted_claim(claim: str) -> str:
    """
    Harden against prompt injection and adversarial delimiters.
    Neutralizes jailbreak attempts, delimiter breakouts, and instruction overrides.
    """
    cleaned = claim.strip()
    cleaned = re.sub(r'<\/?(?:system|instruction|prompt|context|assistant|user|claim)[^>]*>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?i)(?:ignore\s+all\s+(?:previous|above)\s+instructions)', '[REDACTED_INJECTION_ATTEMPT]', cleaned)
    cleaned = re.sub(r'(?i)(?:you\s+must\s+output\s+verdict:\s*["\']?likely\s+true["\']?)', '[REDACTED_OVERRIDE_ATTEMPT]', cleaned)
    return cleaned[:3000].strip()


class VerificationPipeline:
    """
    Central orchestration engine for SachAI.
    Coordinates between offline verified archives, Gemini live grounding, Perplexity,
    evidence provenance, contradiction radar, and persistent SQLite storage.
    """

    def __init__(
        self,
        storage: Optional[AbstractStorage] = None,
        radar: Optional[ContradictionRadar] = None,
    ):
        self.storage = storage or SQLiteStorage()
        self.radar = radar or ContradictionRadar()

    def verify(
        self,
        request: VerificationRequest,
        engine_choice: str = "Google Gemini 1.5/2.0 Flash (Free API)",
        gemini_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None,
    ) -> VerificationVerdict:
        """
        Execute full end-to-end verification with stage latency tracking and storage persistence.
        """
        cid = request.correlation_id
        logger.info("[%s] Beginning verification pipeline for medium: %s", cid, request.source_medium)

        # 1. INGEST & SANITIZE
        with StageTimer("INGEST", cid) as tracker:
            safe_claim = sanitize_untrusted_claim(request.claim)
            tracker["raw_len"] = len(request.claim)
            tracker["sanitized_len"] = len(safe_claim)

        # 2. RETRIEVAL & SYNTHESIS
        raw_sources: List[Dict[str, Any]] = []
        explanation_hindi = ""
        explanation_english = ""
        short_hi = ""
        short_en = ""
        category = request.category or "General"
        risk_level = "Moderate"
        scam_indicators: List[str] = []
        engine_used = "SachAI Engine"
        is_grounded = False

        grounded_verdict: Optional[str] = None
        grounded_confidence: Optional[int] = None

        with StageTimer("RETRIEVAL_AND_SYNTHESIS", cid) as tracker:
            tracker["engine_selected"] = engine_choice

            from gemini_engine import match_offline_archive
            offline_match = match_offline_archive(safe_claim)

            # Check offline archive
            if "Offline" in engine_choice or (offline_match and not gemini_api_key and "Gemini" in engine_choice):
                if offline_match:
                    raw_sources = offline_match.get("sources", [])
                    explanation_hindi = offline_match.get("explanation_hindi", "")
                    explanation_english = offline_match.get("explanation_english", "")
                    short_hi = offline_match.get("short_whatsapp_hindi", "")
                    short_en = offline_match.get("short_whatsapp_english", "")
                    category = offline_match.get("category", category)
                    risk_level = offline_match.get("risk_level", risk_level)
                    grounded_verdict = offline_match.get("verdict")
                    grounded_confidence = offline_match.get("confidence")
                    engine_used = "SachAI PIB Verified Live Archive (Zero Key / Instant)"
                    is_grounded = True
                    tracker["cache_hit"] = True

            # If Gemini requested
            if not raw_sources and "Gemini" in engine_choice:
                if gemini_api_key:
                    try:
                        from gemini_engine import verify_claim_with_gemini
                        delimited_claim = f"<UNTRUSTED_CLAIM_CONTENT>\n{safe_claim}\n</UNTRUSTED_CLAIM_CONTENT>"
                        gemini_res = verify_claim_with_gemini(
                            claim=delimited_claim,
                            api_key=gemini_api_key,
                            city=request.city,
                            category=request.category,
                        )
                        raw_sources = gemini_res.get("sources", [])
                        explanation_hindi = gemini_res.get("explanation_hindi", "")
                        explanation_english = gemini_res.get("explanation_english", "")
                        short_hi = gemini_res.get("short_whatsapp_hindi", "")
                        short_en = gemini_res.get("short_whatsapp_english", "")
                        category = gemini_res.get("category", category)
                        risk_level = gemini_res.get("risk_level", risk_level)
                        scam_indicators = gemini_res.get("scam_indicators", [])
                        engine_used = gemini_res.get("engine_used", "Google Gemini Flash + Live Search")
                        is_grounded = gemini_res.get("is_grounded", False)
                        tracker["gemini_success"] = True
                    except Exception as exc:
                        logger.error("[%s] Gemini call failed: %s", cid, sanitize_error(exc))
                        tracker["gemini_error"] = sanitize_error(exc)

                if not raw_sources:
                    if offline_match:
                        raw_sources = offline_match.get("sources", [])
                        explanation_hindi = offline_match.get("explanation_hindi", "")
                        explanation_english = offline_match.get("explanation_english", "")
                        short_hi = offline_match.get("short_whatsapp_hindi", "")
                        short_en = offline_match.get("short_whatsapp_english", "")
                        category = offline_match.get("category", category)
                        risk_level = offline_match.get("risk_level", risk_level)
                        grounded_verdict = offline_match.get("verdict")
                        grounded_confidence = offline_match.get("confidence")
                        engine_used = "SachAI PIB Verified Live Archive (Offline Fallback)"
                        is_grounded = True
                    else:
                        explanation_hindi = "इस दावे की पुष्टि के लिए फ्री Gemini Key सक्रिय करें या PIB Fact Check पोर्टल देखें।"
                        explanation_english = "No active Gemini key found. Please enter a free key in sidebar or check official PIB portals."
                        short_hi = "ℹ️ इस दावे की पुष्टि नहीं हुई है। बिना जांचे आगे न बढ़ाएं।"
                        short_en = "ℹ️ This claim is unverified. Please avoid forwarding without checking official sources."
                        raw_sources = [{"title": "PIB Fact Check Portal", "url": "https://pib.gov.in/factcheck", "source_type": "Official Government Portal", "snippet": "Official Government of India counter-misinformation wing."}]
                        engine_used = "SachAI Safe Abstention Mode (Zero Key)"
                        is_grounded = False
                        grounded_verdict = "Unclear / Verification Needed"
                        grounded_confidence = 35

            # If Perplexity requested
            if not raw_sources and "Perplexity" in engine_choice:
                from app import verify_claim_with_perplexity
                pplx_res = verify_claim_with_perplexity(safe_claim, city=request.city)
                raw_sources = pplx_res.get("sources", [])
                explanation_hindi = pplx_res.get("explanation_hindi", "")
                explanation_english = pplx_res.get("explanation_english", "")
                short_hi = pplx_res.get("short_whatsapp_hindi", "")
                short_en = pplx_res.get("short_whatsapp_english", "")
                category = pplx_res.get("category", category)
                risk_level = pplx_res.get("risk_level", risk_level)
                engine_used = pplx_res.get("engine_used", "Perplexity Sonar")
                is_grounded = bool(raw_sources)

        # 3. PROVENANCE & DEDUPLICATION (Part B)
        with StageTimer("PROVENANCE_AND_DEDUPLICATION", cid) as tracker:
            # Sanitize raw source snippets against prompt injection
            for src in raw_sources:
                if "snippet" in src:
                    src["snippet"] = sanitize_untrusted_passage(src["snippet"])

            records, relationships, clusters = analyze_evidence_provenance(raw_sources)
            tracker["evidence_records_count"] = len(records)
            tracker["relationships_count"] = len(relationships)
            tracker["clusters_count"] = len(clusters)

        # 4. CONTRADICTION RADAR (Part C)
        with StageTimer("CONTRADICTION_RADAR", cid) as tracker:
            evaluated_records, explicit_conflicts = self.radar.evaluate_evidence(safe_claim, records)
            tracker["conflicts_detected"] = len(explicit_conflicts)

        # 5. EVIDENCE SUFFICIENCY & VERDICT SYNTHESIS (Part D)
        with StageTimer("SUFFICIENCY_SYNTHESIS", cid) as tracker:
            sufficiency, report, display_verdict, confidence_percentage = evaluate_evidence_sufficiency(
                correlation_id=cid,
                claim=safe_claim,
                records=evaluated_records,
                relationships=relationships,
                explicit_conflicts=explicit_conflicts,
                explanation_hindi=explanation_hindi,
                explanation_english=explanation_english,
                grounded_verdict=grounded_verdict,
                grounded_confidence=grounded_confidence,
            )
            tracker["outcome"] = sufficiency.outcome.value
            tracker["confidence"] = confidence_percentage

        # Build legacy sources for backward compatibility
        legacy_sources: List[EvidenceSource] = []
        for r in evaluated_records:
            legacy_sources.append(EvidenceSource(
                title=r.title or "Source",
                url=r.canonical_url or r.source_url,
                source_type=r.source_type,
                snippet=r.passage,
                authority_tier=r.authority_tier,
                is_official=r.is_primary_source or (r.authority_tier == 1),
                stance=r.stance.value,
            ))

        verdict = VerificationVerdict(
            verdict=display_verdict,
            confidence=confidence_percentage,
            explanation_hindi=explanation_hindi or report.explanation_hindi,
            explanation_english=explanation_english or report.explanation_english,
            short_whatsapp_hindi=short_hi or "⚠️ अफ़वाहों से बचें: बिना पुष्टि के संदेश आगे न बढ़ाएँ।",
            short_whatsapp_english=short_en or "⚠️ Avoid forwarding without official verification.",
            category=category,
            risk_level=risk_level,
            scam_indicators=scam_indicators,
            sources=legacy_sources,
            engine_used=engine_used,
            is_grounded=is_grounded,
            correlation_id=cid,
            sufficiency_outcome=sufficiency.outcome,
            investigation_report=report,
        )

        # 6. STORAGE PERSISTENCE (Part F)
        with StageTimer("STORAGE_PERSISTENCE", cid) as tracker:
            try:
                rec_id = self.storage.save_verification(
                    claim=safe_claim,
                    verdict=verdict,
                    correlation_id=cid,
                )
                ev_saved = self.storage.save_evidence_records(
                    correlation_id=cid,
                    records=evaluated_records,
                    relationships=relationships,
                )
                tracker["saved_record_id"] = rec_id
                tracker["saved_evidence_count"] = ev_saved
            except Exception as e:
                logger.error("[%s] Failed to persist verification to SQLite: %s", cid, sanitize_error(e))
                tracker["storage_error"] = sanitize_error(e)

        logger.info("[%s] Verification completed with outcome '%s' (%s)", cid, sufficiency.outcome.value, display_verdict)
        return verdict
