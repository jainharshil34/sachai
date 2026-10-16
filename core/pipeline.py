"""
SachAI Verification Pipeline Orchestrator
Coordinates multimodal ingestion, prompt injection sanitization, evidence retrieval,
provenance enrichment, verdict validation, tracing, and storage persistence.
"""

import logging
import re
from typing import Any, Dict, Optional
from core.models import EvidenceSource, VerificationRequest, VerificationVerdict
from core.provenance import build_evidence_provenance
from core.storage import AbstractStorage, SQLiteStorage
from core.tracing import StageTimer, sanitize_error

logger = logging.getLogger("SachAI.Pipeline")


def sanitize_untrusted_claim(claim: str) -> str:
    """
    Harden against prompt injection and adversarial delimiters.
    Neutralizes jailbreak attempts, delimiter breakouts, and instruction overrides.
    """
    cleaned = claim.strip()
    # Strip dangerous XML/markdown-breaking tags
    cleaned = re.sub(r'<\/?(?:system|instruction|prompt|context|assistant|user|claim)[^>]*>', '', cleaned, flags=re.IGNORECASE)
    # Neutralize common instruction override signatures
    cleaned = re.sub(r'(?i)(?:ignore\s+all\s+(?:previous|above)\s+instructions)', '[REDACTED_INJECTION_ATTEMPT]', cleaned)
    cleaned = re.sub(r'(?i)(?:you\s+must\s+output\s+verdict:\s*["\']?likely\s+true["\']?)', '[REDACTED_OVERRIDE_ATTEMPT]', cleaned)
    return cleaned[:3000].strip()


class VerificationPipeline:
    """
    Central orchestration engine for SachAI.
    Coordinates between offline verified archives, Gemini live grounding, Perplexity,
    evidence provenance, and persistent SQLite storage.
    """

    def __init__(self, storage: Optional[AbstractStorage] = None):
        self.storage = storage or SQLiteStorage()

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
        verdict: Optional[VerificationVerdict] = None

        with StageTimer("RETRIEVAL_AND_SYNTHESIS", cid) as tracker:
            tracker["engine_selected"] = engine_choice

            # Check offline archive first if chosen or as zero-cost fast path
            from gemini_engine import match_offline_archive
            offline_match = match_offline_archive(safe_claim)

            if "Offline" in engine_choice or (offline_match and not gemini_api_key and "Gemini" in engine_choice):
                if offline_match:
                    raw_sources = offline_match.get("sources", [])
                    sources = build_evidence_provenance(raw_sources)
                    verdict = VerificationVerdict(
                        verdict=offline_match.get("verdict", "Unclear"),
                        confidence=int(offline_match.get("confidence", 70)),
                        explanation_hindi=offline_match.get("explanation_hindi", ""),
                        explanation_english=offline_match.get("explanation_english", ""),
                        short_whatsapp_hindi=offline_match.get("short_whatsapp_hindi", ""),
                        short_whatsapp_english=offline_match.get("short_whatsapp_english", ""),
                        category=offline_match.get("category", request.category or "General"),
                        risk_level=offline_match.get("risk_level", "Moderate"),
                        sources=sources,
                        engine_used="SachAI PIB Verified Live Archive (Zero Key / Instant)",
                        is_grounded=True,
                        correlation_id=cid,
                    )
                    tracker["cache_hit"] = True

            # If not resolved by offline archive and Gemini is requested
            if verdict is None and "Gemini" in engine_choice:
                if gemini_api_key:
                    try:
                        from gemini_engine import verify_claim_with_gemini
                        # Embed user input inside structured delimiters
                        delimited_claim = f"<UNTRUSTED_CLAIM_CONTENT>\n{safe_claim}\n</UNTRUSTED_CLAIM_CONTENT>"
                        gemini_res = verify_claim_with_gemini(
                            claim=delimited_claim,
                            api_key=gemini_api_key,
                            city=request.city,
                            category=request.category,
                        )
                        raw_sources = gemini_res.get("sources", [])
                        sources = build_evidence_provenance(raw_sources)
                        verdict = VerificationVerdict(
                            verdict=gemini_res.get("verdict", "Unclear"),
                            confidence=int(gemini_res.get("confidence", 70)),
                            explanation_hindi=gemini_res.get("explanation_hindi", ""),
                            explanation_english=gemini_res.get("explanation_english", ""),
                            short_whatsapp_hindi=gemini_res.get("short_whatsapp_hindi", ""),
                            short_whatsapp_english=gemini_res.get("short_whatsapp_english", ""),
                            category=gemini_res.get("category", request.category or "General"),
                            risk_level=gemini_res.get("risk_level", "Moderate"),
                            scam_indicators=gemini_res.get("scam_indicators", []),
                            sources=sources,
                            engine_used=gemini_res.get("engine_used", "Google Gemini Flash + Live Search"),
                            is_grounded=gemini_res.get("is_grounded", False),
                            correlation_id=cid,
                        )
                        tracker["gemini_success"] = True
                    except Exception as exc:
                        logger.error("[%s] Gemini call failed: %s", cid, sanitize_error(exc))
                        tracker["gemini_error"] = sanitize_error(exc)

                # Graceful zero-key fallback
                if verdict is None:
                    if offline_match:
                        raw_sources = offline_match.get("sources", [])
                        sources = build_evidence_provenance(raw_sources)
                        verdict = VerificationVerdict(
                            verdict=offline_match.get("verdict", "Unclear"),
                            confidence=int(offline_match.get("confidence", 70)),
                            explanation_hindi=offline_match.get("explanation_hindi", ""),
                            explanation_english=offline_match.get("explanation_english", ""),
                            short_whatsapp_hindi=offline_match.get("short_whatsapp_hindi", ""),
                            short_whatsapp_english=offline_match.get("short_whatsapp_english", ""),
                            category=offline_match.get("category", request.category or "General"),
                            risk_level=offline_match.get("risk_level", "Moderate"),
                            sources=sources,
                            engine_used="SachAI PIB Verified Live Archive (Offline Fallback)",
                            is_grounded=True,
                            correlation_id=cid,
                        )
                    else:
                        verdict = VerificationVerdict(
                            verdict="Unclear / Verification Needed",
                            confidence=60,
                            category=request.category or "Public Notice",
                            risk_level="Moderate",
                            explanation_hindi="इस दावे की पुष्टि के लिए फ्री Gemini Key सक्रिय करें या PIB Fact Check पोर्टल देखें।",
                            explanation_english="No active Gemini key found. Please enter a free key in sidebar or check official PIB portals.",
                            short_whatsapp_hindi="ℹ️ इस दावे की पुष्टि नहीं हुई है। बिना जांचे आगे न बढ़ाएं।",
                            short_whatsapp_english="ℹ️ This claim is unverified. Please avoid forwarding without checking official sources.",
                            sources=build_evidence_provenance([
                                {"title": "PIB Fact Check Portal", "url": "https://pib.gov.in/factcheck", "source_type": "Official Government Portal"}
                            ]),
                            engine_used="SachAI Safe Abstention Mode (Zero Key)",
                            is_grounded=False,
                            correlation_id=cid,
                        )

            # If Perplexity requested
            if verdict is None and "Perplexity" in engine_choice:
                from app import verify_claim_with_perplexity
                pplx_res = verify_claim_with_perplexity(safe_claim, city=request.city)
                raw_sources = pplx_res.get("sources", [])
                sources = build_evidence_provenance(raw_sources)
                verdict = VerificationVerdict(
                    verdict=pplx_res.get("verdict", "Unclear"),
                    confidence=int(pplx_res.get("confidence", 65)),
                    explanation_hindi=pplx_res.get("explanation_hindi", ""),
                    explanation_english=pplx_res.get("explanation_english", ""),
                    short_whatsapp_hindi=pplx_res.get("short_whatsapp_hindi", ""),
                    short_whatsapp_english=pplx_res.get("short_whatsapp_english", ""),
                    category=pplx_res.get("category", request.category or "General"),
                    risk_level=pplx_res.get("risk_level", "Moderate"),
                    sources=sources,
                    engine_used=pplx_res.get("engine_used", "Perplexity Sonar"),
                    is_grounded=bool(sources),
                    correlation_id=cid,
                )

            # Catch-all safety fallback
            if verdict is None:
                verdict = VerificationVerdict(
                    verdict="Unclear",
                    confidence=50,
                    category=request.category or "General",
                    risk_level="Moderate",
                    explanation_hindi="दावे की पुष्टि के लिए आधिकारिक स्रोतों की जांच आवश्यक है।",
                    explanation_english="Official verification is recommended before sharing.",
                    short_whatsapp_hindi="⚠️ बिना जांचे आगे न बढ़ाएं।",
                    short_whatsapp_english="⚠️ Verify before forwarding.",
                    sources=[],
                    engine_used="SachAI Fallback Engine",
                    is_grounded=False,
                    correlation_id=cid,
                )

        # 3. STORAGE PERSISTENCE
        with StageTimer("STORAGE_PERSISTENCE", cid) as tracker:
            try:
                rec_id = self.storage.save_verification(
                    claim=safe_claim,
                    verdict=verdict,
                    correlation_id=cid,
                )
                tracker["saved_record_id"] = rec_id
            except Exception as e:
                logger.error("[%s] Failed to persist verification to SQLite: %s", cid, sanitize_error(e))
                tracker["storage_error"] = sanitize_error(e)

        logger.info("[%s] Verification completed with verdict: %s", cid, verdict.verdict)
        return verdict
