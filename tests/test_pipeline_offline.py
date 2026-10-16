"""
Tests for core/pipeline.py offline mode
Validates deterministic offline fact-checking without any API keys or network dependencies.
"""

import tempfile
import os
import pytest
from core.models import VerificationRequest
from core.pipeline import VerificationPipeline
from core.storage import SQLiteStorage


@pytest.fixture
def clean_pipeline():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = SQLiteStorage(db_path=path)
    pipeline = VerificationPipeline(storage=storage)
    yield pipeline, storage
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass



def test_offline_salt_shortage_match(clean_pipeline):
    pipeline, storage = clean_pipeline
    req = VerificationRequest(
        claim="व्हाट्सएप पर खबर है कि मेरठ में नमक की भारी कमी होने वाली है",
        city="Meerut",
        category="Rumour / Panic",
    )
    verdict = pipeline.verify(
        request=req,
        engine_choice="Offline PIB & Cyber Threat Archive (0 Keys Needed)",
    )

    assert verdict.verdict == "Likely False"
    assert verdict.confidence >= 90
    assert "नमक" in verdict.explanation_hindi or "salt" in verdict.explanation_english.lower()
    assert len(verdict.sources) >= 1
    assert verdict.sources[0].authority_tier == 1
    assert verdict.sources[0].is_official is True

    # Check persistence
    records = storage.get_recent_verifications(limit=5)
    assert len(records) == 1
    assert records[0].correlation_id == req.correlation_id


def test_offline_star_500_note_match(clean_pipeline):
    pipeline, _ = clean_pipeline
    req = VerificationRequest(claim="स्टार 500 का नोट अमान्य है")
    verdict = pipeline.verify(
        request=req,
        engine_choice="Offline PIB & Cyber Threat Archive (0 Keys Needed)",
    )

    assert verdict.verdict == "Likely True"
    assert verdict.confidence >= 90
    assert any(s.authority_tier == 1 for s in verdict.sources)


def test_offline_digital_arrest_scam(clean_pipeline):
    pipeline, _ = clean_pipeline
    req = VerificationRequest(claim="Customs police digital arrest Skype call notice")
    verdict = pipeline.verify(
        request=req,
        engine_choice="Offline PIB & Cyber Threat Archive (0 Keys Needed)",
    )

    assert verdict.verdict == "Likely False" or "Scam" in verdict.verdict
    assert verdict.risk_level == "Severe Scam"


def test_unknown_claim_abstains_gracefully(clean_pipeline):
    pipeline, _ = clean_pipeline
    req = VerificationRequest(claim="XYZ random unknown company founded in 1923 by John Doe")
    verdict = pipeline.verify(
        request=req,
        engine_choice="Google Gemini 1.5/2.0 Flash (Free API)",
        gemini_api_key=None,  # Zero-key mode
    )

    assert "Unclear" in verdict.verdict or "Verification Needed" in verdict.verdict
    assert verdict.confidence <= 70
    assert verdict.is_grounded is False
