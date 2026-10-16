"""
Tests for Reliability, Error Handling, and Malformed Output Recovery
Validates source-fetch failure fallbacks, malformed JSON repair,
and graceful degradation without crashing the core verification pipeline.
"""

from unittest.mock import patch, MagicMock
from core.models import VerificationRequest
from core.pipeline import VerificationPipeline
from core.storage import SQLiteStorage
import tempfile
import os
import pytest


@pytest.fixture
def isolated_pipeline():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = SQLiteStorage(db_path=path)
    pipeline = VerificationPipeline(storage=storage)
    yield pipeline, storage
    if os.path.exists(path):
        os.remove(path)


def test_malformed_llm_json_recovers_gracefully():
    from gemini_engine import clean_json_response

    # Text containing JSON with surrounding chat formatting
    raw_response = "Here is the verified analysis:\n```json\n{\"verdict\": \"Likely False\", \"confidence\": 90, \"category\": \"Scam\"}\n```\nHope this helps!"
    parsed = clean_json_response(raw_response)

    assert parsed.get("verdict") == "Likely False"
    assert parsed.get("confidence") == 90


def test_completely_invalid_llm_output_returns_empty_dict():
    from gemini_engine import clean_json_response

    garbage = "I am an AI and I cannot verify this claim due to safety guidelines."
    parsed = clean_json_response(garbage)

    assert isinstance(parsed, dict)
    assert parsed == {}


def test_source_fetch_network_failure_falls_back(isolated_pipeline):
    pipeline, _ = isolated_pipeline
    req = VerificationRequest(claim="Unknown viral claim during internet disconnection")

    # Mock external API failure
    with patch("requests.post", side_effect=Exception("Connection refused / DNS failure")):
        verdict = pipeline.verify(
            request=req,
            engine_choice="Perplexity Sonar (Search + Reasoning)",
            perplexity_api_key="mock-key-12345",
        )

    assert verdict is not None
    assert verdict.verdict is not None
    # Pipeline did not crash, maintained graceful status
    assert verdict.confidence >= 0
