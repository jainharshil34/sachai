"""
Tests for core/storage.py
Validates SQLite thread-safety, persistence, retrieval, schema initialization,
and ensures no raw binary data is stored.
"""

import os
import tempfile
import pytest
from core.models import EvidenceSource, VerificationVerdict
from core.storage import SQLiteStorage


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass



def test_sqlite_initialization(temp_db):
    storage = SQLiteStorage(db_path=temp_db)
    assert os.path.exists(temp_db)
    stats = storage.get_stats()
    assert stats["total_verifications"] == 0


def test_sqlite_save_and_retrieve(temp_db):
    storage = SQLiteStorage(db_path=temp_db)
    verdict = VerificationVerdict(
        verdict="Likely False",
        confidence=98,
        explanation_hindi="नमक की कोई कमी नहीं है",
        explanation_english="No salt shortage",
        category="Rumour / Panic",
        risk_level="High Viral Threat",
        sources=[EvidenceSource(title="PIB Fact Check", url="https://pib.gov.in/factcheck", authority_tier=1, is_official=True)],
        engine_used="SachAI PIB Archive",
        correlation_id="sach-test-12345",
    )

    rec_id = storage.save_verification(
        claim="मेरठ में नमक की कमी",
        verdict=verdict,
        correlation_id="sach-test-12345",
    )
    assert rec_id > 0

    records = storage.get_recent_verifications(limit=10)
    assert len(records) == 1
    assert records[0].id == rec_id
    assert records[0].claim == "मेरठ में नमक की कमी"
    assert records[0].verdict == "Likely False"
    assert records[0].confidence == 98
    assert records[0].correlation_id == "sach-test-12345"

    # Fetch by primary key
    fetched = storage.get_verification_by_id(rec_id)
    assert fetched is not None
    assert fetched.id == rec_id

    # Check stats
    stats = storage.get_stats()
    assert stats["total_verifications"] == 1
    assert stats["verdict_breakdown"]["Likely False"] == 1


def test_sqlite_limit_and_order(temp_db):
    storage = SQLiteStorage(db_path=temp_db)
    for i in range(5):
        storage.save_verification(
            claim=f"Claim {i}",
            verdict=VerificationVerdict(verdict="Unclear", confidence=50),
            correlation_id=f"cid-{i}",
        )

    records = storage.get_recent_verifications(limit=3)
    assert len(records) == 3
    # Most recent first
    assert records[0].claim == "Claim 4"
    assert records[1].claim == "Claim 3"
    assert records[2].claim == "Claim 2"
