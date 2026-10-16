"""
Tests for Part F: Storage Layer Schema v2 & Idempotency
Validates persistence of evidence records, source relationships, schema migrations,
and idempotent repeated ingestion.
"""

import os
import tempfile
import pytest
from core.models import EvidenceRecord, EvidenceRelationship, RelationshipType, StanceType
from core.storage import SQLiteStorage


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = SQLiteStorage(db_path=path)
    yield storage
    if os.path.exists(path):
        os.remove(path)


def test_schema_v2_migration_applied(temp_db):
    with temp_db._get_connection() as conn:
        row = conn.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;").fetchone()
        assert row is not None
        assert row["version"] == 2


def test_save_and_retrieve_evidence_records(temp_db):
    cid = "test-corr-1234"
    records = [
        EvidenceRecord(
            evidence_id="ev-storage-1",
            source_url="https://pib.gov.in/test1",
            canonical_url="https://pib.gov.in/test1",
            domain="pib.gov.in",
            publisher="PIB India",
            title="Press Notice",
            passage="Verified factual passage text.",
            published_at="2026-03-01",
            content_hash="hash-storage-1",
            authority_tier=1,
            is_primary_source=True,
            stance=StanceType.SUPPORTS,
            verification_status="verified",
        )
    ]
    relationships = [
        EvidenceRelationship(
            source_id="ev-storage-1",
            target_id="ev-storage-2",
            relationship_type=RelationshipType.INDEPENDENT,
            similarity_score=0.15,
            rationale="Independent report",
        )
    ]

    saved_count = temp_db.save_evidence_records(cid, records, relationships)
    assert saved_count == 1

    retrieved_ev = temp_db.get_evidence_by_correlation_id(cid)
    assert len(retrieved_ev) == 1
    assert retrieved_ev[0].evidence_id == "ev-storage-1"
    assert retrieved_ev[0].authority_tier == 1
    assert retrieved_ev[0].is_primary_source is True

    retrieved_rel = temp_db.get_relationships_by_correlation_id(cid)
    assert len(retrieved_rel) == 1
    assert retrieved_rel[0].relationship_type == RelationshipType.INDEPENDENT


def test_idempotent_ingestion_of_evidence_records(temp_db):
    cid = "test-corr-idempotent"
    record = EvidenceRecord(
        evidence_id="ev-idemp-1",
        source_url="https://rbi.org.in/notice",
        domain="rbi.org.in",
        passage="Unique passage text.",
        content_hash="unique_hash_1",
        authority_tier=1,
        stance=StanceType.CONTRADICTS,
    )

    # First ingestion
    temp_db.save_evidence_records(cid, [record], [])
    # Second identical ingestion
    temp_db.save_evidence_records(cid, [record], [])

    # Must contain exactly 1 record, not 2
    records = temp_db.get_evidence_by_correlation_id(cid)
    assert len(records) == 1
    assert records[0].evidence_id == "ev-idemp-1"
