"""
Tests for Part A: Evidence Normalization & Integrity
Validates stable evidence IDs, URL canonicalization, tracking parameter stripping,
preservation of missing metadata without invention, and exact passage integrity.
"""

import pytest
from core.models import EvidenceRecord, StanceType
from core.provenance import canonicalize_url, extract_domain, compute_content_hash


def test_evidence_id_deterministic_and_stable():
    url = "https://pib.gov.in/PressReleasePage.aspx?PRID=12345"
    passage = "Department confirms adequate salt stock across all districts."

    id1 = EvidenceRecord.generate_id(url, passage)
    id2 = EvidenceRecord.generate_id(url, passage)

    assert id1.startswith("ev-")
    assert id1 == id2
    assert len(id1) == 19  # 'ev-' + 16 hex chars


def test_canonicalize_url_strips_tracking_safely():
    dirty_url = "https://timesofindia.indiatimes.com/india/story?utm_source=whatsapp&utm_medium=social&fbclid=IwAR123&gclid=xyz&ref=share#top"
    cleaned = canonicalize_url(dirty_url)

    assert "utm_source" not in cleaned
    assert "utm_medium" not in cleaned
    assert "fbclid" not in cleaned
    assert "gclid" not in cleaned
    assert "timesofindia.indiatimes.com/india/story" in cleaned
    assert cleaned.startswith("https://")


def test_canonicalize_url_preserves_legitimate_query_params():
    press_url = "https://pib.gov.in/PressReleasePage.aspx?PRID=1987654"
    cleaned = canonicalize_url(press_url)

    assert "PRID=1987654" in cleaned
    assert cleaned.startswith("https://pib.gov.in/PressReleasePage.aspx")


def test_extract_domain_handles_various_formats():
    assert extract_domain("https://subdomain.pib.gov.in/path") == "subdomain.pib.gov.in"
    assert extract_domain("http://www.rbi.org.in") == "rbi.org.in"
    assert extract_domain("cybercrime.gov.in/report") == "cybercrime.gov.in"
    assert extract_domain("") == ""


def test_missing_metadata_never_invented():
    rec = EvidenceRecord(
        evidence_id="ev-test1234",
        source_url="https://example.com/unverified-report",
        domain="example.com",
        passage="An unverified message claimed unusual military movement.",
        published_at=None,  # Unknown timestamp must stay None
        publisher=None,     # Unknown publisher must stay None
        content_hash=compute_content_hash("An unverified message claimed unusual military movement."),
        stance=StanceType.NEUTRAL_OR_UNRELATED,
        verification_status="unverified",
    )

    assert rec.published_at is None
    assert rec.publisher is None
    assert rec.verification_status == "unverified"
    assert rec.is_primary_source is False


def test_content_hash_identical_for_normalized_whitespace_case():
    t1 = "  Official Announcement: Stocks Are Sufficient!  "
    t2 = "official announcement: stocks are sufficient!"
    t3 = "Official announcement: stocks are different!"

    h1 = compute_content_hash(t1)
    h2 = compute_content_hash(t2)
    h3 = compute_content_hash(t3)

    assert h1 == h2
    assert h1 != h3
