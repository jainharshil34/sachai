"""
Tests for core/provenance.py
Validates authority tiering, URL canonicalization, and evidence deduplication.
"""

from core.provenance import (
    build_evidence_provenance,
    canonicalize_url,
    classify_authority_tier,
    deduplicate_sources,
)
from core.models import EvidenceSource


def test_classify_authority_tier_gov_and_pib():
    assert classify_authority_tier("https://pib.gov.in/factcheck/123") == 1
    assert classify_authority_tier("https://cybercrime.gov.in/default.aspx") == 1
    assert classify_authority_tier("https://rbi.org.in/scripts/press.aspx") == 1
    assert classify_authority_tier("https://echallan.parivahan.gov.in") == 1
    assert classify_authority_tier("https://delhipolice.nic.in") == 1


def test_classify_authority_tier_factcheckers():
    assert classify_authority_tier("https://www.altnews.in/viral-claim-debunk/") == 2
    assert classify_authority_tier("https://boomlive.in/fact-check/story") == 2
    assert classify_authority_tier("https://factly.in/fake-scheme/") == 2


def test_classify_authority_tier_mainstream_news():
    assert classify_authority_tier("https://www.thehindu.com/news/national/story.html") == 3
    assert classify_authority_tier("https://indianexpress.com/article/india/") == 3
    assert classify_authority_tier("https://www.ptinews.com/news/1234") == 3


def test_classify_authority_tier_generic_web():
    assert classify_authority_tier("https://random-rumour-blog.xyz/news") == 4
    assert classify_authority_tier("https://unknown-site.top/free-scheme") == 4
    assert classify_authority_tier("") == 4


def test_canonicalize_url_strips_tracking():
    raw = "https://www.pib.gov.in/factcheck/?utm_source=whatsapp&utm_medium=forward&fbclid=98765"
    cleaned = canonicalize_url(raw)
    assert "utm_source" not in cleaned
    assert "fbclid" not in cleaned
    assert cleaned.startswith("https://www.pib.gov.in/factcheck")


def test_deduplicate_sources():
    sources = [
        EvidenceSource(title="Generic Copy", url="https://pib.gov.in/factcheck?utm_campaign=share", authority_tier=4),
        EvidenceSource(title="PIB Official", url="https://pib.gov.in/factcheck", authority_tier=1, is_official=True),
        EvidenceSource(title="Other News", url="https://thehindu.com/news", authority_tier=3),
    ]
    deduped = deduplicate_sources(sources)
    # The duplicate PIB URL should be merged, retaining the highest authority (tier 1)
    assert len(deduped) == 2
    assert deduped[0].authority_tier == 1
    assert deduped[0].is_official is True


def test_build_evidence_provenance_full_pipeline():
    raw = [
        {"title": "Govt Gazette", "url": "https://mha.gov.in/circular.pdf", "snippet": "Official circular"},
        {"title": "Fact Check", "url": "https://altnews.in/scam-check", "snippet": "Debunking viral rumor"},
        {"title": "Duplicate Gazette", "url": "https://mha.gov.in/circular.pdf?utm_source=twitter", "snippet": "Duplicate"},
    ]
    ev = build_evidence_provenance(raw)
    assert len(ev) == 2
    assert ev[0].authority_tier == 1
    assert ev[0].is_official is True
    assert ev[1].authority_tier == 2
