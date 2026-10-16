"""
Tests for Part B: Source Provenance, Wire Syndication, and Dependency Tracking
Validates exact duplicate hashing, shingle-based near duplicate detection,
upstream wire references (PTI/ANI/PIB), and primary source classification.
"""

from core.models import RelationshipType
from core.provenance import (
    analyze_evidence_provenance,
    classify_authority_tier,
    compute_shingle_jaccard,
    extract_upstream_attributions,
    is_primary_source,
)


def test_exact_duplicate_content_hash_creates_confirmed_dependency():
    raw_sources = [
        {
            "title": "Outlet A News Report",
            "url": "https://outleta.com/story1",
            "snippet": "Department confirms adequate salt stock across all districts in Uttar Pradesh.",
        },
        {
            "title": "Outlet B News Report",
            "url": "https://outletb.com/story2",
            "snippet": "Department confirms adequate salt stock across all districts in Uttar Pradesh.",
        },
    ]

    records, relationships, clusters = analyze_evidence_provenance(raw_sources)

    assert len(records) == 2
    assert len(relationships) == 1
    rel = relationships[0]
    assert rel.relationship_type == RelationshipType.CONFIRMED_DEPENDENCY
    assert rel.similarity_score == 1.0
    assert "Exact duplicate" in rel.rationale
    assert len(clusters) == 1


def test_near_duplicate_syndication_detection_via_jaccard_shingles():
    text1 = "Ministry of Consumer Affairs issued strict clarification today warning citizens that salt shortage rumors are baseless."
    text2 = "Ministry of Consumer Affairs issued strict clarification today cautioning citizens that salt shortage rumors are baseless."
    unrelated = "Cricket board announced the schedule for upcoming international series."

    sim_high = compute_shingle_jaccard(text1, text2, k=3)
    sim_low = compute_shingle_jaccard(text1, unrelated, k=3)

    assert sim_high > 0.60
    assert sim_low < 0.10


def test_extract_upstream_wire_attributions():
    text_pti = "New Delhi, Oct 15 (PTI) - The Supreme Court on Tuesday dismissed the petition."
    text_ani = "According to ANI, security forces intensified border patrolling."
    text_reuters = "Reporting by Reuters: Central bank kept interest rates unchanged."
    text_no_wire = "Local residents held a meeting to discuss water pipeline repairs."

    assert "PTI" in extract_upstream_attributions(text_pti)
    assert "ANI" in extract_upstream_attributions(text_ani)
    assert "REUTERS" in extract_upstream_attributions(text_reuters)
    assert len(extract_upstream_attributions(text_no_wire)) == 0


def test_syndication_cluster_groups_shared_upstream_wire():
    raw_sources = [
        {
            "title": "News Site 1",
            "url": "https://site1.com/pib-report",
            "snippet": "New Delhi (PTI) - The Department confirmed sufficient stock and warned against hoarding.",
        },
        {
            "title": "News Site 2",
            "url": "https://site2.com/pib-report",
            "snippet": "New Delhi (PTI) - The Department confirmed sufficient stock and warned against hoarding.",
        },
    ]

    records, relationships, clusters = analyze_evidence_provenance(raw_sources)

    assert len(records) == 2
    assert len(relationships) == 1
    assert relationships[0].relationship_type == RelationshipType.CONFIRMED_DEPENDENCY
    assert records[0].syndication_cluster_id is not None
    assert records[0].syndication_cluster_id == records[1].syndication_cluster_id


def test_primary_source_classification():
    assert is_primary_source("pib.gov.in", "Government Press Release") is True
    assert is_primary_source("rbi.org.in", "Official Circular") is True
    assert is_primary_source("sci.gov.in", "Court Order") is True
    assert is_primary_source("randomblog.in", "Web Blog") is False
    assert is_primary_source("thehindu.com", "News Article") is False


def test_authority_tier_hierarchy():
    assert classify_authority_tier("https://pib.gov.in/PressRelease") == 1
    assert classify_authority_tier("https://altnews.in/fact-check") == 2
    assert classify_authority_tier("https://thehindu.com/news") == 3
    assert classify_authority_tier("https://random-rumor-blog.xyz/post") == 4
