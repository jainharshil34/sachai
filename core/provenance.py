"""
SachAI Evidence Provenance & Source Authority Engine
Classifies authority tiers, canonicalizes URLs, detects exact/near duplicates,
tracks upstream wire syndications, and builds evidence relationships.
"""

import hashlib
import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple
from core.models import EvidenceRecord, EvidenceRelationship, EvidenceSource, RelationshipType

logger = logging.getLogger("SachAI.Provenance")

# Known authority domain mappings
TIER_1_DOMAINS: Set[str] = {
    "pib.gov.in",
    "gov.in",
    "nic.in",
    "rbi.org.in",
    "cybercrime.gov.in",
    "parivahan.gov.in",
    "echallan.parivahan.gov.in",
    "sbi.co.in",
    "mha.gov.in",
    "who.int",
    "un.org",
    "sci.gov.in",
    "sci.nic.in",
}

TIER_2_DOMAINS: Set[str] = {
    "altnews.in",
    "boomlive.in",
    "factly.in",
    "vishvasnews.com",
    "newsmobile.in",
    "thip.media",
}

TIER_3_DOMAINS: Set[str] = {
    "ptinews.com",
    "aniin.com",
    "thehindu.com",
    "indianexpress.com",
    "bbc.com",
    "reuters.com",
    "hindustantimes.com",
    "ndtv.com",
    "timesofindia.indiatimes.com",
    "livemint.com",
}

KNOWN_WIRE_AGENCIES: Set[str] = {
    "pti",
    "press trust of india",
    "ani",
    "asian news international",
    "reuters",
    "pib",
    "press information bureau",
    "ians",
    "uni",
    "afp",
    "bloomberg",
}


def extract_domain(url: str) -> str:
    """Extract lowercase netloc domain from URL."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def classify_authority_tier(url: str) -> int:
    """
    Classify authority tier (1 to 4):
    Tier 1: Government, regulators, official gazettes, official cybersecurity portals
    Tier 2: IFCN-accredited independent fact-checking organizations
    Tier 3: Established national press agencies and accredited newspapers
    Tier 4: General web, blogs, forums, unverified platforms
    """
    domain = extract_domain(url)
    if not domain:
        return 4

    # Check Tier 1
    if domain.endswith(".gov.in") or domain.endswith(".nic.in") or any(domain == d or domain.endswith(f".{d}") for d in TIER_1_DOMAINS):
        return 1

    # Check Tier 2
    if any(domain == d or domain.endswith(f".{d}") for d in TIER_2_DOMAINS):
        return 2

    # Check Tier 3
    if any(domain == d or domain.endswith(f".{d}") for d in TIER_3_DOMAINS):
        return 3

    return 4


def is_primary_source(domain: str, source_type: str = "") -> bool:
    """
    Check if evidence represents a primary official document, gazette, or agency notice.
    """
    d_lower = domain.lower()
    s_lower = source_type.lower()
    if d_lower.endswith(".gov.in") or d_lower.endswith(".nic.in") or d_lower == "rbi.org.in":
        return True
    if any(kw in s_lower for kw in ["gazette", "notification", "circular", "court order", "advisory"]):
        return True
    return False


def canonicalize_url(url: str) -> str:
    """Strip tracking parameters, fragments, and normalize trailing slash."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        query_dict = urllib.parse.parse_qs(parsed.query)
        filtered_query = {
            k: v for k, v in query_dict.items()
            if not k.startswith("utm_") and k not in {"fbclid", "gclid", "ref", "source"}
        }
        clean_query = urllib.parse.urlencode(filtered_query, doseq=True)
        path = parsed.path.rstrip("/")
        normalized = urllib.parse.urlunparse((
            parsed.scheme.lower() or "https",
            parsed.netloc.lower(),
            path,
            "",
            clean_query,
            "",
        ))
        return normalized
    except Exception:
        return url.strip()


def normalize_text_for_hashing(text: str) -> str:
    """Normalize text (lowercase, whitespace collapse, strip punctuation) for exact duplicate check."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return " ".join(cleaned.split())


def compute_content_hash(text: str) -> str:
    """Deterministic SHA-256 hash of normalized passage content."""
    normalized = normalize_text_for_hashing(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_shingle_jaccard(text1: str, text2: str, k: int = 3) -> float:
    """
    Compute Jaccard similarity over word k-shingles.
    Used for detecting near-duplicate syndicated reporting.
    """
    words1 = normalize_text_for_hashing(text1).split()
    words2 = normalize_text_for_hashing(text2).split()

    if len(words1) < k or len(words2) < k:
        # Fall back to token-level set intersection
        s1, s2 = set(words1), set(words2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    shingles1 = set(" ".join(words1[i:i + k]) for i in range(len(words1) - k + 1))
    shingles2 = set(" ".join(words2[i:i + k]) for i in range(len(words2) - k + 1))

    if not shingles1 or not shingles2:
        return 0.0
    return len(shingles1 & shingles2) / len(shingles1 | shingles2)


def extract_upstream_attributions(text: str) -> List[str]:
    """
    Extract explicit references to upstream wire services (PTI, ANI, Reuters, PIB, etc.).
    e.g. 'Reporting by PTI', 'According to ANI', 'News agency IANS reported'.
    """
    found: List[str] = []
    text_lower = text.lower()

    patterns = [
        r'(?:reported by|reporting by|according to|as per|source:|quoted by|news agency)\s+([a-zA-Z\s]{2,25})',
        r'\(([a-zA-Z\s]{2,15})\)\s*[-—–]',
    ]

    for p in patterns:
        for match in re.finditer(p, text_lower):
            candidate = match.group(1).strip().lower()
            for wire in KNOWN_WIRE_AGENCIES:
                if wire in candidate:
                    canonical_name = wire.upper()
                    if canonical_name not in found:
                        found.append(canonical_name)

    return found


def analyze_evidence_provenance(
    raw_sources: List[Dict[str, Any]]
) -> Tuple[List[EvidenceRecord], List[EvidenceRelationship], Dict[str, List[str]]]:
    """
    Transform raw search/retrieval results into rich EvidenceRecords with:
    1. Stable evidence IDs
    2. Exact duplicate identification
    3. Near-duplicate syndication clustering
    4. Explicit upstream reference discovery
    5. Confirmed vs suspected dependency graphs
    """
    records: List[EvidenceRecord] = []
    relationships: List[EvidenceRelationship] = []
    clusters: Dict[str, List[str]] = {}

    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue
        raw_url = raw.get("url", "").strip()
        canon_url = canonicalize_url(raw_url)
        domain = extract_domain(canon_url or raw_url)
        passage = raw.get("snippet", "").strip() or raw.get("passage", "").strip() or raw.get("title", "").strip()
        if not passage:
            continue

        ev_id = EvidenceRecord.generate_id(canon_url, passage)
        c_hash = compute_content_hash(passage)
        tier = classify_authority_tier(canon_url)
        is_prim = is_primary_source(domain, raw.get("source_type", ""))
        upstream = extract_upstream_attributions(passage)
        pub_at = raw.get("published_at") or raw.get("published_date")  # None if missing; never invent

        rec = EvidenceRecord(
            evidence_id=ev_id,
            source_url=raw_url,
            canonical_url=canon_url,
            publisher=raw.get("publisher") or domain or None,
            domain=domain,
            title=raw.get("title") or "Source",
            passage=passage,
            published_at=pub_at,
            source_type=raw.get("source_type", "Web Citation"),
            authority_tier=tier,
            is_primary_source=is_prim,
            content_hash=c_hash,
            upstream_source_refs=upstream,
            verification_status="verified" if tier in [1, 2] else "unverified",
        )
        records.append(rec)

    # Cross-analyze records for syndication & dependencies
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            r1 = records[i]
            r2 = records[j]

            # 1. Exact Duplicate
            if r1.content_hash == r2.content_hash:
                rel = EvidenceRelationship(
                    source_id=r1.evidence_id,
                    target_id=r2.evidence_id,
                    relationship_type=RelationshipType.CONFIRMED_DEPENDENCY,
                    similarity_score=1.0,
                    rationale="Exact duplicate passage content hash match",
                )
                relationships.append(rel)
                cluster_key = f"cluster-{r1.content_hash[:8]}"
                r1.syndication_cluster_id = cluster_key
                r2.syndication_cluster_id = cluster_key
                clusters.setdefault(cluster_key, []).extend([r1.evidence_id, r2.evidence_id])
                continue

            # 2. Upstream Reference
            common_upstream = set(r1.upstream_source_refs) & set(r2.upstream_source_refs)
            sim = compute_shingle_jaccard(r1.passage, r2.passage)

            if common_upstream and sim >= 0.40:
                rel = EvidenceRelationship(
                    source_id=r1.evidence_id,
                    target_id=r2.evidence_id,
                    relationship_type=RelationshipType.CONFIRMED_DEPENDENCY,
                    similarity_score=round(sim, 3),
                    rationale=f"Both cite shared upstream wire: {', '.join(common_upstream)} with high textual overlap",
                )
                relationships.append(rel)
                cluster_key = f"wire-{list(common_upstream)[0]}"
                r1.syndication_cluster_id = cluster_key
                r2.syndication_cluster_id = cluster_key
                clusters.setdefault(cluster_key, []).extend([r1.evidence_id, r2.evidence_id])

            # 3. Suspected Syndication (high similarity without explicit citation)
            elif sim >= 0.70:
                rel = EvidenceRelationship(
                    source_id=r1.evidence_id,
                    target_id=r2.evidence_id,
                    relationship_type=RelationshipType.SUSPECTED_DEPENDENCY,
                    similarity_score=round(sim, 3),
                    rationale=f"Suspected syndication / reprinted copy (similarity: {int(sim*100)}%)",
                )
                relationships.append(rel)
                cluster_key = f"suspected-{r1.evidence_id[:8]}"
                r1.syndication_cluster_id = cluster_key
                r2.syndication_cluster_id = cluster_key
                clusters.setdefault(cluster_key, []).extend([r1.evidence_id, r2.evidence_id])

            else:
                rel = EvidenceRelationship(
                    source_id=r1.evidence_id,
                    target_id=r2.evidence_id,
                    relationship_type=RelationshipType.INDEPENDENT,
                    similarity_score=round(sim, 3),
                    rationale="Independent reporting with distinct phrasing",
                )
                relationships.append(rel)

    # Deduplicate cluster lists
    for k in clusters:
        clusters[k] = list(set(clusters[k]))

    return records, relationships, clusters


def deduplicate_sources(sources: List[EvidenceSource]) -> List[EvidenceSource]:
    """Legacy-compatible deduplication for EvidenceSource list."""
    seen_urls: Set[str] = set()
    deduped: List[EvidenceSource] = []
    sorted_sources = sorted(sources, key=lambda s: s.authority_tier)

    for src in sorted_sources:
        canon_url = canonicalize_url(src.url)
        key = canon_url if canon_url else f"{src.title}:{src.source_type}"
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(src)

    return deduped


def build_evidence_provenance(raw_sources: List[Dict[str, Any]]) -> List[EvidenceSource]:
    """Legacy-compatible factory transforming raw source dictionaries to EvidenceSource."""
    records, _, _ = analyze_evidence_provenance(raw_sources)
    legacy_sources: List[EvidenceSource] = []

    for r in records:
        legacy_sources.append(EvidenceSource(
            title=r.title or "Official Source",
            url=r.canonical_url or r.source_url,
            source_type=r.source_type,
            snippet=r.passage,
            authority_tier=r.authority_tier,
            is_official=r.is_primary_source or (r.authority_tier == 1),
            stance=r.stance.value,
        ))

    return deduplicate_sources(legacy_sources)
