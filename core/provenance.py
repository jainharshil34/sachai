"""
SachAI Evidence Provenance & Source Authority Engine
Classifies authority tiers, canonicalizes URLs, and deduplicates evidence citations.
"""

import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Set
from core.models import EvidenceSource

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


def canonicalize_url(url: str) -> str:
    """Strip tracking parameters, fragments, and normalize trailing slash."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        # Filter query params
        query_dict = urllib.parse.parse_qs(parsed.query)
        filtered_query = {
            k: v for k, v in query_dict.items()
            if not k.startswith("utm_") and k not in {"fbclid", "gclid", "ref"}
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


def deduplicate_sources(sources: List[EvidenceSource]) -> List[EvidenceSource]:
    """
    Deduplicate sources by canonical URL and domain, keeping the highest authority source.
    """
    seen_urls: Set[str] = set()
    deduped: List[EvidenceSource] = []

    # Sort sources by authority tier (1 first, then 2, etc.)
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
    """
    Transform raw source dictionaries into validated, tiered, and deduplicated EvidenceSource models.
    """
    evidence_list: List[EvidenceSource] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue
        url = raw.get("url", "").strip()
        tier = classify_authority_tier(url)
        is_official = tier == 1

        title = raw.get("title", "").strip() or ("Official Portal" if is_official else "Web Source")
        snippet = raw.get("snippet", "").strip()
        source_type = raw.get("source_type", "")
        if not source_type:
            if tier == 1:
                source_type = "Official Indian Portal"
            elif tier == 2:
                source_type = "Accredited Fact-Checker"
            elif tier == 3:
                source_type = "Established News / Wire"
            else:
                source_type = "Web Citation"

        evidence_list.append(EvidenceSource(
            title=title,
            url=url,
            source_type=source_type,
            snippet=snippet,
            authority_tier=tier,
            is_official=is_official,
        ))

    return deduplicate_sources(evidence_list)
