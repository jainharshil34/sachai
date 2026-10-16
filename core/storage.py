"""
SachAI Storage & Persistence Layer
Abstract storage interface and local SQLite implementation for persisting verification history,
evidence records, source relationships, and investigation snapshots.
"""

from contextlib import contextmanager
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.models import (
    EvidenceRecord,
    EvidenceRelationship,
    RelationshipType,
    StanceType,
    VerificationRecord,
    VerificationVerdict,
)

logger = logging.getLogger("SachAI.Storage")

DEFAULT_DB_PATH = "sachai_history.db"
SCHEMA_VERSION = 2


class AbstractStorage(ABC):
    """Abstract interface defining persistence operations for verification history and provenance."""

    @abstractmethod
    def save_verification(
        self,
        claim: str,
        verdict: VerificationVerdict,
        correlation_id: str,
    ) -> int:
        """Persist a verification snapshot without storing sensitive raw binaries."""
        pass

    @abstractmethod
    def get_recent_verifications(self, limit: int = 20) -> List[VerificationRecord]:
        """Retrieve recent verification records."""
        pass

    @abstractmethod
    def get_verification_by_id(self, record_id: int) -> Optional[VerificationRecord]:
        """Fetch a specific verification record by its database primary key."""
        pass

    @abstractmethod
    def save_evidence_records(
        self,
        correlation_id: str,
        records: List[EvidenceRecord],
        relationships: List[EvidenceRelationship],
    ) -> int:
        """Persist normalized evidence records and source relationships idempotently."""
        pass

    @abstractmethod
    def get_evidence_by_correlation_id(self, correlation_id: str) -> List[EvidenceRecord]:
        """Retrieve all evidence passages associated with a specific correlation ID."""
        pass

    @abstractmethod
    def get_relationships_by_correlation_id(self, correlation_id: str) -> List[EvidenceRelationship]:
        """Retrieve pairwise evidence relationships for a correlation ID."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics (total count, breakdown by verdict)."""
        pass


class SQLiteStorage(AbstractStorage):
    """
    Local SQLite implementation of AbstractStorage.
    Stores metadata, evidence provenance, and structured verdicts.
    Strictly excludes raw media binaries to guarantee DPDP privacy compliance.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize and migrate database schema with proper indexes and versioning."""
        with self._get_connection() as conn:
            # 1. Verification history (Phase 1)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    engine_used TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verif_created ON verification_history(created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verif_corr ON verification_history(correlation_id);")

            # 2. Evidence Records (Phase 2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id TEXT UNIQUE NOT NULL,
                    correlation_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    canonical_url TEXT,
                    domain TEXT NOT NULL,
                    publisher TEXT,
                    title TEXT,
                    passage TEXT NOT NULL,
                    published_at TEXT,
                    retrieved_at TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    authority_tier INTEGER NOT NULL,
                    is_primary_source INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    upstream_source_refs TEXT,
                    syndication_cluster_id TEXT,
                    stance TEXT NOT NULL,
                    stance_method TEXT,
                    verification_status TEXT NOT NULL DEFAULT 'unverified'
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ev_corr ON evidence_records(correlation_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ev_domain ON evidence_records(domain);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ev_hash ON evidence_records(content_hash);")

            # 3. Source Relationships (Phase 2)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    similarity_score REAL DEFAULT 0.0,
                    rationale TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_corr ON source_relationships(correlation_id);")

            # 4. Schema version tracker
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version) VALUES (?);",
                (SCHEMA_VERSION,),
            )
            conn.commit()
        logger.info("SQLite schema initialized/migrated to version %d at %s", SCHEMA_VERSION, self.db_path)

    def save_verification(
        self,
        claim: str,
        verdict: VerificationVerdict,
        correlation_id: str,
    ) -> int:
        clean_claim = claim[:3000].strip()
        payload = verdict.model_dump_json()

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO verification_history (
                    correlation_id, claim, verdict, confidence,
                    category, risk_level, engine_used, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correlation_id,
                    clean_claim,
                    verdict.verdict,
                    verdict.confidence,
                    verdict.category,
                    verdict.risk_level,
                    verdict.engine_used,
                    verdict.created_at,
                    payload,
                ),
            )
            conn.commit()
            record_id = cur.lastrowid or 0
        logger.info("[%s] Saved verification snapshot with ID %d", correlation_id, record_id)
        return record_id

    def save_evidence_records(
        self,
        correlation_id: str,
        records: List[EvidenceRecord],
        relationships: List[EvidenceRelationship],
    ) -> int:
        """Idempotently persist evidence records and pairwise source relationships."""
        inserted_count = 0
        with self._get_connection() as conn:
            for r in records:
                upstream_str = json.dumps(r.upstream_source_refs)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evidence_records (
                        evidence_id, correlation_id, source_url, canonical_url,
                        domain, publisher, title, passage, published_at,
                        retrieved_at, source_type, authority_tier, is_primary_source,
                        content_hash, upstream_source_refs, syndication_cluster_id,
                        stance, stance_method, verification_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.evidence_id,
                        correlation_id,
                        r.source_url,
                        r.canonical_url,
                        r.domain,
                        r.publisher,
                        r.title,
                        r.passage,
                        r.published_at,
                        r.retrieved_at,
                        r.source_type,
                        r.authority_tier,
                        1 if r.is_primary_source else 0,
                        r.content_hash,
                        upstream_str,
                        r.syndication_cluster_id,
                        r.stance.value if hasattr(r.stance, "value") else str(r.stance),
                        r.stance_assessment.method if r.stance_assessment else "default",
                        r.verification_status,
                    ),
                )
                inserted_count += 1

            for rel in relationships:
                conn.execute(
                    """
                    INSERT INTO source_relationships (
                        correlation_id, source_id, target_id, relationship_type,
                        similarity_score, rationale
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        correlation_id,
                        rel.source_id,
                        rel.target_id,
                        rel.relationship_type.value if hasattr(rel.relationship_type, "value") else str(rel.relationship_type),
                        rel.similarity_score,
                        rel.rationale,
                    ),
                )
            conn.commit()

        logger.info("[%s] Idempotently saved %d evidence records and %d relationships", correlation_id, inserted_count, len(relationships))
        return inserted_count

    def get_evidence_by_correlation_id(self, correlation_id: str) -> List[EvidenceRecord]:
        """Fetch all evidence records linked to a correlation ID."""
        records: List[EvidenceRecord] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT evidence_id, source_url, canonical_url, domain, publisher,
                       title, passage, published_at, retrieved_at, source_type,
                       authority_tier, is_primary_source, content_hash,
                       upstream_source_refs, syndication_cluster_id, stance, verification_status
                FROM evidence_records
                WHERE correlation_id = ?
                """,
                (correlation_id,),
            ).fetchall()

        for r in rows:
            upstream = json.loads(r["upstream_source_refs"]) if r["upstream_source_refs"] else []
            stance_val = r["stance"]
            try:
                st = StanceType(stance_val)
            except Exception:
                st = StanceType.NEUTRAL_OR_UNRELATED

            records.append(
                EvidenceRecord(
                    evidence_id=r["evidence_id"],
                    source_url=r["source_url"],
                    canonical_url=r["canonical_url"],
                    domain=r["domain"],
                    publisher=r["publisher"],
                    title=r["title"],
                    passage=r["passage"],
                    published_at=r["published_at"],
                    retrieved_at=r["retrieved_at"],
                    source_type=r["source_type"],
                    authority_tier=r["authority_tier"],
                    is_primary_source=bool(r["is_primary_source"]),
                    content_hash=r["content_hash"],
                    upstream_source_refs=upstream,
                    syndication_cluster_id=r["syndication_cluster_id"],
                    stance=st,
                    verification_status=r["verification_status"],
                )
            )
        return records

    def get_relationships_by_correlation_id(self, correlation_id: str) -> List[EvidenceRelationship]:
        """Fetch all relationships linked to a correlation ID."""
        rels: List[EvidenceRelationship] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT source_id, target_id, relationship_type, similarity_score, rationale
                FROM source_relationships
                WHERE correlation_id = ?
                """,
                (correlation_id,),
            ).fetchall()

        for r in rows:
            try:
                rtype = RelationshipType(r["relationship_type"])
            except Exception:
                rtype = RelationshipType.UNKNOWN

            rels.append(
                EvidenceRelationship(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relationship_type=rtype,
                    similarity_score=r["similarity_score"],
                    rationale=r["rationale"],
                )
            )
        return rels

    def get_recent_verifications(self, limit: int = 20) -> List[VerificationRecord]:
        safe_limit = max(1, min(limit, 100))
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, correlation_id, claim, verdict, confidence,
                       category, risk_level, engine_used, created_at, payload_json
                FROM verification_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        records: List[VerificationRecord] = []
        for r in rows:
            records.append(
                VerificationRecord(
                    id=r["id"],
                    correlation_id=r["correlation_id"],
                    claim=r["claim"],
                    verdict=r["verdict"],
                    confidence=r["confidence"],
                    category=r["category"],
                    risk_level=r["risk_level"],
                    engine_used=r["engine_used"],
                    created_at=r["created_at"],
                    payload_json=r["payload_json"],
                )
            )
        return records

    def get_verification_by_id(self, record_id: int) -> Optional[VerificationRecord]:
        with self._get_connection() as conn:
            r = conn.execute(
                """
                SELECT id, correlation_id, claim, verdict, confidence,
                       category, risk_level, engine_used, created_at, payload_json
                FROM verification_history
                WHERE id = ?
                """,
                (record_id,),
            ).fetchone()

        if not r:
            return None

        return VerificationRecord(
            id=r["id"],
            correlation_id=r["correlation_id"],
            claim=r["claim"],
            verdict=r["verdict"],
            confidence=r["confidence"],
            category=r["category"],
            risk_level=r["risk_level"],
            engine_used=r["engine_used"],
            created_at=r["created_at"],
            payload_json=r["payload_json"],
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM verification_history").fetchone()[0]
            rows = conn.execute(
                "SELECT verdict, COUNT(*) as cnt FROM verification_history GROUP BY verdict"
            ).fetchall()
            ev_count = conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0]

        breakdown = {r["verdict"]: r["cnt"] for r in rows}
        return {
            "total_verifications": total,
            "total_evidence_records": ev_count,
            "verdict_breakdown": breakdown,
            "schema_version": SCHEMA_VERSION,
        }
