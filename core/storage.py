"""
SachAI Storage & Persistence Layer
Abstract storage interface and local SQLite implementation for persisting verification history.
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.models import VerificationRecord, VerificationVerdict

logger = logging.getLogger("SachAI.Storage")

DEFAULT_DB_PATH = "sachai_history.db"


class AbstractStorage(ABC):
    """Abstract interface defining persistence operations for verification history."""

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
    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics (total count, breakdown by verdict)."""
        pass


from contextlib import contextmanager


class SQLiteStorage(AbstractStorage):
    """
    Local SQLite implementation of AbstractStorage.
    Stores metadata and structured verdicts, strictly excluding raw media binaries.
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
        """Initialize database schema with proper indexes."""
        with self._get_connection() as conn:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_verif_created ON verification_history(created_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_verif_corr ON verification_history(correlation_id);"
            )
            conn.commit()
        logger.info("SQLite verification history table initialized at %s", self.db_path)

    def save_verification(
        self,
        claim: str,
        verdict: VerificationVerdict,
        correlation_id: str,
    ) -> int:
        # Sanitize claim to prevent storing runaway payloads
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

        breakdown = {r["verdict"]: r["cnt"] for r in rows}
        return {
            "total_verifications": total,
            "verdict_breakdown": breakdown,
        }
