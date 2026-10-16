"""
SachAI Tracing & Telemetry
Provides correlation IDs, execution timers, structured logging, and error sanitization.
"""

import logging
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("SachAI.Tracing")


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracking a verification request end-to-end."""
    return f"sach-{uuid.uuid4().hex[:12]}"


class TraceEvent(BaseModel):
    """Structured event capturing a lifecycle stage in the verification pipeline."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str
    stage: str
    latency_ms: float
    status: str = "SUCCESS"  # SUCCESS, FAILED, SKIPPED
    metadata: Dict[str, Any] = Field(default_factory=dict)


@contextmanager
def StageTimer(stage: str, correlation_id: str) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager to record stage execution latency and log structured telemetry.
    Usage:
        with StageTimer("RETRIEVAL", correlation_id) as tracker:
            ... perform retrieval ...
            tracker["sources_found"] = 4
    """
    tracker: Dict[str, Any] = {}
    start = time.perf_counter()
    status = "SUCCESS"
    try:
        yield tracker
    except Exception as exc:
        status = "FAILED"
        tracker["error"] = sanitize_error(exc)
        raise
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        event = TraceEvent(
            correlation_id=correlation_id,
            stage=stage,
            latency_ms=latency_ms,
            status=status,
            metadata=tracker,
        )
        logger.info(
            "[%s] Stage '%s' finished in %.2fms with status '%s'",
            correlation_id,
            stage,
            latency_ms,
            status,
        )


def sanitize_error(exc: Exception) -> str:
    """
    Sanitize error message to ensure API keys, Bearer tokens, or credentials
    are never leaked into logs or client-facing responses.
    """
    msg = str(exc)
    # Mask OpenAI / Gemini / Perplexity keys
    msg = re.sub(r'AIzaSy[A-Za-z0-9_-]{33}', '[REDACTED_GEMINI_KEY]', msg)
    msg = re.sub(r'sk-[A-Za-z0-9]{20,}', '[REDACTED_OPENAI_KEY]', msg)
    msg = re.sub(r'pplx-[A-Za-z0-9]{20,}', '[REDACTED_PERPLEXITY_KEY]', msg)
    msg = re.sub(r'key=[A-Za-z0-9_-]+', 'key=[REDACTED]', msg)
    msg = re.sub(r'Bearer\s+[A-Za-z0-9._-]+', 'Bearer [REDACTED]', msg)
    return msg
