"""
privacy/__init__.py
───────────────────
Public API for the FinSight privacy pipeline.

    from privacy import redact_query, redact_chunk, get_redaction_summary, get_engine_status
"""
from __future__ import annotations

from .redactor import redact, presidio_available, active_engine_label
from .audit import get_log_path


def redact_query(query: str) -> tuple[str, list[dict]]:
    """Redact PII from a user chat query. Returns (clean_text, findings)."""
    return redact(query, source="user_query")


def redact_chunk(chunk_text: str) -> tuple[str, list[dict]]:
    """Redact PII from a document chunk before LLM ingestion."""
    return redact(chunk_text, source="document_chunk")


def get_redaction_summary(findings: list[dict]) -> str | None:
    """
    Human-readable badge string for UI display, or None if nothing was redacted.

    Example: "🔒 Privacy layer active — redacted: SSN ×1 [presidio], EMAIL ×1 [regex]"
    """
    if not findings:
        return None
    parts = [f"{f['entity_type']} ×{f['count']} [{f['method']}]" for f in findings]
    return "\U0001f512 Privacy layer active \u2014 redacted: " + ", ".join(parts)


def get_engine_status() -> dict:
    """Status dict for sidebar display."""
    return {
        "active_nlp_engine": active_engine_label(),
        "presidio_available": presidio_available(),
        "audit_log": str(get_log_path()),
    }


__all__ = ["redact_query", "redact_chunk", "get_redaction_summary", "get_engine_status"]
