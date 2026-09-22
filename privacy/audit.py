"""
privacy/audit.py
────────────────
Structured audit logging for every redaction event.

Writes JSON-Lines to logs/redactions.log.
Each line records WHAT type was found and HOW MANY times — never the raw PII value.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── File handler setup (done once at module import) ───────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "redactions.log"

_audit_logger = logging.getLogger("finsight.privacy.audit")
if not _audit_logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _audit_logger.addHandler(_fh)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


def log_redaction(
    *,
    source: str,
    entity_type: str,
    count: int,
    engine: str,
    context_snippet: str = "",
) -> None:
    """
    Append one redaction event to the audit log.

    Parameters
    ----------
    source          : Where the text came from ('user_query' | 'chunk' | …)
    entity_type     : Detected entity label (e.g. 'SSN', 'EMAIL_ADDRESS')
    count           : Number of occurrences replaced
    engine          : Which engine detected it ('presidio' | 'scrubadub' | 'regex')
    context_snippet : First 80 chars of the *original* text for debugging — never
                      the raw matched value.
    """
    _audit_logger.info(json.dumps({
        "ts":          datetime.now(timezone.utc).isoformat(),
        "source":      source,
        "entity_type": entity_type,
        "occurrences": count,
        "engine":      engine,
        "snippet":     context_snippet[:80],
    }))


def get_log_path() -> Path:
    return _LOG_FILE
