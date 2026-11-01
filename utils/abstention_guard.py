"""
utils/abstention_guard.py
==========================
Multi-Layered Abstention Gate (MAG) - Layer 1 & Pre-Retrieval Guardrails.

Detects unanswerable or out-of-scope financial questions before sending them to
the LLM. The "future period" check is driven by the newest period_end actually
present in PostgreSQL, not by a hardcoded year, so it cannot go stale.
"""

import logging
import os
import re
import threading
import time
from datetime import date
from typing import Any, Dict, Optional

from utils.periods import to_date, years_mentioned

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600.0
_latest_cache: Dict[str, tuple[Optional[date], float]] = {}
_cache_lock = threading.Lock()


OUT_OF_SCOPE_RULES = [
    {
        "pattern": r"\b(stock price|share price|closing price|market price|trading price|share quote)\b",
        "category": "market_data",
        "message": (
            "Annual and quarterly filings do not contain daily or real-time market stock prices. "
            "They disclose audited accounting statements (balance sheet, income statement, cash flows) "
            "and per-share accounting metrics such as EPS, not exchange trading quotes."
        ),
    },
    {
        "pattern": r"\b(?:revenue|earnings|sales|profit)\s+(?:guidance|outlook|forecast)\b|"
                   r"\bguidance\s+for\s+(?:fy\s*)?20\d\d\b",
        "category": "future_guidance",
        "message": (
            "SEC filings record historical audited performance. Forward-looking quantitative "
            "targets are issued separately (press releases, earnings calls) and are not part of "
            "this document corpus."
        ),
    },
    {
        "pattern": r"\b(?:number of employees|headcount|staff)\s+(?:per|by|in each)\s+(?:product\s+)?division\b|"
                   r"\bemployees\s+per\s+division\b",
        "category": "non_disclosed_metric",
        "message": (
            "Filings disclose total consolidated headcount but generally do not break headcount "
            "down by individual product division."
        ),
    },
]


def _query_latest_period_end(ticker: Optional[str]) -> Optional[date]:
    """Newest period_end present in PostgreSQL for a ticker (or overall)."""
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        return None
    try:
        import psycopg2

        conn = psycopg2.connect(dsn)
    except Exception as exc:  # noqa: BLE001 - guard must never break the request path
        logger.debug("Abstention guard: no DB connection (%s)", exc)
        return None

    try:
        with conn.cursor() as cur:
            if ticker:
                cur.execute(
                    """
                    SELECT MAX(ff.period_end)
                    FROM financial_facts ff
                    JOIN filings fi   ON fi.filing_id  = ff.filing_id
                    JOIN companies c  ON c.company_id  = fi.company_id
                    WHERE c.ticker = %s
                    """,
                    (ticker.upper().strip(),),
                )
            else:
                cur.execute("SELECT MAX(period_end) FROM financial_facts")
            row = cur.fetchone()
            return to_date(row[0]) if row and row[0] else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Abstention guard: period lookup failed (%s)", exc)
        return None
    finally:
        conn.close()


def _latest_filed_period_end(ticker: Optional[str]) -> Optional[date]:
    key = (ticker or "*").upper()
    now = time.monotonic()
    with _cache_lock:
        cached = _latest_cache.get(key)
        if cached and now - cached[1] < _CACHE_TTL_SECONDS:
            return cached[0]

    value = _query_latest_period_end(ticker)
    with _cache_lock:
        _latest_cache[key] = (value, now)
    return value


def check_pre_retrieval_abstention(
    query: str,
    ticker: Optional[str] = None,
    fiscal_period: Optional[str] = None,
    latest_filed_end: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether a query must be refused / abstained from prior to retrieval.

    Returns:
        Dict with keys {"should_abstain": True, "category": ..., "reason": ...}
        or None if the question is eligible for standard RAG retrieval.
    """
    q_clean = (query or "").strip()
    q_lower = q_clean.lower()

    # Rule A: patterns that no filing can ever answer
    for rule in OUT_OF_SCOPE_RULES:
        if re.search(rule["pattern"], q_lower, re.IGNORECASE):
            return {
                "should_abstain": True,
                "category": rule["category"],
                "reason": rule["message"],
            }

    # Rule B: periods that have not been filed yet — anchored to real data
    latest = latest_filed_end if latest_filed_end is not None else _latest_filed_period_end(ticker)
    if latest is not None:
        future_years = sorted({y for y in years_mentioned(q_lower) if y > latest.year})
        if future_years:
            requested = ", ".join(str(y) for y in future_years)
            return {
                "should_abstain": True,
                "category": "non_existent_period",
                "reason": (
                    f"No audited filing covering {requested} is available in this database. "
                    f"The most recent reported period ends {latest.isoformat()}. "
                    "I can only report periods that have already been filed."
                ),
            }

    return None
