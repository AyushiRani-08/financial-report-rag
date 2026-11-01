"""
utils/periods.py
Single source of truth for fiscal-period labelling.

Every layer — filename heuristics, SEC XBRL facts, vector-store metadata and
prompt routing — derives labels through this module so the text track and the
structured track always speak the same vocabulary.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

_QUARTER_RE = re.compile(r"^\s*Q([1-4])\s*-?\s*(?:FY\s*)?(\d{4})\s*$", re.IGNORECASE)
_FY_RE = re.compile(r"^\s*(?:FY|CY)\s*-?\s*(\d{4})\s*$", re.IGNORECASE)

# Expected length of a reporting period, used to pick between the overlapping
# 3M / 6M / 9M / 12M facts a single filing reports for the same line item.
EXPECTED_DURATION_DAYS = {"FY": 365, "Q": 91}


def to_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def parse_period_label(label: Optional[str]) -> Optional[dict]:
    """'FY2025' -> {kind: FY, quarter: None, year: 2025}; 'Q2FY2026' -> {kind: Q, quarter: 2, year: 2026}."""
    if not label:
        return None
    text = str(label).strip()
    match = _QUARTER_RE.match(text)
    if match:
        return {"kind": "Q", "quarter": int(match.group(1)), "year": int(match.group(2))}
    match = _FY_RE.match(text)
    if match:
        return {"kind": "FY", "quarter": None, "year": int(match.group(1))}
    return None


def expected_duration_days(label: Optional[str]) -> Optional[int]:
    parsed = parse_period_label(label)
    return EXPECTED_DURATION_DAYS.get(parsed["kind"]) if parsed else None


def infer_fiscal_period(period_end, form_type: str = "", fy=None, fp=None) -> Optional[str]:
    """
    Canonical label for a reporting period.

    SEC's own ``fy`` / ``fp`` metadata always wins; the date is only a fallback
    for uploads that carry no filing metadata.
    """
    fp_norm = str(fp or "").strip().upper()
    form = str(form_type or "").upper()
    is_annual_form = bool(re.search(r"(10-K|20-F|40-F|AOC-4)", form))
    is_quarterly_form = "10-Q" in form

    if fy:
        year = int(fy)
        if fp_norm in ("Q1", "Q2", "Q3", "Q4"):
            return f"{fp_norm}FY{year}"
        if fp_norm in ("FY", "CY") and is_annual_form:
            return f"FY{year}"

    end = to_date(period_end)
    if end is None:
        return None
    if is_annual_form:
        return f"FY{end.year}"
    if is_quarterly_form:
        return f"Q{((end.month - 1) // 3) + 1}FY{end.year}"
    return None


def period_from_query(query: str) -> Optional[str]:
    """Best-effort period mentioned in a user question."""
    if not query:
        return None
    text = query.lower()

    match = re.search(
        r"\bq\s*([1-4])\s*(?:of\s*)?(?:fy\s*|fiscal\s*(?:year\s*)?)?(\d{4})\b", text
    )
    if match:
        return f"Q{match.group(1)}FY{match.group(2)}"

    match = re.search(r"\b(?:fy|fiscal\s*(?:year\s*)?)\s*(\d{4})\b", text)
    if match:
        return f"FY{match.group(1)}"

    match = re.search(r"\b(20\d{2})\b", text)
    if match:
        return f"FY{match.group(1)}"
    return None


def years_mentioned(text: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(20\d{2})\b", text or "")]


def period_sort_key(label: Optional[str]) -> tuple[int, int]:
    """
    Ordering key for labels when no dates are available. Prefer ordering by
    ``period_end`` from the database; this is only a fallback.
    """
    parsed = parse_period_label(label)
    if not parsed:
        return (0, 0)
    within_year = 4 if parsed["kind"] == "FY" else int(parsed["quarter"])
    return (parsed["year"], within_year)


__all__ = [
    "EXPECTED_DURATION_DAYS",
    "expected_duration_days",
    "infer_fiscal_period",
    "parse_period_label",
    "period_from_query",
    "period_sort_key",
    "to_date",
    "years_mentioned",
]
