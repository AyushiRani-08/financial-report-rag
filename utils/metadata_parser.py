# utils/metadata_parser.py
import re
from datetime import date
from typing import Optional, Dict

from utils.periods import infer_fiscal_period

COMPANY_TICKER_MAP = {
    "tesla": "TSLA",
    "tsla": "TSLA",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "apple": "AAPL",
    "aapl": "AAPL",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "jpmc": "JPM",
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "boeing": "BA",
    "ba": "BA",
    "berkshire": "BRK-B",
    "asml": "ASML",
    "baba": "BABA",
    "alibaba": "BABA",
}

_FILENAME_RE = re.compile(r"^([a-zA-Z]+)[-_](\d{4})(\d{2})(\d{2})")


def _detect_form_type(clean: str) -> Optional[str]:
    if "10-k" in clean or "_10k" in clean or "-10k" in clean:
        return "10-K"
    if "10-q" in clean or "_10q" in clean or "-10q" in clean:
        return "10-Q"
    if "8-k" in clean or "_8k" in clean or "-8k" in clean:
        return "8-K"
    return None


def _looks_annual(period_end: date) -> bool:
    """True when the date looks like a fiscal year end rather than a quarter end."""
    return period_end.month in (6, 9, 12) and period_end.day >= 25


def parse_filing_metadata(doc_name: str) -> Dict[str, Optional[str]]:
    """
    Parses ticker, fiscal_period, and form_type from a document name or filename.
    Period labelling is delegated to utils.periods so it matches the XBRL track.

    Examples:
      'aapl-20250927'      -> {'ticker': 'AAPL', 'fiscal_period': 'FY2025', 'form_type': '10-K'}
      'ba_20260630_10q'    -> {'ticker': 'BA',   'fiscal_period': 'Q2FY2026', 'form_type': '10-Q'}
      'jpm_20251231_10k'   -> {'ticker': 'JPM',  'fiscal_period': 'FY2025', 'form_type': '10-K'}
      'microsoft'          -> {'ticker': 'MSFT', 'fiscal_period': None,     'form_type': None}
    """
    clean = (
        doc_name.lower()
        .replace(".html", "")
        .replace(".htm", "")
        .replace(".pdf", "")
        .replace(".md", "")
        .replace(".txt", "")
        .strip()
    )

    form_type = _detect_form_type(clean)

    if clean in COMPANY_TICKER_MAP:
        return {
            "ticker": COMPANY_TICKER_MAP[clean],
            "fiscal_period": None,
            "form_type": form_type,
        }

    # Standard SEC filing pattern: ticker[-_]YYYYMMDD
    m = _FILENAME_RE.match(clean)
    if m:
        raw_prefix = m.group(1).lower()
        ticker = COMPANY_TICKER_MAP.get(raw_prefix, m.group(1).upper())
        try:
            period_end = date(int(m.group(2)), int(m.group(3)), int(m.group(4)))
        except ValueError:
            period_end = None

        effective_form = form_type or (
            "10-K" if period_end and _looks_annual(period_end) else "10-Q"
        )
        period = infer_fiscal_period(period_end, effective_form)

        return {
            "ticker": ticker,
            "fiscal_period": period,
            "form_type": effective_form,
        }

    # Partial keyword match for tickers in composite names
    for comp, tick in COMPANY_TICKER_MAP.items():
        if comp in clean:
            return {
                "ticker": tick,
                "fiscal_period": None,
                "form_type": form_type,
            }

    # Plain ticker code (e.g. "AAPL", "MSFT")
    if len(clean) in (1, 2, 3, 4, 5) and clean.isalpha():
        return {
            "ticker": clean.upper(),
            "fiscal_period": None,
            "form_type": form_type,
        }

    return {
        "ticker": None,
        "fiscal_period": None,
        "form_type": form_type,
    }
