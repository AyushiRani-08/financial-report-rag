# utils/metadata_parser.py
import re
from typing import Optional, Tuple, Dict, Any

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


def parse_filing_metadata(doc_name: str) -> Dict[str, Optional[str]]:
    """
    Parses ticker, fiscal_period, and form_type from a document name or filename.
    Examples:
      'aapl-20250927'      -> {'ticker': 'AAPL', 'fiscal_period': 'FY2025', 'form_type': '10-K'}
      'ba_20260630_10q'    -> {'ticker': 'BA',   'fiscal_period': 'FY2026', 'form_type': '10-Q'}
      'jpm_20251231_10k'   -> {'ticker': 'JPM',  'fiscal_period': 'FY2025', 'form_type': '10-K'}
      'microsoft'          -> {'ticker': 'MSFT', 'fiscal_period': None,     'form_type': None}
      'tesla'              -> {'ticker': 'TSLA', 'fiscal_period': None,     'form_type': None}
    """
    clean = doc_name.lower().replace(".html", "").replace(".htm", "").replace(".pdf", "").replace(".md", "").replace(".txt", "").strip()

    # Form type detection
    form_type = None
    if "10-k" in clean or "_10k" in clean or "-10k" in clean:
        form_type = "10-K"
    elif "10-q" in clean or "_10q" in clean or "-10q" in clean:
        form_type = "10-Q"
    elif "8-k" in clean or "_8k" in clean or "-8k" in clean:
        form_type = "8-K"

    # Direct company name lookup
    if clean in COMPANY_TICKER_MAP:
        return {
            "ticker": COMPANY_TICKER_MAP[clean],
            "fiscal_period": None,
            "form_type": form_type,
        }

    # Standard SEC filing pattern: ticker[-_]YYYYMMDD
    m = re.match(r"^([a-zA-Z]+)[-_](\d{4})(\d{2})(\d{2})", clean)
    if m:
        raw_prefix = m.group(1).lower()
        ticker = COMPANY_TICKER_MAP.get(raw_prefix, m.group(1).upper())
        year, month, day = int(m.group(2)), int(m.group(3)), int(m.group(4))

        # Fiscal period estimation
        if month in (6, 9, 12) and day >= 28:
            fy = year if month >= 4 else year - 1
            period = f"FY{fy}"
        elif month == 3 and day >= 28:
            period = f"Q1FY{year}"
        elif month in (5, 6):
            period = f"Q2FY{year}"
        else:
            period = f"FY{year}"

        if not form_type:
            # Default to 10-K for annual period filings, else 10-Q
            form_type = "10-K" if period.startswith("FY") else "10-Q"

        return {
            "ticker": ticker,
            "fiscal_period": period,
            "form_type": form_type,
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
