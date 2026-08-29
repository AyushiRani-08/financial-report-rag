"""
ingestion/xbrl_parser.py
Parse SEC XBRL (.xml) files and insert structured facts into PostgreSQL
using the normalized schema: companies → filings → financial_facts.
"""

import os
import logging
from datetime import date
from typing import Optional

import psycopg2
from xbrl import XBRLParser, GAAP, GAAPSerializer  # python-xbrl 1.1.1

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical field mapping: raw US-GAAP tags → normalized field names
# Add more tags as needed; these cover the core retail-investor metrics.
# ---------------------------------------------------------------------------
CANONICAL_MAP: dict[str, str] = {
    # Revenue
    "us-gaap_Revenues":                          "revenue",
    "us-gaap_RevenueFromContractWithCustomer":   "revenue",
    "us-gaap_SalesRevenueNet":                   "revenue",
    # Net income
    "us-gaap_NetIncomeLoss":                     "net_income",
    "us-gaap_ProfitLoss":                        "net_income",
    # Operating income
    "us-gaap_OperatingIncomeLoss":               "operating_income",
    # EPS
    "us-gaap_EarningsPerShareBasic":             "eps_basic",
    "us-gaap_EarningsPerShareDiluted":           "eps_diluted",
    # Balance sheet
    "us-gaap_Assets":                            "total_assets",
    "us-gaap_Liabilities":                       "total_liabilities",
    "us-gaap_StockholdersEquity":                "stockholders_equity",
    "us-gaap_LongTermDebt":                      "long_term_debt",
    "us-gaap_CashAndCashEquivalentsAtCarrying":  "cash",
    # Cash flow
    "us-gaap_NetCashProvidedByUsedInOperating":  "operating_cash_flow",
    "us-gaap_NetCashProvidedByUsedInInvesting":  "investing_cash_flow",
    "us-gaap_NetCashProvidedByUsedInFinancing":  "financing_cash_flow",
}


def _get_or_create_company(
    cur,
    ticker: str,
    name: str,
    cik: Optional[str],
    market: str,
    sector: Optional[str],
) -> int:
    """Return existing company_id or insert and return new one."""
    cur.execute("SELECT company_id FROM companies WHERE ticker = %s", (ticker,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        """
        INSERT INTO companies (ticker, name, cik, market, sector)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING company_id
        """,
        (ticker, name, cik, market, sector),
    )
    return cur.fetchone()[0]


def _get_or_create_filing(
    cur,
    company_id: int,
    form_type: str,
    fiscal_period: str,
    filed_date: Optional[date],
    source_url: Optional[str],
) -> int:
    """Return existing filing_id or insert and return new one."""
    cur.execute(
        """
        SELECT filing_id FROM filings
        WHERE company_id = %s AND form_type = %s AND fiscal_period = %s
        """,
        (company_id, form_type, fiscal_period),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        """
        INSERT INTO filings (company_id, form_type, fiscal_period, filed_date, source_url)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING filing_id
        """,
        (company_id, form_type, fiscal_period, filed_date, source_url),
    )
    return cur.fetchone()[0]


def parse_xbrl_to_db(
    xbrl_path: str,
    ticker: str,
    company_name: str,
    form_type: str,       # e.g. '10-K', '10-Q'
    fiscal_period: str,   # e.g. 'FY2025', 'Q3-2025'
    market: str = "US",
    cik: Optional[str] = None,
    sector: Optional[str] = None,
    filed_date: Optional[date] = None,
    dsn: Optional[str] = None,
) -> int:
    """
    Parse an XBRL file and upsert structured financial facts into PostgreSQL.

    Returns the number of fact rows inserted.
    """
    dsn = dsn or os.environ["POSTGRES_DSN"]
    conn = psycopg2.connect(dsn)

    try:
        parser = XBRLParser(precision=0)
        xbrl_doc = parser.parse(open(xbrl_path))
        gaap_obj = parser.parseGAAP(
            xbrl_doc,
            doc_date=fiscal_period,
            context="current",
            ignore_errors=True,
        )
        gaap_data: dict = GAAPSerializer(gaap_obj).data  # flat dict of tag→value

        with conn:
            cur = conn.cursor()

            company_id = _get_or_create_company(
                cur, ticker, company_name, cik, market, sector
            )
            filing_id = _get_or_create_filing(
                cur, company_id, form_type, fiscal_period, filed_date, xbrl_path
            )

            inserted = 0
            for raw_tag, value in gaap_data.items():
                if value is None:
                    continue
                canonical = CANONICAL_MAP.get(raw_tag, raw_tag.lower().replace("-", "_"))
                cur.execute(
                    """
                    INSERT INTO financial_facts
                        (filing_id, tag, canonical_field, value, unit, period_end, source_page)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        filing_id,
                        raw_tag,
                        canonical,
                        float(value) if value else None,
                        "USD" if market == "US" else "INR",
                        filed_date,
                        xbrl_path,
                    ),
                )
                inserted += 1

        logger.info(
            "Inserted %d facts for %s (%s %s)", inserted, ticker, form_type, fiscal_period
        )
        return inserted

    finally:
        conn.close()
