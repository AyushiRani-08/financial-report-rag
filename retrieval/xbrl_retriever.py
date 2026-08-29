"""
retrieval/xbrl_retriever.py
Query structured financial facts from PostgreSQL to augment RAG prompts.
Uses the normalized schema: companies → filings → financial_facts.
"""

import os
import logging
from typing import Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Core investor-facing facts to surface in RAG context
DEFAULT_CANONICAL_FIELDS = [
    "revenue",
    "net_income",
    "operating_income",
    "eps_basic",
    "eps_diluted",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "long_term_debt",
    "cash",
    "operating_cash_flow",
]


class XBRLRetriever:
    """
    Retrieves structured financial facts from PostgreSQL for use in RAG prompts.
    Supports filtering by ticker, fiscal period, and canonical field names.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.environ["POSTGRES_DSN"]
        self._conn: Optional[psycopg2.extensions.connection] = None

    @property
    def conn(self) -> psycopg2.extensions.connection:
        """Lazy connection with auto-reconnect."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def get_key_financials(
        self,
        ticker: str,
        fiscal_period: Optional[str] = None,
        canonical_fields: Optional[list[str]] = None,
        limit_periods: int = 1,
    ) -> list[dict]:
        """
        Fetch financial facts for a ticker from PostgreSQL.

        Args:
            ticker:           Stock ticker symbol (e.g. 'AAPL').
            fiscal_period:    e.g. 'FY2025' or 'Q3-2025'. None = latest available.
            canonical_fields: List of normalized field names to retrieve.
                              Defaults to DEFAULT_CANONICAL_FIELDS.
            limit_periods:    How many distinct filing periods to return (default 1).

        Returns:
            List of dicts with keys: canonical_field, value, unit, period_end,
            form_type, fiscal_period, tag.
        """
        fields = canonical_fields or DEFAULT_CANONICAL_FIELDS

        period_filter = ""
        params: list = [ticker, fields]
        if fiscal_period:
            period_filter = "AND f.fiscal_period = %s"
            params.append(fiscal_period)

        params.append(limit_periods)

        query = f"""
            SELECT
                ff.canonical_field,
                ff.value,
                ff.unit,
                ff.period_end,
                fi.form_type,
                fi.fiscal_period,
                ff.tag,
                ff.source_page
            FROM financial_facts ff
            JOIN filings fi ON fi.filing_id = ff.filing_id
            JOIN companies c  ON c.company_id = fi.company_id
            WHERE c.ticker = %s
              AND ff.canonical_field = ANY(%s)
              {period_filter}
            ORDER BY fi.fiscal_period DESC, ff.canonical_field
            LIMIT (
                SELECT COUNT(DISTINCT ff2.canonical_field)
                FROM financial_facts ff2
                JOIN filings fi2 ON fi2.filing_id = ff2.filing_id
                JOIN companies c2 ON c2.company_id = fi2.company_id
                WHERE c2.ticker = %s
            ) * %s
        """
        # Simpler flat query:
        query = f"""
            SELECT
                ff.canonical_field,
                ff.value,
                ff.unit,
                ff.period_end,
                fi.form_type,
                fi.fiscal_period,
                ff.tag,
                ff.source_page
            FROM financial_facts ff
            JOIN filings fi ON fi.filing_id = ff.filing_id
            JOIN companies c  ON c.company_id = fi.company_id
            WHERE c.ticker = %s
              AND ff.canonical_field = ANY(%s)
              {period_filter}
            ORDER BY fi.fiscal_period DESC, ff.canonical_field
        """

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def get_available_periods(self, ticker: str) -> list[str]:
        """Return all fiscal periods available for a ticker, newest first."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT fi.fiscal_period
                FROM filings fi
                JOIN companies c ON c.company_id = fi.company_id
                WHERE c.ticker = %s
                ORDER BY fi.fiscal_period DESC
                """,
                (ticker,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_available_tickers(self) -> list[dict]:
        """Return all indexed companies with their market and sector."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT ticker, name, market, sector FROM companies ORDER BY ticker"
            )
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # RAG context formatting
    # ------------------------------------------------------------------

    def to_context_string(
        self,
        ticker: str,
        fiscal_period: Optional[str] = None,
        canonical_fields: Optional[list[str]] = None,
    ) -> str:
        """
        Format structured XBRL facts as a plain-text block for injection
        into a RAG prompt context window.

        Returns empty string if no facts are found (graceful degradation).
        """
        facts = self.get_key_financials(ticker, fiscal_period, canonical_fields)
        if not facts:
            logger.debug("No XBRL facts found for ticker=%s period=%s", ticker, fiscal_period)
            return ""

        period_label = fiscal_period or (facts[0]["fiscal_period"] if facts else "unknown")
        lines = [
            f"[Structured Financial Data | {ticker} | {period_label}]",
            "-" * 55,
        ]
        for f in facts:
            val = f["value"]
            unit = f["unit"] or ""
            label = f["canonical_field"].replace("_", " ").title()
            if val is not None:
                # Format large numbers with commas
                formatted = f"{val:,.2f}" if abs(val) >= 1 else f"{val:.4f}"
                lines.append(f"  {label:<30} {formatted:>18} {unit}")
            else:
                lines.append(f"  {label:<30} {'N/A':>18}")

        lines.append("-" * 55)
        return "\n".join(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
