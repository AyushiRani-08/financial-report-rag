"""
retrieval/xbrl_retriever.py
Query structured financial facts from PostgreSQL to augment RAG prompts.
Uses the normalized schema: companies → filings → financial_facts.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv()

logger = logging.getLogger(__name__)

# Core investor-facing facts to surface in RAG context
DEFAULT_CANONICAL_FIELDS = [
    "revenue",
    "gross_profit",
    "cost_of_revenue",
    "operating_income",
    "operating_expenses",
    "rd_expense",
    "sga_expense",
    "pretax_income",
    "income_tax_expense",
    "net_income",
    "eps_basic",
    "eps_diluted",
    "shares_outstanding",
    "total_assets",
    "current_assets",
    "cash",
    "cash_and_investments",
    "total_liabilities",
    "current_liabilities",
    "long_term_debt",
    "stockholders_equity",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capex",
    "share_repurchases",
    "dividends_paid",
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
            period_filter = "AND fi.fiscal_period = %s"
            params.append(fiscal_period)

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
        query: Optional[str] = None,
    ) -> str:
        """
        Format structured XBRL facts as plain-text blocks for injection
        into a RAG prompt context window. Supports single-period, multi-period,
        and query-based fiscal period resolution.
        """
        facts = self.get_key_financials(ticker, fiscal_period, canonical_fields)
        if not facts:
            logger.debug("No XBRL facts found for ticker=%s period=%s", ticker, fiscal_period)
            return ""

        # Group facts by fiscal_period -> canonical_field
        by_period: dict[str, dict[str, dict]] = {}
        for f in facts:
            p = f["fiscal_period"] or "Latest"
            if p not in by_period:
                by_period[p] = {}
            field = f["canonical_field"]
            p_end = str(f.get("period_end") or "")
            if field not in by_period[p] or p_end > str(by_period[p][field].get("period_end") or ""):
                by_period[p][field] = f

        # Determine which periods to format
        selected_periods: list[str] = []
        if fiscal_period and fiscal_period in by_period:
            selected_periods = [fiscal_period]
        elif query:
            q_lower = query.lower()
            # Match specific years and quarters in query
            for p in by_period.keys():
                p_lower = p.lower()
                # Check for explicit year match (e.g. 2024, 2023)
                for y in ("2026", "2025", "2024", "2023", "2022", "2021", "2020"):
                    if y in q_lower and y in p_lower:
                        if ("q1" in q_lower or "first quarter" in q_lower) and "q1" in p_lower:
                            if p not in selected_periods: selected_periods.append(p)
                        elif ("q2" in q_lower or "second quarter" in q_lower) and "q2" in p_lower:
                            if p not in selected_periods: selected_periods.append(p)
                        elif ("q3" in q_lower or "third quarter" in q_lower) and "q3" in p_lower:
                            if p not in selected_periods: selected_periods.append(p)
                        elif ("q4" in q_lower or "fourth quarter" in q_lower) and "q4" in p_lower:
                            if p not in selected_periods: selected_periods.append(p)
                        elif not any(q in q_lower for q in ("q1", "q2", "q3", "q4", "first quarter", "second quarter", "third quarter", "fourth quarter")):
                            if p not in selected_periods: selected_periods.append(p)
            
            # Also check if query asks for prior year / YoY comparison
            if any(term in q_lower for term in ("prior year", "compare", "yoy", "growth", "previous year", "last year")):
                for p in list(by_period.keys()):
                    if ("2023" in p or "2022" in p) and p not in selected_periods:
                        selected_periods.append(p)

        if not selected_periods:
            # Default to top 3 most recent periods
            selected_periods = list(by_period.keys())[:3]

        blocks = []
        for p in selected_periods:
            if p not in by_period:
                continue
            lines = [
                f"[Structured Financial Data | {ticker} | {p}]",
                "-" * 55,
            ]
            for field, f in sorted(by_period[p].items()):
                val = f["value"]
                unit = f["unit"] or ""
                label = field.replace("_", " ").title()
                if val is not None:
                    formatted = f"{val:,.2f}" if abs(val) >= 1 else f"{val:.4f}"
                    lines.append(f"  {label:<30} {formatted:>18} {unit}")
                else:
                    lines.append(f"  {label:<30} {'N/A':>18}")
            lines.append("-" * 55)
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
