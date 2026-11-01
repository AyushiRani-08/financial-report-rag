"""
retrieval/xbrl_retriever.py
Query structured financial facts from PostgreSQL to augment RAG prompts.
Uses the normalized schema: companies -> filings -> financial_facts.

Facts are selected deterministically: for a given fiscal period the reader
picks the row whose duration matches the period (91 days for a quarter,
365 for a year) instead of relying on insertion order.
"""
import os
import re
import logging
from datetime import date
from typing import Optional

from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import psycopg2.pool

from utils.canonical_map import DEFAULT_CANONICAL_FIELDS
from utils.periods import (
    expected_duration_days,
    parse_period_label,
    to_date,
    years_mentioned,
)

load_dotenv()

logger = logging.getLogger(__name__)


def _pick_best_fact(rows: list[dict], label: str) -> dict:
    """
    Choose one row per (field, period).

    A single filing reports the same line item over several overlapping
    durations (3M / 6M / 9M / 12M) plus prior-year comparatives. Pick the row
    whose duration is closest to the period label, then the latest period end.
    """
    expected = expected_duration_days(label)

    def sort_key(row: dict):
        duration = row.get("duration_days")
        if duration is None or expected is None:
            penalty = 0
        else:
            penalty = abs(int(duration) - expected)
        period_end = to_date(row.get("period_end"))
        return (penalty, -(period_end.toordinal() if period_end else 0))

    return min(rows, key=sort_key)


class XBRLRetriever:
    """
    Retrieves structured financial facts from PostgreSQL for use in RAG prompts.
    Supports filtering by ticker, fiscal period, and canonical field names.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.environ["POSTGRES_DSN"]
        # Initialize a thread‑safe connection pool (min 1, max 5 connections)
        self._pool = psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=self.dsn)
        self._conn: Optional[psycopg2.extensions.connection] = None

    @property
    def conn(self) -> psycopg2.extensions.connection:
        """Obtain a connection from the pool. Caller must return it via self._pool.putconn()."""
        conn = self._pool.getconn()
        # Ensure the connection is healthy; reconnect if needed
        if conn.closed:
            conn = self._pool.getconn()
        elif conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            conn.rollback()
        return conn

    def close(self):
        # Close all pooled connections gracefully
        if hasattr(self, "_pool"):
            self._pool.closeall()

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
            fiscal_period:    e.g. 'FY2025' or 'Q3FY2025'. None = latest available.
            canonical_fields: List of normalized field names to retrieve.
                              Defaults to DEFAULT_CANONICAL_FIELDS.
            limit_periods:    How many distinct filing periods to return (default 1).

        Returns:
            List of dicts with keys: canonical_field, value, unit, period_start,
            period_end, duration_days, is_instant, form_type, fiscal_period, tag.
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
                ff.period_start,
                ff.period_end,
                ff.duration_days,
                ff.is_instant,
                fi.form_type,
                fi.fiscal_period,
                ff.tag
            FROM financial_facts ff
            JOIN filings fi ON fi.filing_id = ff.filing_id
            JOIN companies c  ON c.company_id = fi.company_id
            WHERE c.ticker = %s
              AND ff.canonical_field = ANY(%s)
              AND ff.value IS NOT NULL
              {period_filter}
            ORDER BY ff.period_end DESC NULLS LAST,
                     ff.duration_days DESC NULLS LAST,
                     ff.fact_id DESC
        """

        conn = self.conn
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        finally:
            # Return the connection to the pool
            self._pool.putconn(conn)

    def get_available_periods(self, ticker: str) -> list[str]:
        """Return all fiscal periods available for a ticker, newest first."""
        try:
            conn = self.conn
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT fi.fiscal_period
                        FROM filings fi
                        JOIN companies c ON c.company_id = fi.company_id
                        LEFT JOIN financial_facts ff ON ff.filing_id = fi.filing_id
                        WHERE c.ticker = %s
                        GROUP BY fi.fiscal_period
                        ORDER BY MAX(ff.period_end) DESC NULLS LAST,
                                 fi.fiscal_period DESC
                        """,
                        (ticker,),
                    )
                    return [row[0] for row in cur.fetchall()]
            finally:
                self._pool.putconn(conn)
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

    def get_available_tickers(self) -> list[dict]:
        """Return all indexed companies with their market and sector."""
        try:
            conn = self.conn
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT ticker, name, market, sector FROM companies ORDER BY ticker"
                    )
                    return [dict(row) for row in cur.fetchall()]
            finally:
                self._pool.putconn(conn)
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

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

        # Group -> pick exactly one deterministic fact per (period, field)
        grouped: dict[str, dict[str, list[dict]]] = {}
        for f in facts:
            period = f["fiscal_period"] or "Latest"
            grouped.setdefault(period, {}).setdefault(f["canonical_field"], []).append(f)

        by_period: dict[str, dict[str, dict]] = {
            period: {field: _pick_best_fact(rows, period) for field, rows in fields.items()}
            for period, fields in grouped.items()
        }

        def _period_end(period: str) -> date:
            ends = [to_date(r.get("period_end")) for r in by_period[period].values()]
            ends = [e for e in ends if e]
            return max(ends) if ends else date.min

        ordered_periods = sorted(by_period.keys(), key=_period_end, reverse=True)

        selected_periods: list[str] = []
        if fiscal_period and fiscal_period in by_period:
            selected_periods = [fiscal_period]
        elif query:
            q_lower = query.lower()
            wanted_years = set(years_mentioned(q_lower))
            quarter_match = re.search(r"\bq([1-4])\b", q_lower)
            wants_quarter = int(quarter_match.group(1)) if quarter_match else None

            for period in ordered_periods:
                parsed = parse_period_label(period)
                if not parsed:
                    continue
                if wanted_years and parsed["year"] not in wanted_years:
                    continue
                if (
                    wants_quarter
                    and parsed["kind"] == "Q"
                    and parsed["quarter"] != wants_quarter
                ):
                    continue
                selected_periods.append(period)

        if not selected_periods:
            # Default to the most recent periods by actual period end date
            selected_periods = ordered_periods[:3]

        blocks = []
        for period in selected_periods:
            if period not in by_period:
                continue

            sample = next(iter(by_period[period].values()), {})
            form_type = sample.get("form_type") or ""
            period_end = _period_end(period)
            header = f"[Structured Financial Data | {ticker} | {period}"
            if form_type:
                header += f" | {form_type}"
            header += "]"

            lines = [header, "-" * 62]
            if period_end != date.min:
                lines.append(f"  Period end: {period_end.isoformat()}")

            for field, fact in sorted(by_period[period].items()):
                value = fact["value"]
                unit = fact["unit"] or ""
                label = field.replace("_", " ").title()
                if value is not None:
                    numeric = float(value)
                    formatted = f"{numeric:,.2f}" if abs(numeric) >= 1 else f"{numeric:.4f}"
                    lines.append(f"  {label:<30} {formatted:>18} {unit}")
                else:
                    lines.append(f"  {label:<30} {'N/A':>18}")

            lines.append("-" * 62)
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
