"""
ingestion/xbrl_parser.py
Parse SEC XBRL instance documents (.xml) directly with lxml.
No dependency on python-xbrl or arelle — works on Python 3.13+.

Inserts structured facts into PostgreSQL:
  companies → filings → financial_facts
"""

import os
import logging
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
import psycopg2
from lxml import etree

load_dotenv()

logger = logging.getLogger(__name__)

# ── XML namespace map ──────────────────────────────────────────────────────────
NS = {
    "xbrli":   "http://www.xbrl.org/2003/instance",
    "us-gaap": "http://fasb.org/us-gaap/2025",
    "dei":     "http://xbrl.sec.gov/dei/2025",
    "link":    "http://www.xbrl.org/2003/linkbase",
    "xlink":   "http://www.w3.org/1999/xlink",
}

# ── Raw US-GAAP tag → normalized canonical field ──────────────────────────────
# Canonical tag -> field mapping (single source of truth)
from utils.canonical_map import CANONICAL_MAP



# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _local_tag(element) -> str:
    """Strip namespace URI from tag: {http://...}TagName → TagName"""
    tag = element.tag
    if "}" in tag:
        return tag.split("}")[1]
    return tag


def _ns_prefix(element) -> str:
    """Return the namespace prefix: {http://fasb.org/us-gaap/2025}X → us-gaap

    Matched by URI prefix so new taxonomy years (2026, 2027, …) keep working
    instead of silently dropping every fact.
    """
    tag = element.tag
    if "}" not in tag:
        return ""
    uri = tag[1:tag.index("}")]
    if uri.startswith("http://fasb.org/us-gaap/") or uri.startswith("https://fasb.org/us-gaap/"):
        return "us-gaap"
    if uri.startswith("http://xbrl.sec.gov/dei/") or uri.startswith("https://xbrl.sec.gov/dei/"):
        return "dei"
    if uri.startswith("http://xbrl.ifrs.org/taxonomy/"):
        return "ifrs-full"
    return ""


def _build_context_map(root) -> dict[str, dict]:
    """
    Build a dict of contextId → {period_start, period_end, is_instant, segment}
    Only keep non-segment (consolidated) contexts.
    """
    contexts = {}
    ns_xbrli = "http://www.xbrl.org/2003/instance"

    for ctx in root.iter(f"{{{ns_xbrli}}}context"):
        ctx_id = ctx.get("id", "")

        # Skip dimensional contexts (segment/scenario) — we want consolidated only
        if ctx.find(f"{{{ns_xbrli}}}entity/{{{ns_xbrli}}}segment") is not None:
            continue

        period = ctx.find(f"{{{ns_xbrli}}}period")
        if period is None:
            continue

        instant = period.find(f"{{{ns_xbrli}}}instant")
        start   = period.find(f"{{{ns_xbrli}}}startDate")
        end     = period.find(f"{{{ns_xbrli}}}endDate")

        if instant is not None:
            contexts[ctx_id] = {
                "period_start": None,
                "period_end":   _parse_date(instant.text),
                "is_instant":   True,
                "duration_days": None,
            }
        elif start is not None and end is not None:
            start_d = _parse_date(start.text)
            end_d = _parse_date(end.text)
            duration = (end_d - start_d).days if (start_d and end_d) else None
            contexts[ctx_id] = {
                "period_start": start_d,
                "period_end":   end_d,
                "is_instant":   False,
                "duration_days": duration,
            }

    return contexts


def _build_unit_map(root) -> dict[str, str]:
    """Build a dict of unitId → unit label ('USD', 'USD/share', 'shares')."""
    units = {}
    ns_xbrli = "http://www.xbrl.org/2003/instance"

    for unit in root.iter(f"{{{ns_xbrli}}}unit"):
        uid = unit.get("id", "")
        measures = [
            measure.text.split(":")[-1]
            for measure in unit.iter(f"{{{ns_xbrli}}}measure")
            if measure.text
        ]
        if not measures:
            continue
        if len(measures) >= 2:
            denominator = measures[1]
            if denominator.lower() in ("shares", "share"):
                denominator = "share"
            units[uid] = f"{measures[0]}/{denominator}"
        else:
            units[uid] = measures[0]

    return units


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_or_create_company(cur, ticker, name, cik, market, sector) -> int:
    cur.execute("SELECT company_id FROM companies WHERE ticker = %s", (ticker,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO companies (ticker, name, cik, market, sector) "
        "VALUES (%s,%s,%s,%s,%s) RETURNING company_id",
        (ticker, name, cik, market, sector),
    )
    return cur.fetchone()[0]


def _get_or_create_filing(cur, company_id, form_type, fiscal_period, filed_date, source_url) -> int:
    cur.execute(
        "SELECT filing_id FROM filings "
        "WHERE company_id=%s AND form_type=%s AND fiscal_period=%s",
        (company_id, form_type, fiscal_period),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO filings (company_id, form_type, fiscal_period, filed_date, source_url) "
        "VALUES (%s,%s,%s,%s,%s) RETURNING filing_id",
        (company_id, form_type, fiscal_period, filed_date, source_url),
    )
    return cur.fetchone()[0]


# ── Main function ──────────────────────────────────────────────────────────────

def parse_xbrl_to_db(
    xbrl_path: str,
    ticker: str,
    company_name: str,
    form_type: str,
    fiscal_period: str,
    market: str = "US",
    cik: Optional[str] = None,
    sector: Optional[str] = None,
    filed_date: Optional[date] = None,
    dsn: Optional[str] = None,
) -> int:
    """
    Parse an XBRL instance document and insert financial facts into PostgreSQL.

    Uses lxml to parse directly — no python-xbrl or arelle dependency.
    Skips dimensional (segment) facts and keeps only consolidated figures.

    Returns the number of fact rows inserted.
    """
    dsn = dsn or os.environ["POSTGRES_DSN"]
    default_unit = "USD" if market == "US" else "INR"

    logger.info("Parsing XBRL: %s", xbrl_path)
    tree = etree.parse(xbrl_path)
    root = tree.getroot()

    context_map = _build_context_map(root)
    unit_map    = _build_unit_map(root)
    logger.info("Contexts (consolidated): %d | Units: %d", len(context_map), len(unit_map))

    conn = psycopg2.connect(dsn)
    inserted = 0

    try:
        with conn:
            cur = conn.cursor()

            company_id = _get_or_create_company(cur, ticker, company_name, cik, market, sector)
            filing_id  = _get_or_create_filing(
                cur, company_id, form_type, fiscal_period, filed_date, xbrl_path
            )

            # Iterate every element — XBRL facts are direct children of root
            for elem in root:
                tag_local  = _local_tag(elem)
                ns_prefix  = _ns_prefix(elem)

                # Only care about us-gaap and dei facts
                if ns_prefix not in ("us-gaap", "dei"):
                    continue

                ctx_ref  = elem.get("contextRef", "")
                unit_ref = elem.get("unitRef", "")
                decimals = elem.get("decimals", "")
                value_str = (elem.text or "").strip()

                # Skip facts with no value or non-consolidated contexts
                if not value_str or ctx_ref not in context_map:
                    continue

                # Parse numeric value
                try:
                    value = float(value_str)
                except ValueError:
                    continue  # skip non-numeric (string) facts

                ctx   = context_map[ctx_ref]
                unit  = unit_map.get(unit_ref, default_unit)
                canonical = CANONICAL_MAP.get(tag_local, tag_local[0].lower() + tag_local[1:])
                raw_tag   = f"{ns_prefix}:{tag_local}"

                cur.execute(
                    """
                    INSERT INTO financial_facts
                        (filing_id, tag, canonical_field, value, unit,
                         period_start, period_end, duration_days, is_instant, source_page)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        filing_id,
                        raw_tag,
                        canonical,
                        value,
                        unit,
                        ctx["period_start"],
                        ctx["period_end"],
                        ctx["duration_days"],
                        ctx["is_instant"],
                        xbrl_path,
                    ),
                )
                if cur.rowcount:
                    inserted += 1

        logger.info("Inserted %d facts for %s (%s %s)", inserted, ticker, form_type, fiscal_period)

    finally:
        conn.close()

    return inserted
