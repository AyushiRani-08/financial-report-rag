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

import psycopg2
from lxml import etree

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
CANONICAL_MAP: dict[str, str] = {
    # Revenue
    "Revenues":                                    "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet":                             "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    # Net income
    "NetIncomeLoss":                               "net_income",
    "ProfitLoss":                                  "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",
    # Operating income
    "OperatingIncomeLoss":                         "operating_income",
    # EPS
    "EarningsPerShareBasic":                       "eps_basic",
    "EarningsPerShareDiluted":                     "eps_diluted",
    # Balance sheet
    "Assets":                                      "total_assets",
    "Liabilities":                                 "total_liabilities",
    "StockholdersEquity":                          "stockholders_equity",
    "LiabilitiesAndStockholdersEquity":            "total_liabilities_equity",
    "LongTermDebt":                                "long_term_debt",
    "LongTermDebtNoncurrent":                      "long_term_debt",
    "CashAndCashEquivalentsAtCarryingValue":       "cash",
    "CashCashEquivalentsAndShortTermInvestments":  "cash_and_investments",
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities":  "operating_cash_flow",
    "NetCashProvidedByUsedInInvestingActivities":  "investing_cash_flow",
    "NetCashProvidedByUsedInFinancingActivities":  "financing_cash_flow",
    # Shares
    "CommonStockSharesOutstanding":                "shares_outstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic": "weighted_avg_shares_basic",
    # Gross profit
    "GrossProfit":                                 "gross_profit",
    "CostOfRevenue":                               "cost_of_revenue",
    "CostOfGoodsAndServicesSold":                  "cost_of_revenue",
    # R&D / SG&A
    "ResearchAndDevelopmentExpense":               "rd_expense",
    "SellingGeneralAndAdministrativeExpense":      "sga_expense",
    # DEI fields
    "EntityCommonStockSharesOutstanding":          "shares_outstanding",
}


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
    """Return the namespace prefix: {http://fasb.org/us-gaap/2025}X → us-gaap"""
    tag = element.tag
    if "}" not in tag:
        return ""
    uri = tag[1:tag.index("}")]
    # Map known URIs to prefixes
    uri_map = {
        "http://fasb.org/us-gaap/2025": "us-gaap",
        "http://fasb.org/us-gaap/2024": "us-gaap",
        "http://fasb.org/us-gaap/2023": "us-gaap",
        "http://xbrl.sec.gov/dei/2025": "dei",
        "http://xbrl.sec.gov/dei/2024": "dei",
    }
    return uri_map.get(uri, "")


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
            }
        elif start is not None and end is not None:
            contexts[ctx_id] = {
                "period_start": _parse_date(start.text),
                "period_end":   _parse_date(end.text),
                "is_instant":   False,
            }

    return contexts


def _build_unit_map(root) -> dict[str, str]:
    """Build a dict of unitId → unit label (USD, shares, etc.)"""
    units = {}
    ns_xbrli = "http://www.xbrl.org/2003/instance"

    for unit in root.iter(f"{{{ns_xbrli}}}unit"):
        uid = unit.get("id", "")
        measure = unit.find(f"{{{ns_xbrli}}}measure")
        if measure is not None and measure.text:
            # 'iso4217:USD' → 'USD', 'xbrli:shares' → 'shares'
            label = measure.text.split(":")[-1]
            units[uid] = label

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
                         period_start, period_end, source_page)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        filing_id,
                        raw_tag,
                        canonical,
                        value,
                        unit,
                        ctx["period_start"],
                        ctx["period_end"],
                        xbrl_path,
                    ),
                )
                inserted += 1

        logger.info("Inserted %d facts for %s (%s %s)", inserted, ticker, form_type, fiscal_period)

    finally:
        conn.close()

    return inserted
