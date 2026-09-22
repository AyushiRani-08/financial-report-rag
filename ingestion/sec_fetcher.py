"""
ingestion/sec_fetcher.py
Fetch XBRL facts from SEC EDGAR public API by ticker. No file download needed.

SEC APIs used (free, no auth):
  Ticker->CIK : https://www.sec.gov/files/company_tickers.json
  Facts JSON  : https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EDGAR_HEADERS = {
    "User-Agent": "FinSight financial-report-rag research@finsight.dev",
    "Accept-Encoding": "gzip, deflate",
}

# US-GAAP tag -> normalized canonical field
CANONICAL_MAP = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "GrossProfit": "gross_profit",
    "CostOfRevenue": "cost_of_revenue",
    "CostOfGoodsAndServicesSold": "cost_of_revenue",
    "OperatingIncomeLoss": "operating_income",
    "OperatingExpenses": "operating_expenses",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
    "NetIncomeLoss": "net_income",
    "ProfitLoss": "net_income",
    "IncomeTaxExpenseBenefit": "income_tax_expense",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "pretax_income",
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "PropertyPlantAndEquipmentNet": "ppe_net",
    "Goodwill": "goodwill",
    "InventoryNet": "inventory",
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "Liabilities": "total_liabilities",
    "LiabilitiesCurrent": "current_liabilities",
    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "long_term_debt",
    "StockholdersEquity": "stockholders_equity",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
    "NetCashProvidedByUsedInFinancingActivities": "financing_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsForRepurchaseOfCommonStock": "share_repurchases",
    "PaymentsOfDividendsCommonStock": "dividends_paid",
    "PaymentsOfDividends": "dividends_paid",
}


COMPANY_ALIASES = {
    "jpmorgan": "JPM",
    "jpmorgan chase": "JPM",
    "jp morgan": "JPM",
    "jpmc": "JPM",
    "boeing": "BA",
    "boeing co": "BA",
    "the boeing company": "BA",
    "tesla": "TSLA",
    "tesla motors": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "berkshire": "BRK-A",
}


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Resolve ticker or company name -> zero-padded 10-digit CIK. Returns None if not found."""
    clean = ticker.strip().lower()
    clean_ticker = COMPANY_ALIASES.get(clean, ticker.upper().strip())

    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    # 1. Exact ticker match
    for entry in data.values():
        if entry.get("ticker", "").upper() == clean_ticker:
            return str(entry["cik_str"]).zfill(10)

    # 2. Company title match
    for entry in data.values():
        title = entry.get("title", "").lower()
        if clean in title or title in clean:
            return str(entry["cik_str"]).zfill(10)

    return None


def fetch_company_facts(cik: str) -> dict:
    """Fetch all XBRL facts from SEC companyfacts API for a CIK."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _infer_fiscal_period(
    period_end: str,
    form: str,
    fy: Optional[int] = None,
    fp: Optional[str] = None,
) -> Optional[str]:
    """Convert period end date + SEC fy/fp + form type to a canonical fiscal period label e.g. FY2025, Q1FY2024."""
    if fy and fp:
        fp_clean = str(fp).strip().upper()
        if fp_clean in ("Q1", "Q2", "Q3", "Q4"):
            return f"{fp_clean}FY{fy}"
        elif fp_clean in ("FY", "CY"):
            return f"FY{fy}"

    try:
        d = datetime.strptime(period_end, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    year, month = d.year, d.month
    if "10-K" in form or "20-F" in form:
        return f"FY{year}"
    elif "10-Q" in form:
        if month <= 3:   return f"Q1FY{year}"
        elif month <= 6: return f"Q2FY{year}"
        elif month <= 9: return f"Q3FY{year}"
        else:            return f"Q4FY{year}"
    return None


def _get_or_create_company(cur, ticker, name, cik, market="US") -> int:
    cur.execute("SELECT company_id FROM companies WHERE ticker = %s", (ticker,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO companies (ticker, name, cik, market, sector) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING company_id",
        (ticker, name, cik, market, None),
    )
    return cur.fetchone()[0]


def _get_or_create_filing(cur, company_id, form_type, fiscal_period, source_url) -> int:
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
        "VALUES (%s, %s, %s, %s, %s) RETURNING filing_id",
        (company_id, form_type, fiscal_period, None, source_url),
    )
    return cur.fetchone()[0]


def fetch_and_store_xbrl(
    ticker: str,
    fiscal_period: Optional[str] = None,
    form_type: str = "10-K",
    market: str = "US",
    dsn: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Fetch XBRL facts from SEC EDGAR for a ticker and insert into PostgreSQL.

    Args:
        ticker:            Stock ticker e.g. 'MSFT'
        fiscal_period:     e.g. 'FY2025'. None = import all available periods.
        form_type:         '10-K' (annual) or '10-Q' (quarterly)
        market:            'US' or 'IN'
        dsn:               PostgreSQL DSN. Defaults to POSTGRES_DSN env var.
        progress_callback: Optional fn(msg) for live UI progress updates.

    Returns:
        dict with ticker, cik, company_name, facts_inserted, periods_found.
    """
    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    dsn = dsn or os.environ["POSTGRES_DSN"]
    ticker = ticker.upper().strip()

    _log(f"Looking up CIK for {ticker}...")
    cik = get_cik_for_ticker(ticker)
    if not cik:
        raise ValueError(
            f"Ticker '{ticker}' not found on SEC EDGAR. "
            "Verify the symbol is a US-listed company."
        )
    _log(f"CIK: {cik}")

    _log("Fetching financial facts from SEC EDGAR API...")
    facts_json = fetch_company_facts(cik)
    company_name = facts_json.get("entityName", ticker)
    _log(f"Company: {company_name}")

    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        raise ValueError(
            f"No US-GAAP XBRL facts found for {ticker}. "
            "Company may not file in XBRL format with the SEC."
        )

    # For 10-Q filings, fiscal_period like "FY2025" will never match
    # quarter-inferred labels like "Q3FY2024". Only apply period filter for 10-K.
    effective_period = fiscal_period if form_type == "10-K" else None

    conn = psycopg2.connect(dsn)
    total_inserted = 0
    periods_found: set = set()
    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    try:
        with conn:
            cur = conn.cursor()
            company_id = _get_or_create_company(cur, ticker, company_name, cik, market)

            for tag_name, tag_data in us_gaap.items():
                canonical = CANONICAL_MAP.get(
                    tag_name,
                    tag_name[0].lower() + tag_name[1:],
                )
                for unit_label, fact_list in tag_data.get("units", {}).items():
                    unit = unit_label.split("/")[-1] if "/" in unit_label else unit_label

                    for fact in fact_list:
                        fact_form = fact.get("form", "")
                        if form_type not in fact_form:
                            continue
                        period_end = fact.get("end") or fact.get("instant", "")
                        inferred = _infer_fiscal_period(
                            period_end, fact_form, fact.get("fy"), fact.get("fp")
                        )
                        if not inferred:
                            continue
                        if fiscal_period and inferred != fiscal_period:
                            continue
                        if effective_period and inferred != effective_period:
                            continue

                        periods_found.add(inferred)
                        filing_id = _get_or_create_filing(
                            cur, company_id, fact_form, inferred, source_url
                        )
                        try:
                            value = float(fact.get("val", 0))
                            cur.execute(
                                "INSERT INTO financial_facts "
                                "(filing_id, tag, canonical_field, value, unit, "
                                "period_start, period_end, source_page) "
                                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                                (
                                    filing_id,
                                    f"us-gaap:{tag_name}",
                                    canonical,
                                    value,
                                    unit,
                                    fact.get("start"),
                                    period_end,
                                    None,
                                ),
                            )
                            total_inserted += 1
                        except Exception:
                            conn.rollback()

        _log(f"Done: {total_inserted:,} facts inserted | Periods: {sorted(periods_found)}")

    finally:
        conn.close()

    return {
        "ticker": ticker,
        "cik": cik,
        "company_name": company_name,
        "facts_inserted": total_inserted,
        "periods_found": sorted(periods_found),
    }


def get_primary_filing_info(
    ticker: str,
    form_type: str = "10-K",
    fiscal_year: Optional[int] = None,
) -> Optional[dict]:
    """
    Look up the primary HTML filing document from SEC EDGAR Submissions API.
    Handles:
      - 10-K, 10-Q, 20-F, and amendment forms (10-K/A, 10-Q/A)
      - Fiscal year matching across 'recent' and submission partition files
      - Graceful fallback to the latest available filing of that type if specific year is not found
    """
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return None
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    clean_form = form_type.upper().strip()
    acceptable_forms = [clean_form]
    if clean_form == "10-K":
        acceptable_forms.extend(["10-K/A", "20-F", "20-F/A"])
    elif clean_form == "10-Q":
        acceptable_forms.extend(["10-Q/A", "6-K"])

    def _extract_filing(src_dict, idx):
        fdate = src_dict["filingDate"][idx]
        rdate = src_dict.get("reportDate", [""] * len(src_dict["filingDate"]))[idx] or fdate
        acc_num = src_dict["accessionNumber"][idx].replace("-", "")
        doc_name = src_dict["primaryDocument"][idx]
        cik_int = int(cik)
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_num}/{doc_name}"
        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "company_name": data.get("name", ticker),
            "form": src_dict["form"][idx],
            "filing_date": fdate,
            "report_date": rdate,
            "doc_name": doc_name,
            "doc_url": doc_url,
        }

    recent = data.get("filings", {}).get("recent", {})
    recent_forms = recent.get("form", [])

    latest_fallback = None

    for i, form in enumerate(recent_forms):
        if form in acceptable_forms:
            filing = _extract_filing(recent, i)
            if latest_fallback is None:
                latest_fallback = filing
            if fiscal_year:
                if str(fiscal_year) in filing["report_date"] or str(fiscal_year) in filing["filing_date"]:
                    return filing
            else:
                return filing

    # If fiscal_year was specified and not found in 'recent', scan older partition files
    if fiscal_year:
        partition_files = data.get("filings", {}).get("files", [])
        for p_info in partition_files[:10]:
            p_name = p_info.get("name")
            if not p_name:
                continue
            try:
                p_url = f"https://data.sec.gov/submissions/{p_name}"
                p_resp = requests.get(p_url, headers=EDGAR_HEADERS, timeout=15)
                if p_resp.status_code == 200:
                    p_data = p_resp.json()
                    p_forms = p_data.get("form", [])
                    for i, form in enumerate(p_forms):
                        if form in acceptable_forms:
                            filing = _extract_filing(p_data, i)
                            if str(fiscal_year) in filing["report_date"] or str(fiscal_year) in filing["filing_date"]:
                                return filing
            except Exception as e:
                logger.warning(f"Error checking partition {p_name}: {e}")

    if latest_fallback:
        if fiscal_year:
            logger.warning(
                f"Requested {form_type} for FY{fiscal_year} not found for {ticker}; "
                f"falling back to latest available ({latest_fallback['report_date']})"
            )
        return latest_fallback

    return None


def download_sec_filing(
    ticker: str,
    form_type: str = "10-K",
    save_dir: Optional[Path] = None,
    fiscal_year: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Automatically downloads the complete primary SEC filing (.htm) for a company.
    Saves it into data/raw_pdfs/ with a clean standardized name.
    """
    def _log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _log(f"Resolving SEC EDGAR filings for {ticker} ({form_type})...")
    info = get_primary_filing_info(ticker, form_type, fiscal_year)
    if not info:
        raise ValueError(
            f"No {form_type} filing found on SEC EDGAR for '{ticker}'. "
            f"Please verify the company name/ticker (e.g. MSFT, AAPL, JPM, TSLA, BA)."
        )

    save_dir = save_dir or (Path(__file__).resolve().parent.parent / "data" / "raw_pdfs")
    save_dir.mkdir(parents=True, exist_ok=True)

    clean_file_name = f"{ticker.lower()}_{info['report_date']}_{form_type.lower()}.htm".replace("-", "")
    target_path = save_dir / clean_file_name

    _log(f"Downloading primary report ({info['doc_name']}) from SEC EDGAR...")
    resp = requests.get(info["doc_url"], headers=EDGAR_HEADERS, timeout=60)
    resp.raise_for_status()

    with open(target_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(resp.text)

    file_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)
    _log(f"Downloaded {file_size_mb} MB -> saved as `{clean_file_name}`")

    return {
        **info,
        "file_path": str(target_path),
        "file_name": clean_file_name,
        "size_mb": file_size_mb,
    }

