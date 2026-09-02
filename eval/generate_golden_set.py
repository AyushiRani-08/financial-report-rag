"""
eval/generate_golden_set.py
Auto-generates a golden benchmark dataset from PostgreSQL XBRL facts + qualitative criteria.
Saves to eval/golden_set.json.
"""

import os
import sys
import json
from pathlib import Path
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Allow loading modules from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"

# Question templates per canonical field
QUESTION_TEMPLATES = {
    "revenue": [
        "What was {name}'s ({ticker}) total revenue in {period}?",
        "How much net sales / revenue did {ticker} generate for {period}?",
    ],
    "net_income": [
        "What was the net income for {ticker} in {period}?",
        "How much profit / net income did {name} report for {period}?",
    ],
    "operating_income": [
        "What was {ticker}'s operating income in {period}?",
        "How much operating profit did {name} generate in {period}?",
    ],
    "gross_profit": [
        "What was {ticker}'s gross profit in {period}?",
    ],
    "eps_basic": [
        "What was the basic earnings per share (EPS) for {ticker} in {period}?",
    ],
    "operating_cash_flow": [
        "What was the operating cash flow reported by {ticker} for {period}?",
    ],
    "long_term_debt": [
        "What was the total long-term debt for {ticker} in {period}?",
    ],
    "total_assets": [
        "What was the total asset value reported by {ticker} in {period}?",
    ],
}

# Curated qualitative benchmark questions for filings
QUALITATIVE_BENCHMARKS = [
    {
        "id": "aapl_fy25_risk_supply_chain",
        "ticker": "AAPL",
        "fiscal_period": "FY2025",
        "category": "qualitative",
        "question": "What major risk factors does Apple disclose regarding supply chain and component manufacturing?",
        "key_concepts": [
            "single-source suppliers",
            "concentration of manufacturing",
            "geopolitical tensions",
            "logistics disruptions"
        ],
        "expected_section": "Item 1A. Risk Factors"
    },
    {
        "id": "aapl_fy25_mda_services_growth",
        "ticker": "AAPL",
        "fiscal_period": "FY2025",
        "category": "qualitative",
        "question": "How did Apple's Services division perform and what were the key drivers of growth according to MD&A?",
        "key_concepts": [
            "Services revenue",
            "App Store",
            "Cloud services",
            "Advertising",
            "Payment services"
        ],
        "expected_section": "Item 7. Management's Discussion and Analysis"
    },
    {
        "id": "aapl_fy25_capital_allocation",
        "ticker": "AAPL",
        "fiscal_period": "FY2025",
        "category": "qualitative",
        "question": "What is Apple's capital return program, including share buybacks and dividend distributions?",
        "key_concepts": [
            "share repurchases",
            "dividends",
            "capital return",
            "common stock"
        ],
        "expected_section": "Liquidity and Capital Resources"
    }
]


def format_human_value(val: float, field: str, unit: str) -> str:
    """Format numeric values cleanly for financial presentation."""
    if field == "eps_basic":
        return f"${val:,.2f}"
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f} trillion ({val:,.0f} {unit})"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f} billion ({val:,.0f} {unit})"
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f} million ({val:,.0f} {unit})"
    return f"{val:,.2f} {unit}"


def generate_golden_set(dsn: str | None = None) -> list[dict]:
    dsn = dsn or os.environ.get("POSTGRES_DSN")
    if not dsn:
        print("[ERROR] POSTGRES_DSN not set.")
        sys.exit(1)

    print("[*] Connecting to PostgreSQL to extract XBRL ground truth facts...")
    conn = psycopg2.connect(dsn)

    query = """
        SELECT
            c.ticker,
            c.name               AS company_name,
            c.market,
            fi.form_type,
            fi.fiscal_period,
            ff.canonical_field,
            MAX(ff.value)        AS value,
            ff.unit,
            ff.period_start,
            ff.period_end
        FROM financial_facts ff
        JOIN filings fi   ON fi.filing_id   = ff.filing_id
        JOIN companies c  ON c.company_id   = fi.company_id
        WHERE ff.canonical_field = ANY(%s)
          AND ff.value IS NOT NULL
          AND ff.period_end IS NOT NULL
        GROUP BY c.ticker, c.name, c.market, fi.form_type, fi.fiscal_period,
                 ff.canonical_field, ff.unit, ff.period_start, ff.period_end
        ORDER BY c.ticker, fi.fiscal_period, ff.canonical_field
    """

    fields_to_track = list(QUESTION_TEMPLATES.keys())
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (fields_to_track,))
        rows = cur.fetchall()

    conn.close()
    print(f"[*] Retrieved {len(rows)} raw fact groups from PostgreSQL.")

    # Deduplicate facts: pick the annual / longest duration period per (ticker, period, field)
    fact_map = {}
    for r in rows:
        ticker = r["ticker"]
        period = r["fiscal_period"]
        field  = r["canonical_field"]
        key = (ticker, period, field)

        p_start = r["period_start"]
        p_end   = r["period_end"]
        duration = (p_end - p_start).days if (p_start and p_end) else 0

        val = float(r["value"])
        if key not in fact_map or duration > fact_map[key]["duration"]:
            fact_map[key] = {
                "ticker": ticker,
                "company_name": r["company_name"],
                "market": r["market"],
                "form_type": r["form_type"],
                "fiscal_period": period,
                "canonical_field": field,
                "ground_truth_value": val,
                "unit": r["unit"],
                "period_start": str(p_start) if p_start else None,
                "period_end": str(p_end) if p_end else None,
                "duration": duration,
            }

    golden_cases = []
    case_counter = 1

    # Generate quantitative test cases
    for (ticker, period, field), fact in sorted(fact_map.items()):
        templates = QUESTION_TEMPLATES.get(field, ["What was the {field} for {ticker} in {period}?"])
        # Use first template for primary benchmark
        question_text = templates[0].format(
            name=fact["company_name"],
            ticker=ticker,
            period=period,
            field=field.replace("_", " "),
        )

        formatted_answer = format_human_value(fact["ground_truth_value"], field, fact["unit"])

        case = {
            "id": f"{ticker.lower()}_{period.lower()}_{field}_{case_counter:03d}",
            "ticker": ticker,
            "company_name": fact["company_name"],
            "fiscal_period": period,
            "category": "quantitative",
            "question": question_text,
            "canonical_field": field,
            "ground_truth_value": fact["ground_truth_value"],
            "ground_truth_unit": fact["unit"],
            "ground_truth_formatted": formatted_answer,
            "relative_tolerance": 0.015,  # 1.5% tolerance for rounding (e.g. 416.2B vs 416.16B)
            "period_end": fact["period_end"],
        }
        golden_cases.append(case)
        case_counter += 1

    # Append qualitative benchmark questions for available tickers
    available_tickers = {fact["ticker"] for fact in fact_map.values()}
    for q_case in QUALITATIVE_BENCHMARKS:
        if q_case["ticker"] in available_tickers:
            golden_cases.append(q_case)

    # Save to golden_set.json
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_cases, f, indent=2)

    print(f"[OK] Generated {len(golden_cases)} golden test cases.")
    print(f"     Saved to: {GOLDEN_SET_PATH.resolve()}")
    return golden_cases


if __name__ == "__main__":
    generate_golden_set()
