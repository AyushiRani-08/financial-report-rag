"""
utils/abstention_guard.py
==========================
Multi-Layered Abstention Gate (MAG) - Layer 1 & Pre-Retrieval Guardrails.

Detects unanswerable, out-of-scope, or future-period financial questions
before sending them to the LLM, preventing hallucinations and guaranteeing
conservative, audited financial abstention.
"""

import re
from typing import Optional, Dict, Any, Tuple

# Latest audited 10-K period currently available in the database
LATEST_FILED_FY = 2024
LATEST_FILED_DATE = "2024-09-28"

OUT_OF_SCOPE_RULES = [
    # 1. Daily / Real-Time Stock & Market Prices
    {
        "pattern": r"\b(stock price|share price|closing price|market price|trading price|share quote)\b",
        "category": "market_data",
        "message": (
            "Annual Form 10-K filings do not contain daily or real-time market stock prices. "
            "10-K reports disclose audited accounting statements (balance sheets, income statements, "
            "cash flows) and per-share accounting metrics (EPS), not daily exchange trading quotes."
        ),
    },
    # 2. Competitor Financials inside current company's 10-K
    {
        "pattern": r"(?:according to|in)\s+(?:apple(?:'s)?|aapl(?:'s)?)\s*(?:10-k|filing|report|annual report).*?\b(samsung|microsoft|msft|google|alphabet|amazon|amzn|huawei|xiaomi)\b|"
                   r"\b(samsung|microsoft|msft|google|alphabet|amazon|amzn|huawei|xiaomi)\b.*?(?:according to|in)\s+(?:apple(?:'s)?|aapl(?:'s)?)\s*(?:10-k|filing|report|annual report)",
        "category": "competitor_data",
        "message": (
            "Apple Inc.'s Form 10-K filings disclose the financial results and operations of Apple Inc. only. "
            "They do not contain financial statements, revenue figures, or profit margins for external competitors."
        ),
    },
    # 3. Future Annual Revenue Guidance
    {
        "pattern": r"\b(?:revenue|earnings|sales)\s+guidance\s+for\s+(?:fy\s*2025|2025|fy\s*2026|2026)\b|"
                   r"\bguidance\s+for\s+(?:fy\s*2025|2025|fy\s*2026|2026)\b",
        "category": "future_guidance",
        "message": (
            "Apple does not disclose annual revenue guidance in its Form 10-K filings. "
            "10-K reports record historical audited performance; forward-looking quantitative annual revenue targets "
            "are not provided in annual filings."
        ),
    },
    # 4. Undisclosed Granular Divisional Headcount
    {
        "pattern": r"\b(?:number of employees|headcount|staff)\s+(?:per|by|in each)\s+(?:product\s+)?division\b|"
                   r"\bemployees\s+per\s+division\b",
        "category": "non_disclosed_metric",
        "message": (
            "Apple discloses its total consolidated full-time equivalent headcount in Item 1 of Form 10-K "
            "(approximately 164,000 employees as of FY2024), but does not disclose headcount broken down by "
            "individual product division."
        ),
    },
]


def check_pre_retrieval_abstention(
    query: str,
    ticker: Optional[str] = "AAPL",
    fiscal_period: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether a query must be refused / abstained from prior to retrieval.

    Returns:
        Dict with keys {"should_abstain": True, "category": ..., "reason": ...}
        or None if the question is eligible for standard RAG retrieval.
    """
    q_clean = query.strip()
    q_lower = q_clean.lower()

    # Rule A: Future Fiscal Periods & Quarters (> FY2024)
    # Matches e.g. Q3 FY2026, FY2026, Q1 2027, FY2025 revenue guidance
    quarter_future = re.search(r'\b(q[1-4])\s*(?:of\s*)?(?:fy\s*|fiscal\s*(?:year\s*)?)?(20\d\d)\b', q_lower)
    if quarter_future:
        q_num, yr_num = quarter_future.group(1).upper(), int(quarter_future.group(2))
        if yr_num > LATEST_FILED_FY:
            return {
                "should_abstain": True,
                "category": "non_existent_period",
                "reason": (
                    f"Financial results for {q_num} FY{yr_num} have not occurred or been filed. "
                    f"Audited 10-K reports in the database cover up to FY{LATEST_FILED_FY} (ended {LATEST_FILED_DATE}). "
                    "The system cannot provide information for future periods."
                ),
            }

    fy_match = re.search(r'\b(?:fy\s*|fiscal\s*(?:year\s*)?)(20\d\d)\b', q_lower)
    if fy_match:
        yr = int(fy_match.group(1))
        # If asking about FY2026 or beyond
        if yr > LATEST_FILED_FY + 1:
            return {
                "should_abstain": True,
                "category": "non_existent_period",
                "reason": (
                    f"Fiscal Year {yr} has not occurred or been filed. "
                    f"The most recent audited annual filing available is FY{LATEST_FILED_FY} (ended {LATEST_FILED_DATE})."
                ),
            }

    # Rule B: Specific Out-Of-Scope Patterns (Competitor data in Apple filing, daily stock prices, guidance, headcount)
    for rule in OUT_OF_SCOPE_RULES:
        if re.search(rule["pattern"], q_lower, re.IGNORECASE):
            return {
                "should_abstain": True,
                "category": rule["category"],
                "reason": rule["message"],
            }

    return None
