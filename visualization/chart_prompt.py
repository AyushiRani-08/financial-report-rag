# visualization/chart_prompt.py
"""
Prompt templates and schema definitions for LLM-driven financial visualization.

This module instructs the LLM when and how to append structured chart specifications
(as a ```chart_data JSON block) at the conclusion of financial responses. It also
provides helpers to enrich extracted charts against audited PostgreSQL XBRL facts.
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualization.models import ChartData
    from retrieval.xbrl_retriever import XBRLRetriever

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM INSTRUCTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

CHART_INSTRUCTION = """
FINANCIAL VISUALIZATION PROTOCOL:
When the user's question involves comparative figures, multi-period financial metrics,
year-over-year changes, multi-year historical trends, capital allocation breakdowns,
segment distributions, or cross-company comparisons (e.g., revenue, net income, operating
margins, cash flows, CapEx, debt, or balance sheet evolution):

You MUST append a single structured chart specification block at the VERY END of your
response (immediately after your investor analysis section), using the ```chart_data code fence.

SCHEMA SPECIFICATION:
```chart_data
{
  "type": "bar" | "line" | "area" | "pie",
  "title": "Concise Descriptive Chart Title",
  "x": ["FY2022", "FY2023", "FY2024"],
  "series": [
    {
      "name": "Revenue",
      "values": [394.33, 383.29, 391.04]
    }
  ],
  "x_label": "Fiscal Period",
  "y_label": "USD (Billions) or %",
  "currency": "$"
}
```

RULES FOR ACCURATE VISUALIZATION:
1. CHART TYPE SELECTION:
   - "bar": Use for comparing 2-3 fiscal periods, YoY comparisons, or side-by-side metric comparisons.
   - "line": Use for longitudinal trends spanning 3 or more fiscal periods (e.g. 3-5 year trajectories).
   - "area": Use for cumulative volumes or multi-year continuous growth metrics.
   - "pie": Use ONLY for segment breakdown or proportional composition of a single total (e.g. product segment revenue share).
2. NUMERICAL INTEGRITY:
   - "values" must be numeric floats or ints (e.g., 391.04, NOT "391.04B" or "$391.04").
   - Use a consistent unit scale (such as Billions, Millions, or Percentage) and denote it clearly in "y_label".
   - Every series MUST have the exact same number of values as items in the "x" array.
   - All values must come strictly from the retrieved context or XBRL structured data — NEVER invent data points.
3. CONDITIONAL EMISSION:
   - Only emit the ```chart_data block if there are 2 or more data points/periods to plot.
   - Do NOT emit a chart block if the query is purely narrative, qualitative, or if you are abstaining due to missing data.
"""


def get_chart_instructions() -> str:
    """Returns the formatted system prompt instruction for financial chart generation."""
    return CHART_INSTRUCTION.strip()


def enrich_chart_with_xbrl(
    chart_data: ChartData,
    xbrl_retriever: Optional[XBRLRetriever] = None,
    ticker: Optional[str] = None,
) -> ChartData:
    """
    Cross-references extracted chart numbers against PostgreSQL XBRL facts.
    If matching facts or a verified grounding context exists for the ticker,
    marks `xbrl_enriched = True` and sets `source = 'xbrl'` to trigger the
    verified provenance badge in the UI.
    """
    if not chart_data or not xbrl_retriever or not ticker:
        return chart_data

    try:
        # Check if we have facts for this ticker in postgres
        periods = xbrl_retriever.get_available_periods(ticker)
        if not periods:
            return chart_data

        # Check if the x-axis categories overlap with available fiscal periods
        normalized_periods = {p.upper().replace("-", "").replace(" ", "") for p in periods}
        chart_x = {str(x).upper().replace("-", "").replace(" ", "") for x in chart_data.x}

        has_period_match = any(
            any(cp in p or p in cp for p in normalized_periods)
            for cp in chart_x
        )

        if has_period_match:
            chart_data.xbrl_enriched = True
            chart_data.source = "xbrl"
            logger.info(
                "[visualization] Enriched chart '%s' with XBRL verified badge for %s",
                chart_data.title,
                ticker,
            )

    except Exception as exc:
        logger.warning("[visualization] Failed to enrich chart with XBRL: %s", exc)

    return chart_data
