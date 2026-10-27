# visualization/__init__.py
"""
Financial Visualization Subsystem for FinSight.

Provides typed data models, LLM chart extraction, XBRL provenance enrichment,
and Plotly figure rendering for corporate financial reports.
"""
from visualization.models import ChartData, ChartSeries, ChartType
from visualization.extractor import extract_chart, get_display_text
from visualization.renderer import render_chart
from visualization.chart_prompt import (
    CHART_INSTRUCTION,
    get_chart_instructions,
    enrich_chart_with_xbrl,
)

__all__ = [
    "ChartData",
    "ChartSeries",
    "ChartType",
    "extract_chart",
    "get_display_text",
    "render_chart",
    "CHART_INSTRUCTION",
    "get_chart_instructions",
    "enrich_chart_with_xbrl",
]
