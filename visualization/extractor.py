# visualization/extractor.py
"""
Extracts and validates ChartData from raw LLM text output.

The LLM is instructed (via chart_prompt.py) to append a ```chart_data
JSON block at the end of its response. This module:
  1. Locates and extracts that block via regex.
  2. Parses and validates the JSON into a ChartData model.
  3. Returns None on any failure — chart extraction must never crash the app.

Additionally exposes get_display_text() which strips the JSON block
from the text so it is never shown raw to the user during streaming.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from visualization.models import ChartData, ChartSeries, ChartType

logger = logging.getLogger(__name__)

# Matches a complete ```chart_data ... ``` block (non-greedy, DOTALL)
_CHART_BLOCK_RE = re.compile(
    r"```(?:chart_data)\s*\n?(.*?)```",
    re.DOTALL,
)

# Matches an unclosed or partial trailing chart_data block (e.g. truncated generation)
_UNCLOSED_CHART_RE = re.compile(
    r"```chart_data\s*\n?(\{.*)\s*$",
    re.DOTALL,
)

# Fallback: matches any ```json block that contains financial chart series keys
_JSON_CHART_RE = re.compile(
    r"```(?:json)?\s*\n?(\{\s*\"type\"\s*:.*?\"series\"\s*:.*?)\s*```",
    re.DOTALL,
)

# Matches a partial (still-streaming) chart_data block that hasn't been closed yet
_PARTIAL_CHART_RE = re.compile(
    r"```chart_data.*$",
    re.DOTALL,
)

_VALID_CHART_TYPES = {ct.value for ct in ChartType}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_chart(raw_text: str) -> Optional[ChartData]:
    """
    Parse a ChartData object from the LLM's raw output.

    Returns None if:
    - No chart_data block is present.
    - The block contains malformed JSON.
    - The parsed data fails ChartData validation.

    Never raises — all errors are logged and swallowed.
    """
    if not raw_text:
        return None

    # 1. Try standard ```chart_data ... ```
    match = _CHART_BLOCK_RE.search(raw_text)
    if not match:
        # 2. Try unclosed chart_data block at end of text
        match = _UNCLOSED_CHART_RE.search(raw_text)
    if not match:
        # 3. Try standard json block containing chart keys
        match = _JSON_CHART_RE.search(raw_text)

    if not match:
        return None

    json_str = match.group(1).strip()

    # Clean common LLM JSON syntax issues (e.g. trailing commas before } or ])
    json_str = re.sub(r",\s*([\]\}])", r"\1", json_str)

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("[visualization] Failed to parse chart_data JSON: %s", exc)
        return None

    return _build_chart_data(payload)


def get_display_text(raw_text: str) -> str:
    """
    Strip the chart_data block (complete or partial) from the LLM output
    so the raw JSON is never shown to the user.

    Safe to call on partial (still-streaming) text.
    """
    # Remove completed blocks first
    text = _CHART_BLOCK_RE.sub("", raw_text)
    # Remove any partial block that started but hasn't been closed yet
    text = _PARTIAL_CHART_RE.sub("", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_chart_data(payload: dict) -> Optional[ChartData]:
    """Convert raw dict from JSON → validated ChartData. Returns None on error."""
    try:
        chart_type_str = str(payload.get("type", "bar")).lower()
        if chart_type_str not in _VALID_CHART_TYPES:
            logger.warning(
                "[visualization] Unknown chart type '%s'; defaulting to 'bar'.",
                chart_type_str,
            )
            chart_type_str = "bar"

        raw_series = payload.get("series", [])
        if not isinstance(raw_series, list) or not raw_series:
            logger.warning("[visualization] chart_data has no series; skipping.")
            return None

        series = [
            ChartSeries(
                name=str(s.get("name", f"Series {i+1}")),
                values=_coerce_values(s.get("values", [])),
                color=s.get("color"),
            )
            for i, s in enumerate(raw_series)
        ]

        x_labels = [str(v) for v in payload.get("x", [])]
        if not x_labels:
            logger.warning("[visualization] chart_data.x is empty; skipping.")
            return None

        return ChartData(
            type       = ChartType(chart_type_str),
            title      = str(payload.get("title", "Financial Metrics")),
            x          = x_labels,
            series     = series,
            x_label    = payload.get("x_label"),
            y_label    = payload.get("y_label"),
            currency   = payload.get("currency"),
        )

    except Exception as exc:
        logger.warning("[visualization] ChartData construction failed: %s", exc)
        return None


def _coerce_values(raw: list) -> list[float]:
    """Convert mixed int/float/str values to float, dropping non-numeric entries."""
    result = []
    for v in raw:
        try:
            result.append(float(str(v).replace(",", "").strip()))
        except (ValueError, TypeError):
            result.append(0.0)
    return result
