# visualization/renderer.py
"""
Renders ChartData objects as Plotly figures inside Streamlit.

Design principles:
- One public entry point: render_chart(chart_data).
- Dark theme is hardcoded to match the app's design system.
- Each chart type has a dedicated private builder for clean separation.
- All rendering errors are caught and shown as a graceful fallback
  warning — never as a crash.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from visualization.models import ChartData, ChartType

logger = logging.getLogger(__name__)

# ── Design tokens (mirrors 1_Dashboard.py and research_workspace.py) ─────────
_PALETTE = [
    "#3b82f6",   # blue
    "#10b981",   # emerald
    "#f59e0b",   # amber
    "#f87171",   # rose
    "#818cf8",   # indigo
    "#fbbf24",   # yellow
    "#34d399",   # teal
    "#fb923c",   # orange
]

_LAYOUT_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(255,255,255,0.02)",
    font          = dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    legend        = dict(
        bgcolor     = "rgba(0,0,0,0)",
        bordercolor = "rgba(255,255,255,0.08)",
        borderwidth = 1,
        font        = dict(color="#cbd5e1", size=11),
    ),
    margin        = dict(l=12, r=12, t=48, b=12),
    hoverlabel    = dict(
        bgcolor   = "#1e293b",
        font      = dict(family="Inter, sans-serif", color="#f8fafc"),
        bordercolor = "rgba(255,255,255,0.12)",
    ),
    xaxis = dict(
        gridcolor     = "rgba(255,255,255,0.05)",
        linecolor     = "rgba(255,255,255,0.1)",
        tickfont      = dict(color="#94a3b8"),
        title_font    = dict(color="#64748b"),
        showgrid      = False,
    ),
    yaxis = dict(
        gridcolor     = "rgba(255,255,255,0.05)",
        linecolor     = "rgba(255,255,255,0.1)",
        tickfont      = dict(color="#94a3b8"),
        title_font    = dict(color="#64748b"),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def render_chart(chart_data: ChartData) -> None:
    """
    Render chart_data as a Plotly figure in the current Streamlit context.
    Shows a provenance badge (LLM-extracted vs XBRL-verified).
    Swallows all exceptions to protect the parent chat message.
    """
    try:
        _render_provenance_badge(chart_data)
        fig = _build_figure(chart_data)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:
        logger.warning("[visualization] render_chart failed: %s", exc)
        st.caption("⚠️ Chart could not be rendered for this response.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure(chart_data: ChartData) -> go.Figure:
    builders = {
        ChartType.BAR:  _build_bar,
        ChartType.LINE: _build_line,
        ChartType.AREA: _build_area,
        ChartType.PIE:  _build_pie,
    }
    builder = builders.get(chart_data.type, _build_bar)
    fig = builder(chart_data)
    _apply_layout(fig, chart_data)
    return fig


def _build_bar(chart_data: ChartData) -> go.Figure:
    fig = go.Figure()
    for i, series in enumerate(chart_data.series):
        color = series.color or _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Bar(
            name        = series.name,
            x           = chart_data.x,
            y           = series.values,
            marker_color= color,
            marker_line = dict(color="rgba(0,0,0,0)", width=0),
            hovertemplate = (
                f"<b>%{{x}}</b><br>{series.name}: "
                f"{chart_data.value_suffix}%{{y:,.2f}}<extra></extra>"
            ),
        ))
    # Group bars side-by-side when there are multiple series
    if len(chart_data.series) > 1:
        fig.update_layout(barmode="group")
    return fig


def _build_line(chart_data: ChartData) -> go.Figure:
    fig = go.Figure()
    for i, series in enumerate(chart_data.series):
        color = series.color or _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            name        = series.name,
            x           = chart_data.x,
            y           = series.values,
            mode        = "lines+markers",
            line        = dict(color=color, width=2),
            marker      = dict(color=color, size=6),
            hovertemplate = (
                f"<b>%{{x}}</b><br>{series.name}: "
                f"{chart_data.value_suffix}%{{y:,.2f}}<extra></extra>"
            ),
        ))
    return fig


def _build_area(chart_data: ChartData) -> go.Figure:
    fig = go.Figure()
    for i, series in enumerate(chart_data.series):
        color = series.color or _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            name        = series.name,
            x           = chart_data.x,
            y           = series.values,
            mode        = "lines",
            fill        = "tozeroy",
            line        = dict(color=color, width=2),
            fillcolor   = color.replace(")", ", 0.15)").replace("rgb", "rgba")
                          if color.startswith("rgb") else color + "26",
            hovertemplate = (
                f"<b>%{{x}}</b><br>{series.name}: "
                f"{chart_data.value_suffix}%{{y:,.2f}}<extra></extra>"
            ),
        ))
    return fig


def _build_pie(chart_data: ChartData) -> go.Figure:
    # For pie charts, use first series values as the segments
    series = chart_data.series[0]
    fig = go.Figure(go.Pie(
        labels          = chart_data.x,
        values          = series.values,
        marker          = dict(colors=_PALETTE[:len(chart_data.x)]),
        hovertemplate   = "<b>%{label}</b><br>%{percent}<extra></extra>",
        textfont        = dict(color="#f8fafc"),
        hole            = 0.35,   # donut style
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Layout and badge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_layout(fig: go.Figure, chart_data: ChartData) -> None:
    layout = copy.deepcopy(_LAYOUT_BASE)
    layout["title"] = dict(
        text      = chart_data.title,
        font      = dict(color="#e2e8f0", size=14, family="Inter, sans-serif"),
        x         = 0,
        xanchor   = "left",
        pad       = dict(l=4),
    )
    if chart_data.x_label:
        layout["xaxis"]["title"] = dict(text=chart_data.x_label, font=dict(color="#64748b"))
    if chart_data.y_label:
        layout["yaxis"]["title"] = dict(text=chart_data.y_label, font=dict(color="#64748b"))

    fig.update_layout(**layout)


def _render_provenance_badge(chart_data: ChartData) -> None:
    """
    Renders a small badge indicating whether chart values came from
    LLM text extraction or verified XBRL structured data.
    """
    if chart_data.xbrl_enriched:
        badge_html = (
            '<span style="'
            "display:inline-block;padding:0.15rem 0.5rem;"
            "background:rgba(16,185,129,0.12);color:#6ee7b7;"
            "border:1px solid rgba(16,185,129,0.25);border-radius:4px;"
            "font-size:0.7rem;font-weight:600;letter-spacing:0.04em;"
            "font-family:JetBrains Mono,monospace;"
            '">✓ XBRL VERIFIED</span>'
        )
    else:
        badge_html = (
            '<span style="'
            "display:inline-block;padding:0.15rem 0.5rem;"
            "background:rgba(59,130,246,0.1);color:#93c5fd;"
            "border:1px solid rgba(59,130,246,0.2);border-radius:4px;"
            "font-size:0.7rem;font-weight:600;letter-spacing:0.04em;"
            "font-family:JetBrains Mono,monospace;"
            '">◈ EXTRACTED FROM FILING</span>'
        )
    st.markdown(badge_html, unsafe_allow_html=True)
