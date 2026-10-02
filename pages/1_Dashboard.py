"""
pages/1_Dashboard.py
Financial metrics dashboard — charts trends from PostgreSQL XBRL data.
"""

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv

from auth.session import require_auth, render_sidebar_user

load_dotenv()



# ── Auth Gate ─────────────────────────────────────────────────────────────
current_user = require_auth()
render_sidebar_user(current_user, page_key="dashboard")

# ── Formal Dark Theme Styles ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.stApp { background-color: #0b0f19 !important; color: #f1f5f9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid rgba(255, 255, 255, 0.08) !important; }

[data-testid="stMetric"] {
    background: #151d2d !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
    padding: 0.75rem 0.95rem !important;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 600 !important;
    font-size: 1.35rem !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.dash-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.dash-sub {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.25rem;
}
.section-heading {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}
hr { border-color: rgba(255, 255, 255, 0.08) !important; margin: 1.25rem 0 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=10, r=10, t=40, b=10),
)

COLORS = {
    "revenue":           "#3b82f6",
    "gross_profit":      "#10b981",
    "operating_income":  "#f59e0b",
    "net_income":        "#f87171",
    "operating_cash_flow": "#818cf8",
    "eps_basic":         "#fbbf24",
    "long_term_debt":    "#ef4444",
    "total_assets":      "#60a5fa",
}


# ── DB connection ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        return None
    try:
        conn = psycopg2.connect(dsn)
        return conn
    except Exception:
        return None


def _ensure_clean_conn(conn):
    if conn and not conn.closed:
        if conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            conn.rollback()


@st.cache_data(ttl=60)
def get_tickers(_conn) -> list[str]:
    if _conn is None:
        return []
    try:
        _ensure_clean_conn(_conn)
        with _conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ticker FROM companies ORDER BY ticker")
            return [r[0] for r in cur.fetchall()]
    except Exception:
        try:
            if _conn and not _conn.closed:
                _conn.rollback()
        except Exception:
            pass
        return []


@st.cache_data(ttl=30)
def get_facts(_conn, ticker: str, fields: list[str]) -> pd.DataFrame:
    """
    Fetch deduplicated facts: one row per (canonical_field, period_end).
    Uses MAX(value) to handle duplicates from double-ingestion.
    """
    if _conn is None:
        return pd.DataFrame()
    query = """
        SELECT
            ff.canonical_field,
            MAX(ff.value)       AS value,
            ff.unit,
            fi.fiscal_period,
            ff.period_start,
            ff.period_end
        FROM financial_facts ff
        JOIN filings fi   ON fi.filing_id   = ff.filing_id
        JOIN companies c  ON c.company_id   = fi.company_id
        WHERE c.ticker = %s
          AND ff.canonical_field = ANY(%s)
          AND ff.period_end IS NOT NULL
          AND ff.value IS NOT NULL
        GROUP BY ff.canonical_field, ff.unit, fi.fiscal_period, ff.period_start, ff.period_end
        ORDER BY ff.period_end
    """
    try:
        _ensure_clean_conn(_conn)
        with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (ticker, fields))
            rows = cur.fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
        if not df.empty and "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
        return df
    except Exception:
        try:
            if _conn and not _conn.closed:
                _conn.rollback()
        except Exception:
            pass
        return pd.DataFrame()


def fmt_billions(v) -> str:
    if v is None:
        return "N/A"
    v = float(v)
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:.2f}"


def pct_change(a, b) -> str:
    if a is None or b is None:
        return "N/A"
    a, b = float(a), float(b)
    if a == 0:
        return "N/A"
    c = (b - a) / abs(a) * 100
    prefix = "+" if c > 0 else ""
    color = "#34d399" if c > 0 else "#f87171"
    return f'<span style="color:{color}">{prefix}{c:.1f}%</span>'


# ── Page Content ───────────────────────────────────────────────────────────────
conn = get_conn()

st.markdown('<div class="dash-header">Financial Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">Multi-year operational metrics synthesized from verified SEC XBRL records</div>', unsafe_allow_html=True)

if conn is None:
    st.error("PostgreSQL connection unavailable. Verify that POSTGRES_DSN is configured in .env and the database service is operational.")
    st.stop()

tickers = get_tickers(conn)
if not tickers:
    st.info("No corporate entities ingested yet. Use the main workspace to ingest an SEC report or XBRL dataset.")
    st.stop()

# ── Ticker selector ──
col_sel, col_spacer = st.columns([1.5, 3.5])
with col_sel:
    selected_ticker = st.selectbox("Select Reporting Entity", tickers, key="dash_ticker")

INCOME_FIELDS   = ["revenue", "gross_profit", "operating_income", "net_income"]
CASHFLOW_FIELDS = ["operating_cash_flow"]
EPS_FIELDS      = ["eps_basic"]
BALANCE_FIELDS  = ["total_assets", "long_term_debt"]
ALL_FIELDS      = list(set(INCOME_FIELDS + CASHFLOW_FIELDS + EPS_FIELDS + BALANCE_FIELDS))

df = get_facts(conn, selected_ticker, ALL_FIELDS)

if df.empty:
    st.warning(f"No reported facts located for symbol: {selected_ticker}.")
    st.stop()

# Deduplicate: keep longest period per fiscal_period per field
df["period_end"] = pd.to_datetime(df["period_end"])
df["period_start"] = pd.to_datetime(df["period_start"])
df["duration"] = (df["period_end"] - df["period_start"]).dt.days.fillna(0)
df = (
    df.sort_values("duration", ascending=False)
    .drop_duplicates(subset=["canonical_field", "fiscal_period"])
    .sort_values("period_end")
)

periods = df["fiscal_period"].unique().tolist()

st.divider()

# ── SECTION 1: Key Metrics ──
st.markdown('<div class="section-heading">Key Financial Indicators (Latest vs Prior Period)</div>', unsafe_allow_html=True)

kpi_fields = ["revenue", "net_income", "operating_income", "eps_basic"]
kpi_labels = {
    "revenue": "Revenue",
    "net_income": "Net Income",
    "operating_income": "Operating Income",
    "eps_basic": "Basic EPS"
}

kpi_cols = st.columns(len(kpi_fields))
for col, field in zip(kpi_cols, kpi_fields):
    sub = df[df["canonical_field"] == field].sort_values("period_end")
    if sub.empty:
        col.metric(kpi_labels[field], "N/A")
        continue
    latest = sub.iloc[-1]
    val = float(latest["value"])
    label = f"${val:,.2f}" if field == "eps_basic" else fmt_billions(val)
    delta = None
    if len(sub) >= 2:
        prev = float(sub.iloc[-2]["value"])
        pct = (val - prev) / abs(prev) * 100 if prev != 0 else 0
        delta = f"{pct:+.1f}% YoY"
    col.metric(kpi_labels[field], label, delta)

st.divider()

# ── SECTION 2: Income Statement Trends ──
st.markdown('<div class="section-heading">Income Statement Trends</div>', unsafe_allow_html=True)

income_df = df[df["canonical_field"].isin(INCOME_FIELDS)]
if not income_df.empty:
    fig = go.Figure()
    for field in INCOME_FIELDS:
        sub = income_df[income_df["canonical_field"] == field].sort_values("period_end")
        if sub.empty:
            continue
        label = field.replace("_", " ").title()
        fig.add_trace(go.Bar(
            x=sub["fiscal_period"],
            y=sub["value"] / 1e9,
            name=label,
            marker_color=COLORS.get(field, "#3b82f6"),
            opacity=0.88,
        ))
    fig.update_layout(
        **PLOTLY_THEME,
        barmode="group",
        title=dict(text="Revenue, Gross Profit, Operating Income, and Net Income (USD Billions)", font=dict(size=12, color="#94a3b8")),
        yaxis_title="USD Billions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 3: Margin & EPS Analysis ──
st.markdown('<div class="section-heading">Profit Margins and Per-Share Metrics</div>', unsafe_allow_html=True)
margin_cols = st.columns(2)

with margin_cols[0]:
    rev_df = df[df["canonical_field"] == "revenue"].set_index("fiscal_period")["value"]
    gp_df  = df[df["canonical_field"] == "gross_profit"].set_index("fiscal_period")["value"]
    ni_df  = df[df["canonical_field"] == "net_income"].set_index("fiscal_period")["value"]
    op_df  = df[df["canonical_field"] == "operating_income"].set_index("fiscal_period")["value"]

    common_idx = rev_df.index.intersection(gp_df.index).intersection(ni_df.index)
    if len(common_idx) > 0:
        gross_margin = (gp_df[common_idx] / rev_df[common_idx] * 100).round(2)
        net_margin   = (ni_df[common_idx] / rev_df[common_idx] * 100).round(2)
        op_margin    = (op_df.reindex(common_idx) / rev_df[common_idx] * 100).round(2)

        fig2 = go.Figure()
        for series, name, color in [
            (gross_margin, "Gross Margin", "#10b981"),
            (op_margin,    "Operating Margin", "#f59e0b"),
            (net_margin,   "Net Margin", "#f87171"),
        ]:
            fig2.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines+markers+text",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                text=[f"{v:.1f}%" for v in series.values],
                textposition="top center",
                textfont=dict(size=9, color=color),
            ))
        fig2.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Operating and Profit Margins (%)", font=dict(size=12, color="#94a3b8")),
            yaxis_title="Percent",
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

with margin_cols[1]:
    eps_sub = df[df["canonical_field"] == "eps_basic"].sort_values("period_end")
    if not eps_sub.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=eps_sub["fiscal_period"],
            y=eps_sub["value"],
            mode="lines+markers+text",
            line=dict(color="#fbbf24", width=2),
            marker=dict(size=7, color="#fbbf24"),
            fill="tozeroy",
            fillcolor="rgba(251,191,36,0.06)",
            text=[f"${v:.2f}" for v in eps_sub["value"]],
            textposition="top center",
            textfont=dict(size=9, color="#fbbf24"),
            name="Basic EPS",
        ))
        fig3.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Basic Earnings Per Share (USD)", font=dict(size=12, color="#94a3b8")),
            yaxis_title="USD",
            yaxis_tickprefix="$",
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── SECTION 4: Cash Flow & Balance Sheet ──
st.markdown('<div class="section-heading">Cash Flow and Balance Sheet Structure</div>', unsafe_allow_html=True)
bs_cols = st.columns(2)

with bs_cols[0]:
    cf_sub = df[df["canonical_field"] == "operating_cash_flow"].sort_values("period_end")
    ni_sub = df[df["canonical_field"] == "net_income"].sort_values("period_end")
    if not cf_sub.empty:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=cf_sub["fiscal_period"], y=cf_sub["value"] / 1e9,
            name="Operating Cash Flow", marker_color="#818cf8", opacity=0.85,
        ))
        if not ni_sub.empty:
            common = cf_sub["fiscal_period"].values
            ni_match = ni_sub[ni_sub["fiscal_period"].isin(common)]
            fig4.add_trace(go.Scatter(
                x=ni_match["fiscal_period"], y=ni_match["value"] / 1e9,
                mode="lines+markers", name="Net Income",
                line=dict(color="#f87171", width=1.8), marker=dict(size=6),
            ))
        fig4.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Operating Cash Flow vs Net Income (USD Billions)", font=dict(size=12, color="#94a3b8")),
            yaxis_title="USD Billions", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig4, use_container_width=True)

with bs_cols[1]:
    assets_sub = df[df["canonical_field"] == "total_assets"].sort_values("period_end")
    debt_sub   = df[df["canonical_field"] == "long_term_debt"].sort_values("period_end")
    if not assets_sub.empty or not debt_sub.empty:
        fig5 = go.Figure()
        if not assets_sub.empty:
            fig5.add_trace(go.Bar(
                x=assets_sub["fiscal_period"], y=assets_sub["value"] / 1e9,
                name="Total Assets", marker_color="#60a5fa", opacity=0.85,
            ))
        if not debt_sub.empty:
            fig5.add_trace(go.Bar(
                x=debt_sub["fiscal_period"], y=debt_sub["value"] / 1e9,
                name="Long-Term Debt", marker_color="#ef4444", opacity=0.85,
            ))
        fig5.update_layout(
            **PLOTLY_THEME,
            barmode="group",
            title=dict(text="Total Assets vs Long-Term Debt (USD Billions)", font=dict(size=12, color="#94a3b8")),
            yaxis_title="USD Billions", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── SECTION 5: Raw Data Table ──
with st.expander("Structured Financial Facts Table (PostgreSQL Record Audit)"):
    display = df[["canonical_field", "fiscal_period", "value", "unit", "period_end"]].copy()
    display["value_fmt"] = display.apply(
        lambda r: f"${float(r['value']):,.2f}" if r["canonical_field"] == "eps_basic"
        else fmt_billions(r["value"]), axis=1
    )
    display = display.rename(columns={
        "canonical_field": "Field",
        "fiscal_period": "Period",
        "value_fmt": "Value",
        "unit": "Unit",
        "period_end": "Period End",
    })
    st.dataframe(
        display[["Field", "Period", "Value", "Unit", "Period End"]],
        use_container_width=True,
        hide_index=True,
    )
