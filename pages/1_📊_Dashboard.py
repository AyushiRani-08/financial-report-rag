"""
pages/1_📊_Dashboard.py
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

load_dotenv()

st.set_page_config(
    page_title="FinSight — Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Dark theme styles (matches main app) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0b0f19 !important; color: #f1f5f9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid rgba(255, 255, 255, 0.08) !important; }
[data-testid="stMetric"] { background: #161f30 !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 12px !important; padding: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 700 !important; }
.dash-header { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #818cf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.25rem; }
.dash-sub { font-size: 0.9rem; color: #94a3b8; margin-bottom: 1.5rem; }
hr { border-color: rgba(255, 255, 255, 0.08) !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=10, r=10, t=40, b=10),
)

COLORS = {
    "revenue":           "#63b3ed",
    "gross_profit":      "#68d391",
    "operating_income":  "#f6e05e",
    "net_income":        "#fc8181",
    "operating_cash_flow": "#b794f4",
    "eps_basic":         "#fbd38d",
    "long_term_debt":    "#fc8181",
    "total_assets":      "#63b3ed",
}


# ── DB connection ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        return None
    try:
        return psycopg2.connect(dsn)
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_tickers(_conn) -> list[str]:
    if _conn is None:
        return []
    try:
        with _conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ticker FROM companies ORDER BY ticker")
            return [r[0] for r in cur.fetchall()]
    except Exception:
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
        with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (ticker, fields))
            rows = cur.fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
        if not df.empty and "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
        return df
    except Exception:
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
    arrow = "▲" if c > 0 else "▼"
    color = "green" if c > 0 else "red"
    return f'<span style="color:{color}">{arrow} {abs(c):.1f}%</span>'


# ── Page ───────────────────────────────────────────────────────────────────────
conn = get_conn()

st.markdown('<div class="dash-header">📊 Financial Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">Multi-year trend analysis from verified XBRL data in PostgreSQL</div>', unsafe_allow_html=True)

if conn is None:
    st.error("Cannot connect to PostgreSQL. Make sure `POSTGRES_DSN` is set in `.env` and Docker container is running.")
    st.stop()

tickers = get_tickers(conn)
if not tickers:
    st.info("No companies ingested yet. Go to the main app and ingest an XBRL file first.")
    st.stop()

# ── Ticker selector ──
col_sel, col_spacer = st.columns([1, 3])
with col_sel:
    selected_ticker = st.selectbox("Select Company", tickers, key="dash_ticker")

INCOME_FIELDS   = ["revenue", "gross_profit", "operating_income", "net_income"]
CASHFLOW_FIELDS = ["operating_cash_flow"]
EPS_FIELDS      = ["eps_basic"]
BALANCE_FIELDS  = ["total_assets", "long_term_debt"]
ALL_FIELDS      = list(set(INCOME_FIELDS + CASHFLOW_FIELDS + EPS_FIELDS + BALANCE_FIELDS))

df = get_facts(conn, selected_ticker, ALL_FIELDS)

if df.empty:
    st.warning(f"No facts found for `{selected_ticker}`. Check ingestion.")
    st.stop()

# ── Deduplicate: keep longest period (annual) per fiscal_period per field ──
df["period_end"] = pd.to_datetime(df["period_end"])
df["period_start"] = pd.to_datetime(df["period_start"])
df["duration"] = (df["period_end"] - df["period_start"]).dt.days.fillna(0)
# For each (canonical_field, fiscal_period), keep the row with longest duration
df = (
    df.sort_values("duration", ascending=False)
    .drop_duplicates(subset=["canonical_field", "fiscal_period"])
    .sort_values("period_end")
)

periods = df["fiscal_period"].unique().tolist()

st.divider()

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — KPI Cards (latest vs prev year)
# ═══════════════════════════════════════════════════════════════
st.markdown("### 📌 Key Metrics — Latest vs Prior Year")

kpi_fields = ["revenue", "net_income", "operating_income", "eps_basic"]
kpi_labels = {"revenue": "Revenue", "net_income": "Net Income",
               "operating_income": "Operating Income", "eps_basic": "EPS (Basic)"}

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

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — Income Statement Trend
# ═══════════════════════════════════════════════════════════════
st.markdown("### 📈 Income Statement Trends")

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
            marker_color=COLORS.get(field, "#63b3ed"),
            opacity=0.88,
        ))
    fig.update_layout(
        **PLOTLY_THEME,
        barmode="group",
        title=dict(text="Revenue, Gross Profit, Operating Income, Net Income (USD Billions)", font=dict(size=13, color="#94a3b8")),
        yaxis_title="USD Billions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — Margin Analysis
# ═══════════════════════════════════════════════════════════════
st.markdown("### 📉 Profit Margin Trends")

margin_cols = st.columns(2)

# Gross margin
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
            (gross_margin, "Gross Margin", "#68d391"),
            (op_margin,    "Operating Margin", "#f6e05e"),
            (net_margin,   "Net Margin", "#fc8181"),
        ]:
            fig2.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines+markers+text",
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
                text=[f"{v:.1f}%" for v in series.values],
                textposition="top center",
                textfont=dict(size=10, color=color),
            ))
        fig2.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Margin % Over Time", font=dict(size=13, color="#94a3b8")),
            yaxis_title="%",
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

# EPS trend
with margin_cols[1]:
    eps_sub = df[df["canonical_field"] == "eps_basic"].sort_values("period_end")
    if not eps_sub.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=eps_sub["fiscal_period"],
            y=eps_sub["value"],
            mode="lines+markers+text",
            line=dict(color="#fbd38d", width=2.5),
            marker=dict(size=9, color="#fbd38d"),
            fill="tozeroy",
            fillcolor="rgba(251,211,141,0.08)",
            text=[f"${v:.2f}" for v in eps_sub["value"]],
            textposition="top center",
            textfont=dict(size=10, color="#fbd38d"),
            name="EPS Basic",
        ))
        fig3.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Earnings Per Share (Basic)", font=dict(size=13, color="#94a3b8")),
            yaxis_title="USD",
            yaxis_tickprefix="$",
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — Cash Flow & Balance Sheet
# ═══════════════════════════════════════════════════════════════
st.markdown("### 💰 Cash Flow & Balance Sheet")

bs_cols = st.columns(2)

with bs_cols[0]:
    cf_sub = df[df["canonical_field"] == "operating_cash_flow"].sort_values("period_end")
    ni_sub = df[df["canonical_field"] == "net_income"].sort_values("period_end")
    if not cf_sub.empty:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=cf_sub["fiscal_period"], y=cf_sub["value"] / 1e9,
            name="Operating Cash Flow", marker_color="#b794f4", opacity=0.85,
        ))
        if not ni_sub.empty:
            common = cf_sub["fiscal_period"].values
            ni_match = ni_sub[ni_sub["fiscal_period"].isin(common)]
            fig4.add_trace(go.Scatter(
                x=ni_match["fiscal_period"], y=ni_match["value"] / 1e9,
                mode="lines+markers", name="Net Income",
                line=dict(color="#fc8181", width=2), marker=dict(size=7),
            ))
        fig4.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Operating Cash Flow vs Net Income (USD B)", font=dict(size=13, color="#94a3b8")),
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
                name="Total Assets", marker_color="#63b3ed", opacity=0.85,
            ))
        if not debt_sub.empty:
            fig5.add_trace(go.Bar(
                x=debt_sub["fiscal_period"], y=debt_sub["value"] / 1e9,
                name="Long-Term Debt", marker_color="#fc8181", opacity=0.85,
            ))
        fig5.update_layout(
            **PLOTLY_THEME,
            barmode="group",
            title=dict(text="Total Assets vs Long-Term Debt (USD B)", font=dict(size=13, color="#94a3b8")),
            yaxis_title="USD Billions", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — Raw Data Table
# ═══════════════════════════════════════════════════════════════
with st.expander("🔢 View Raw Financial Facts from PostgreSQL"):
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
