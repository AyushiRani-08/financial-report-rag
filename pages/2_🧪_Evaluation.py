"""
pages/2_🧪_Evaluation.py
Interactive Evaluation Dashboard — inspect evaluation results, metrics, and audit logs.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="FinSight — Evaluation Audit",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0b0f19 !important; color: #f1f5f9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid rgba(255, 255, 255, 0.08) !important; }
[data-testid="stMetric"] { background: #161f30 !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 12px !important; padding: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 700 !important; }
.dash-header { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #818cf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.25rem; }
.dash-sub { font-size: 0.9rem; color: #94a3b8; margin-bottom: 1.5rem; }
.pill { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pill-pass { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.pill-fail { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.pill-blue { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
hr { border-color: rgba(255, 255, 255, 0.08) !important; }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

st.markdown('<div class="dash-header">🧪 Evaluation & Accuracy Audit</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">Verification of Ground Truth facts, numerical precision, and LLM-as-a-Judge telemetry</div>', unsafe_allow_html=True)

if not RESULTS_DIR.exists():
    st.info("No evaluation runs found yet. Run `python eval/run_eval.py` to generate an evaluation audit.")
    st.stop()

result_files = sorted(list(RESULTS_DIR.glob("eval_*.json")), reverse=True)

if not result_files:
    st.info("No evaluation JSON files found in `eval/results/`.")
    st.stop()

# Select evaluation run
col_file, col_space = st.columns([2, 2])
with col_file:
    selected_file = st.selectbox(
        "Select Evaluation Run",
        result_files,
        format_func=lambda p: f"{p.name} ({p.stat().st_size} bytes)",
    )

with open(selected_file, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

summary = eval_data.get("summary", {})
results = eval_data.get("results", [])
timestamp = eval_data.get("run_timestamp", "Unknown")

st.caption(f"Run Timestamp: `{timestamp}` | Total Test Cases: `{len(results)}`")

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
quant_results = [r for r in results if r.get("mode") == "quantitative"]
qual_results  = [r for r in results if r.get("mode") == "qualitative"]

with col1:
    q_pass_rate = summary.get("quantitative_pass_rate", 0)
    st.metric("Quantitative Pass Rate", f"{q_pass_rate}%")

with col2:
    q_passed = sum(1 for r in quant_results if r.get("passed"))
    st.metric("Facts Verified", f"{q_passed} / {len(quant_results)}")

with col3:
    errors = [r["relative_error_pct"] for r in quant_results if r.get("relative_error_pct") is not None]
    avg_err = f"{sum(errors)/len(errors):.2f}%" if errors else "0.0%"
    st.metric("Avg Numerical Error", avg_err)

with col4:
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    avg_lat = f"{sum(latencies)/len(latencies)/1000:.1f}s" if latencies else "N/A"
    st.metric("Avg Latency", avg_lat)

st.divider()

# Error Distribution Bar Chart
if quant_results:
    st.subheader("📊 Metric Precision Breakdown")
    plot_data = []
    for r in quant_results:
        plot_data.append({
            "Metric": r.get("canonical_field", "").replace("_", " ").title(),
            "Relative Error (%)": r.get("relative_error_pct", 0) if r.get("relative_error_pct") is not None else 100,
            "Status": "PASS" if r.get("passed") else "FAIL",
        })
    df_plot = pd.DataFrame(plot_data)

    fig = px.bar(
        df_plot,
        x="Metric",
        y="Relative Error (%)",
        color="Status",
        color_discrete_map={"PASS": "#34d399", "FAIL": "#f87171"},
        title="Relative Numerical Error vs Ground Truth (Lower is Better)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Detailed Audit Table
st.subheader("🔍 Detailed Test Case Audit")

for i, r in enumerate(results, 1):
    passed = r.get("passed", False)
    status_pill = (
        '<span class="pill pill-pass">PASS</span>'
        if passed else
        '<span class="pill pill-fail">FAIL</span>'
    )
    field_pill = f'<span class="pill pill-blue">{r.get("canonical_field", r.get("category", "General"))}</span>'
    period_pill = f'<span class="pill pill-blue">{r.get("fiscal_period", "")}</span>'

    expander_title = f"{'✅' if passed else '❌'} Case #{i}: {r.get('canonical_field', '').replace('_', ' ').title()} ({r.get('ticker', '')} {r.get('fiscal_period', '')})"
    
    with st.expander(expander_title, expanded=not passed):
        st.markdown(f"{status_pill} &nbsp; {field_pill} &nbsp; {period_pill}", unsafe_allow_html=True)
        st.markdown(f"**Question:** {r.get('question')}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            gt = r.get("ground_truth_value")
            gt_fmt = f"{gt:,.2f}" if isinstance(gt, (int, float)) else str(gt)
            st.markdown(f"**PostgreSQL Ground Truth:** `{gt_fmt} {r.get('ground_truth_unit', '')}`")
        with c2:
            bm = r.get("best_match")
            bm_fmt = f"{bm:,.2f}" if isinstance(bm, (int, float)) else str(bm)
            st.markdown(f"**Extracted AI Answer:** `{bm_fmt}`")
        with c3:
            err = r.get("relative_error_pct")
            err_fmt = f"{err:.2f}%" if err is not None else "N/A"
            st.markdown(f"**Relative Error:** `{err_fmt}` (Tol: `{r.get('tolerance_pct', 2)}%`)")
        
        st.markdown("**Full AI Model Response:**")
        st.caption(r.get("model_answer", "No answer recorded."))
