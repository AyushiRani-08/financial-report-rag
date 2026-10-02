"""
pages/2_Evaluation.py
Evaluation Dashboard — inspect automated evaluation results, precision metrics, and audit logs.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from auth.session import require_auth, render_sidebar_user

st.set_page_config(
    page_title="FinSight — Evaluation and Accuracy Audit",
    layout="wide",
)

# ── Auth Gate ─────────────────────────────────────────────────────────────
current_user = require_auth()
render_sidebar_user(current_user, page_key="eval")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
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

.formal-tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.73rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}
.formal-tag-pass { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
.formal-tag-fail { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
.formal-tag-blue { background: rgba(37, 99, 235, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }

hr { border-color: rgba(255, 255, 255, 0.08) !important; margin: 1.25rem 0 !important; }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

st.markdown('<div class="dash-header">Evaluation and Accuracy Audit</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">Verification of ground truth financial facts, numerical precision, and automated evaluation telemetry</div>', unsafe_allow_html=True)

if not RESULTS_DIR.exists():
    st.info("No evaluation runs found. Execute `python eval/run_eval.py` to generate an evaluation benchmark run.")
    st.stop()

result_files = sorted(list(RESULTS_DIR.glob("eval_*.json")), reverse=True)

if not result_files:
    st.info("No evaluation JSON files located in eval/results/.")
    st.stop()

# Select evaluation run
col_file, col_space = st.columns([2, 2])
with col_file:
    selected_file = st.selectbox(
        "Select Evaluation Benchmark Run",
        result_files,
        format_func=lambda p: f"{p.name} ({p.stat().st_size:,} bytes)",
    )

with open(selected_file, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

summary = eval_data.get("summary", {})
results = eval_data.get("results", [])
timestamp = eval_data.get("run_timestamp", "Unknown")

st.caption(f"Benchmark Timestamp: {timestamp} | Total Test Cases Evaluated: {len(results)}")

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
    st.metric("Average Numerical Error", avg_err)

with col4:
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    avg_lat = f"{sum(latencies)/len(latencies)/1000:.1f}s" if latencies else "N/A"
    st.metric("Average Latency", avg_lat)

st.divider()

# Error Distribution Bar Chart
if quant_results:
    st.markdown('<div class="section-heading">Metric Precision Breakdown vs Ground Truth</div>', unsafe_allow_html=True)
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
        color_discrete_map={"PASS": "#10b981", "FAIL": "#ef4444"},
        title="Relative Numerical Error vs Ground Truth (Lower is Better)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Detailed Audit Table
st.markdown('<div class="section-heading">Detailed Case-by-Case Test Audit</div>', unsafe_allow_html=True)

for i, r in enumerate(results, 1):
    passed = r.get("passed", False)
    status_tag = (
        '<span class="formal-tag formal-tag-pass">PASS</span>'
        if passed else
        '<span class="formal-tag formal-tag-fail">FAIL</span>'
    )
    field_tag = f'<span class="formal-tag formal-tag-blue">{r.get("canonical_field", r.get("category", "General"))}</span>'
    period_tag = f'<span class="formal-tag formal-tag-blue">{r.get("fiscal_period", "")}</span>'

    expander_title = f"[{'PASS' if passed else 'FAIL'}] Case {i}: {r.get('canonical_field', '').replace('_', ' ').title()} ({r.get('ticker', '')} {r.get('fiscal_period', '')})"
    
    with st.expander(expander_title, expanded=not passed):
        st.markdown(f"{status_tag} &nbsp; {field_tag} &nbsp; {period_tag}", unsafe_allow_html=True)
        st.markdown(f"**Query:** {r.get('question')}")
        
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
            st.markdown(f"**Relative Error:** `{err_fmt}` (Tolerance: `{r.get('tolerance_pct', 2)}%`)")
        
        st.markdown("**Model Output:**")
        st.caption(r.get("model_answer", "No response recorded."))
