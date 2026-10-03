"""
pages/2_🧪_Evaluation.py
=========================
Interactive Evaluation Dashboard for the 50-question, 5-dimension RAG evaluation.

Accuracy dimensions displayed:
  1. Retrieval Accuracy    — right chunks retrieved?
  2. Numerical Accuracy    — numbers within tolerance?
  3. Calculation Accuracy  — derived metrics correct?
  4. Groundedness Accuracy — claims grounded in context?
  5. Abstention Accuracy   — correctly refused when needed?
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

# ── Design System ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #080d16 !important; color: #f1f5f9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important; padding: 1rem !important;
    backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 700 !important; font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, rgba(96,165,250,0.12), rgba(129,140,248,0.12), rgba(167,139,250,0.12));
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 18px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.page-sub { font-size: 0.88rem; color: #64748b; margin-top: 0.25rem; }

/* Dimension score cards */
.dim-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1rem 1.25rem;
    text-align: center;
}
.dim-card-label { font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }
.dim-card-pct { font-size: 1.9rem; font-weight: 800; }
.dim-card-sub { font-size: 0.7rem; color: #475569; margin-top: 0.2rem; }

/* Accuracy colour scale */
.acc-high  { color: #34d399; }
.acc-med   { color: #fbbf24; }
.acc-low   { color: #f87171; }

/* Pills */
.pill { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        font-size:0.7rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.pill-pass   { background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); }
.pill-fail   { background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.3); }
.pill-blue   { background:rgba(99,179,237,0.15); color:#63b3ed; border:1px solid rgba(99,179,237,0.3); }
.pill-purple { background:rgba(167,139,250,0.15);color:#a78bfa; border:1px solid rgba(167,139,250,0.3); }
.pill-amber  { background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }

/* Progress bar */
.prog-wrap { background:rgba(255,255,255,0.06); border-radius:999px; height:8px; overflow:hidden; margin-top:0.3rem; }
.prog-fill  { height:8px; border-radius:999px; transition: width 0.4s ease; }

hr { border-color:rgba(255,255,255,0.07) !important; }

/* Expander tweaks */
[data-testid="stExpander"] {
    background:rgba(15,23,42,0.6) !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:12px !important; margin-bottom:0.5rem;
}
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

DIMENSIONS = [
    ("retrieval_accuracy",    "Retrieval",    "#60a5fa"),
    ("numerical_accuracy",    "Numerical",    "#34d399"),
    ("calculation_accuracy",  "Calculation",  "#a78bfa"),
    ("groundedness_accuracy", "Groundedness", "#f59e0b"),
    ("abstention_accuracy",   "Abstention",   "#f472b6"),
]

CATEGORY_LABELS = {
    "direct_fact_retrieval":     "Direct Fact Retrieval",
    "yoy_comparison":            "Year-over-Year Comparison",
    "multi_year_trend":          "Multi-Year Trend",
    "cross_company_comparison":  "Cross-Company Comparison",
    "financial_ratio":           "Financial Ratio / Derived",
    "semantic_retrieval":        "Semantic / Paraphrase",
    "time_period_understanding": "Time-Period Understanding",
    "multi_hop_reasoning":       "Multi-Hop Reasoning",
    "hallucination_unavailable": "Hallucination / Unavailable",
    "adversarial_hard":          "Hard / Adversarial",
}


def acc_color(pct_val: float) -> str:
    if pct_val >= 75:
        return "acc-high"
    if pct_val >= 50:
        return "acc-med"
    return "acc-low"


def pct(lst):
    valid = [v for v in lst if v is not None]
    if not valid:
        return 0.0
    return round(100 * sum(1 for v in valid if v is True) / len(valid), 1)


# ── Page header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">🧪 RAG Evaluation Dashboard</div>
    <div class="page-sub">
        50-question test suite &nbsp;·&nbsp;
        5 accuracy dimensions: Retrieval · Numerical · Calculation · Groundedness · Abstention
    </div>
</div>
""", unsafe_allow_html=True)

# ── File selector ─────────────────────────────────────────────────────────────
if not RESULTS_DIR.exists():
    st.info("No evaluation runs found. Run `python eval/run_eval_50.py` to generate results.")
    st.stop()

# Support both old eval_*.json and new eval50_*.json files
result_files = sorted(list(RESULTS_DIR.glob("eval*.json")), reverse=True)
if not result_files:
    st.info("No evaluation JSON files found in `eval/results/`.")
    st.stop()

col_sel, col_meta = st.columns([2, 3])
with col_sel:
    selected_file = st.selectbox(
        "📂 Select Evaluation Run",
        result_files,
        format_func=lambda p: f"{p.name}  ({p.stat().st_size // 1024} KB)",
    )

with open(selected_file, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

results    = eval_data.get("results", [])
summary    = eval_data.get("summary", {})
timestamp  = eval_data.get("run_timestamp", "Unknown")
model_name = eval_data.get("model", "Unknown")

with col_meta:
    st.caption(
        f"🕐 **Run:** `{timestamp}`  &nbsp;|&nbsp;  "
        f"🤖 **Model:** `{model_name}`  &nbsp;|&nbsp;  "
        f"📝 **Cases:** `{len(results)}`"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OVERALL KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Overall Performance")

overall_pass_rate = pct([r.get("overall_pass") for r in results])
halluc_count      = sum(1 for r in results if r.get("has_hallucination") is True)
abstention_needed = [r for r in results if r.get("abstention_expected")]
abstention_ok     = sum(1 for r in abstention_needed if r.get("abstention_accuracy") is True)
lats              = [r["latency_ms"] for r in results if r.get("latency_ms")]
avg_lat           = f"{sum(lats)/len(lats)/1000:.2f}s" if lats else "N/A"

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Overall Pass Rate", f"{overall_pass_rate}%",
              delta=f"{sum(1 for r in results if r.get('overall_pass'))} / {len(results)} passed")
with k2:
    st.metric("Hallucinations",    f"{halluc_count}",
              delta=f"out of {len(results)} questions", delta_color="inverse")
with k3:
    st.metric("Abstention Rate",   f"{abstention_ok}/{len(abstention_needed)} correct",
              delta="needed / got right")
with k4:
    st.metric("Avg Latency",       avg_lat)
with k5:
    num_err = [r["relative_error_pct"] for r in results if r.get("relative_error_pct") is not None]
    avg_err = f"{sum(num_err)/len(num_err):.2f}%" if num_err else "N/A"
    st.metric("Avg Numerical Error", avg_err)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — 5 DIMENSION ACCURACY CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🎯 Accuracy by Dimension")
dim_cols = st.columns(5)
for col, (dim_key, dim_label, color) in zip(dim_cols, DIMENSIONS):
    applicable = [r.get(dim_key) for r in results if r.get(dim_key) is not None]
    passed     = sum(1 for v in applicable if v is True)
    acc_pct    = round(100 * passed / len(applicable), 1) if applicable else 0.0
    bar_width  = int(acc_pct)
    css_cls    = acc_color(acc_pct)

    with col:
        st.markdown(f"""
        <div class="dim-card">
            <div class="dim-card-label">{dim_label}</div>
            <div class="dim-card-pct {css_cls}">{acc_pct}%</div>
            <div class="dim-card-sub">{passed} / {len(applicable)} passed</div>
            <div class="prog-wrap">
                <div class="prog-fill" style="width:{bar_width}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RADAR / SPIDER CHART (5 dimensions)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🕸️ Accuracy Radar")

categories_radar = [dl for _, dl, _ in DIMENSIONS]
values_radar = []
for dim_key, _, _ in DIMENSIONS:
    applicable = [r.get(dim_key) for r in results if r.get(dim_key) is not None]
    values_radar.append(round(100 * sum(1 for v in applicable if v is True) / len(applicable), 1) if applicable else 0.0)

values_radar_closed = values_radar + [values_radar[0]]
categories_radar_closed = categories_radar + [categories_radar[0]]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=values_radar_closed,
    theta=categories_radar_closed,
    fill="toself",
    fillcolor="rgba(96,165,250,0.15)",
    line=dict(color="#60a5fa", width=2),
    name="Accuracy %",
    hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#94a3b8", size=10),
                        gridcolor="rgba(255,255,255,0.07)"),
        angularaxis=dict(tickfont=dict(color="#e2e8f0", size=12), gridcolor="rgba(255,255,255,0.07)"),
        bgcolor="rgba(0,0,0,0)",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    height=350, margin=dict(l=50, r=50, t=30, b=30),
    showlegend=False,
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HEATMAP: Category × Dimension
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Accuracy Heatmap — Category × Dimension")

cats_in_data = list(dict.fromkeys(r["category"] for r in results))
heatmap_rows = []
for cat in cats_in_data:
    cat_results = [r for r in results if r["category"] == cat]
    row = {"Category": CATEGORY_LABELS.get(cat, cat)}
    for dim_key, dim_label, _ in DIMENSIONS:
        applicable = [r.get(dim_key) for r in cat_results if r.get(dim_key) is not None]
        row[dim_label] = round(100 * sum(1 for v in applicable if v is True) / len(applicable), 1) if applicable else None
    heatmap_rows.append(row)

df_heat = pd.DataFrame(heatmap_rows).set_index("Category")
dim_labels_list = [dl for _, dl, _ in DIMENSIONS]

fig_heat = px.imshow(
    df_heat[dim_labels_list].astype(float),
    text_auto=".0f",
    color_continuous_scale=[[0, "#1e1b4b"], [0.5, "#c2410c"], [1, "#34d399"]],
    range_color=[0, 100],
    labels=dict(color="Accuracy %"),
    aspect="auto",
)
fig_heat.update_traces(textfont=dict(size=12, color="white"))
fig_heat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    height=400,
    margin=dict(l=10, r=10, t=20, b=10),
    coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
    xaxis=dict(side="top", tickfont=dict(size=11, color="#e2e8f0")),
    yaxis=dict(tickfont=dict(size=10, color="#94a3b8")),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CATEGORY PASS-RATE BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Pass Rate by Question Category")

bar_data = []
for cat in cats_in_data:
    cat_results = [r for r in results if r["category"] == cat]
    cat_pct     = pct([r.get("overall_pass") for r in cat_results])
    bar_data.append({"Category": CATEGORY_LABELS.get(cat, cat), "Pass Rate (%)": cat_pct, "N": len(cat_results)})

df_bar = pd.DataFrame(bar_data).sort_values("Pass Rate (%)", ascending=True)
fig_bar = px.bar(
    df_bar, x="Pass Rate (%)", y="Category", orientation="h",
    color="Pass Rate (%)",
    color_continuous_scale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#34d399"]],
    range_color=[0, 100], text="Pass Rate (%)",
    hover_data={"N": True},
)
fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      textfont=dict(color="#e2e8f0", size=11))
fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    height=400, showlegend=False,
    margin=dict(l=10, r=50, t=20, b=10),
    xaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#e2e8f0", size=11)),
    coloraxis_showscale=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — HALLUCINATION & ABSTENTION DETAILS
# ─────────────────────────────────────────────────────────────────────────────
col_abst, col_halluc = st.columns(2)

with col_abst:
    st.markdown("#### 🚫 Abstention Analysis")
    abst_needed  = [r for r in results if r.get("abstention_expected")]
    abst_answer  = [r for r in results if not r.get("abstention_expected")]
    abst_correct = sum(1 for r in abst_needed if r.get("abstention_accuracy") is True)
    false_abst   = sum(1 for r in abst_answer  if r.get("abstention_accuracy") is False)

    fig_abst = go.Figure(data=[go.Pie(
        labels=["Correctly abstained", "Failed to abstain", "False abstentions (over-refusal)", "Correctly answered"],
        values=[
            abst_correct,
            len(abst_needed) - abst_correct,
            false_abst,
            len(abst_answer) - false_abst,
        ],
        hole=0.55,
        marker=dict(colors=["#34d399", "#f87171", "#fbbf24", "#60a5fa"]),
        textfont=dict(color="#e2e8f0", size=11),
    )])
    fig_abst.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
        height=270, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_abst, use_container_width=True)

with col_halluc:
    st.markdown("#### 🔍 Groundedness Scores Distribution")
    gs_vals = [r.get("groundedness_score") for r in results if r.get("groundedness_score") is not None]
    if gs_vals:
        gs_counts = {i: gs_vals.count(i) for i in range(1, 6)}
        fig_gs = px.bar(
            x=list(gs_counts.keys()), y=list(gs_counts.values()),
            labels={"x": "Score (1=fabricated, 5=fully grounded)", "y": "Count"},
            color=list(gs_counts.keys()),
            color_continuous_scale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#34d399"]],
            range_color=[1, 5],
        )
        fig_gs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            height=270, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(tickmode="linear", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_gs, use_container_width=True)
    else:
        st.info("No groundedness scores in this result file (legacy format).")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FILTERS + DETAILED CASE AUDIT TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Detailed Case Audit")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
with filter_col1:
    show_only = st.selectbox("Show", ["All", "PASS only", "FAIL only", "Abstention cases", "Hallucination cases"])
with filter_col2:
    cat_options = ["All"] + list(dict.fromkeys(r["category"] for r in results))
    cat_filter  = st.selectbox("Category", cat_options, format_func=lambda c: CATEGORY_LABELS.get(c, c))
with filter_col3:
    dim_options = ["All"] + [dl for _, dl, _ in DIMENSIONS]
    dim_filter  = st.selectbox("Dimension", dim_options)
with filter_col4:
    search_q = st.text_input("Search question", placeholder="keyword...")

# Apply filters
filtered = results
if show_only == "PASS only":
    filtered = [r for r in filtered if r.get("overall_pass")]
elif show_only == "FAIL only":
    filtered = [r for r in filtered if not r.get("overall_pass")]
elif show_only == "Abstention cases":
    filtered = [r for r in filtered if r.get("abstention_expected")]
elif show_only == "Hallucination cases":
    filtered = [r for r in filtered if r.get("has_hallucination") is True]

if cat_filter != "All":
    filtered = [r for r in filtered if r.get("category") == cat_filter]
if dim_filter != "All":
    dim_key = next((dk for dk, dl, _ in DIMENSIONS if dl == dim_filter), None)
    if dim_key:
        filtered = [r for r in filtered if r.get(dim_key) is not None]
if search_q:
    filtered = [r for r in filtered if search_q.lower() in r.get("question", "").lower()]

st.caption(f"Showing {len(filtered)} / {len(results)} cases")

for r in filtered:
    passed       = r.get("overall_pass", False)
    cat_label    = CATEGORY_LABELS.get(r.get("category", ""), r.get("category", ""))
    sub_cat      = r.get("sub_category", "")
    q_id         = r.get("id", "?")
    abstain_exp  = r.get("abstention_expected", False)

    # Build dim score pills for title
    dim_badges = ""
    for dim_key, dim_label, color in DIMENSIONS:
        v = r.get(dim_key)
        if v is not None:
            badge_class = "pill-pass" if v else "pill-fail"
            dim_badges += f'<span class="pill {badge_class}">{dim_label[:3].upper()}</span> '

    status_icon = "✅" if passed else "❌"
    expander_title = (
        f"{status_icon} [{q_id}] {cat_label} — {sub_cat.replace('_', ' ').title()}"
    )

    with st.expander(expander_title, expanded=not passed):
        # Header pills
        status_pill = (
            '<span class="pill pill-pass">PASS</span>' if passed
            else '<span class="pill pill-fail">FAIL</span>'
        )
        abst_pill = (
            '<span class="pill pill-amber">ABSTAIN EXPECTED</span>' if abstain_exp else ""
        )
        hall_pill = (
            '<span class="pill pill-fail">HALLUCINATION</span>'
            if r.get("has_hallucination") else ""
        )
        st.markdown(
            f"{status_pill} &nbsp; {dim_badges} &nbsp; {abst_pill} {hall_pill}",
            unsafe_allow_html=True
        )

        st.markdown(f"**Question:** {r.get('question')}")

        # Ground truth vs model answer
        c1, c2, c3 = st.columns(3)
        with c1:
            gt = r.get("ground_truth_value")
            if gt is not None and isinstance(gt, (int, float)):
                gt_fmt = f"{gt:,.2f}"
            else:
                gt_fmt = str(gt) if gt else r.get("ground_truth_formatted", "Qualitative")
            unit = r.get("ground_truth_unit", "")
            st.markdown(f"**Ground Truth:** `{gt_fmt} {unit}`")
            st.caption(r.get("ground_truth_formatted", ""))
        with c2:
            bm = r.get("best_match")
            bm_fmt = f"{bm:,.2f}" if isinstance(bm, (int, float)) else "N/A"
            st.markdown(f"**Extracted Number:** `{bm_fmt}`")
            err = r.get("relative_error_pct")
            if err is not None:
                st.markdown(f"Relative error: `{err:.2f}%`  (tol: `{r.get('tolerance_pct', 2)}%`)")
        with c3:
            gs  = r.get("groundedness_score", "—")
            fs  = r.get("faithfulness_score",  "—")
            rs  = r.get("relevance_score",     "—")
            st.markdown(f"**Judge Scores:** G={gs}/5 · F={fs}/5 · R={rs}/5")
            st.caption(r.get("judge_reasoning", ""))

        # 5-Dimension grid
        st.markdown("**Accuracy Dimensions:**")
        dim_grid = st.columns(5)
        for col_d, (dk, dl, color) in zip(dim_grid, DIMENSIONS):
            v = r.get(dk)
            with col_d:
                if v is True:
                    st.markdown(f"<div style='text-align:center;color:#34d399;font-size:1.2rem;'>✔</div><div style='text-align:center;font-size:0.7rem;color:#64748b;'>{dl}</div>", unsafe_allow_html=True)
                elif v is False:
                    st.markdown(f"<div style='text-align:center;color:#f87171;font-size:1.2rem;'>✘</div><div style='text-align:center;font-size:0.7rem;color:#64748b;'>{dl}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center;color:#475569;font-size:1.2rem;'>–</div><div style='text-align:center;font-size:0.7rem;color:#475569;'>{dl}</div>", unsafe_allow_html=True)

        # Model answer
        st.markdown("**Model Answer:**")
        st.caption(r.get("model_answer", "No answer recorded."))

        # Retrieved sources info
        n_src = r.get("sources_returned", 0)
        avg_s = r.get("avg_similarity", 0)
        lat   = r.get("latency_ms", 0)
        st.markdown(
            f"<small style='color:#475569;'>Sources: {n_src} &nbsp;|&nbsp; "
            f"Avg similarity: {avg_s:.3f} &nbsp;|&nbsp; Latency: {lat}ms</small>",
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — NUMERICAL ERROR SCATTER
# ─────────────────────────────────────────────────────────────────────────────
num_results = [r for r in results if r.get("relative_error_pct") is not None]
if num_results:
    st.markdown("### 📉 Numerical Error Distribution")
    scatter_data = [
        {
            "ID": r["id"],
            "Category": CATEGORY_LABELS.get(r["category"], r["category"]),
            "Relative Error (%)": r["relative_error_pct"],
            "Status": "PASS" if r.get("overall_pass") else "FAIL",
            "Question": r["question"][:60] + "...",
        }
        for r in num_results
    ]
    df_scatter = pd.DataFrame(scatter_data)
    fig_sc = px.scatter(
        df_scatter, x="ID", y="Relative Error (%)",
        color="Status", color_discrete_map={"PASS": "#34d399", "FAIL": "#f87171"},
        symbol="Category", hover_data=["Question", "Category"],
        title="Relative Numerical Error per Question (lower = better)",
    )
    fig_sc.add_hline(y=2, line_dash="dot", line_color="#fbbf24",
                     annotation_text="2% tolerance", annotation_font_color="#fbbf24")
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter", color="#94a3b8"), height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RAW DATA TABLE (downloadable)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📄 Full Results Table (all columns)", expanded=False):
    df_full = pd.DataFrame(results)
    cols_show = [
        "id", "category", "question", "overall_pass",
        "retrieval_accuracy", "numerical_accuracy", "calculation_accuracy",
        "groundedness_accuracy", "abstention_accuracy",
        "groundedness_score", "faithfulness_score", "relevance_score",
        "has_hallucination", "abstention_expected",
        "relative_error_pct", "best_match", "sources_returned",
        "avg_similarity", "latency_ms",
    ]
    cols_show = [c for c in cols_show if c in df_full.columns]
    st.dataframe(df_full[cols_show], use_container_width=True)
    csv = df_full[cols_show].to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "eval_results.csv", "text/csv")

st.caption(
    "FinSight Evaluation Dashboard · 50 Questions · 5 Accuracy Dimensions · "
    "LLM-as-Judge via Groq"
)
