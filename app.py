# app.py
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import os

import sys
import importlib

import ingestion.sec_fetcher
import retrieval.retriever
import retrieval.xbrl_retriever
import generation.generator
import indexing.vector_store

importlib.reload(ingestion.sec_fetcher)
importlib.reload(retrieval.retriever)
importlib.reload(retrieval.xbrl_retriever)
importlib.reload(generation.generator)
importlib.reload(indexing.vector_store)

from generation.generator import RAGGenerator
from indexing.vector_store import VectorStore
from ingestion.sec_fetcher import download_sec_filing, fetch_and_store_xbrl
from privacy import redact_query, get_redaction_summary, get_engine_status

load_dotenv()

st.set_page_config(
    page_title="FinSight — Financial Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PREMIUM DARK UI STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global resets & typography ── */
* {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stBottom"] {
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8;
    font-size: 0.82rem;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}
.stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    background: #1e293b !important;
    color: #f1f5f9 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #60a5fa !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1.5px dashed rgba(99, 179, 237, 0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
    color: #94a3b8 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-1px);
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: #161f30 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    margin-bottom: 0.85rem !important;
    padding: 1rem 1.25rem !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #161f30 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-weight: 700 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}

/* ── Header ── */
.finsight-header {
    background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(99,179,237,0.08));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.finsight-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.finsight-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
}

/* ── Status pill ── */
.pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}
.pill-blue  { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.pill-green { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.pill-amber { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.4); }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# COMPONENT INIT
# ─────────────────────────────────────────────
@st.cache_resource
def get_components():
    store = VectorStore(persist_directory="data/chroma_db")
    generator = RAGGenerator(retriever=None, model_name="openai/gpt-oss-20b")
    return store, generator

vector_store, rag_generator = get_components()

if not hasattr(vector_store, "add_document") or not hasattr(rag_generator, "generate_standard_report"):
    st.cache_resource.clear()
    st.rerun()

@st.cache_data(ttl=5)  # Short TTL so new docs appear quickly
def get_indexed_documents(_collection):
    try:
        metas = _collection.get(include=["metadatas"])["metadatas"]
        return sorted(list({m["paper_id"] for m in metas if m and "paper_id" in m}))
    except Exception:
        return []


COMPANY_TICKER_MAP = {
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "jpmc": "JPM",
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "boeing": "BA",
    "berkshire": "BRK-B",
    "asml": "ASML",
    "baba": "BABA",
    "alibaba": "BABA",
}


def _parse_ticker_and_period(doc_name: str):
    """
    Auto-detect ticker and period from SEC filing filename or company name.
    Examples:
      msft-20240331   -> (MSFT, Q3FY2024)   [10-Q]
      aapl-20250927   -> (AAPL, FY2025)      [10-K, Sep year-end]
      msft-20250630   -> (MSFT, FY2025)      [10-K, Jun year-end]
      tesla           -> (TSLA, None)
      tsla            -> (TSLA, None)
    Returns (ticker, period) or (None, None) if not parseable.
    """
    import re
    clean = doc_name.lower().replace(".html", "").replace(".htm", "").replace(".pdf", "").strip()

    # Direct company name match (e.g. "tesla", "microsoft")
    if clean in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[clean], None

    # Standard SEC filing regex: ticker-YYYYMMDD
    m = re.match(r'^([a-zA-Z]+)[-_](\d{4})(\d{2})(\d{2})', doc_name)
    if not m:
        for comp, tick in COMPANY_TICKER_MAP.items():
            if comp in clean:
                return tick, None
        if len(clean) in (3, 4, 5) and clean.isalpha():
            return clean.upper(), None
        return None, None

    raw_prefix = m.group(1).lower()
    ticker = COMPANY_TICKER_MAP.get(raw_prefix, m.group(1).upper())
    year, month, day = int(m.group(2)), int(m.group(3)), int(m.group(4))
    if month in (6, 9, 12) and day >= 28:
        fy = year if month >= 4 else year - 1
        period = f"FY{fy}"
    elif month == 3:
        period = f"Q3FY{year - 1}" if year > 2000 else None
    else:
        period = None
    return ticker, period


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size:1.3rem; font-weight:800; color:#63b3ed; letter-spacing:-0.3px;">
            📈 FinSight
        </div>
        <div style="font-size:0.75rem; color:#475569; margin-top:2px;">Financial Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: 1-Click Company Search & Ingest ──
    st.markdown('<div class="section-label">⚡ 1-Click Company Loader</div>', unsafe_allow_html=True)
    with st.expander("🔍 Search & Auto-Load Company", expanded=True):
        st.caption("Type any US company. FinSight auto-downloads the full 150+ page filing & verified numbers in 1 click.")
        auto_comp_input = st.text_input(
            "Company Name or Ticker",
            placeholder="e.g. Boeing, BA, Apple, TSLA, JPM, MSFT",
            key="auto_company_input",
        ).strip()
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            auto_form_sel = st.selectbox("Filing Type", ["10-K (Annual)", "10-Q (Quarterly)"], key="auto_form_sel")
        with col_f2:
            auto_year_input = st.text_input("Fiscal Year (opt)", placeholder="e.g. 2024", key="auto_year_input").strip()

        form_code = "10-Q" if "10-Q" in auto_form_sel else "10-K"
        year_val = int(auto_year_input) if auto_year_input.isdigit() else None

        if st.button("🚀 Load Complete Filing & Facts", use_container_width=True, key="btn_auto_load_company", disabled=not auto_comp_input):
            clean_comp = auto_comp_input.lower()
            resolved_ticker = COMPANY_TICKER_MAP.get(clean_comp, auto_comp_input.upper())

            prog_box = st.empty()
            p_logs = []
            def _p_log(msg):
                p_logs.append(msg)
                prog_box.info("\n\n".join(p_logs))

            with st.spinner(f"Auto-fetching official SEC report & XBRL facts for **{resolved_ticker}**..."):
                try:
                    f_info = download_sec_filing(
                        ticker=resolved_ticker,
                        form_type=form_code,
                        fiscal_year=year_val,
                        progress_callback=_p_log,
                    )
                    _p_log(f"Embedding {f_info['file_name']} ({f_info['size_mb']} MB) into vector database...")
                    chunk_count = vector_store.add_document(f_info["file_path"])
                    st.cache_data.clear()

                    _p_log(f"Loading official SEC XBRL facts for {resolved_ticker}...")
                    x_res = fetch_and_store_xbrl(
                        ticker=resolved_ticker,
                        form_type=form_code,
                        progress_callback=_p_log,
                    )

                    prog_box.empty()
                    st.success(
                        f"✅ Loaded **{f_info['company_name']}** ({resolved_ticker})  \n"
                        f"📄 Document: `{f_info['file_name']}` (**{chunk_count} chunks**, {f_info['size_mb']} MB)  \n"
                        f"📊 Verified Numbers: **{x_res['facts_inserted']:,} facts** loaded"
                    )
                    st.session_state["auto_ticker"] = resolved_ticker
                    periods = x_res.get("periods_found", [])
                    st.session_state["auto_period"] = periods[-1] if periods else None
                    st.rerun()
                except Exception as e:
                    prog_box.empty()
                    st.error(f"Auto-load failed: {e}")
                    st.exception(e)

    # ── Section 2: Custom Document Upload ──
    with st.expander("📁 Upload Custom Filing (PDF / HTML)", expanded=False):
        st.caption("Optional: Upload custom downloaded financial PDFs or HTML filings.")
        uploaded_doc = st.file_uploader(
            "Drop PDF or HTML here",
            type=["pdf", "html", "htm"],
            key="doc_uploader",
            label_visibility="collapsed",
        )
        if uploaded_doc and st.button("⬆ Index Document", use_container_width=True, key="btn_index_doc"):
            raw_dir = Path(__file__).resolve().parent / "data" / "raw_pdfs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            clean_name = "".join(c for c in uploaded_doc.name if c.isalnum() or c in (".", "_", "-"))
            save_path = raw_dir / clean_name
            with open(save_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            with st.spinner(f"Parsing & embedding `{clean_name}`... (this may take 30–60s)"):
                try:
                    count = vector_store.add_document(str(save_path))
                    st.cache_data.clear()
                    doc_stem = Path(clean_name).stem
                    auto_ticker, auto_period = _parse_ticker_and_period(doc_stem)
                    st.success(f"✅ Indexed **{count}** chunks from `{clean_name}`")
                    if auto_ticker:
                        st.info(f"🔗 Auto-detected: **{auto_ticker}** · {auto_period or 'unknown period'}")
                        st.session_state["auto_ticker"] = auto_ticker
                        st.session_state["auto_period"] = auto_period
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    st.exception(e)


    with st.expander("⚡ Fetch XBRL from SEC EDGAR", expanded=False):
        st.caption("Auto-downloads verified financial facts for any US public company.")
        fetch_ticker = st.text_input(
            "Ticker Symbol", placeholder="e.g. MSFT, TSLA, GOOGL",
            key="fetch_ticker_input"
        ).strip().upper()
        fetch_period = st.text_input(
            "Fiscal Period (optional)", placeholder="e.g. FY2025 — leave blank for all",
            key="fetch_period_input"
        ).strip() or None
        fetch_form = st.selectbox(
            "Form Type", ["10-K", "10-Q"], key="fetch_form_type"
        )

        if st.button(
            "🌐 Fetch from SEC EDGAR",
            use_container_width=True,
            key="btn_fetch_sec",
            disabled=not fetch_ticker,
        ):
            # Warn if period format looks wrong for 10-Q
            if fetch_form == "10-Q" and fetch_period and fetch_period.startswith("FY"):
                st.warning(
                    "⚠️ For 10-Q, leave the period blank to get all quarters. "
                    "FY-format periods only apply to 10-K annual filings. Fetching all quarters now..."
                )
            from ingestion.sec_fetcher import fetch_and_store_xbrl
            progress_box = st.empty()
            logs = []
            def _progress(msg):
                logs.append(msg)
                progress_box.info("\n\n".join(logs))

            with st.spinner(f"Fetching XBRL facts for **{fetch_ticker}** from SEC..."):
                try:
                    result = fetch_and_store_xbrl(
                        ticker=fetch_ticker,
                        fiscal_period=fetch_period,
                        form_type=fetch_form,
                        progress_callback=_progress,
                    )
                    progress_box.empty()
                    st.success(
                        f"✅ **{result['company_name']}** ({result['ticker']})  \n"
                        f"Inserted **{result['facts_inserted']:,}** facts  \n"
                        f"Periods: `{'`, `'.join(result['periods_found']) or 'none'}`"
                    )
                    st.caption(f"CIK: `{result['cik']}` · Source: SEC EDGAR companyfacts API")
                    # Activate XBRL grounding immediately — don't wait for filename detection
                    if result["facts_inserted"] > 0:
                        st.session_state["auto_ticker"] = result["ticker"]
                        # For 10-Q: activate the most recent quarter fetched
                        # For 10-K: activate the most recent fiscal year
                        periods = result["periods_found"]
                        st.session_state["auto_period"] = periods[-1] if periods else None
                        st.rerun()
                except Exception as e:
                    progress_box.empty()
                    st.error(f"Fetch failed: {e}")
                    st.exception(e)



    st.divider()

    # ── Section 2: Indexed Documents ──
    st.markdown('<div class="section-label">📂 Indexed Documents</div>', unsafe_allow_html=True)

    available_docs = get_indexed_documents(vector_store.collection)
    total_chunks = vector_store.collection.count()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Chunks", total_chunks)
    with col_m2:
        st.metric("Docs", len(available_docs))

    if not available_docs:
        st.caption("📤 Upload or auto-load a filing above to get started.")

    doc_options = ["🌐 All Documents (Auto-Route)"] + available_docs
    selected_doc = st.selectbox(
        "Active Document Scope",
        options=doc_options,
        index=0,
        key="doc_filter",
        help="Choose 'All Documents' to let FinSight automatically search all filings and pull the relevant company's numbers, or select a specific filing.",
    )
    is_global = selected_doc.startswith("🌐") or selected_doc == "All Documents"
    target_paper = None if is_global else selected_doc
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)

    # ── Auto XBRL: detect ticker+period from selected document or query ──
    if not is_global:
        auto_t, auto_p = _parse_ticker_and_period(selected_doc)
        xbrl_ticker = auto_t or None
        xbrl_period = auto_p or None
    else:
        xbrl_ticker = None
        xbrl_period = None

    if is_global:
        st.markdown(
            '<span class="pill pill-blue">🌐 Global Search</span>&nbsp;'
            '<span class="pill pill-green">⚡ Auto-Routes XBRL</span>',
            unsafe_allow_html=True,
        )
        st.caption("Auto-detects company from your question & retrieves matching verified facts.")
    elif xbrl_ticker:
        st.markdown(
            f'<span class="pill pill-green">● XBRL</span>&nbsp;'
            f'<span class="pill pill-blue">{xbrl_ticker}</span>&nbsp;'
            f'<span class="pill pill-amber">{xbrl_period or "all periods"}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Strictly filtering to **{selected_doc}**.")
    else:
        st.markdown(
            '<span class="pill pill-amber">○ Text Only</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Filtering text chunks to **{selected_doc}**.")

    # ── Manage Indexed Documents & Session ──
    with st.expander("⚙️ Session & Document Tools", expanded=False):
        if available_docs:
            doc_to_del = st.selectbox("Select document to delete", available_docs, key="sel_doc_del")
            if st.button("🗑️ Remove Document", use_container_width=True, key="btn_del_doc"):
                del_count = vector_store.delete_document(doc_to_del)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Removed `{doc_to_del}` ({del_count} chunks deleted).")
                st.rerun()
        if st.button("🧹 Clear Chat History", use_container_width=True, key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Reset Cache", use_container_width=True, key="btn_reset_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # ── Privacy Shield Status ──
    st.divider()
    try:
        _ps = get_engine_status()
        _engine_label = _ps["active_nlp_engine"].upper()
        _engine_color = "#10b981" if _engine_label != "REGEX-ONLY" else "#f59e0b"
    except Exception:
        _engine_label = "REGEX-ONLY"
        _engine_color = "#f59e0b"
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(59,130,246,0.06));
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-top: 0.25rem;
    ">
        <div style="font-size:0.75rem; font-weight:700; color:#10b981; letter-spacing:0.05em; margin-bottom:4px;">
            🔒 PRIVACY SHIELD ACTIVE
        </div>
        <div style="font-size:0.72rem; color:#64748b; line-height:1.6;">
            In-flight redaction on every query.<br>
            PII &amp; sensitive financial data masked before the AI sees it.<br>
            <span style="color:{_engine_color}; font-weight:600;">Engine: {_engine_label}</span>
            &nbsp;+&nbsp;<span style="color:#3b82f6; font-weight:600;">REGEX</span><br>
            <span style="color:#475569;">Audit log → <code>logs/redactions.log</code></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="finsight-header">
    <div style="display:flex; align-items:center; gap:0.75rem;">
        <div>
            <p class="finsight-title">📈 FinSight</p>
            <p class="finsight-subtitle">
                AI-powered financial intelligence for retail investors &mdash;
                combining SEC filings, XBRL structured data, and advanced RAG.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Action Row ──
col_label, col_btn = st.columns([4, 1])
with col_label:
    if is_global:
        active_doc_label = '🌐 <span class="pill pill-blue">All Indexed Documents</span> &nbsp;·&nbsp; <span class="pill pill-green">Auto-Routes Company & Facts</span>'
    else:
        active_doc_label = f"📂 `{selected_doc}`"
        if xbrl_ticker:
            active_doc_label += f" &nbsp;·&nbsp; " \
                f'<span class="pill pill-green">{xbrl_ticker}</span> ' \
                f'<span class="pill pill-blue">{xbrl_period or "latest"}</span>'
    st.markdown(
        f'<div style="color:#64748b; font-size:0.85rem; padding-top:0.6rem;">'
        f'Scope: {active_doc_label}</div>',
        unsafe_allow_html=True,
    )
with col_btn:
    generate_report_btn = st.button(
        "⚡ Standard Report", use_container_width=True, key="btn_report"
    )

st.divider()

# ─────────────────────────────────────────────
# CHAT STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome state
if not st.session_state.messages:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 3rem 1rem;
        color: #475569;
    ">
        <div style="font-size:3rem; margin-bottom:1rem;">💬</div>
        <div style="font-size:1.1rem; font-weight:600; color:#64748b; margin-bottom:0.5rem;">
            Start your financial analysis
        </div>
        <div style="font-size:0.85rem; color:#475569;">
            Upload a filing in the sidebar, then ask questions or generate a standard report.
        </div>
        <div style="margin-top:1.5rem; display:flex; justify-content:center; gap:0.75rem; flex-wrap:wrap;">
            <span class="pill pill-blue">What was revenue growth?</span>
            <span class="pill pill-blue">Summarize key risk factors</span>
            <span class="pill pill-blue">Analyze cash flow health</span>
            <span class="pill pill-blue">What is the debt-to-equity ratio?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Render chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 View {len(msg['sources'])} retrieved source(s)"):
                for i, src in enumerate(msg["sources"], start=1):
                    meta = src["metadata"]
                    st.markdown(
                        f'<span class="pill pill-blue">Source {i}</span> &nbsp;'
                        f'`{meta.get("paper_id")}` &nbsp;·&nbsp; '
                        f'Page `{meta.get("page_number")}` &nbsp;·&nbsp; '
                        f'Score `{src.get("similarity_score")}`',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["text"])
                    if i < len(msg["sources"]):
                        st.divider()


# ─────────────────────────────────────────────
# GENERATE STANDARD REPORT
# ─────────────────────────────────────────────
if generate_report_btn:
    user_msg = f"⚡ Generate Standard Financial Analysis Report for `{selected_doc}`"
    with st.chat_message("user"):
        st.markdown(user_msg)
    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing retail investor report..."):
            report_data = rag_generator.generate_standard_report(
                paper_id=target_paper,
                ticker=xbrl_ticker,
                fiscal_period=xbrl_period,
            )
            st.markdown(report_data["answer"])
            if report_data["sources"]:
                with st.expander(f"🔍 View {len(report_data['sources'])} retrieved source(s)"):
                    for i, src in enumerate(report_data["sources"], start=1):
                        meta = src["metadata"]
                        st.markdown(
                            f'<span class="pill pill-blue">Source {i}</span> &nbsp;'
                            f'`{meta.get("paper_id")}` &nbsp;·&nbsp; '
                            f'Page `{meta.get("page_number")}` &nbsp;·&nbsp; '
                            f'Score `{src.get("similarity_score")}`',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["text"])
                        if i < len(report_data["sources"]):
                            st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": report_data["answer"],
        "sources": report_data["sources"],
    })


# ─────────────────────────────────────────────
# INLINE CHAT ATTACHMENT & INPUT
# ─────────────────────────────────────────────
active_attached_doc = st.session_state.get("attached_doc")

col_att1, col_att2 = st.columns([2.2, 4.8])
with col_att1:
    with st.popover("📎 Attach Filing / XBRL / CSV", use_container_width=True):
        st.markdown("**Upload financial document directly into chat:**")
        st.caption("Supports SEC 10-K/10-Q (PDF/HTML), XBRL JSON, CSV financial tables, TXT, MD.")
        inline_file = st.file_uploader(
            "Drop file here",
            type=["pdf", "html", "htm", "json", "csv", "txt", "md", "xml"],
            key="inline_chat_uploader",
            label_visibility="collapsed",
        )
        if inline_file:
            raw_dir = Path(__file__).resolve().parent / "data" / "raw_pdfs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            clean_name = "".join(c for c in inline_file.name if c.isalnum() or c in (".", "_", "-"))
            save_path = raw_dir / clean_name
            with open(save_path, "wb") as f:
                f.write(inline_file.getbuffer())
            
            if st.button("⚡ Index & Chat with File", use_container_width=True, key="btn_index_inline"):
                with st.spinner(f"Indexing `{clean_name}` into vector store..."):
                    c_count = vector_store.add_document(str(save_path))
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    doc_stem = Path(clean_name).stem
                    st.session_state["attached_doc"] = doc_stem
                    auto_t, auto_p = _parse_ticker_and_period(doc_stem)
                    if auto_t:
                        st.session_state["auto_ticker"] = auto_t
                    st.success(f"✅ Ready! `{clean_name}` ({c_count} chunks indexed).")
                    st.rerun()

with col_att2:
    if active_attached_doc:
        st.markdown(
            f'<div style="padding-top:0.4rem;">'
            f'<span class="pill pill-green">📎 Attached: <b>{active_attached_doc}</b></span> '
            f'<span style="color:#94a3b8; font-size:0.8rem; margin-left:0.5rem;">Ready for analysis</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Target document prioritizing attached doc if present
effective_paper_id = active_attached_doc or target_paper

if prompt := st.chat_input("Ask about revenue, debt, risks, margins, EPS or your attached file..."):
    # ── Privacy: redact PII / sensitive data from user input ──
    clean_prompt, pii_findings = redact_query(prompt)
    privacy_badge = get_redaction_summary(pii_findings)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        if privacy_badge:
            st.caption(privacy_badge)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating analyst response..."):
            response_data = rag_generator.generate_answer(
                query=clean_prompt,          # use sanitised query
                top_k=top_k,
                paper_id=effective_paper_id,
                ticker=xbrl_ticker,
                fiscal_period=xbrl_period,
            )
            st.markdown(response_data["answer"])
            if response_data["sources"]:
                with st.expander(f"🔍 View {len(response_data['sources'])} retrieved source(s)"):
                    for i, src in enumerate(response_data["sources"], start=1):
                        meta = src["metadata"]
                        st.markdown(
                            f'<span class="pill pill-blue">Source {i}</span> &nbsp;'
                            f'`{meta.get("paper_id")}` &nbsp;·&nbsp; '
                            f'Page `{meta.get("page_number")}` &nbsp;·&nbsp; '
                            f'Score `{src.get("similarity_score")}`',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["text"])
                        if i < len(response_data["sources"]):
                            st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["answer"],
        "sources": response_data["sources"],
    })