# app.py
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import os

from generation.generator import RAGGenerator
from retrieval.retriever import Retriever
from indexing.vector_store import VectorStore
from ingestion.sec_fetcher import download_sec_filing, fetch_and_store_xbrl
from privacy import redact_query, get_redaction_summary, get_engine_status
from auth.session import require_auth, logout, get_current_user, render_sidebar_user

load_dotenv()

st.set_page_config(
    page_title="FinSight — Financial Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# FORMAL THEME STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Input Fields */
.stTextInput input, .stSelectbox select {
    background-color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 6px !important;
    color: #f8fafc !important;
}
.stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding-left: 12px;
    padding-right: 12px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #94a3b8;
    border-radius: 4px 4px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #f8fafc !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #151d2d !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    overflow: hidden;
    margin-bottom: 0.75rem;
}
[data-testid="stExpander"] summary {
    background: #151d2d !important;
    color: #e2e8f0 !important;
    font-size: 0.86rem;
    font-weight: 500;
}
[data-testid="stExpander"] summary:hover {
    color: #93c5fd !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 0.9rem !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    transform: none !important;
}

/* Secondary Button Style */
.stButton > button[kind="secondary"] {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #334155 !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: #141c2c !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    margin-bottom: 0.85rem !important;
    padding: 0.9rem 1.1rem !important;
}

/* Chat Input */
[data-testid="stChatInput"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #151d2d !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
    padding: 0.65rem 0.85rem !important;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Clean UI Badges */
.formal-tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.73rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.formal-tag-blue {
    background: rgba(37, 99, 235, 0.15);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.3);
}
.formal-tag-green {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.formal-tag-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Section Header */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 1.1rem;
    margin-bottom: 0.45rem;
}

/* Top App Header */
.app-header {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
}
.app-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
}

/* Scope Status Bar */
.scope-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #131b2a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    margin-bottom: 1.25rem;
    font-size: 0.82rem;
    color: #cbd5e1;
}

/* Source Box */
.source-card {
    background: #0f1624;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.65rem;
    font-size: 0.82rem;
    color: #cbd5e1;
}

/* Divider */
hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
    margin: 1.1rem 0 !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTH GATE — must be authenticated to proceed
# ─────────────────────────────────────────────
current_user = require_auth()
user_id   = current_user["id"]
user_email = current_user["email"]
user_name  = current_user["name"]

# ─────────────────────────────────────────────
# COMPONENT INITIALIZATION (per-user)
# ─────────────────────────────────────────────
@st.cache_resource
def get_components(uid: str):
    store = VectorStore(persist_directory="data/chroma_db", user_id=uid)
    retriever = Retriever(vector_store=store, user_id=uid)
    generator = RAGGenerator(retriever=retriever, model_name="openai/gpt-oss-20b")
    return store, generator

vector_store, rag_generator = get_components(user_id)

if not hasattr(vector_store, "add_document") or not hasattr(rag_generator, "generate_standard_report"):
    st.cache_resource.clear()
    st.rerun()

@st.cache_data(ttl=5)
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
    Detect ticker and period from filing filename or company identifier.
    Examples:
      msft-20240331 -> (MSFT, Q3FY2024)
      aapl-20250927 -> (AAPL, FY2025)
      tesla         -> (TSLA, None)
    """
    import re
    clean = doc_name.lower().replace(".html", "").replace(".htm", "").replace(".pdf", "").strip()

    if clean in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[clean], None

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
render_sidebar_user(current_user, page_key="app")

with st.sidebar:
    # ── Section 1: Ingestion Interface ──
    st.markdown('<div class="sidebar-section-title">Data Ingestion</div>', unsafe_allow_html=True)
    tab_sec, tab_upload = st.tabs(["SEC EDGAR", "File Upload"])

    with tab_sec:
        auto_comp_input = st.text_input(
            "Company Identifier or Ticker",
            placeholder="e.g. MSFT, AAPL, NVDA, BA",
            key="auto_company_input",
            help="Enter a stock symbol or company name for automated SEC EDGAR retrieval."
        ).strip()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            auto_form_sel = st.selectbox(
                "Filing Form",
                ["10-K (Annual)", "10-Q (Quarterly)"],
                key="auto_form_sel"
            )
        with col_f2:
            auto_year_input = st.text_input(
                "Fiscal Year",
                placeholder="Latest or e.g. 2024",
                key="auto_year_input"
            ).strip()

        form_code = "10-Q" if "10-Q" in auto_form_sel else "10-K"
        year_val = int(auto_year_input) if auto_year_input.isdigit() else None

        fetch_facts_only = st.checkbox(
            "Quantitative facts only (XBRL)",
            value=False,
            key="chk_facts_only",
            help="Select to load only structured database facts without downloading the textual filing."
        )

        btn_label = "Load SEC Facts" if fetch_facts_only else "Load Complete Filing & Facts"
        if st.button(btn_label, use_container_width=True, key="btn_auto_load_company", disabled=not auto_comp_input):
            clean_comp = auto_comp_input.lower()
            resolved_ticker = COMPANY_TICKER_MAP.get(clean_comp, auto_comp_input.upper())

            prog_box = st.empty()
            p_logs = []
            def _p_log(msg):
                p_logs.append(msg)
                prog_box.info("\n\n".join(p_logs))

            with st.spinner(f"Retrieving data for {resolved_ticker} from SEC EDGAR..."):
                try:
                    if not fetch_facts_only:
                        f_info = download_sec_filing(
                            ticker=resolved_ticker,
                            form_type=form_code,
                            fiscal_year=year_val,
                            progress_callback=_p_log,
                        )
                        _p_log(f"Indexing {f_info['file_name']} into vector store...")
                        chunk_count = vector_store.add_document(f_info["file_path"])
                        st.cache_data.clear()

                    _p_log(f"Extracting structured XBRL financial facts for {resolved_ticker}...")
                    x_res = fetch_and_store_xbrl(
                        ticker=resolved_ticker,
                        form_type=form_code,
                        progress_callback=_p_log,
                    )

                    prog_box.empty()
                    if not fetch_facts_only:
                        st.success(
                            f"Successfully ingested {f_info['company_name']} ({resolved_ticker}). "
                            f"Filing indexed: {chunk_count} chunks ({f_info['size_mb']} MB). "
                            f"Verified facts loaded: {x_res['facts_inserted']:,} records."
                        )
                    else:
                        st.success(
                            f"Loaded {x_res['facts_inserted']:,} verified facts for {resolved_ticker}."
                        )

                    st.session_state["auto_ticker"] = resolved_ticker
                    periods = x_res.get("periods_found", [])
                    st.session_state["auto_period"] = periods[-1] if periods else None
                    st.rerun()
                except Exception as e:
                    prog_box.empty()
                    st.error(f"SEC Ingestion Error: {e}")

    with tab_upload:
        uploaded_doc = st.file_uploader(
            "Select Financial Document",
            type=["pdf", "html", "htm", "txt"],
            key="doc_uploader",
            help="Upload an annual or quarterly report in PDF or HTML format."
        )
        if uploaded_doc and st.button("Index Uploaded Document", use_container_width=True, key="btn_index_doc"):
            raw_dir = Path(__file__).resolve().parent / "data" / "raw_pdfs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            clean_name = "".join(c for c in uploaded_doc.name if c.isalnum() or c in (".", "_", "-"))
            save_path = raw_dir / clean_name
            with open(save_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())

            with st.spinner(f"Parsing and embedding {clean_name}..."):
                try:
                    count = vector_store.add_document(str(save_path))
                    st.cache_data.clear()
                    doc_stem = Path(clean_name).stem
                    auto_ticker, auto_period = _parse_ticker_and_period(doc_stem)
                    st.success(f"Indexed {count} chunks from {clean_name}.")
                    if auto_ticker:
                        st.session_state["auto_ticker"] = auto_ticker
                        st.session_state["auto_period"] = auto_period
                    st.rerun()
                except Exception as e:
                    st.error(f"Document Indexing Error: {e}")

    st.divider()

    # ── Section 2: Retrieval Scope & Document Selection ──
    st.markdown('<div class="sidebar-section-title">Query Scope & Retrieval</div>', unsafe_allow_html=True)

    available_docs = get_indexed_documents(vector_store.collection)
    total_chunks = vector_store.chunk_count

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Total Chunks", total_chunks)
    with col_m2:
        st.metric("Documents", len(available_docs))

    doc_options = ["All Indexed Documents (Automated Routing)"] + available_docs
    selected_doc = st.selectbox(
        "Active Scope",
        options=doc_options,
        index=0,
        key="doc_filter",
        help="Select a specific filing to restrict retrieval, or choose All Indexed Documents for automated query routing."
    )
    is_global = selected_doc.startswith("All Indexed Documents")
    target_paper = None if is_global else selected_doc
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)

    if not is_global:
        auto_t, auto_p = _parse_ticker_and_period(selected_doc)
        xbrl_ticker = auto_t or None
        xbrl_period = auto_p or None
    else:
        xbrl_ticker = None
        xbrl_period = None

    if is_global:
        st.markdown(
            '<span class="formal-tag formal-tag-blue">Automated Routing</span> '
            '<span class="formal-tag formal-tag-green">XBRL Fact Grounding</span>',
            unsafe_allow_html=True,
        )
    elif xbrl_ticker:
        st.markdown(
            f'<span class="formal-tag formal-tag-green">XBRL: {xbrl_ticker}</span> '
            f'<span class="formal-tag formal-tag-blue">{xbrl_period or "All Periods"}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="formal-tag">Text Search Only</span>',
            unsafe_allow_html=True,
        )

    # ── Section 3: Document and Session Management ──
    st.divider()
    with st.expander("Session and Document Management", expanded=False):
        if available_docs:
            doc_to_del = st.selectbox("Select document to remove", available_docs, key="sel_doc_del")
            if st.button("Delete Selected Document", use_container_width=True, key="btn_del_doc"):
                del_count = vector_store.delete_document(doc_to_del)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Removed {doc_to_del} ({del_count} chunks deleted).")
                st.rerun()

        if st.button("Clear Chat History", use_container_width=True, key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Reset Application Cache", use_container_width=True, key="btn_reset_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # ── Section 4: Privacy & Compliance Status ──
    try:
        _ps = get_engine_status()
        _engine_label = _ps["active_nlp_engine"].upper()
    except Exception:
        _engine_label = "REGEX"

    st.markdown(f"""
    <div style="
        background: #131b2a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        margin-top: 1rem;
    ">
        <div style="font-size: 0.72rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">
            Data Privacy and Compliance
        </div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px; line-height: 1.45;">
            In-flight redaction active. All queries sanitized prior to inference.<br>
            <span style="color: #94a3b8;">Sanitization Engine: {_engine_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN WORKSPACE HEADER
# ─────────────────────────────────────────────
col_head, col_action = st.columns([3.8, 1.2])

with col_head:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">FinSight Financial Intelligence</div>
        <div class="app-subtitle">
            SEC Filing Retrieval-Augmented Generation with Structured XBRL Fact Verification
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_action:
    generate_report_btn = st.button(
        "Generate Financial Report",
        use_container_width=True,
        key="btn_report",
        help="Synthesizes a structured analyst report covering performance, margins, risks, and financial condition."
    )

# ── Active Scope Indicator Bar ──
if is_global:
    scope_text = "All Indexed Documents &mdash; Automated Query Routing Active"
else:
    scope_text = f"Scope: <b>{selected_doc}</b>"
    if xbrl_ticker:
        scope_text += f" &nbsp;|&nbsp; Verified XBRL Grounding: <b>{xbrl_ticker} ({xbrl_period or 'Latest'})</b>"

st.markdown(f'<div class="scope-bar">{scope_text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT STATE AND EMPTY-STATE INTERACTION
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handler for suggested starter queries
query_to_execute = None

if not st.session_state.messages:
    st.markdown("""
    <div style="
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.35rem;">
            Financial Analysis Workspace
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5; margin-bottom: 1.25rem;">
            Submit inquiries regarding annual or quarterly SEC filings. The retrieval engine combines vector text grounding
            with verified XBRL quantitative facts to deliver auditable financial analysis.
        </div>
        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.6rem;">
            Standard Analysis Queries
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("Analyze revenue growth and operating margin trajectory", use_container_width=True, key="sq1"):
            query_to_execute = "Analyze revenue growth and operating margin trajectory over the reported periods."
        if st.button("Summarize principal risk factors and disclosures", use_container_width=True, key="sq2"):
            query_to_execute = "Summarize the principal risk factors and material legal disclosures."
    with col_q2:
        if st.button("Evaluate liquidity, debt maturity, and operating cash flow", use_container_width=True, key="sq3"):
            query_to_execute = "Evaluate liquidity, debt maturity profile, and operating cash flow."
        if st.button("Report basic and diluted earnings per share (EPS) metrics", use_container_width=True, key="sq4"):
            query_to_execute = "Report basic and diluted earnings per share (EPS) and capital expenditure details."


# ── Render Chat History ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"Retrieved Source References ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], start=1):
                    meta = src.get("metadata", {})
                    p_id = meta.get("paper_id", "Unknown Document")
                    p_num = meta.get("page_number", "N/A")
                    st.markdown(f"**Reference {i}** &mdash; Filing: `{p_id}` | Page: `{p_num}`")
                    st.markdown(f"> {src.get('text', '')}")
                    if i < len(msg["sources"]):
                        st.divider()


# ─────────────────────────────────────────────
# EXECUTION: STANDARD REPORT
# ─────────────────────────────────────────────
if generate_report_btn:
    doc_target_label = selected_doc if not is_global else "all indexed filings"
    user_prompt_text = f"Generate standard financial analysis report for {doc_target_label}."
    
    st.session_state.messages.append({"role": "user", "content": user_prompt_text})
    with st.chat_message("user"):
        st.markdown(user_prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing comprehensive financial report from filings and XBRL facts..."):
            report_data = rag_generator.generate_standard_report(
                paper_id=target_paper,
                ticker=xbrl_ticker,
                fiscal_period=xbrl_period,
            )
            st.markdown(report_data["answer"])
            if report_data["sources"]:
                with st.expander(f"Retrieved Source References ({len(report_data['sources'])})"):
                    for i, src in enumerate(report_data["sources"], start=1):
                        meta = src.get("metadata", {})
                        p_id = meta.get("paper_id", "Unknown Document")
                        p_num = meta.get("page_number", "N/A")
                        st.markdown(f"**Reference {i}** &mdash; Filing: `{p_id}` | Page: `{p_num}`")
                        st.markdown(f"> {src.get('text', '')}")
                        if i < len(report_data["sources"]):
                            st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": report_data["answer"],
        "sources": report_data["sources"],
    })


# ─────────────────────────────────────────────
# INLINE ATTACHMENT AND INPUT HANDLING
# ─────────────────────────────────────────────
active_attached_doc = st.session_state.get("attached_doc")

col_attach, col_status = st.columns([1.5, 3.5])
with col_attach:
    with st.popover("Attach Session Document", use_container_width=True):
        st.caption("Upload a financial report directly for targeted session query analysis.")
        session_upload = st.file_uploader(
            "Upload Filing",
            type=["pdf", "html", "htm", "json", "csv", "txt"],
            key="session_uploader",
            label_visibility="collapsed",
        )
        if session_upload:
            raw_dir = Path(__file__).resolve().parent / "data" / "raw_pdfs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            clean_name = "".join(c for c in session_upload.name if c.isalnum() or c in (".", "_", "-"))
            save_path = raw_dir / clean_name
            with open(save_path, "wb") as f:
                f.write(session_upload.getbuffer())

            if st.button("Index and Attach", use_container_width=True, key="btn_session_index"):
                with st.spinner(f"Indexing {clean_name}..."):
                    try:
                        c_count = vector_store.add_document(str(save_path))
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        doc_stem = Path(clean_name).stem
                        st.session_state["attached_doc"] = doc_stem
                        auto_t, auto_p = _parse_ticker_and_period(doc_stem)
                        if auto_t:
                            st.session_state["auto_ticker"] = auto_t
                        st.success(f"Attached {clean_name} ({c_count} chunks indexed).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload Error: {e}")

with col_status:
    if active_attached_doc:
        st.markdown(
            f'<div style="padding-top: 0.35rem;">'
            f'<span class="formal-tag formal-tag-blue">Session Attachment: {active_attached_doc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

effective_paper_id = active_attached_doc or target_paper

chat_prompt = st.chat_input("Enter financial analysis inquiry or question regarding filing...")
final_prompt = query_to_execute or chat_prompt

if final_prompt:
    clean_prompt, pii_findings = redact_query(final_prompt)
    privacy_badge = get_redaction_summary(pii_findings)

    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)
        if privacy_badge:
            st.caption(privacy_badge)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving filing context and generating financial analysis..."):
            response_data = rag_generator.generate_answer(
                query=clean_prompt,
                top_k=top_k,
                paper_id=effective_paper_id,
                ticker=xbrl_ticker,
                fiscal_period=xbrl_period,
            )
            st.markdown(response_data["answer"])
            if response_data["sources"]:
                with st.expander(f"Retrieved Source References ({len(response_data['sources'])})"):
                    for i, src in enumerate(response_data["sources"], start=1):
                        meta = src.get("metadata", {})
                        p_id = meta.get("paper_id", "Unknown Document")
                        p_num = meta.get("page_number", "N/A")
                        st.markdown(f"**Reference {i}** &mdash; Filing: `{p_id}` | Page: `{p_num}`")
                        st.markdown(f"> {src.get('text', '')}")
                        if i < len(response_data["sources"]):
                            st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["answer"],
        "sources": response_data["sources"],
    })