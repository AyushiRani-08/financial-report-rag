# app.py
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import os

from generation.generator import RAGGenerator
from indexing.vector_store import VectorStore

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

@st.cache_data(ttl=60)
def get_indexed_documents(_collection):
    try:
        metas = _collection.get(include=["metadatas"])["metadatas"]
        return sorted(list({m["paper_id"] for m in metas if m and "paper_id" in m}))
    except Exception:
        return []


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

    # ── Section 1: Document Ingestion ──
    st.markdown('<div class="section-label">📄 Document Ingestion</div>', unsafe_allow_html=True)

    with st.expander("Upload Filing (PDF / HTML)", expanded=True):
        st.caption("SEC 10-K, 10-Q, Annual Reports, HTML filings")
        uploaded_doc = st.file_uploader(
            "Drop PDF or HTML here",
            type=["pdf", "html", "htm"],
            key="doc_uploader",
            label_visibility="collapsed",
        )
        if uploaded_doc and st.button("⬆ Index Document", use_container_width=True, key="btn_index_doc"):
            raw_dir = Path("data/raw_pdfs")
            raw_dir.mkdir(parents=True, exist_ok=True)
            clean_name = "".join(c for c in uploaded_doc.name if c.isalnum() or c in (".", "_", "-"))
            save_path = raw_dir / clean_name
            with open(save_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            with st.spinner(f"Parsing & embedding `{clean_name}`..."):
                try:
                    count = vector_store.add_document(str(save_path))
                    st.cache_data.clear()
                    st.success(f"Indexed **{count}** chunks from `{clean_name}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    with st.expander("Upload XBRL (.xml) for Exact Numbers", expanded=False):
        st.caption("Links verified SEC financial figures to PostgreSQL")
        uploaded_xbrl = st.file_uploader(
            "Drop XBRL XML here",
            type=["xml"],
            key="xbrl_uploader",
            label_visibility="collapsed",
        )
        xbrl_ticker_input = st.text_input(
            "Ticker", placeholder="AAPL", key="xbrl_ticker_input"
        ).strip().upper()
        xbrl_name_input = st.text_input(
            "Company Name", placeholder="Apple Inc.", key="xbrl_name_input"
        ).strip()
        xbrl_form_input = st.selectbox(
            "Form Type", ["10-K", "10-Q", "AOC-4", "Annual Report"], key="xbrl_form"
        )
        xbrl_period_input = st.text_input(
            "Fiscal Period", placeholder="FY2024", key="xbrl_period_input"
        ).strip()
        xbrl_market_input = st.selectbox("Market", ["US", "IN"], key="xbrl_market")

        can_ingest_xbrl = (
            uploaded_xbrl and xbrl_ticker_input and xbrl_name_input and xbrl_period_input
        )
        if st.button(
            "⚡ Ingest XBRL into Database",
            use_container_width=True,
            key="btn_ingest_xbrl",
            disabled=not can_ingest_xbrl,
        ):
            xbrl_dir = Path("data/xbrl")
            xbrl_dir.mkdir(parents=True, exist_ok=True)
            xbrl_path = xbrl_dir / uploaded_xbrl.name
            with open(xbrl_path, "wb") as f:
                f.write(uploaded_xbrl.getbuffer())
            with st.spinner(f"Parsing XBRL for **{xbrl_ticker_input}**..."):
                try:
                    from ingestion.xbrl_parser import parse_xbrl_to_db
                    count = parse_xbrl_to_db(
                        xbrl_path=str(xbrl_path),
                        ticker=xbrl_ticker_input,
                        company_name=xbrl_name_input,
                        form_type=xbrl_form_input,
                        fiscal_period=xbrl_period_input,
                        market=xbrl_market_input,
                    )
                    st.success(f"Inserted **{count}** facts for `{xbrl_ticker_input}` into PostgreSQL")
                except Exception as e:
                    st.error(f"XBRL ingestion failed: {e}")
                    st.exception(e)

    st.divider()

    # ── Section 2: Query Controls ──
    st.markdown('<div class="section-label">🔍 Query Controls</div>', unsafe_allow_html=True)

    available_docs = get_indexed_documents(vector_store.collection)
    total_chunks = vector_store.collection.count()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Chunks", total_chunks)
    with col_m2:
        st.metric("Docs", len(available_docs))

    selected_doc = st.selectbox(
        "Filter Document",
        options=["All Documents"] + available_docs,
        key="doc_filter",
    )
    target_paper = None if selected_doc == "All Documents" else selected_doc
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)

    st.divider()

    # ── Section 3: XBRL Grounding ──
    st.markdown('<div class="section-label">🗄️ XBRL Grounding</div>', unsafe_allow_html=True)
    st.caption("Link a ticker to inject verified numbers into every answer.")

    xbrl_ticker = st.text_input(
        "Active Ticker", placeholder="e.g. AAPL", key="active_ticker"
    ).strip().upper() or None

    xbrl_period = st.text_input(
        "Active Period", placeholder="e.g. FY2024", key="active_period"
    ).strip() or None

    if xbrl_ticker:
        st.markdown(
            f'<span class="pill pill-green">● XBRL ON</span>&nbsp;'
            f'<span class="pill pill-blue">{xbrl_ticker}</span>&nbsp;'
            f'<span class="pill pill-amber">{xbrl_period or "latest"}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="pill pill-amber">○ XBRL OFF</span>&emsp;'
            '<span style="color:#475569;font-size:0.75rem;">Set ticker to activate</span>',
            unsafe_allow_html=True,
        )


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
    active_doc_label = f"📂 `{selected_doc}`"
    if xbrl_ticker:
        active_doc_label += f" &nbsp;·&nbsp; " \
            f'<span class="pill pill-green">{xbrl_ticker}</span> ' \
            f'<span class="pill pill-blue">{xbrl_period or "latest"}</span>'
    st.markdown(
        f'<div style="color:#64748b; font-size:0.85rem; padding-top:0.6rem;">'
        f'Filtering: {active_doc_label}</div>',
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
# CUSTOM Q&A CHAT INPUT
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask about revenue, debt, risks, margins, EPS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating analyst response..."):
            response_data = rag_generator.generate_answer(
                query=prompt,
                top_k=top_k,
                paper_id=target_paper,
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