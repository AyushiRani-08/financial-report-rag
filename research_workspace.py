# research_workspace.py — FinSight main chat/research interface
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
from visualization import render_chart
from ui import apply_theme, render_settings_section, render_scope_bar

load_dotenv()

# ─────────────────────────────────────────────
# CLEAN THEME STYLES (Light & Dark Support)
# ─────────────────────────────────────────────
apply_theme()

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


# ── File Upload Validation Helpers ──────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 100
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/html",
    "application/xhtml+xml",
    "application/json",
    "text/csv",
    "text/plain",
}
ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".json", ".csv", ".txt"}


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters."""
    # Remove path components
    name = Path(filename).name
    # Keep only alphanumeric, dots, underscores, hyphens
    clean = "".join(c for c in name if c.isalnum() or c in (".", "_", "-"))
    # Ensure extension is preserved and allowed
    suffix = Path(clean).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        # Default to .txt if extension not recognized
        clean = Path(clean).stem + ".txt"
    # Limit length
    if len(clean) > 255:
        stem = Path(clean).stem[:255 - len(suffix)]
        clean = stem + suffix
    return clean or "upload.txt"


def _validate_uploaded_file(uploaded_file) -> str | None:
    """Validate uploaded file size and type. Returns error message or None if valid."""
    # Check file size
    if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return f"File too large: {uploaded_file.size / (1024*1024):.1f} MB (max {MAX_UPLOAD_SIZE_MB} MB)"

    # Check MIME type
    mime_type = getattr(uploaded_file, "type", None)
    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        return f"Unsupported file type: {mime_type}. Allowed: PDF, HTML, JSON, CSV, TXT"

    # Check extension
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file extension: {ext}. Allowed: .pdf, .html, .htm, .json, .csv, .txt"

    return None


def _user_friendly_error(exc: Exception) -> str:
    """Convert exception to user-friendly error message."""
    msg = str(exc).lower()
    if "file not found" in msg:
        return "The uploaded file could not be found. Please try uploading again."
    if "permission denied" in msg:
        return "Permission denied saving the file. Please contact support."
    if "no space left" in msg or "disk full" in msg:
        return "Server storage is full. Please try again later."
    if "timeout" in msg or "timed out" in msg:
        return "Processing timed out. The file may be too large or complex."
    if "memory" in msg or "oom" in msg:
        return "Not enough memory to process this file. Try a smaller file."
    if "tesseract" in msg or "ocr" in msg:
        return "OCR processing failed. Ensure the PDF has selectable text (not scanned images)."
    if "chromadb" in msg or "vector" in msg or "embedding" in msg:
        return "Failed to index document. The vector store may be unavailable."
    if "postgres" in msg or "database" in msg or "psycopg" in msg:
        return "Database error while storing structured data. Please try again."
    if "groq" in msg or "llm" in msg or "api" in msg:
        return "AI service temporarily unavailable. Please try again in a moment."
    return f"Upload failed: {exc}"


def _log_upload_error(exc: Exception, filename: str) -> None:
    """Log upload error with context for debugging."""
    import logging
    logger = logging.getLogger(__name__)
    logger.error(
        "Upload failed for %s: %s: %s",
        filename,
        type(exc).__name__,
        exc,
        exc_info=True,
    )


# ─────────────────────────────────────────────
# SIDEBAR (Minimalist, ChatGPT-Style Layout)
# ─────────────────────────────────────────────
render_sidebar_user(current_user, page_key="app")

with st.sidebar:
    # ── 1. Primary Filter: Filing Scope ──
    st.markdown("##### Analysis Scope")
    available_docs = get_indexed_documents(vector_store.collection)
    doc_options = ["All Indexed Documents (Auto Routing)"] + available_docs
    selected_doc = st.selectbox(
        "Active Filing",
        options=doc_options,
        index=0,
        key="doc_filter",
        help="Select a specific filing to restrict search, or choose All Indexed Documents for automated routing.",
        label_visibility="collapsed",
    )
    is_global = selected_doc.startswith("All Indexed Documents")
    target_paper = None if is_global else selected_doc

    if not is_global:
        auto_t, auto_p = _parse_ticker_and_period(selected_doc)
        xbrl_ticker = auto_t or None
        xbrl_period = auto_p or None
    else:
        xbrl_ticker = None
        xbrl_period = None

    st.markdown("---")

    # ── 2. Collapsible Data Ingestion ──
    with st.expander("Import Filings", expanded=False):
        tab_sec, tab_upload = st.tabs(["SEC EDGAR", "File Upload"])

        with tab_sec:
            auto_comp_input = st.text_input(
                "Company Ticker / Name",
                placeholder="e.g. MSFT, AAPL, NVDA",
                key="auto_company_input",
            ).strip()

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                auto_form_sel = st.selectbox(
                    "Form Type",
                    ["10-K (Annual)", "10-Q (Quarterly)"],
                    key="auto_form_sel"
                )
            with col_f2:
                auto_year_input = st.text_input(
                    "Fiscal Year",
                    placeholder="e.g. 2024",
                    key="auto_year_input"
                ).strip()

            form_code = "10-Q" if "10-Q" in auto_form_sel else "10-K"
            year_val = int(auto_year_input) if auto_year_input.isdigit() else None

            fetch_facts_only = st.checkbox(
                "Facts only (XBRL)",
                value=False,
                key="chk_facts_only",
                help="Loads financial numbers without full textual report."
            )

            btn_label = "Load Facts" if fetch_facts_only else "Download & Index"
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
                                f"Ingested {f_info['company_name']} ({resolved_ticker}). "
                                f"Indexed: {chunk_count} chunks. Facts: {x_res['facts_inserted']:,} records."
                            )
                        else:
                            st.success(f"Loaded {x_res['facts_inserted']:,} facts for {resolved_ticker}.")

                        st.session_state["auto_ticker"] = resolved_ticker
                        periods = x_res.get("periods_found", [])
                        st.session_state["auto_period"] = periods[-1] if periods else None
                        st.rerun()
                    except Exception as e:
                        prog_box.empty()
                        st.error(f"SEC Ingestion Error: {e}")

        with tab_upload:
            uploaded_doc = st.file_uploader(
                "Upload Financial Document",
                type=["pdf", "html", "htm", "txt"],
                key="doc_uploader",
                help="Upload an annual or quarterly report in PDF or HTML format."
            )
            # Validation
            if uploaded_doc:
                # MIME/type validation
                if uploaded_doc.type not in ["application/pdf", "text/html", "text/htm", "text/plain"]:
                    st.error("Unsupported file type. Please upload PDF, HTML, HTM, or TXT files.")
                    uploaded_doc = None
                # Size limit 50 MB
                elif uploaded_doc.size > 50 * 1024 * 1024:
                    st.error("File size exceeds 50 MB limit.")
                    uploaded_doc = None
            if uploaded_doc and st.button("Index Document", use_container_width=True, key="btn_index_doc"):
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

    # ── 3. Modular Settings Section (Contains Theme Toggle, Search Depth, Actions & Sign Out) ──
    def _clear_chat_action():
        st.session_state.messages = []

    def _reset_cache_action():
        st.cache_data.clear()
        st.cache_resource.clear()

    def _delete_doc_action(doc_name: str):
        vector_store.delete_document(doc_name)
        st.cache_data.clear()
        st.cache_resource.clear()

    render_settings_section(
        available_docs=available_docs,
        on_clear_chat=_clear_chat_action,
        on_reset_cache=_reset_cache_action,
        on_delete_doc=_delete_doc_action,
        on_logout=logout,
    )

    top_k = st.session_state.get("slider_top_k", 4)

    # Muted compliance status
    try:
        _ps = get_engine_status()
        _engine_label = _ps["active_nlp_engine"].upper()
    except Exception:
        _engine_label = "REGEX"
    st.caption(f"Privacy redaction active ({_engine_label}) • Auditable session")


# ─────────────────────────────────────────────
# MAIN WORKSPACE HEADER
# ─────────────────────────────────────────────
col_head, col_action = st.columns([3.6, 1.4])

with col_head:
    st.markdown("""
    <div class="app-main-header">
        <h2 class="app-main-title">FinSight Research Workspace</h2>
        <div class="app-main-subtitle">SEC Filing Analysis &amp; Verified XBRL Financial Fact Grounding</div>
    </div>
    """, unsafe_allow_html=True)

with col_action:
    generate_report_btn = st.button(
        "Generate Report",
        use_container_width=True,
        key="btn_report",
        help="Synthesizes a structured analyst report covering performance, margins, risks, and financial condition."
    )

# ── Active Scope Indicator Bar ──
if is_global:
    scope_text = "All Documents &bull; Automated query routing"
    grounding_info = "SEC Filings &amp; Verified Facts"
else:
    scope_text = f"{selected_doc}"
    grounding_info = f"XBRL: {xbrl_ticker} ({xbrl_period or 'Latest'})" if xbrl_ticker else "Text Grounding Only"

render_scope_bar(scope_text, grounding_info)


# ─────────────────────────────────────────────
# CHAT STATE AND STARTER QUERIES
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

query_to_execute = None

if not st.session_state.messages:
    st.caption("Select a filing or prompt to begin analysis:")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("Analyze revenue growth & operating margin trajectory", use_container_width=True, key="sq1", type="secondary"):
            query_to_execute = "Analyze revenue growth and operating margin trajectory over the reported periods."
        if st.button("Summarize principal risk factors & legal disclosures", use_container_width=True, key="sq2", type="secondary"):
            query_to_execute = "Summarize the principal risk factors and material legal disclosures."
    with col_q2:
        if st.button("Evaluate liquidity, debt maturity & operating cash flow", use_container_width=True, key="sq3", type="secondary"):
            query_to_execute = "Evaluate liquidity, debt maturity profile, and operating cash flow."
        if st.button("Report earnings per share (EPS) & capex details", use_container_width=True, key="sq4", type="secondary"):
            query_to_execute = "Report basic and diluted earnings per share (EPS) and capital expenditure details."


# ── Render Chat History ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_data"):
            render_chart(msg["chart_data"])
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
# EXECUTION: STANDARD REPORT (streamed)
# ─────────────────────────────────────────────
if generate_report_btn:
    doc_target_label = selected_doc if not is_global else "all indexed filings"
    user_prompt_text = f"Generate standard financial analysis report for {doc_target_label}."

    st.session_state.messages.append({"role": "user", "content": user_prompt_text})
    with st.chat_message("user"):
        st.markdown(user_prompt_text)

    with st.chat_message("assistant"):
        report_sources = []
        report_charts = []
        report_answer = st.write_stream(
            rag_generator.generate_standard_report_stream(
                paper_id=target_paper,
                ticker=xbrl_ticker,
                fiscal_period=xbrl_period,
                sources_out=report_sources,
                chart_out=report_charts,
            )
        )

        if report_charts:
            render_chart(report_charts[0])

        safe_label = doc_target_label.replace(" ", "_").replace("/", "-")
        st.download_button(
            label="Export Report (.md)",
            data=report_answer,
            file_name=f"FinSight_Report_{safe_label}.md",
            mime="text/markdown",
            key="btn_export_report",
        )

        if report_sources:
            with st.expander(f"Retrieved Source References ({len(report_sources)})"):
                for i, src in enumerate(report_sources, start=1):
                    meta = src.get("metadata", {})
                    p_id = meta.get("paper_id", "Unknown Document")
                    p_num = meta.get("page_number", "N/A")
                    st.markdown(f"**Reference {i}** &mdash; Filing: `{p_id}` | Page: `{p_num}`")
                    st.markdown(f"> {src.get('text', '')}")
                    if i < len(report_sources):
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": report_answer,
        "chart_data": report_charts[0] if report_charts else None,
        "sources": report_sources,
    })


# ─────────────────────────────────────────────
# INLINE ATTACHMENT AND INPUT HANDLING
# ─────────────────────────────────────────────
active_attached_doc = st.session_state.get("attached_doc")

col_attach, col_status = st.columns([1.5, 3.5])
with col_attach:
    with st.popover("Attach Document", use_container_width=True):
        st.caption("Upload a financial report for session-level analysis.")
        session_upload = st.file_uploader(
            "Upload Filing",
            type=["pdf", "html", "htm", "json", "csv", "txt"],
            key="session_uploader",
            label_visibility="collapsed",
        )
        if session_upload:
            # Validate file
            validation_error = _validate_uploaded_file(session_upload)
            if validation_error:
                st.error(validation_error)
            else:
                raw_dir = Path(__file__).resolve().parent / "data" / "raw_pdfs"
                raw_dir.mkdir(parents=True, exist_ok=True)
                clean_name = _sanitize_filename(session_upload.name)
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
                            _log_upload_error(e, clean_name)
                            st.error(_user_friendly_error(e))

with col_status:
    if active_attached_doc:
        st.caption(f"Active session attachment: **{active_attached_doc}**")

effective_paper_id = active_attached_doc or target_paper

chat_prompt = st.chat_input("Enter your financial inquiry regarding this filing...")
final_prompt = query_to_execute or chat_prompt


# ─────────────────────────────────────────────
# CHAT RESPONSE (streamed word-by-word)
# ─────────────────────────────────────────────
if final_prompt:
    clean_prompt, pii_findings = redact_query(final_prompt)
    privacy_badge = get_redaction_summary(pii_findings)

    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)
        if privacy_badge:
            st.caption(privacy_badge)

    with st.chat_message("assistant"):
        chat_sources = []
        chat_charts = []
        answer = st.write_stream(
            rag_generator.generate_answer_stream(
                query=clean_prompt,
                top_k=top_k,
                paper_id=effective_paper_id,
                ticker=xbrl_ticker,
                fiscal_period=xbrl_period,
                sources_out=chat_sources,
                chart_out=chat_charts,
            )
        )

        if chat_charts:
            render_chart(chat_charts[0])

        if chat_sources:
            with st.expander(f"Retrieved Source References ({len(chat_sources)})"):
                for i, src in enumerate(chat_sources, start=1):
                    meta = src.get("metadata", {})
                    p_id = meta.get("paper_id", "Unknown Document")
                    p_num = meta.get("page_number", "N/A")
                    st.markdown(f"**Reference {i}** &mdash; Filing: `{p_id}` | Page: `{p_num}`")
                    st.markdown(f"> {src.get('text', '')}")
                    if i < len(chat_sources):
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chart_data": chat_charts[0] if chat_charts else None,
        "sources": chat_sources,
    })
