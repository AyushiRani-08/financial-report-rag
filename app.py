# app.py
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from generation.generator import RAGGenerator
from indexing.vector_store import VectorStore

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Report RAG Assistant",
    page_icon="📊",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #1E293B; }
    .sub-text { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .citation-badge { background-color: #F1F5F9; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_components():
  """Loads vector store and generator once and caches them in memory."""
  store = VectorStore(persist_directory="data/chroma_db")
  generator = RAGGenerator(retriever=None, model_name="openai/gpt-oss-20b")
  return store, generator


vector_store, rag_generator = get_components()

# Self-healing check: automatically clear cache if an outdated cached instance is detected
if not hasattr(vector_store, "add_document") or not hasattr(rag_generator, "generate_standard_report"):
  st.cache_resource.clear()
  st.rerun()

@st.cache_data(ttl=60)
def get_indexed_documents(_collection):
  """Fetch indexed document list cached for 60 seconds to avoid disk reads on every rerun."""
  try:
    metas = _collection.get(include=["metadatas"])["metadatas"]
    docs = sorted(list({m["paper_id"] for m in metas if m and "paper_id" in m}))
    return docs
  except Exception:
    return []


# --- SIDEBAR: Financial Report Management ---
with st.sidebar:
  st.title("📊 Report Indexer")
  st.write("Upload PDF or HTML financial filings (10-K, 10-Q, Annual Reports).")

  uploaded_file = st.file_uploader(
      "Upload Document (.pdf, .html, .htm)",
      type=["pdf", "html", "htm"],
      key="doc_uploader",
  )
  if uploaded_file and st.button("Index Financial Report", use_container_width=True):
    raw_dir = Path("data/raw_pdfs")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename (removes spaces, parentheses, special characters)
    clean_name = "".join(
        c for c in uploaded_file.name if c.isalnum() or c in (".", "_", "-")
    )
    save_path = raw_dir / clean_name

    with open(save_path, "wb") as f:
      f.write(uploaded_file.getbuffer())

    with st.spinner(f"Parsing, chunking, and embedding '{clean_name}'..."):
      try:
        added_count = vector_store.add_document(str(save_path))
        st.cache_data.clear()  # Invalidate document list cache
        st.success(
            f"✅ Successfully indexed `{clean_name}` ({added_count} chunks added)!"
        )
        st.rerun()
      except Exception as e:
        st.error(f"❌ Indexing Failed: {e}")
        st.exception(e)

  st.divider()

  # Collection Stats
  total_chunks = vector_store.collection.count()
  st.metric(label="Total Chunks in Vector DB", value=total_chunks)

  # Cached Document List
  available_docs = get_indexed_documents(vector_store.collection)

  selected_doc = st.selectbox(
      "Filter by Document ID",
      options=["All Documents"] + available_docs,
      help="Filter search to a specific financial report",
  )
  target_paper = None if selected_doc == "All Documents" else selected_doc

  # Top-K Slider
  top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)

# --- MAIN CHAT INTERFACE ---
st.markdown(
    '<div class="main-header">📊 Financial Report RAG Assistant for Retail Investors</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-text">Analyze corporate filings, quarterly earnings, and annual reports. Ask custom questions or generate standardized retail investor reports.</div>',
    unsafe_allow_html=True,
)

# Standard Report Generation Quick Action
col1, col2 = st.columns([3, 1])
with col1:
  st.caption(f"Currently filtering: **{selected_doc}**")
with col2:
  generate_report_btn = st.button("⚡ Generate Standard Financial Report", use_container_width=True)

# Chat session state initialization
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if "sources" in msg and msg["sources"]:
      with st.expander("🔍 View Retrieved Financial Sources"):
        for i, src in enumerate(msg["sources"], start=1):
          meta = src["metadata"]
          st.markdown(
              f"**Source {i}:** Document `{meta.get('paper_id')}` | Page/Section"
              f" `{meta.get('page_number')}` | Similarity Score:"
              f" `{src.get('similarity_score')}`"
          )
          st.caption(src["text"])
          st.divider()

# Action: Generate Standard Report
if generate_report_btn:
  with st.chat_message("user"):
    st.markdown(f"⚡ Generate Standard Financial Analysis Report for `{selected_doc}`")
  st.session_state.messages.append(
      {"role": "user", "content": f"⚡ Generate Standard Financial Analysis Report for `{selected_doc}`"}
  )

  with st.chat_message("assistant"):
    with st.spinner("Analyzing financial filing and synthesizing retail investor report..."):
      report_data = rag_generator.generate_standard_report(paper_id=target_paper)
      st.markdown(report_data["answer"])

      if report_data["sources"]:
        with st.expander("🔍 View Retrieved Financial Sources"):
          for i, src in enumerate(report_data["sources"], start=1):
            meta = src["metadata"]
            st.markdown(
                f"**Source {i}:** Document `{meta.get('paper_id')}` | Page/Section"
                f" `{meta.get('page_number')}` | Similarity Score:"
                f" `{src.get('similarity_score')}`"
            )
            st.caption(src["text"])
            st.divider()

  st.session_state.messages.append({
      "role": "assistant",
      "content": report_data["answer"],
      "sources": report_data["sources"],
  })

# Action: Custom Q&A Prompt
if prompt := st.chat_input("Ask a question about revenue, debt, risks, margins..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Retrieving financial context and generating analyst response..."):
      response_data = rag_generator.generate_answer(
          query=prompt, top_k=top_k, paper_id=target_paper
      )

      st.markdown(response_data["answer"])

      if response_data["sources"]:
        with st.expander("🔍 View Retrieved Financial Sources"):
          for i, src in enumerate(response_data["sources"], start=1):
            meta = src["metadata"]
            st.markdown(
                f"**Source {i}:** Document `{meta.get('paper_id')}` | Page/Section"
                f" `{meta.get('page_number')}` | Similarity Score:"
                f" `{src.get('similarity_score')}`"
            )
            st.caption(src["text"])
            st.divider()

  st.session_state.messages.append({
      "role": "assistant",
      "content": response_data["answer"],
      "sources": response_data["sources"],
  })