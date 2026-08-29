import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retriever import Retriever
from retrieval.xbrl_retriever import XBRLRetriever
from utils.logger import log_query, Timer

load_dotenv()


class RAGGenerator:

  def __init__(
      self,
      retriever=None,
      model_name: str = "openai/gpt-oss-20b",
      xbrl_retriever: Optional[XBRLRetriever] = None,
  ):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
      raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    self.retriever = retriever or Retriever()
    self.model_name = model_name
    self.client = Groq(api_key=api_key)

    # Optional: structured XBRL retriever backed by PostgreSQL.
    # Auto-initializes if POSTGRES_DSN is set; silently disabled otherwise.
    if xbrl_retriever is not None:
      self.xbrl_retriever: Optional[XBRLRetriever] = xbrl_retriever
    elif os.getenv("POSTGRES_DSN"):
      try:
        self.xbrl_retriever = XBRLRetriever()
      except Exception:
        self.xbrl_retriever = None
    else:
      self.xbrl_retriever = None

  def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with citations into a unified context block."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
      page = chunk["metadata"].get("page_number", "1")
      paper_id = chunk["metadata"].get("paper_id", "Unknown Document")
      text = chunk["text"].strip()
      context_parts.append(
          f"[Source {i} | Document: {paper_id} | Page/Section: {page}]\n{text}"
      )
    return "\n\n".join(context_parts)

  def _get_xbrl_context(self, ticker: Optional[str], fiscal_period: Optional[str] = None) -> str:
    """Fetch structured XBRL facts from PostgreSQL as a formatted text block.
    Returns empty string if XBRL retriever is unavailable or ticker is None.
    """
    if not self.xbrl_retriever or not ticker:
      return ""
    try:
      return self.xbrl_retriever.to_context_string(ticker=ticker, fiscal_period=fiscal_period)
    except Exception:
      return ""

  def generate_answer(
      self, query: str, top_k: int = 3, paper_id=None,
      ticker: Optional[str] = None, fiscal_period: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Retrieves relevant chunks and generates a grounded response with page citations."""
    total_start = time.perf_counter()

    # --- Retrieval with timing ---
    with Timer() as retrieval_timer:
      chunks = self.retriever.retrieve(query=query, top_k=top_k, paper_id=paper_id)

    if not chunks:
      log_query(
          query=query, query_type="qa", paper_id=paper_id, top_k=top_k,
          num_chunks_retrieved=0, similarity_scores=[],
          retrieval_latency_ms=retrieval_timer.elapsed_ms,
          llm_latency_ms=0, total_latency_ms=retrieval_timer.elapsed_ms,
          model_name=self.model_name, answer_length=0,
      )
      return {
          "answer": "No relevant financial documents found in the database matching your search.",
          "sources": [],
      }

    context = self.format_context(chunks)
    similarity_scores = [c["similarity_score"] for c in chunks]

    # --- Structured XBRL facts from PostgreSQL (prepended for grounding) ---
    xbrl_block = self._get_xbrl_context(ticker=ticker, fiscal_period=fiscal_period)
    xbrl_section = f"{xbrl_block}\n\n" if xbrl_block else ""

    system_prompt = (
        "You are an expert Senior Financial Analyst specialized in explaining corporate financial reports (10-K, 10-Q, quarterly earnings) to retail investors.\n"
        "Rules:\n"
        "1. Answer clearly, accurately, and concisely using ONLY the provided CONTEXT.\n"
        "2. If STRUCTURED FINANCIAL DATA is provided at the top, treat those figures as ground truth — they come directly from official XBRL filings.\n"
        "3. Explain complex financial jargon (e.g. EBITDA, Free Cash Flow, Diluted EPS) in accessible terms for retail investors when relevant.\n"
        "4. If the context does not contain the answer, state: 'I cannot find sufficient information in the indexed financial report.'\n"
        "5. Always cite specific page or section numbers using [Page X] notation whenever citing metrics or statements."
    )

    user_prompt = f"""{xbrl_section}CONTEXT FROM FINANCIAL REPORT:
{context}

RETAIL INVESTOR QUESTION:
{query}

ANALYST RESPONSE:"""

    # --- LLM call with timing ---
    with Timer() as llm_timer:
      response = self.client.chat.completions.create(
          model=self.model_name,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.1,
      )

    answer = response.choices[0].message.content
    total_ms = (time.perf_counter() - total_start) * 1000

    # --- Log the query ---
    log_query(
        query=query, query_type="qa", paper_id=paper_id, top_k=top_k,
        num_chunks_retrieved=len(chunks), similarity_scores=similarity_scores,
        retrieval_latency_ms=retrieval_timer.elapsed_ms,
        llm_latency_ms=llm_timer.elapsed_ms,
        total_latency_ms=total_ms,
        model_name=self.model_name, answer_length=len(answer),
    )

    return {"answer": answer, "sources": chunks}

  def generate_standard_report(
      self, paper_id=None,
      ticker: Optional[str] = None, fiscal_period: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Generates a comprehensive 5-part standard retail investor financial report."""
    total_start = time.perf_counter()

    report_query = (
        "Provide a comprehensive financial summary including revenue growth, net income, "
        "profitability margins, free cash flow, major risk factors, and overall strategic guidance."
    )

    with Timer() as retrieval_timer:
      chunks = self.retriever.retrieve(query=report_query, top_k=6, paper_id=paper_id)

    if not chunks:
      log_query(
          query=report_query, query_type="standard_report", paper_id=paper_id, top_k=6,
          num_chunks_retrieved=0, similarity_scores=[],
          retrieval_latency_ms=retrieval_timer.elapsed_ms,
          llm_latency_ms=0, total_latency_ms=retrieval_timer.elapsed_ms,
          model_name=self.model_name, answer_length=0,
      )
      return {
          "answer": "No financial documents found to generate a report. Please upload a PDF or HTML report first.",
          "sources": [],
      }

    context = self.format_context(chunks)
    similarity_scores = [c["similarity_score"] for c in chunks]

    # --- Structured XBRL facts from PostgreSQL (prepended for grounding) ---
    xbrl_block = self._get_xbrl_context(ticker=ticker, fiscal_period=fiscal_period)
    xbrl_section = f"{xbrl_block}\n\n" if xbrl_block else ""

    system_prompt = (
        "You are a Senior Retail Financial Analyst.\n"
        "Generate a structured, professional Standard Financial Analysis Report formatted in clean Markdown.\n"
        "Your report MUST include the following 5 sections:\n"
        "### 1. Executive Summary & Core Highlights\n"
        "### 2. Revenue & Earnings Performance\n"
        "### 3. Balance Sheet & Cash Flow Health\n"
        "### 4. Key Risk Factors & Market Headwinds\n"
        "### 5. Retail Investor Takeaway\n\n"
        "If STRUCTURED FINANCIAL DATA is provided at the top, use those exact figures in sections 2 and 3 — they are verified XBRL values from official filings.\n"
        "Base all other statements strictly on the provided CONTEXT and cite pages/sections as [Page X]."
    )

    user_prompt = f"""{xbrl_section}CONTEXT FROM FINANCIAL FILING:
{context}

Generate the Standard Financial Analysis Report now."""

    with Timer() as llm_timer:
      response = self.client.chat.completions.create(
          model=self.model_name,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.2,
      )

    answer = response.choices[0].message.content
    total_ms = (time.perf_counter() - total_start) * 1000

    log_query(
        query=report_query, query_type="standard_report", paper_id=paper_id, top_k=6,
        num_chunks_retrieved=len(chunks), similarity_scores=similarity_scores,
        retrieval_latency_ms=retrieval_timer.elapsed_ms,
        llm_latency_ms=llm_timer.elapsed_ms,
        total_latency_ms=total_ms,
        model_name=self.model_name, answer_length=len(answer),
    )

    return {"answer": answer, "sources": chunks}


if __name__ == "__main__":
  generator = RAGGenerator()
  print("Generator initialized successfully.")
