import os
import time
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retriever import Retriever
from retrieval.xbrl_retriever import XBRLRetriever
from utils.logger import log_query, Timer
from utils.metadata_parser import parse_filing_metadata, COMPANY_TICKER_MAP
from utils.abstention_guard import check_pre_retrieval_abstention

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

  # Models tried in order when the primary is at capacity (503/429)
  _FALLBACK_MODELS = [
      "llama-3.3-70b-versatile",
      "llama-3.1-8b-instant",
      "gemma2-9b-it",
  ]

  def _call_llm(
      self,
      messages: list,
      temperature: float = 0.1,
      max_retries: int = 3,
  ) -> str:
      """
      Wraps client.chat.completions.create with:
      • Exponential-backoff retry (3 attempts, 2s / 4s / 8s)
      • Automatic fallback to alternative models on 503 / 429

      Returns the final answer string.
      """
      models_to_try = [self.model_name] + [
          m for m in self._FALLBACK_MODELS if m != self.model_name
      ]

      last_exc = None
      for model in models_to_try:
          for attempt in range(max_retries):
              try:
                  resp = self.client.chat.completions.create(
                      model=model,
                      messages=messages,
                      temperature=temperature,
                  )
                  content = resp.choices[0].message.content
                  if model != self.model_name:
                      logging.getLogger(__name__).warning(
                          f"[Generator] Primary model '{self.model_name}' unavailable — "
                          f"used fallback '{model}' (attempt {attempt+1})"
                      )
                  return content or ""
              except Exception as exc:
                  last_exc = exc
                  err_str = str(exc).lower()
                  is_capacity = any(code in err_str for code in ["503", "429", "capacity", "overloaded", "rate"])
                  if is_capacity and attempt < max_retries - 1:
                      wait = 2 ** (attempt + 1)   # 2s, 4s, 8s
                      logging.getLogger(__name__).warning(
                          f"[Generator] {model} returned capacity error — retrying in {wait}s "
                          f"(attempt {attempt+1}/{max_retries})"
                      )
                      time.sleep(wait)
                  else:
                      break   # non-capacity error or exhausted retries → try next model

      raise RuntimeError(
          f"All models exhausted. Last error: {last_exc}"
      )

  def _call_llm_stream(
      self,
      messages: list,
      temperature: float = 0.1,
  ):
      """
      Streams tokens from the LLM, falling back across models on 503/429.
      Yields string token chunks as they arrive from the Groq API.
      """
      models_to_try = [self.model_name] + [
          m for m in self._FALLBACK_MODELS if m != self.model_name
      ]
      for model in models_to_try:
          try:
              stream = self.client.chat.completions.create(
                  model=model,
                  messages=messages,
                  temperature=temperature,
                  stream=True,
              )
              for chunk in stream:
                  delta = chunk.choices[0].delta.content
                  if delta:
                      yield delta
              return  # streaming finished successfully
          except Exception as exc:
              err_str = str(exc).lower()
              is_capacity = any(
                  c in err_str for c in ["503", "429", "capacity", "overloaded", "rate"]
              )
              if is_capacity:
                  logging.getLogger(__name__).warning(
                      f"[Generator] {model} capacity error during stream — trying next model."
                  )
                  continue
              raise
      raise RuntimeError("All models exhausted during streaming.")

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

  def _get_xbrl_context(
      self,
      ticker: Optional[str],
      fiscal_period: Optional[str] = None,
      query: Optional[str] = None,
  ) -> str:
    """Fetch structured XBRL facts from PostgreSQL as a formatted text block.
    Returns empty string if XBRL retriever is unavailable or ticker is None.
    """
    if not self.xbrl_retriever or not ticker:
      return ""
    try:
      return self.xbrl_retriever.to_context_string(
          ticker=ticker, fiscal_period=fiscal_period, query=query
      )
    except Exception:
      return ""

  def generate_answer(
      self, query: str, top_k: int = 3, paper_id=None,
      ticker: Optional[str] = None, fiscal_period: Optional[str] = None,
      session_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Retrieves relevant chunks and generates a grounded response with page citations."""
    total_start = time.perf_counter()

    # --- Pre-Retrieval Scope Resolution (Retrieval Isolation) ---
    active_ticker = ticker
    active_period = fiscal_period

    if paper_id:
        inferred = parse_filing_metadata(paper_id)
        if not active_ticker:
            active_ticker = inferred.get("ticker")
        if not active_period:
            active_period = inferred.get("fiscal_period")

    if not active_ticker:
        q_lower = query.lower()
        company_hints = {
            "boeing": "BA", " ba ": "BA",
            "jpmorgan": "JPM", "jp morgan": "JPM", "jpmc": "JPM", " jpm ": "JPM",
            "tesla": "TSLA", " tsla ": "TSLA",
            "apple": "AAPL", " aapl ": "AAPL",
            "microsoft": "MSFT", " msft ": "MSFT",
            "google": "GOOGL", "alphabet": "GOOGL", " googl ": "GOOGL",
            "amazon": "AMZN", " amzn ": "AMZN",
            "nvidia": "NVDA", " nvda ": "NVDA",
        }
        for hint, t_code in company_hints.items():
            if hint in q_lower:
                active_ticker = t_code
                break

    if not active_period:
        import re
        fy_match = re.search(r'\b(fy\s*20\d\d|20\d\d|q[1-4]\s*fy\s*20\d\d)\b', query.lower())
        if fy_match:
            raw_period = fy_match.group(1).replace(" ", "").upper()
            if raw_period.isdigit():
                active_period = f"FY{raw_period}"
            else:
                active_period = raw_period

    # --- Layer 1: Pre-Retrieval Abstention Gate ---
    abstention = check_pre_retrieval_abstention(query, ticker=active_ticker, fiscal_period=active_period)
    if abstention:
        log_query(
            query=query, query_type="qa_abstain", paper_id=paper_id, top_k=0,
            num_chunks_retrieved=0, similarity_scores=[],
            retrieval_latency_ms=0, llm_latency_ms=0, total_latency_ms=1,
            model_name=self.model_name, answer_length=len(abstention["reason"]),
        )
        return {
            "answer": abstention["reason"],
            "sources": [],
            "abstained": True,
        }

    # --- Isolated Retrieval with timing ---
    with Timer() as retrieval_timer:
      chunks = self.retriever.retrieve(
          query=query,
          top_k=top_k,
          paper_id=paper_id,
          ticker=active_ticker,
          fiscal_period=active_period,
          statement_scope="consolidated",
          session_id=session_id,
      )

    # Graceful fallback: if strictly isolated search yielded 0 results, relax period or ticker
    if not chunks and active_ticker and not paper_id:
        with Timer() as retrieval_timer:
            chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                paper_id=None,
                session_id=session_id,
            )

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

    # If active_ticker wasn't determined beforehand, try inferring from top retrieved chunk
    if not active_ticker and chunks:
        top_paper = chunks[0]["metadata"].get("paper_id", "")
        inferred = parse_filing_metadata(top_paper)
        active_ticker = inferred.get("ticker")

    # --- Structured XBRL facts from PostgreSQL (prepended for grounding) ---
    xbrl_block = self._get_xbrl_context(
        ticker=active_ticker, fiscal_period=active_period, query=query
    )
    xbrl_section = f"{xbrl_block}\n\n" if xbrl_block else ""

    system_prompt = (
        "You are an expert Senior Financial Analyst specialized in explaining corporate financial reports (10-K, 10-Q, quarterly earnings) to retail investors.\n\n"
        "CORE REPORTING & ACCURACY RULES:\n"
        "1. GROUNDING & EVIDENCE: Answer clearly, accurately, and concisely using ONLY the provided CONTEXT. Never speculate or fabricate.\n"
        "2. STANDALONE VS. CONSOLIDATED: Clearly state the reporting scope. Annual 10-K filings report consolidated corporate operations. Label group figures as '[Consolidated]' (e.g. '[Consolidated] Total Net Sales: $391.04B'). If parent-only, label as '[Standalone]'.\n"
        "3. XBRL VERIFIED FACTS: If STRUCTURED FINANCIAL DATA is provided at the top, treat those numbers as audited ground truth directly from SEC XBRL filings.\n"
        "4. ABSTENTION PROTOCOL: If the context does not contain the answer (e.g. non-disclosed divisional headcounts, daily stock prices, competitor financials, or forward-looking targets), state: 'The provided financial reports do not disclose [topic].' Do not guess.\n"
        "5. CITATIONS: Always cite specific page or section numbers using [Page X] notation whenever citing metrics or statements.\n\n"
        "RETAIL INVESTOR ANALYSIS SECTION:\n"
        "Unless you are abstaining/refusing due to missing data, conclude your answer with an investor analysis section formatted as follows:\n\n"
        "### 💡 What Does This Mean for an Investor?\n"
        "• 🟢 **Positive Signals**: 1-2 concise bullet points highlighting financial strengths, margin expansion, or capital returns shown in the data.\n"
        "• 🟡 **Things to Monitor**: 1-2 concise bullet points highlighting key operational risks, cost pressures, or trends to watch.\n"
        "• 📌 **Key Takeaway**: One sentence summarizing the overall implication for shareholders."
    )

    user_prompt = f"""{xbrl_section}CONTEXT FROM FINANCIAL REPORT:
{context}

RETAIL INVESTOR QUESTION:
{query}

ANALYST RESPONSE:"""

    # --- LLM call with timing ---
    with Timer() as llm_timer:
      answer = self._call_llm(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ],
          temperature=0.1,
      )

    if not answer or not answer.strip():
        answer = "I was unable to generate a response. Please try rephrasing your question or check that the correct document is selected."
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

  def generate_answer_stream(
      self,
      query: str,
      top_k: int = 3,
      paper_id=None,
      ticker: Optional[str] = None,
      fiscal_period: Optional[str] = None,
      session_id: Optional[str] = None,
      sources_out: Optional[list] = None,
  ):
      """
      Streaming version of generate_answer.
      Yields LLM tokens live. If sources_out is provided, it is populated
      with the retrieved source chunks so the caller can display citations.
      """
      import re
      active_ticker = ticker
      active_period = fiscal_period

      if paper_id:
          inferred = parse_filing_metadata(paper_id)
          if not active_ticker:
              active_ticker = inferred.get("ticker")
          if not active_period:
              active_period = inferred.get("fiscal_period")

      if not active_ticker:
          q_lower = query.lower()
          company_hints = {
              "boeing": "BA", " ba ": "BA",
              "jpmorgan": "JPM", "jp morgan": "JPM", "jpmc": "JPM", " jpm ": "JPM",
              "tesla": "TSLA", " tsla ": "TSLA",
              "apple": "AAPL", " aapl ": "AAPL",
              "microsoft": "MSFT", " msft ": "MSFT",
              "google": "GOOGL", "alphabet": "GOOGL", " googl ": "GOOGL",
              "amazon": "AMZN", " amzn ": "AMZN",
              "nvidia": "NVDA", " nvda ": "NVDA",
          }
          for hint, t_code in company_hints.items():
              if hint in q_lower:
                  active_ticker = t_code
                  break

      if not active_period:
          fy_match = re.search(r'\b(fy\s*20\d\d|20\d\d|q[1-4]\s*fy\s*20\d\d)\b', query.lower())
          if fy_match:
              raw_period = fy_match.group(1).replace(" ", "").upper()
              active_period = f"FY{raw_period}" if raw_period.isdigit() else raw_period

      # Layer 1: Pre-Retrieval Abstention Gate
      abstention = check_pre_retrieval_abstention(query, ticker=active_ticker, fiscal_period=active_period)
      if abstention:
          if sources_out is not None:
              sources_out.clear()
          yield abstention["reason"]
          return

      # Retrieval
      chunks = self.retriever.retrieve(
          query=query, top_k=top_k, paper_id=paper_id,
          ticker=active_ticker, fiscal_period=active_period,
          statement_scope="consolidated", session_id=session_id,
      )
      if not chunks and active_ticker and not paper_id:
          chunks = self.retriever.retrieve(query=query, top_k=top_k, session_id=session_id)

      if not chunks:
          if sources_out is not None:
              sources_out.clear()
          yield "No relevant financial documents found in the database matching your search."
          return

      if sources_out is not None:
          sources_out.extend(chunks)

      context = self.format_context(chunks)

      if not active_ticker and chunks:
          active_ticker = parse_filing_metadata(
              chunks[0]["metadata"].get("paper_id", "")
          ).get("ticker")

      xbrl_block = self._get_xbrl_context(
          ticker=active_ticker, fiscal_period=active_period, query=query
      )
      xbrl_section = f"{xbrl_block}\n\n" if xbrl_block else ""

      system_prompt = (
          "You are an expert Senior Financial Analyst specialized in explaining corporate financial reports (10-K, 10-Q, quarterly earnings) to retail investors.\n\n"
          "CORE REPORTING & ACCURACY RULES:\n"
          "1. GROUNDING & EVIDENCE: Answer clearly, accurately, and concisely using ONLY the provided CONTEXT. Never speculate or fabricate.\n"
          "2. STANDALONE VS. CONSOLIDATED: Clearly state the reporting scope. Annual 10-K filings report consolidated corporate operations. Label group figures as '[Consolidated]' (e.g. '[Consolidated] Total Net Sales: $391.04B'). If parent-only, label as '[Standalone]'.\n"
          "3. XBRL VERIFIED FACTS: If STRUCTURED FINANCIAL DATA is provided at the top, treat those numbers as audited ground truth directly from SEC XBRL filings.\n"
          "4. ABSTENTION PROTOCOL: If the context does not contain the answer (e.g. non-disclosed divisional headcounts, daily stock prices, competitor financials, or forward-looking targets), state: 'The provided financial reports do not disclose [topic].' Do not guess.\n"
          "5. CITATIONS: Always cite specific page or section numbers using [Page X] notation whenever citing metrics or statements.\n\n"
          "RETAIL INVESTOR ANALYSIS SECTION:\n"
          "Unless you are abstaining/refusing due to missing data, conclude your answer with an investor analysis section formatted as follows:\n\n"
          "### 💡 What Does This Mean for an Investor?\n"
          "• 🟢 **Positive Signals**: 1-2 concise bullet points highlighting financial strengths, margin expansion, or capital returns shown in the data.\n"
          "• 🟡 **Things to Monitor**: 1-2 concise bullet points highlighting key operational risks, cost pressures, or trends to watch.\n"
          "• 📌 **Key Takeaway**: One sentence summarizing the overall implication for shareholders."
      )
      user_prompt = (
          f"{xbrl_section}CONTEXT FROM FINANCIAL REPORT:\n{context}\n\n"
          f"RETAIL INVESTOR QUESTION:\n{query}\n\nANALYST RESPONSE:"
      )

      yield from self._call_llm_stream(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ],
          temperature=0.1,
      )

  def generate_standard_report(
      self, paper_id=None,
      ticker: Optional[str] = None, fiscal_period: Optional[str] = None,
      session_id: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Generates a comprehensive 5-part standard retail investor financial report."""
    total_start = time.perf_counter()

    active_ticker = ticker
    active_period = fiscal_period
    if paper_id:
        inferred = parse_filing_metadata(paper_id)
        if not active_ticker:
            active_ticker = inferred.get("ticker")
        if not active_period:
            active_period = inferred.get("fiscal_period")

    report_query = (
        "Provide a comprehensive financial summary including revenue growth, net income, "
        "profitability margins, free cash flow, major risk factors, and overall strategic guidance."
    )

    with Timer() as retrieval_timer:
      chunks = self.retriever.retrieve(
          query=report_query,
          top_k=6,
          paper_id=paper_id,
          ticker=active_ticker,
          fiscal_period=active_period,
          session_id=session_id,
      )

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
      answer = self._call_llm(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ],
          temperature=0.2,
      )
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

  def generate_standard_report_stream(
      self,
      paper_id=None,
      ticker: Optional[str] = None,
      fiscal_period: Optional[str] = None,
      session_id: Optional[str] = None,
      sources_out: Optional[list] = None,
  ):
      """
      Streaming version of generate_standard_report.
      Yields LLM tokens live. Populates sources_out if provided.
      """
      active_ticker = ticker
      active_period = fiscal_period
      if paper_id:
          inferred = parse_filing_metadata(paper_id)
          if not active_ticker:
              active_ticker = inferred.get("ticker")
          if not active_period:
              active_period = inferred.get("fiscal_period")

      report_query = (
          "Provide a comprehensive financial summary including revenue growth, net income, "
          "profitability margins, free cash flow, major risk factors, and overall strategic guidance."
      )

      chunks = self.retriever.retrieve(
          query=report_query, top_k=6, paper_id=paper_id,
          ticker=active_ticker, fiscal_period=active_period, session_id=session_id,
      )

      if not chunks:
          if sources_out is not None:
              sources_out.clear()
          yield "No financial documents found to generate a report. Please upload a PDF or HTML report first."
          return

      if sources_out is not None:
          sources_out.extend(chunks)

      context = self.format_context(chunks)
      xbrl_block = self._get_xbrl_context(ticker=active_ticker, fiscal_period=active_period)
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
          "If STRUCTURED FINANCIAL DATA is provided at the top, use those exact figures in "
          "sections 2 and 3 — they are verified XBRL values from official filings.\n"
          "Base all other statements strictly on the provided CONTEXT and cite pages/sections as [Page X]."
      )
      user_prompt = (
          f"{xbrl_section}CONTEXT FROM FINANCIAL FILING:\n{context}\n\n"
          "Generate the Standard Financial Analysis Report now."
      )

      yield from self._call_llm_stream(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ],
          temperature=0.2,
      )


if __name__ == "__main__":
  generator = RAGGenerator()
  print("Generator initialized successfully.")
