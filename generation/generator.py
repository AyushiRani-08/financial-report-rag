import os
import time
import threading
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
from visualization import (
    extract_chart,
    get_display_text,
    CHART_INSTRUCTION,
    enrich_chart_with_xbrl,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Rate Limiter  –  simple token-bucket, one bucket per session_id
# Default: 60 requests per 60-second window (1 req/s burst-free)
# ──────────────────────────────────────────────────────────────
_RATE_LIMIT_CAPACITY = int(os.getenv("GROQ_RATE_LIMIT_RPM", "60"))  # tokens = max requests per window
_RATE_LIMIT_WINDOW   = 60.0  # seconds
_rate_buckets: Dict[str, Dict] = {}  # session_id → {tokens, last_refill}
_rate_lock = threading.Lock()


def _check_rate_limit(session_id: str) -> None:
    """Raise RuntimeError if this session has exceeded the Groq rate limit."""
    if not session_id:
        return
    with _rate_lock:
        now = time.monotonic()
        bucket = _rate_buckets.setdefault(
            session_id, {"tokens": float(_RATE_LIMIT_CAPACITY), "last_refill": now}
        )
        elapsed = now - bucket["last_refill"]
        # Refill proportionally to elapsed time
        refill = elapsed * (_RATE_LIMIT_CAPACITY / _RATE_LIMIT_WINDOW)
        bucket["tokens"] = min(float(_RATE_LIMIT_CAPACITY), bucket["tokens"] + refill)
        bucket["last_refill"] = now

        if bucket["tokens"] < 1.0:
            wait_s = (1.0 - bucket["tokens"]) * (_RATE_LIMIT_WINDOW / _RATE_LIMIT_CAPACITY)
            raise RuntimeError(
                f"Rate limit exceeded for session '{session_id}'. "
                f"Please wait ~{wait_s:.1f}s before sending another request."
            )
        bucket["tokens"] -= 1.0


# ──────────────────────────────────────────────────────────────
# Circuit Breaker  –  one breaker per model name
# Opens after FAILURE_THRESHOLD consecutive failures;
# enters half-open after RECOVERY_TIMEOUT seconds.
# ──────────────────────────────────────────────────────────────
_CB_FAILURE_THRESHOLD = int(os.getenv("GROQ_CB_FAILURE_THRESHOLD", "5"))
_CB_RECOVERY_TIMEOUT  = float(os.getenv("GROQ_CB_RECOVERY_TIMEOUT", "60"))

class _CircuitState:
    CLOSED    = "CLOSED"     # normal – calls pass through
    OPEN      = "OPEN"       # tripped – calls are blocked
    HALF_OPEN = "HALF_OPEN"  # probing – one call allowed through

_circuit_breakers: Dict[str, Dict] = {}
_cb_lock = threading.Lock()


def _cb_get(model: str) -> Dict:
    with _cb_lock:
        return _circuit_breakers.setdefault(
            model,
            {"state": _CircuitState.CLOSED, "failures": 0, "opened_at": 0.0}
        )


def _cb_allow(model: str) -> bool:
    """Return True if the circuit allows a call to model."""
    cb = _cb_get(model)
    with _cb_lock:
        if cb["state"] == _CircuitState.CLOSED:
            return True
        if cb["state"] == _CircuitState.OPEN:
            if time.monotonic() - cb["opened_at"] >= _CB_RECOVERY_TIMEOUT:
                cb["state"] = _CircuitState.HALF_OPEN
                logger.info("[CircuitBreaker] %s → HALF_OPEN (probing)", model)
                return True
            return False
        # HALF_OPEN: allow one probe
        return True


def _cb_record_success(model: str) -> None:
    cb = _cb_get(model)
    with _cb_lock:
        cb["failures"] = 0
        if cb["state"] != _CircuitState.CLOSED:
            logger.info("[CircuitBreaker] %s → CLOSED (recovered)", model)
        cb["state"] = _CircuitState.CLOSED


def _cb_record_failure(model: str) -> None:
    cb = _cb_get(model)
    with _cb_lock:
        cb["failures"] += 1
        if cb["state"] == _CircuitState.HALF_OPEN or cb["failures"] >= _CB_FAILURE_THRESHOLD:
            cb["state"]     = _CircuitState.OPEN
            cb["opened_at"] = time.monotonic()
            logger.warning(
                "[CircuitBreaker] %s → OPEN after %d failure(s). Will retry in %.0fs.",
                model, cb["failures"], _CB_RECOVERY_TIMEOUT
            )


def _get_secret(section: str, key: str) -> str:
    """Read a value from st.secrets (Streamlit Cloud), returning '' on any failure."""
    try:
        import streamlit as st
        return st.secrets[section][key] or ""
    except Exception:
        return ""


class RAGGenerator:

  def __init__(
      self,
      retriever=None,
      model_name: str = "openai/gpt-oss-20b",
      xbrl_retriever: Optional[XBRLRetriever] = None,
  ):
    api_key = os.getenv("GROQ_API_KEY") or _get_secret("app", "groq_api_key")
    if not api_key:
      raise ValueError("GROQ_API_KEY not found. Set GROQ_API_KEY env var or add groq_api_key under [app] in .streamlit/secrets.toml.")

    self.retriever = retriever or Retriever()
    self.model_name = model_name
    self.client = Groq(api_key=api_key, timeout=120.0)

    # Optional: structured XBRL retriever backed by PostgreSQL.
    # Auto-initializes if POSTGRES_DSN is set; silently disabled otherwise.
    if xbrl_retriever is not None:
      self.xbrl_retriever: Optional[XBRLRetriever] = xbrl_retriever
    elif os.getenv("POSTGRES_DSN") or _get_secret("postgres", "dsn"):
      try:
        self.xbrl_retriever = XBRLRetriever()
      except Exception:
        self.xbrl_retriever = None
    else:
      self.xbrl_retriever = None

  # Models tried in order when the primary is at capacity, times out, or fails
  _FALLBACK_MODELS = [
      "qwen/qwen3.8-27b",
      "openai/gpt-oss-120b",
      "llama-3.3-70b-versatile",
      "llama-3.1-8b-instant",
  ]

  def _call_llm(
      self,
      messages: list,
      temperature: float = 0.1,
      max_retries: int = 3,
      session_id: Optional[str] = None,
  ) -> str:
      """
      Wraps client.chat.completions.create with:
      • Per-session token-bucket rate limiting (60 req/min by default)
      • Per-model circuit breaker (opens after 5 failures, resets after 60s)
      • Exponential-backoff retry (3 attempts, 2s / 4s / 8s)
      • Automatic fallback to alternative models on 503 / 429 / timeout

      Returns the final answer string.
      """
      # Guard: enforce per-session rate limit before any API call
      _check_rate_limit(session_id or "")

      models_to_try = [self.model_name] + [
          m for m in self._FALLBACK_MODELS if m != self.model_name
      ]

      last_exc = None
      for model in models_to_try:
          # Skip immediately if the circuit is open for this model
          if not _cb_allow(model):
              logger.warning("[CircuitBreaker] %s is OPEN — skipping.", model)
              continue

          for attempt in range(max_retries):
              try:
                  resp = self.client.chat.completions.create(
                      model=model,
                      messages=messages,
                      temperature=temperature,
                      max_tokens=4096,
                  )
                  content = resp.choices[0].message.content
                  _cb_record_success(model)
                  if model != self.model_name:
                      logger.warning(
                          "[Generator] Primary model '%s' unavailable — "
                          "used fallback '%s' (attempt %d)",
                          self.model_name, model, attempt + 1,
                      )
                  return content or ""
              except Exception as exc:
                  last_exc = exc
                  err_str = str(exc).lower()
                  is_recoverable = any(
                      code in err_str
                      for code in ["503", "429", "capacity", "overloaded", "rate", "timeout", "timed out", "connection"]
                  )
                  _cb_record_failure(model)
                  if is_recoverable and attempt < max_retries - 1:
                      wait = 2 ** (attempt + 1)   # 2s, 4s, 8s
                      logger.warning(
                          "[Generator] %s returned retryable error (%s) — retrying in %ds (attempt %d/%d)",
                          model, exc, wait, attempt + 1, max_retries,
                      )
                      time.sleep(wait)
                  else:
                      break   # non-recoverable or exhausted → try next model

      raise RuntimeError(
          f"All models exhausted. Last error: {last_exc}"
      )

  def _call_llm_stream(
      self,
      messages: list,
      temperature: float = 0.1,
      session_id: Optional[str] = None,
  ):
      """
      Streams tokens from the LLM, falling back across models on 503/429/timeout.
      Integrates per-session rate limiting and per-model circuit breaker.
      Yields string token chunks as they arrive from the Groq API.
      """
      # Guard: enforce per-session rate limit before any streaming call
      _check_rate_limit(session_id or "")

      models_to_try = [self.model_name] + [
          m for m in self._FALLBACK_MODELS if m != self.model_name
      ]
      last_exc = None
      for model in models_to_try:
          if not _cb_allow(model):
              logger.warning("[CircuitBreaker] %s is OPEN — skipping (stream).", model)
              continue

          tokens_yielded = False
          try:
              stream = self.client.chat.completions.create(
                  model=model,
                  messages=messages,
                  temperature=temperature,
                  stream=True,
                  max_tokens=4096,
              )
              for chunk in stream:
                  delta = chunk.choices[0].delta.content
                  if delta:
                      tokens_yielded = True
                      yield delta
              _cb_record_success(model)
              return  # streaming finished successfully
          except Exception as exc:
              last_exc = exc
              _cb_record_failure(model)
              err_str = str(exc).lower()
              is_recoverable = any(
                  c in err_str
                  for c in ["503", "429", "capacity", "overloaded", "rate", "timeout", "timed out", "connection"]
              )
              if not tokens_yielded and is_recoverable:
                  logger.warning(
                      "[Generator] %s encountered recoverable error (%s) before streaming — trying fallback.",
                      model, exc,
                  )
                  continue
              raise
      raise RuntimeError(f"All models exhausted during streaming. Last error: {last_exc}")

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
        "• 📌 **Key Takeaway**: One sentence summarizing the overall implication for shareholders.\n\n"
        f"{CHART_INSTRUCTION}"
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

    chart_data = extract_chart(answer)
    if chart_data and self.xbrl_retriever and active_ticker:
        chart_data = enrich_chart_with_xbrl(chart_data, self.xbrl_retriever, active_ticker)

    clean_answer = get_display_text(answer) if chart_data else answer

    # --- Log the query ---
    log_query(
        query=query, query_type="qa", paper_id=paper_id, top_k=top_k,
        num_chunks_retrieved=len(chunks), similarity_scores=similarity_scores,
        retrieval_latency_ms=retrieval_timer.elapsed_ms,
        llm_latency_ms=llm_timer.elapsed_ms,
        total_latency_ms=total_ms,
        model_name=self.model_name, answer_length=len(clean_answer),
    )

    return {"answer": clean_answer, "sources": chunks, "chart_data": chart_data}

  def generate_answer_stream(
      self,
      query: str,
      top_k: int = 3,
      paper_id=None,
      ticker: Optional[str] = None,
      fiscal_period: Optional[str] = None,
      session_id: Optional[str] = None,
      sources_out: Optional[list] = None,
      chart_out: Optional[list] = None,
  ):
      """
      Streaming version of generate_answer.
      Yields clean LLM text tokens live, suppressing any ```chart_data JSON block
      from the user-visible stream while parsing and returning it via chart_out.
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
          "• 📌 **Key Takeaway**: One sentence summarizing the overall implication for shareholders.\n\n"
          f"{CHART_INSTRUCTION}"
      )
      user_prompt = (
          f"{xbrl_section}CONTEXT FROM FINANCIAL REPORT:\n{context}\n\n"
          f"RETAIL INVESTOR QUESTION:\n{query}\n\nANALYST RESPONSE:"
      )

      raw_tokens: List[str] = []
      in_chart = False
      buffer = ""

      for delta in self._call_llm_stream(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ],
          temperature=0.1,
      ):
          raw_tokens.append(delta)
          if in_chart:
              # Resume normal streaming if chart block has closed with ```
              chart_tail = "".join(raw_tokens).split("```chart_data", 1)[-1]
              if "```" in chart_tail:
                  in_chart = False
                  buffer = ""
              continue

          buffer += delta
          if "```chart_data" in buffer:
              parts = buffer.split("```chart_data", 1)
              pre_text = parts[0]
              if pre_text:
                  yield pre_text
              in_chart = True
              buffer = ""
          elif "```" in buffer:
              idx = buffer.find("```")
              after = buffer[idx:]
              if "```chart_data".startswith(after):
                  continue
              else:
                  yield buffer
                  buffer = ""
          else:
              yield buffer
              buffer = ""

      if buffer and not in_chart:
          yield buffer

      # Extract chart and enrich with XBRL
      full_raw = "".join(raw_tokens)
      chart_data = extract_chart(full_raw)
      if chart_data and self.xbrl_retriever and active_ticker:
          chart_data = enrich_chart_with_xbrl(chart_data, self.xbrl_retriever, active_ticker)

      if chart_out is not None and chart_data:
          chart_out.append(chart_data)

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
        "Base all other statements strictly on the provided CONTEXT and cite pages/sections as [Page X].\n\n"
        f"{CHART_INSTRUCTION}"
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

    chart_data = extract_chart(answer)
    if chart_data and self.xbrl_retriever and active_ticker:
        chart_data = enrich_chart_with_xbrl(chart_data, self.xbrl_retriever, active_ticker)

    clean_answer = get_display_text(answer) if chart_data else answer

    log_query(
        query=report_query, query_type="standard_report", paper_id=paper_id, top_k=6,
        num_chunks_retrieved=len(chunks), similarity_scores=similarity_scores,
        retrieval_latency_ms=retrieval_timer.elapsed_ms,
        llm_latency_ms=llm_timer.elapsed_ms,
        total_latency_ms=total_ms,
        model_name=self.model_name, answer_length=len(clean_answer),
    )

    return {"answer": clean_answer, "sources": chunks, "chart_data": chart_data}

  def generate_standard_report_stream(
      self,
      paper_id=None,
      ticker: Optional[str] = None,
      fiscal_period: Optional[str] = None,
      session_id: Optional[str] = None,
      sources_out: Optional[list] = None,
      chart_out: Optional[list] = None,
  ):
      """
      Streaming version of generate_standard_report.
      Yields clean LLM text tokens live, suppressing any ```chart_data JSON block
      from the user-visible stream while parsing and returning it via chart_out.
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
          "Base all other statements strictly on the provided CONTEXT and cite pages/sections as [Page X].\n\n"
          f"{CHART_INSTRUCTION}"
      )
      user_prompt = (
          f"{xbrl_section}CONTEXT FROM FINANCIAL FILING:\n{context}\n\n"
          "Generate the Standard Financial Analysis Report now."
      )

      raw_tokens: List[str] = []
      in_chart = False
      buffer = ""

      for delta in self._call_llm_stream(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ],
          temperature=0.2,
      ):
          raw_tokens.append(delta)
          if in_chart:
              # Resume normal streaming if chart block has closed with ```
              chart_tail = "".join(raw_tokens).split("```chart_data", 1)[-1]
              if "```" in chart_tail:
                  in_chart = False
                  buffer = ""
              continue

          buffer += delta
          if "```chart_data" in buffer:
              parts = buffer.split("```chart_data", 1)
              pre_text = parts[0]
              if pre_text:
                  yield pre_text
              in_chart = True
              buffer = ""
          elif "```" in buffer:
              idx = buffer.find("```")
              after = buffer[idx:]
              if "```chart_data".startswith(after):
                  continue
              else:
                  yield buffer
                  buffer = ""
          else:
              yield buffer
              buffer = ""

      if buffer and not in_chart:
          yield buffer

      # Extract chart and enrich with XBRL
      full_raw = "".join(raw_tokens)
      chart_data = extract_chart(full_raw)
      if chart_data and self.xbrl_retriever and active_ticker:
          chart_data = enrich_chart_with_xbrl(chart_data, self.xbrl_retriever, active_ticker)

      if chart_out is not None and chart_data:
          chart_out.append(chart_data)


if __name__ == "__main__":
  generator = RAGGenerator()
  print("Generator initialized successfully.")
