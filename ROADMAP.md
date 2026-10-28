# Project Roadmap & Engineering Goals

This document outlines the strategic engineering roadmap for **Financial Report RAG**, structured according to industry standards with clear prioritization (**P0: Near-Term / Critical**, **P1: Mid-Term / High Impact**, and **P2: Long-Term / Enterprise Hardening**).

---

## Priority Matrix Overview

| Priority | Category | Key Objective | Target Outcome |
|---|---|---|---|
| **P0** | **Evaluation & Guardrails** | LLM Judge Calibration & G-Eval Rubric | Eliminate judge verbosity bias and fail-open fallbacks |
| **P0** | **Dataset Expansion** | Multi-Sector Benchmark Scaling (150+ Qs) | Extend test cases across Banking, Retail, & Auto beyond AAPL |
| **P1** | **Ingestion & Parsing** | Layout & Table-Aware Semantic Chunking | Prevent financial tables and balance sheets from mid-row splitting |
| **P1** | **Taxonomy & Ingestion** | Dynamic XBRL Extension Tag Handling | Ingest custom company-specific SEC extension tags & IFRS metrics |
| **P1** | **Search Optimization** | Hybrid Search (BM25 Sparse + Dense RRF) | Guarantee exact-match retrieval for financial acronyms & covenants |
| **P2** | **LLMOps & CI/CD** | Automated Evaluation PR Gating | Run automated regression test suites on GitHub Actions |
| **P2** | **Production Reliability** | Multi-Provider Fallback & Local Model Serving | Resilient routing across Groq, Claude, OpenAI, and local Ollama |

---

## P0: Near-Term & Critical Priorities

### 1. LLM Judge Calibration & Bias Mitigation
* **Context**: Current qualitative evaluation (`eval/run_eval.py`) relies on single-turn LLM judge calls (`openai/gpt-oss-20b` via Groq) with 400-character context truncation and default passing fallbacks.
* **Goals**:
  - [ ] **Implement G-Eval Rubric**: Upgrade judge prompts to use structured Chain-of-Thought (CoT) reasoning. The judge must explicitly cite sentences and state grading rationales *before* generating numerical scores (1–5).
  - [ ] **Full Context Pass-Through**: Remove the arbitrary 400-character chunk truncation (`[:400]`) so the judge assesses complete context without penalizing answers for missing data.
  - [ ] **Fail-Closed Evaluation Error Handling**: Replace default passing scores (`faithfulness: 4`, `has_hallucination: false`) on API timeouts with explicit `ERROR / UNGRADED` flags to avoid false confidence.
  - [ ] **Multi-Judge Consensus**: Form an ensemble evaluation step (e.g., Llama 3.3 70B + Claude 3.5 Sonnet / GPT-4o-mini) for ambiguous or borderline answers, flagging score discrepancies $>1.0$ for inspection.

### 2. Benchmark Dataset Scaling (FinanceBench Alignment)
* **Context**: The existing golden test suite (`eval/golden_set_50.json`) is heavily weighted toward Apple Inc. (`AAPL`) 10-K filings.
* **Goals**:
  - [ ] **5-Sector Benchmark Expansion**: Ingest 10-K filings across 5 distinct sectors to evaluate domain-specific financial structures:
    - **Tech**: Apple (`AAPL`), Microsoft (`MSFT`)
    - **Banking / Financials**: JPMorgan Chase (`JPM`) — testing Net Interest Margin, CET1 capital ratios
    - **Retail / Consumer**: Walmart (`WMT`) — testing inventory turnover and retail square footage metrics
    - **Industrial / Auto**: General Motors (`GM`) — testing vehicle delivery margins and warranty reserves
    - **Energy**: ExxonMobil (`XOM`) — testing upstream/downstream EBITDA and proved reserves
  - [ ] **Integrate FinanceBench Test Cases**: Ingest 30+ standardized questions from the open-source **FinanceBench** benchmark to establish verifiable, reproducible baseline comparisons against top industry RAG architectures.
  - [ ] **Scale to 150+ Structured Questions**: Expand golden test cases to include multi-year historical series (2020–2025) and SEC amendment filings (10-K/A).

---

## P1: Mid-Term & High Impact Priorities

### 3. Layout- & Table-Aware Semantic Chunking
* **Context**: The current ingestion pipeline (`ingestion/pdf_parser.py`) applies a naive sliding window of 500 words with 10% overlap, which risks slicing complex financial tables and footnotes mid-row.
* **Goals**:
  - [ ] **Table-Preserving Chunking**: Integrate layout-aware PDF parsers (e.g., PyMuPDF structure tags or `pdfplumber`) to detect HTML/Markdown tables and isolate them as distinct, indivisible chunks.
  - [ ] **Header Inheritance**: Automatically prepend parent section titles (e.g., *Item 7. MD&A - Liquidity and Capital Resources*) to child chunks to preserve hierarchical context in dense vector space.

### 4. Dynamic XBRL Extension Tag & IFRS Normalization
* **Context**: The `CANONICAL_MAP` covers ~69 standard US-GAAP tags, but public companies often use custom extension tags (e.g., customized segment revenue, adjusted EBITDA), and foreign filers report under IFRS.
* **Goals**:
  - [ ] **Fuzzy & LLM-Assisted Tag Canonicalization**: Implement a fallback classifier that maps unfamiliar SEC XBRL extension tags to canonical financial line items using XBRL calculation linkbases.
  - [ ] **IFRS Taxonomy Support**: Map primary International Financial Reporting Standards (IFRS) tags to unified canonical fields for cross-border comparative analysis.

### 5. Hybrid Search (Sparse BM25 + Dense Vector RRF)
* **Context**: Dense embeddings (`bge-small-en-v1.5`) excel at semantic queries but occasionally miss exact financial jargon, credit covenant formulas (e.g., `SOFR + 150 bps`), or regulatory clause IDs.
* **Goals**:
  - [ ] **Integrate In-Memory BM25**: Implement `rank_bm25` across filing chunks.
  - [ ] **Reciprocal Rank Fusion (RRF)**: Combine dense vector similarity scores with BM25 sparse scores using reciprocal rank fusion ($k=60$) before feeding candidate chunks to the FlashRank cross-encoder.

---

## P2: Long-Term & Enterprise Hardening

### 6. Automated Evaluation & CI/CD Regression Gating
* **Context**: Evaluation scripts are currently executed manually via the command line.
* **Goals**:
  - [ ] **GitHub Actions Automated Eval**: Create a `.github/workflows/eval.yml` action that runs the 50-question smoke test suite on every PR modifying retrieval, ingestion, or generation code.
  - [ ] **PR Regression Thresholds**: Automatically block PR merges if numerical accuracy drops below 95% or hallucination rates exceed 2%.
  - [ ] **LLMOps Platform Integration**: Export evaluation traces and latency distributions directly to **Arize Phoenix**, **Ragas**, or **LangSmith** dashboards.

### 7. Multi-Provider Fallback & Resilient Inference
* **Context**: The generation layer relies entirely on the Groq Cloud API, creating a single point of failure during rate limits or service interruptions.
* **Goals**:
  - [ ] **Automated Provider Failover**: Implement an adapter pattern that falls back from Groq to Anthropic Claude 3.5 Sonnet, OpenAI GPT-4o-mini, or a local Ollama instance on HTTP 429/500 errors.
  - [ ] **Streaming Response Generation**: Support token streaming in the Streamlit UI to reduce perceived time-to-first-token (TTFT) for long reports.

---

## Summary Tracking

- [x] Initial Two-Track Hybrid RAG (XBRL + Dense Vector)
- [x] Canonical Map expansion to 69+ US-GAAP tags
- [x] FlashRank ONNX cross-encoder re-ranking integration
- [x] Multi-tenant Supabase pgvector + local ChromaDB dual backend
- [x] 50-Question Multi-Dimension Evaluation Suite (`golden_set_50.json`)
- [ ] **Phase 1 (Next Up)**: LLM Judge G-Eval calibration + FinanceBench dataset expansion
- [ ] **Phase 2**: Layout-aware table chunking + BM25/RRF hybrid retrieval
- [ ] **Phase 3**: Automated GitHub Actions eval gating + LLMOps dashboard tracking
