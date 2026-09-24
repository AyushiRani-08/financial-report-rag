# 📊 Financial Report RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant designed for retail investors to analyze corporate financial filings (`.pdf`, `.html`, `.htm`).

---

## ✨ Features

- **Two-Track Hybrid Ingestion**:
  - **Dense Narrative Track**: Embeds unstructured text chunks via ChromaDB and `BAAI/bge-small-en-v1.5` embeddings for deep qualitative Q&A.
  - **Structured Ground-Truth Track**: Parses SEC XBRL instance documents directly with `lxml` into PostgreSQL across 69+ US-GAAP canonical metrics.
- **High-Precision Q&A**: Zero-hallucination numeric retrieval backed by structured database facts combined with exact `[Page X]` and `[Source N]` text citations.
- **Dynamic Evaluation Test Bench**: Evaluates both quantitative facts (against live PostgreSQL ground truth) and qualitative synthesis (LLM-as-a-Judge), achieving **95.7% pass rate** with **0.40% average numerical error**.
- **1-Click Standard Report**: Generate a 5-point retail investor financial summary.
- **Automated CI**: GitHub Actions workflow verifies code quality and syntax on every push.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Engine**: Groq API (`openai/gpt-oss-20b` / `llama-3.3-70b-versatile`)
- **Vector DB**: ChromaDB (`BAAI/bge-small-en-v1.5`)
- **Structured Database**: PostgreSQL (psycopg2)
- **Document & XBRL Parsers**: `lxml`, PyMuPDF, BeautifulSoup4
- **Evaluation**: Custom dynamic test harness (`eval/run_eval.py`)

---

## 📈 Accuracy & Evaluation Benchmarks

The system is continuously benchmarked using dynamic evaluation against audited SEC filings:

| Benchmark | Result | Description |
|---|---|---|
| **Quantitative Accuracy** | **95.7% (22/23 Passed)** | Tested against ground-truth facts extracted from SEC 10-K instance documents. |
| **Avg Numerical Error** | **0.40%** | Average relative numerical variance on line items (R&D, CapEx, buybacks, taxes, etc.). |
| **Numeric Hallucination** | **0.0%** | Zero hallucinations on mapped XBRL line items via direct PostgreSQL fact injection. |

---

## 🏗️ Architecture & Engineering Log

Detailed documentation on the two-track architecture, data flow, embedding experiments (`bge-small` vs `bge-base`), and how accuracy increased is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🚀 Quickstart

### Option A: Run with Docker Compose (Recommended)

1. **Configure environment**:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY in .env
   ```

2. **Start all services (App + PostgreSQL)**:
   ```bash
   docker compose up --build
   ```
   - Automatically initializes PostgreSQL schema from `db/schema.sql`.
   - Access the dashboard at **`http://localhost:8080`**.
   - PostgreSQL is also accessible at `localhost:5432`.

---

### Option B: Local Python Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AyushiRani-08/paper-rag.git
   cd paper-rag
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Add environment variables**:
   Create a `.env` file from `.env.example`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   POSTGRES_DSN=postgresql://postgres:password@localhost:5432/financial_rag
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

