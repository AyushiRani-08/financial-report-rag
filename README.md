# 📊 Financial Report RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant designed for retail investors to analyze corporate financial filings (`.pdf`, `.html`, `.htm`).

---

## ✨ Features

- **Document Ingestion**: Upload SEC 10-K, 10-Q, annual reports, or HTML filings.
- **1-Click Standard Report**: Generate a 5-point retail investor financial summary.
- **Interactive Q&A**: Ask custom financial questions with exact `[Page X]` citations.
- **Fast Vector Search**: Built with ChromaDB and `BAAI/bge-small-en-v1.5` embeddings.
- **Automated CI**: GitHub Actions workflow verifies code quality on every push.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Engine**: Groq API (`openai/gpt-oss-20b`)
- **Vector DB**: ChromaDB
- **Embeddings**: `BAAI/bge-small-en-v1.5`
- **Parsers**: PyMuPDF, BeautifulSoup4, PyTesseract

---

## 🚀 Quickstart

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
   ```

4. **Add environment variable**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```
