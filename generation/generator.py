import os
from pathlib import Path
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv
from groq import Groq


sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retriever import Retriever


load_dotenv()


class RAGGenerator:

  def __init__(
      self,
      retriever: Retriever | None = None,
      model_name: str = "openai/gpt-oss-20b"
  ):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
      raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    self.retriever = retriever or Retriever()
    self.model_name = model_name
    self.client = Groq(api_key=api_key)

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

  def generate_answer(
      self, query: str, top_k: int = 3, paper_id: str | None = None
  ) -> Dict[str, Any]:
    """Retrieves relevant chunks and generates a grounded response with page citations."""
    chunks = self.retriever.retrieve(query=query, top_k=top_k, paper_id=paper_id)

    if not chunks:
      return {
          "answer": "No relevant financial documents found in the database matching your search.",
          "sources": [],
      }

    context = self.format_context(chunks)

    system_prompt = (
        "You are an expert Senior Financial Analyst specialized in explaining corporate financial reports (10-K, 10-Q, quarterly earnings) to retail investors.\n"
        "Rules:\n"
        "1. Answer clearly, accurately, and concisely using ONLY the provided CONTEXT.\n"
        "2. Explain complex financial jargon (e.g. EBITDA, Free Cash Flow, Diluted EPS) in accessible terms for retail investors when relevant.\n"
        "3. If the context does not contain the answer, state: 'I cannot find sufficient information in the indexed financial report.'\n"
        "4. Always cite specific page or section numbers using [Page X] notation whenever citing metrics or statements."
    )

    user_prompt = f"""CONTEXT FROM FINANCIAL REPORT:
{context}

RETAIL INVESTOR QUESTION:
{query}

ANALYST RESPONSE:"""

    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }

  def generate_standard_report(self, paper_id: str | None = None) -> Dict[str, Any]:
    """Generates a comprehensive 5-part standard retail investor financial report."""
    report_query = (
        "Provide a comprehensive financial summary including revenue growth, net income, "
        "profitability margins, free cash flow, major risk factors, and overall strategic guidance."
    )
    # Fetch top 6 chunks for broad document coverage
    chunks = self.retriever.retrieve(query=report_query, top_k=6, paper_id=paper_id)

    if not chunks:
      return {
          "answer": "No financial documents found to generate a report. Please upload a PDF or HTML report first.",
          "sources": [],
      }

    context = self.format_context(chunks)

    system_prompt = (
        "You are a Senior Retail Financial Analyst.\n"
        "Generate a structured, professional Standard Financial Analysis Report formatted in clean Markdown.\n"
        "Your report MUST include the following 5 sections:\n"
        "### 1. Executive Summary & Core Highlights\n"
        "### 2. Revenue & Earnings Performance\n"
        "### 3. Balance Sheet & Cash Flow Health\n"
        "### 4. Key Risk Factors & Market Headwinds\n"
        "### 5. Retail Investor Takeaway\n\n"
        "Base all statements strictly on the provided CONTEXT and cite pages/sections as [Page X]."
    )

    user_prompt = f"""CONTEXT FROM FINANCIAL FILING:
{context}

Generate the Standard Financial Analysis Report now."""

    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


if __name__ == "__main__":
  generator = RAGGenerator()
  print("Generator initialized successfully.")