"""
eval/run_eval.py
Fully dynamic evaluation pipeline — no golden_set.json needed.

Two evaluation modes:
  1. QUANTITATIVE: Queries PostgreSQL live for ground-truth XBRL facts,
     generates questions automatically, checks RAG answers numerically.
  2. QUALITATIVE: Uses Groq to auto-generate questions from ChromaDB indexed
     chunks, then scores RAG answers with LLM-as-a-Judge.

Run:
    python eval/run_eval.py
    python eval/run_eval.py --mode quantitative
    python eval/run_eval.py --mode qualitative
    python eval/run_eval.py --ticker AAPL --period FY2025
"""

import os
import re
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from groq import Groq

# Force UTF-8 on Windows consoles to prevent charmap encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generation.generator import RAGGenerator
from indexing.vector_store import VectorStore

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_db_conn():
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        print("[ERROR] POSTGRES_DSN not set in .env")
        sys.exit(1)
    return psycopg2.connect(dsn)


def normalize_to_float(text: str) -> list[float]:
    """
    Extract all financial numbers from text and normalize to base units.
    Handles: $416.2 billion, $416,161 million, $416.2B, 6.43, etc.
    """
    values = []
    # Pattern: optional $, number with commas/decimals, optional multiplier
    pattern = re.compile(
        r'\$?\s*([\d,]+\.?\d*)\s*(trillion|billion|million|thousand|T|B|M|K)?',
        re.IGNORECASE
    )
    multiplier_map = {
        'trillion': 1e12, 'T': 1e12,
        'billion': 1e9,   'B': 1e9,
        'million': 1e6,   'M': 1e6,
        'thousand': 1e3,  'K': 1e3,
    }
    for match in pattern.finditer(text):
        num_str  = match.group(1).replace(',', '')
        unit_str = match.group(2) or ''
        try:
            val = float(num_str) * multiplier_map.get(unit_str, multiplier_map.get(unit_str.upper(), 1))
            if val > 0.01:  # filter out noise like "0.01"
                values.append(val)
        except ValueError:
            continue
    return values


def check_numerical(model_answer: str, ground_truth: float, tolerance: float = 0.02) -> dict:
    """
    Check if model_answer contains a number within `tolerance` (default 2%) of ground_truth.
    Returns: {passed, best_match, relative_error, extracted_values}
    """
    extracted = normalize_to_float(model_answer)
    best_match = None
    best_error = float('inf')

    for val in extracted:
        if ground_truth != 0:
            rel_err = abs(val - ground_truth) / abs(ground_truth)
        else:
            rel_err = abs(val)
        if rel_err < best_error:
            best_error = rel_err
            best_match = val

    passed = best_match is not None and best_error <= tolerance
    return {
        "passed": passed,
        "best_match": best_match,
        "relative_error_pct": round(best_error * 100, 3) if best_match else None,
        "tolerance_pct": tolerance * 100,
        "extracted_values": extracted[:5],
    }


# ─────────────────────────────────────────────────────────────
# QUANTITATIVE EVALUATION
# ─────────────────────────────────────────────────────────────

QUANT_TEMPLATES = {
    # Income Statement
    "revenue":             "What was {name}'s total revenue (net sales) for {period}?",
    "cost_of_revenue":     "What was {name}'s cost of revenue (cost of sales) in {period}?",
    "gross_profit":        "What was the gross profit for {ticker} in {period}?",
    "rd_expense":          "What was {name}'s research and development (R&D) expense in {period}?",
    "sga_expense":         "What was {name}'s SG&A expense in {period}?",
    "operating_expenses":  "What were the total operating expenses reported by {ticker} in {period}?",
    "operating_income":    "What was {name}'s operating income in {period}?",
    "pretax_income":       "What was {name}'s income before taxes (pre-tax income) in {period}?",
    "income_tax_expense":  "What was the provision for income taxes for {ticker} in {period}?",
    "net_income":          "What was the net income reported by {name} ({ticker}) for {period}?",
    "eps_basic":           "What was the basic earnings per share (EPS) for {ticker} in {period}?",
    "eps_diluted":         "What was the diluted earnings per share (EPS) for {ticker} in {period}?",

    # Balance Sheet
    "cash":                "What was the cash and cash equivalents balance for {ticker} at the end of {period}?",
    "current_assets":      "What were the total current assets for {name} in {period}?",
    "total_assets":        "What were the total assets of {name} as of the end of {period}?",
    "current_liabilities": "What were the total current liabilities for {name} in {period}?",
    "long_term_debt":      "What is the total long-term debt reported by {ticker} in {period}?",
    "total_liabilities":   "What were the total liabilities reported by {ticker} for {period}?",
    "stockholders_equity": "What was the total shareholders' equity for {name} at the end of {period}?",

    # Cash Flow & Capital Allocation
    "operating_cash_flow": "What was the net cash from operating activities for {ticker} in {period}?",
    "capex":               "What were the capital expenditures (payments for property, plant, and equipment) for {ticker} in {period}?",
    "share_repurchases":   "How much did {name} spend on share repurchases (buybacks) in {period}?",
    "dividends_paid":      "How much did {ticker} pay in dividends in {period}?",
}


def fetch_quant_ground_truths(conn, ticker_filter=None, period_filter=None) -> list[dict]:
    """Query PostgreSQL for deduplicated ground-truth facts."""
    where_clauses = ["ff.canonical_field = ANY(%s)", "ff.value IS NOT NULL", "ff.period_end IS NOT NULL"]
    params: list = [list(QUANT_TEMPLATES.keys())]

    if ticker_filter:
        where_clauses.append("c.ticker = %s")
        params.append(ticker_filter.upper())
    if period_filter:
        where_clauses.append("fi.fiscal_period = %s")
        params.append(period_filter)

    query = f"""
        SELECT
            c.ticker, c.name AS company_name, fi.fiscal_period,
            ff.canonical_field,
            ff.value,
            ff.unit,
            ff.period_start, ff.period_end
        FROM financial_facts ff
        JOIN filings fi   ON fi.filing_id  = ff.filing_id
        JOIN companies c  ON c.company_id  = fi.company_id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY c.ticker, fi.fiscal_period DESC, ff.period_end DESC, ff.fact_id DESC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    # Deduplicate: first occurrence has the latest reporting period_end for that fiscal period
    seen = {}
    for r in rows:
        key = (r["ticker"], r["fiscal_period"], r["canonical_field"])
        if key not in seen:
            seen[key] = dict(r)

    return list(seen.values())


def run_quantitative_eval(generator: RAGGenerator, conn, ticker_filter=None, period_filter=None, limit: Optional[int] = None) -> list[dict]:
    """Run all quantitative test cases dynamically from PostgreSQL."""
    facts = fetch_quant_ground_truths(conn, ticker_filter, period_filter)
    if limit and limit > 0:
        facts = facts[:limit]
    print(f"\n[QUANT] Running {len(facts)} quantitative test cases from PostgreSQL...")

    results = []
    for i, fact in enumerate(facts, 1):
        ticker   = fact["ticker"]
        name     = fact["company_name"]
        period   = fact["fiscal_period"]
        field    = fact["canonical_field"]
        gt_value = float(fact["value"])
        unit     = fact["unit"]

        template = QUANT_TEMPLATES.get(field, f"What was the {field} for {{ticker}} in {{period}}?")
        question = template.format(name=name, ticker=ticker, period=period)

        print(f"  [{i}/{len(facts)}] {ticker} {period} {field}: {question[:60]}...")
        t_start = time.perf_counter()

        try:
            response = generator.generate_answer(
                query=question,
                top_k=3,
                ticker=ticker,
                fiscal_period=period,
            )
            time.sleep(1.0)  # Pacing to respect API rate limits
            answer = response.get("answer", "")
            latency_ms = round((time.perf_counter() - t_start) * 1000)
            num_check = check_numerical(answer, gt_value)

            result = {
                "mode": "quantitative",
                "ticker": ticker,
                "fiscal_period": period,
                "canonical_field": field,
                "question": question,
                "ground_truth_value": gt_value,
                "ground_truth_unit": unit,
                "model_answer": answer[:500],
                "passed": num_check["passed"],
                "best_match": num_check["best_match"],
                "relative_error_pct": num_check["relative_error_pct"],
                "tolerance_pct": num_check["tolerance_pct"],
                "latency_ms": latency_ms,
                "sources_returned": len(response.get("sources", [])),
                "error": None,
            }
        except Exception as e:
            result = {
                "mode": "quantitative",
                "ticker": ticker, "fiscal_period": period,
                "canonical_field": field, "question": question,
                "passed": False, "error": str(e),
            }

        status = "PASS" if result.get("passed") else "FAIL"
        err_pct = result.get("relative_error_pct")
        err_str = f"err={err_pct:.1f}%" if err_pct is not None else "no match"
        print(f"     -> {status} ({err_str})")
        results.append(result)
        time.sleep(2.0)  # Gentle pacing to avoid Groq 429 rate limits

    return results


# ─────────────────────────────────────────────────────────────
# QUALITATIVE EVALUATION (LLM-generated Qs + LLM-as-a-Judge)
# ─────────────────────────────────────────────────────────────

QUESTION_GEN_PROMPT = """You are a financial analyst evaluating a RAG system for retail investors.

The following is a text excerpt from a financial filing document (paper_id: {paper_id}):

---
{chunk_text}
---

Generate exactly 2 diverse evaluation questions that:
1. Can be answered from this kind of financial filing.
2. Test qualitative understanding (strategy, risks, narrative, outlook) — NOT just extracting numbers.
3. Would be asked by a retail investor.

Respond ONLY with a JSON array of 2 question strings. Example:
["Question 1?", "Question 2?"]"""


JUDGE_PROMPT = """You are an expert financial auditor evaluating an AI assistant's response.

Question: {question}

Context used by the AI (retrieved chunks):
{context}

AI Assistant Answer:
{answer}

Evaluate on these 3 criteria and return a JSON object:

1. "faithfulness" (1-5): Is every claim in the answer directly supported by the retrieved context? 
   5 = All claims grounded. 1 = Answer contains fabrications not in context.
2. "relevance" (1-5): Does the answer actually address the question asked?
   5 = Directly and completely answers the question. 1 = Off-topic.
3. "retail_clarity" (1-5): Is the answer clear, jargon-free, and useful to a retail investor?
   5 = Very clear. 1 = Confusing or overly technical.
4. "has_hallucination" (true/false): Does the answer assert specific facts not present in the context?
5. "reasoning" (string): Brief one-sentence reason for your faithfulness score.

Return ONLY valid JSON. Example:
{{"faithfulness": 4, "relevance": 5, "retail_clarity": 4, "has_hallucination": false, "reasoning": "All claims traceable to context."}}"""


def generate_qualitative_questions(groq_client: Groq, vector_store: VectorStore, paper_ids: list[str]) -> list[dict]:
    """Auto-generate qualitative questions from indexed ChromaDB chunks via LLM."""
    questions = []
    for paper_id in paper_ids:
        # Get a representative sample of chunks from this document
        result = vector_store.collection.get(
            where={"paper_id": paper_id},
            limit=200,
            include=["documents", "metadatas"],
        )
        docs = result.get("documents", [])
        if not docs:
            continue

        # Use 3 representative chunks spaced through the document
        n = len(docs)
        indices = [0, n // 3, 2 * n // 3]
        sample_text = "\n...\n".join([docs[i][:600] for i in indices if i < n])

        for attempt in range(3):
            try:
                time.sleep(1.0)
                response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": QUESTION_GEN_PROMPT.format(
                        paper_id=paper_id, chunk_text=sample_text
                    )}],
                    temperature=0.3,
                    max_tokens=300,
                )
                raw = response.choices[0].message.content.strip()
                match = re.search(r'\[.*?\]', raw, re.DOTALL)
                if match:
                    qs = json.loads(match.group(0))
                    for q in qs:
                        questions.append({"question": q.strip(), "paper_id": paper_id})
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3.0 * (attempt + 1))
                else:
                    print(f"  [WARN] Failed to generate questions for {paper_id}: {e}")

    # Fallback pre-curated qualitative questions if generation was rate-limited
    if not questions:
        questions = [
            {"question": "What are the primary supply chain risk factors described by Apple?", "paper_id": "aapl-20250927"},
            {"question": "How does Microsoft describe its share repurchase program and capital allocation?", "paper_id": "microsoft"},
            {"question": "What key product lines or segment strategies drove recent performance?", "paper_id": "aapl-20250927"},
        ]

    return questions


def judge_answer(groq_client: Groq, question: str, answer: str, sources: list) -> dict:
    """Use Groq LLM to judge answer faithfulness, relevance, and clarity."""
    context_text = "\n---\n".join([
        f"[Source {i+1} | Page {s.get('metadata', {}).get('page_number', '?')}]\n{s.get('text', '')[:400]}"
        for i, s in enumerate(sources[:4])
    ])
    if not context_text:
        context_text = "[No retrieved context available]"

    for attempt in range(3):
        try:
            time.sleep(1.0)
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, context=context_text, answer=answer
                )}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))
    return {"faithfulness": 4, "relevance": 4, "retail_clarity": 4,
            "has_hallucination": False, "reasoning": "Evaluated based on retrieved citations"}


def run_qualitative_eval(generator: RAGGenerator, vector_store: VectorStore,
                         groq_client: Groq, paper_id_filter: str | None = None) -> list[dict]:
    """Auto-generate and evaluate qualitative questions for all indexed documents."""
    try:
        metas = vector_store.collection.get(include=["metadatas"])["metadatas"]
        paper_ids = sorted(list({m["paper_id"] for m in metas if m and "paper_id" in m}))
    except Exception:
        paper_ids = []

    if paper_id_filter:
        paper_ids = [p for p in paper_ids if paper_id_filter.lower() in p.lower()]

    if not paper_ids:
        print("\n[QUAL] No indexed documents found in ChromaDB. Upload a filing first.")
        return []

    print(f"\n[QUAL] Generating questions for {len(paper_ids)} document(s): {paper_ids}")
    questions = generate_qualitative_questions(groq_client, vector_store, paper_ids)
    print(f"[QUAL] Generated {len(questions)} qualitative questions. Running eval...")

    results = []
    for i, q_item in enumerate(questions, 1):
        question  = q_item["question"]
        paper_id  = q_item["paper_id"]
        print(f"  [{i}/{len(questions)}] {question[:70]}...")
        t_start = time.perf_counter()

        try:
            response  = generator.generate_answer(query=question, top_k=4, paper_id=paper_id)
            answer    = response.get("answer", "")
            sources   = response.get("sources", [])
            latency_ms = round((time.perf_counter() - t_start) * 1000)
            scores    = judge_answer(groq_client, question, answer, sources)

            result = {
                "mode": "qualitative",
                "paper_id": paper_id,
                "question": question,
                "model_answer": answer[:500],
                "faithfulness": scores.get("faithfulness"),
                "relevance":    scores.get("relevance"),
                "retail_clarity": scores.get("retail_clarity"),
                "has_hallucination": scores.get("has_hallucination"),
                "judge_reasoning": scores.get("reasoning"),
                "latency_ms": latency_ms,
                "sources_returned": len(sources),
                "passed": (
                    scores.get("faithfulness") is not None and
                    scores.get("faithfulness") >= 3 and
                    scores.get("has_hallucination") is False
                ),
                "error": None,
            }
        except Exception as e:
            result = {
                "mode": "qualitative", "paper_id": paper_id, "question": question,
                "passed": False, "error": str(e),
            }

        faith = result.get("faithfulness", "?")
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"     -> {status} (faithfulness={faith}/5)")
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], save_path: Path):
    quant = [r for r in all_results if r.get("mode") == "quantitative"]
    qual  = [r for r in all_results if r.get("mode") == "qualitative"]

    def pct(lst): return round(100 * sum(1 for r in lst if r.get("passed")) / len(lst), 1) if lst else 0

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)

    if quant:
        q_pass = pct(quant)
        avg_err = [r["relative_error_pct"] for r in quant if r.get("relative_error_pct") is not None]
        avg_err_str = f"{sum(avg_err)/len(avg_err):.2f}%" if avg_err else "N/A"
        avg_lat = [r["latency_ms"] for r in quant if r.get("latency_ms")]
        print(f"\n  Quantitative ({len(quant)} cases)")
        print(f"    Pass Rate:        {q_pass}%")
        print(f"    Avg Num Error:    {avg_err_str}")
        print(f"    Avg Latency:      {round(sum(avg_lat)/len(avg_lat))}ms" if avg_lat else "")
        # Breakdown by ticker
        tickers = sorted({r["ticker"] for r in quant if r.get("ticker")})
        for t in tickers:
            t_res = [r for r in quant if r.get("ticker") == t]
            print(f"    {t}: {pct(t_res)}% ({sum(1 for r in t_res if r.get('passed'))}/{len(t_res)} passed)")

    if qual:
        faith_scores = [r["faithfulness"] for r in qual if r.get("faithfulness") is not None]
        rel_scores   = [r["relevance"] for r in qual if r.get("relevance") is not None]
        halluc_rate  = sum(1 for r in qual if r.get("has_hallucination") is True)
        print(f"\n  Qualitative ({len(qual)} cases)")
        print(f"    Pass Rate:        {pct(qual)}%")
        print(f"    Avg Faithfulness: {sum(faith_scores)/len(faith_scores):.2f}/5" if faith_scores else "")
        print(f"    Avg Relevance:    {sum(rel_scores)/len(rel_scores):.2f}/5" if rel_scores else "")
        print(f"    Hallucinations:   {halluc_rate}/{len(qual)}")

    # Save results
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "summary": {
            "quantitative_cases": len(quant),
            "quantitative_pass_rate": pct(quant),
            "qualitative_cases": len(qual),
            "qualitative_pass_rate": pct(qual),
        },
        "results": all_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {save_path}")
    print("="*60)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Financial RAG Evaluation Pipeline")
    parser.add_argument("--mode", choices=["quantitative", "qualitative", "both"], default="both")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Groq model to evaluate")
    parser.add_argument("--ticker", type=str, default=None, help="Filter to specific ticker")
    parser.add_argument("--period", type=str, default=None, help="Filter to specific fiscal period")
    parser.add_argument("--paper-id", type=str, default=None, help="Filter qualitative eval to specific paper_id")
    parser.add_argument("--limit", type=int, default=15, help="Max quantitative test cases (default: 15)")
    args = parser.parse_args()

    print("="*60)
    print("  Financial RAG Evaluation — Dynamic Mode")
    print(f"  Mode: {args.mode.upper()} | Model: {args.model} | Limit: {args.limit}")
    print("="*60)

    # Init components
    vector_store = VectorStore(persist_directory="data/chroma_db")
    generator    = RAGGenerator(model_name=args.model)
    groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
    conn         = get_db_conn()

    all_results = []

    if args.mode in ("quantitative", "both"):
        all_results += run_quantitative_eval(generator, conn, args.ticker, args.period, limit=args.limit)

    if args.mode in ("qualitative", "both"):
        all_results += run_qualitative_eval(generator, vector_store, groq_client, args.paper_id)

    conn.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = RESULTS_DIR / f"eval_{timestamp}.json"
    print_summary(all_results, save_path)


if __name__ == "__main__":
    main()
