"""
eval/run_eval_50.py
====================
Comprehensive 50-question evaluation pipeline with 5 accuracy dimensions:

  1. RETRIEVAL ACCURACY    — Did the system find the right chunks?
  2. NUMERICAL ACCURACY    — Are the numbers correct within tolerance?
  3. CALCULATION ACCURACY  — Are derived/computed metrics correct?
  4. GROUNDEDNESS ACCURACY — Are claims supported by retrieved context? (LLM-as-Judge)
  5. ABSTENTION ACCURACY   — Does the system correctly refuse to answer
                             when information is unavailable?

Question categories covered (10 types × 5 each = 50 total):
  1. Direct fact retrieval
  2. Year-over-year comparisons
  3. Multi-year trend questions
  4. Cross-company comparisons
  5. Financial ratio / derived metrics
  6. Semantic / paraphrase retrieval
  7. Time-period and date understanding
  8. Multi-hop reasoning
  9. Hallucination / unavailable information
 10. Hard / adversarial questions

Usage:
    python eval/run_eval_50.py
    python eval/run_eval_50.py --limit 10
    python eval/run_eval_50.py --category direct_fact_retrieval
    python eval/run_eval_50.py --dimension abstention_accuracy
"""

import os
import re
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── stdout encoding fix for Windows consoles ──────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from groq import Groq
from generation.generator import RAGGenerator
from indexing.vector_store import VectorStore

load_dotenv()

GOLDEN_SET   = Path(__file__).parent / "golden_set_50.json"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ABSTENTION KEYWORDS — phrases indicating the model declined to answer
# ─────────────────────────────────────────────────────────────────────────────
ABSTENTION_PHRASES = [
    "i don't have", "i do not have", "not available", "not disclosed",
    "not provided", "not found", "cannot find", "cannot determine",
    "no information", "not in the filing", "not in the document",
    "cannot answer", "unable to answer", "not mentioned", "not reported",
    "does not provide", "does not disclose", "not specified", "not stated",
    "outside the scope", "not covered", "insufficient information",
    "no data", "no record", "not accessible", "beyond the information",
    "not part of", "i cannot", "i'm unable", "insufficient context",
    "please note that", "the filing does not",
]

# ─────────────────────────────────────────────────────────────────────────────
# GROUNDEDNESS JUDGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────
GROUNDEDNESS_JUDGE_PROMPT = """\
You are an expert financial auditor evaluating an AI assistant's response for groundedness.

QUESTION: {question}

RETRIEVED CONTEXT (what the AI had access to):
{context}

AI ANSWER:
{answer}

Evaluate the answer on these 4 criteria. Return ONLY valid JSON — no markdown fences, no extra text.

{{
  "groundedness_score": <1-5 integer>,
  "faithfulness_score": <1-5 integer>,
  "relevance_score": <1-5 integer>,
  "has_hallucination": <true|false>,
  "correctly_abstained": <true|false>,
  "reasoning": "<one sentence>"
}}

Scoring guide:
- groundedness_score: 5 = every claim is directly in context; 1 = claims fabricated not in context
- faithfulness_score:  5 = all facts match; 1 = contradicts source
- relevance_score:     5 = fully answers question; 1 = off-topic
- has_hallucination:   true if answer asserts specific facts NOT in the retrieved context
- correctly_abstained: true if the question required abstention AND the answer appropriately declined
"""

# ─────────────────────────────────────────────────────────────────────────────
# NUMBER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_numbers(text: str) -> List[float]:
    """
    Extract all financial numbers from text, normalising to base units.
    Handles: $416.2 billion, $416,161 million, $416.2B, 6.43%, etc.
    """
    MULTIPLIERS = {
        "trillion": 1e12, "t": 1e12,
        "billion":  1e9,  "b": 1e9,
        "million":  1e6,  "m": 1e6,
        "thousand": 1e3,  "k": 1e3,
    }
    pattern = re.compile(
        r"\$?\s*([\d,]+\.?\d*)\s*"
        r"(trillion|billion|million|thousand|[tbmk])?",
        re.IGNORECASE,
    )
    values = []
    for match in pattern.finditer(text):
        raw = match.group(1).replace(",", "")
        unit = (match.group(2) or "").lower()
        try:
            val = float(raw) * MULTIPLIERS.get(unit, 1)
            if val > 0.001:
                values.append(val)
        except ValueError:
            continue
    return values


def check_numerical(answer: str, ground_truth: float, tol_pct: float = 2.0) -> Dict:
    """
    Check if the answer contains a number within tol_pct % of the ground truth.
    Returns a result dict with passed, best_match, relative_error_pct.
    """
    extracted = extract_numbers(answer)
    best_match, best_error = None, float("inf")
    for val in extracted:
        rel_err = abs(val - ground_truth) / abs(ground_truth) if ground_truth != 0 else abs(val)
        if rel_err < best_error:
            best_error, best_match = rel_err, val
    passed = best_match is not None and best_error <= tol_pct / 100
    return {
        "passed":             passed,
        "best_match":         best_match,
        "relative_error_pct": round(best_error * 100, 3) if best_match else None,
        "tolerance_pct":      tol_pct,
        "extracted_values":   extracted[:6],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ABSTENTION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_abstention(answer: str) -> bool:
    """Returns True if the answer appears to abstain / decline."""
    lower = answer.lower()
    return any(phrase in lower for phrase in ABSTENTION_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE (Groundedness)
# ─────────────────────────────────────────────────────────────────────────────

def judge_groundedness(
    groq_client: Groq, question: str, answer: str,
    sources: List[Dict], abstention_expected: bool
) -> Dict:
    """Use Groq as a judge to score groundedness, faithfulness and relevance."""
    context_text = "\n---\n".join([
        f"[Source {i+1} | Page {s.get('metadata', {}).get('page_number', '?')}]\n"
        f"{s.get('text', '')[:500]}"
        for i, s in enumerate(sources[:5])
    ]) or "[No retrieved context available]"

    prompt = GROUNDEDNESS_JUDGE_PROMPT.format(
        question=question, context=context_text, answer=answer
    )

    for attempt in range(3):
        try:
            time.sleep(1.0)
            resp = groq_client.chat.completions.create(
                model=os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=350,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip possible markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"```$", "", raw)
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                scores = json.loads(match.group(0))
                return {
                    "groundedness_score": scores.get("groundedness_score"),
                    "faithfulness_score": scores.get("faithfulness_score"),
                    "relevance_score":    scores.get("relevance_score"),
                    "has_hallucination":  scores.get("has_hallucination", False),
                    "correctly_abstained":scores.get("correctly_abstained", False),
                    "judge_reasoning":    scores.get("reasoning", ""),
                }
        except Exception as exc:
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))
            else:
                print(f"      [JUDGE WARN] {exc}")

    # Fallback if judge fails
    return {
        "groundedness_score": 3, "faithfulness_score": 3, "relevance_score": 3,
        "has_hallucination": False, "correctly_abstained": abstention_expected,
        "judge_reasoning": "Judge unavailable — default scores assigned",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-QUESTION SCORING — 5 ACCURACY DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────

def score_question(
    case: Dict,
    answer: str,
    sources: List[Dict],
    judge_scores: Dict,
) -> Dict[str, Any]:
    """
    Score one question across all 5 evaluation dimensions.

    Returns a dict:
    {
        retrieval_accuracy:    bool | None,
        numerical_accuracy:    bool | None,
        calculation_accuracy:  bool | None,
        groundedness_accuracy: bool | None,
        abstention_accuracy:   bool | None,
    }
    """
    dims = case.get("eval_dimensions", [])
    gt   = case.get("ground_truth_value")
    tol  = case.get("tolerance_pct", 2.0)
    abstention_expected = case.get("abstention_expected", False)
    actually_abstained  = check_abstention(answer)

    result: Dict[str, Any] = {
        "retrieval_accuracy":    None,
        "numerical_accuracy":    None,
        "calculation_accuracy":  None,
        "groundedness_accuracy": None,
        "abstention_accuracy":   None,
    }

    # ── 1. RETRIEVAL ACCURACY ─────────────────────────────────────────────
    # Passes if the system returned at least 1 source AND those sources have
    # reasonable similarity AND (for qualitative) judge relevance >= 3
    if "retrieval_accuracy" in dims:
        has_sources   = len(sources) > 0
        avg_sim       = (
            sum(s.get("similarity_score", 0) for s in sources) / len(sources)
            if sources else 0
        )
        judge_relevant = (judge_scores.get("relevance_score") or 0) >= 3
        # For abstention-expected, retrieval can still pass if sources were found
        result["retrieval_accuracy"] = (
            has_sources and avg_sim >= 0.3 and (judge_relevant or abstention_expected)
        )

    # ── 2. NUMERICAL ACCURACY ─────────────────────────────────────────────
    # Only scored when ground_truth_value is a number
    if "numerical_accuracy" in dims and gt is not None and isinstance(gt, (int, float)):
        num_check = check_numerical(answer, float(gt), tol or 2.0)
        result["numerical_accuracy"] = num_check["passed"]
        result["_num_check"] = num_check          # store for reporting

    # ── 3. CALCULATION ACCURACY ────────────────────────────────────────────
    # For derived metrics: checks if computed value is within tolerance.
    # Uses the same numeric check but flags it as a calculation result.
    if "calculation_accuracy" in dims and gt is not None and isinstance(gt, (int, float)):
        if "_num_check" not in result:
            num_check = check_numerical(answer, float(gt), tol or 5.0)
            result["_num_check"] = num_check
        result["calculation_accuracy"] = result["_num_check"]["passed"]
    elif "calculation_accuracy" in dims:
        # For qualitative calculations: judge's faithfulness >= 3 is the proxy
        result["calculation_accuracy"] = (judge_scores.get("faithfulness_score") or 0) >= 3

    # ── 4. GROUNDEDNESS ACCURACY ──────────────────────────────────────────
    # Passes when groundedness_score >= 4 AND no hallucination detected
    if "groundedness_accuracy" in dims:
        gs = judge_scores.get("groundedness_score") or 0
        no_halluc = not judge_scores.get("has_hallucination", True)
        result["groundedness_accuracy"] = gs >= 4 and no_halluc

    # ── 5. ABSTENTION ACCURACY ────────────────────────────────────────────
    # PASS scenarios:
    #   abstention_expected=True  → model should abstain → passes if actually_abstained
    #   abstention_expected=False → model should answer  → passes if NOT abstained
    if "abstention_accuracy" in dims:
        if abstention_expected:
            result["abstention_accuracy"] = actually_abstained
        else:
            result["abstention_accuracy"] = not actually_abstained

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    golden_set: List[Dict],
    generator: RAGGenerator,
    groq_client: Groq,
    vector_store: VectorStore,
    limit: Optional[int] = None,
    category_filter: Optional[str] = None,
    dimension_filter: Optional[str] = None,
) -> List[Dict]:

    if category_filter:
        golden_set = [c for c in golden_set if c.get("category") == category_filter]
    if dimension_filter:
        golden_set = [c for c in golden_set if dimension_filter in c.get("eval_dimensions", [])]
    if limit:
        golden_set = golden_set[:limit]

    total = len(golden_set)
    print(f"\n[EVAL] Running {total} test cases ...\n{'='*65}")

    results = []
    for idx, case in enumerate(golden_set, 1):
        q_id      = case["id"]
        category  = case["category"]
        question  = case["question"]
        ticker    = case.get("ticker", "")
        period    = case.get("fiscal_period", "")

        # Strip compound tickers for retrieval
        retrieval_ticker = ticker.split("_")[0] if "_" in ticker else ticker

        print(f"  [{idx:02d}/{total}] {q_id} [{category}]")
        print(f"         Q: {question[:80]}...")

        t0 = time.perf_counter()
        try:
            response = generator.generate_answer(
                query=question,
                top_k=5,
                ticker=retrieval_ticker or None,
                fiscal_period=period if "_to_" not in period and period not in
                              ("multi_year", "most_recent", "ambiguous_fy_vs_cy") else None,
            )
            answer  = response.get("answer", "")
            sources = response.get("sources", [])
        except Exception as exc:
            answer  = ""
            sources = []
            print(f"      [GEN ERROR] {exc}")

        latency_ms = round((time.perf_counter() - t0) * 1000)
        time.sleep(1.5)  # Groq rate-limit pacing

        # LLM-as-Judge
        judge = judge_groundedness(
            groq_client, question, answer, sources,
            abstention_expected=case.get("abstention_expected", False)
        )
        time.sleep(1.5)

        # Score across 5 dimensions
        dim_scores = score_question(case, answer, sources, judge)
        num_info   = dim_scores.pop("_num_check", {})

        # Overall pass: ALL applicable dimensions must pass
        applicable = [v for v in dim_scores.values() if v is not None]
        overall_pass = bool(applicable) and all(applicable)

        result = {
            # Identification
            "id":              q_id,
            "category":        category,
            "sub_category":    case.get("sub_category", ""),
            "ticker":          ticker,
            "fiscal_period":   period,
            "question":        question,
            "abstention_expected": case.get("abstention_expected", False),

            # Ground truth
            "ground_truth_value":     case.get("ground_truth_value"),
            "ground_truth_unit":      case.get("ground_truth_unit"),
            "ground_truth_formatted": case.get("ground_truth_formatted"),
            "tolerance_pct":          case.get("tolerance_pct"),

            # Response
            "model_answer":       answer[:600],
            "sources_returned":   len(sources),
            "avg_similarity":     round(
                sum(s.get("similarity_score", 0) for s in sources) / len(sources), 4
            ) if sources else 0.0,
            "latency_ms":         latency_ms,

            # Numerical check (if applicable)
            "best_match":         num_info.get("best_match"),
            "relative_error_pct": num_info.get("relative_error_pct"),
            "extracted_values":   num_info.get("extracted_values", []),

            # Judge scores
            "groundedness_score":  judge.get("groundedness_score"),
            "faithfulness_score":  judge.get("faithfulness_score"),
            "relevance_score":     judge.get("relevance_score"),
            "has_hallucination":   judge.get("has_hallucination"),
            "correctly_abstained": judge.get("correctly_abstained"),
            "judge_reasoning":     judge.get("judge_reasoning"),

            # 5-Dimension accuracy scores
            "retrieval_accuracy":    dim_scores["retrieval_accuracy"],
            "numerical_accuracy":    dim_scores["numerical_accuracy"],
            "calculation_accuracy":  dim_scores["calculation_accuracy"],
            "groundedness_accuracy": dim_scores["groundedness_accuracy"],
            "abstention_accuracy":   dim_scores["abstention_accuracy"],

            # Composite
            "overall_pass":    overall_pass,
            "eval_dimensions": case.get("eval_dimensions", []),
        }

        # Console status
        dim_str = " | ".join([
            f"{k.split('_')[0].upper()[:3]}={'P' if v is True else 'F' if v is False else '-'}"
            for k, v in dim_scores.items() if v is not None
        ])
        mark = "PASS" if overall_pass else "FAIL"
        print(f"      [{mark}] {dim_str} | latency={latency_ms}ms")

        results.append(result)
        time.sleep(2.0)  # Extra pacing between questions

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES = [
    "direct_fact_retrieval", "yoy_comparison", "multi_year_trend",
    "cross_company_comparison", "financial_ratio", "semantic_retrieval",
    "time_period_understanding", "multi_hop_reasoning",
    "hallucination_unavailable", "adversarial_hard",
]

DIMENSIONS = [
    "retrieval_accuracy", "numerical_accuracy", "calculation_accuracy",
    "groundedness_accuracy", "abstention_accuracy",
]

DIM_DESCRIPTIONS = {
    "retrieval_accuracy":    "Retrieval Accuracy    — Right chunks found?",
    "numerical_accuracy":    "Numerical Accuracy    — Numbers within tolerance?",
    "calculation_accuracy":  "Calculation Accuracy  — Derived metrics correct?",
    "groundedness_accuracy": "Groundedness Accuracy — Claims grounded in context?",
    "abstention_accuracy":   "Abstention Accuracy   — Correctly refused/answered?",
}


def pct(lst: List[bool]) -> float:
    if not lst:
        return 0.0
    return round(100 * sum(1 for x in lst if x is True) / len(lst), 1)


def compute_summary(results: List[Dict]) -> Dict:
    summary: Dict[str, Any] = {}

    # 1. Overall pass rate
    summary["total_cases"]   = len(results)
    summary["overall_pass"]  = pct([r["overall_pass"] for r in results])

    # 2. Per-dimension accuracy
    summary["dimensions"] = {}
    for dim in DIMENSIONS:
        applicable = [r[dim] for r in results if r.get(dim) is not None]
        summary["dimensions"][dim] = {
            "total":   len(applicable),
            "passed":  sum(1 for v in applicable if v is True),
            "accuracy": pct(applicable),
        }

    # 3. Per-category breakdown
    summary["categories"] = {}
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        summary["categories"][cat] = {
            "total":       len(cat_results),
            "overall_pass": pct([r["overall_pass"] for r in cat_results]),
            "dimensions": {
                dim: pct([r[dim] for r in cat_results if r.get(dim) is not None])
                for dim in DIMENSIONS
            },
        }

    # 4. Abstention stats
    abstention_cases = [r for r in results if r["abstention_expected"]]
    answer_cases     = [r for r in results if not r["abstention_expected"]]
    summary["abstention"] = {
        "questions_requiring_abstention": len(abstention_cases),
        "correctly_abstained": sum(1 for r in abstention_cases if r.get("abstention_accuracy") is True),
        "false_abstentions":  sum(1 for r in answer_cases     if r.get("abstention_accuracy") is False),
    }

    # 5. Hallucination stats
    summary["hallucination"] = {
        "total_judged":    len([r for r in results if r.get("has_hallucination") is not None]),
        "detected_count":  sum(1 for r in results if r.get("has_hallucination") is True),
    }

    # 6. Latency stats
    lats = [r["latency_ms"] for r in results if r.get("latency_ms")]
    summary["latency"] = {
        "avg_ms":  round(sum(lats) / len(lats)) if lats else 0,
        "min_ms":  min(lats) if lats else 0,
        "max_ms":  max(lats) if lats else 0,
    }

    return summary


def print_summary(summary: Dict) -> None:
    W = 65
    print(f"\n{'='*W}")
    print("  FINANCIAL RAG — 50-QUESTION EVALUATION SUMMARY")
    print(f"{'='*W}")
    print(f"  Total cases : {summary['total_cases']}")
    print(f"  Overall PASS: {summary['overall_pass']}%")

    print(f"\n{'─'*W}")
    print("  ACCURACY BY DIMENSION")
    print(f"{'─'*W}")
    for dim, info in summary["dimensions"].items():
        bar_len = int(info["accuracy"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {DIM_DESCRIPTIONS[dim]}")
        print(f"    [{bar}] {info['accuracy']}%  ({info['passed']}/{info['total']} passed)")

    print(f"\n{'─'*W}")
    print("  ACCURACY BY QUESTION CATEGORY")
    print(f"{'─'*W}")
    for cat, info in summary["categories"].items():
        label = cat.replace("_", " ").title()
        print(f"  {label:38s} {info['overall_pass']:5.1f}%  ({info['total']} Qs)")

    abst = summary["abstention"]
    print(f"\n{'─'*W}")
    print("  ABSTENTION & HALLUCINATION")
    print(f"{'─'*W}")
    print(f"  Questions needing abstention : {abst['questions_requiring_abstention']}")
    print(f"  Correctly abstained          : {abst['correctly_abstained']}")
    print(f"  False abstentions (over-refusal): {abst['false_abstentions']}")
    halluc = summary["hallucination"]
    print(f"  Hallucinations detected      : {halluc['detected_count']} / {halluc['total_judged']}")

    lat = summary["latency"]
    print(f"\n{'─'*W}")
    print("  LATENCY")
    print(f"{'─'*W}")
    print(f"  Avg: {lat['avg_ms']}ms  |  Min: {lat['min_ms']}ms  |  Max: {lat['max_ms']}ms")
    print(f"{'='*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG — 50-Question 5-Dimension Evaluation"
    )
    parser.add_argument("--model",     default=os.getenv("EVAL_MODEL", "openai/gpt-oss-20b"))
    parser.add_argument("--limit",     type=int, default=None, help="Run first N questions only")
    parser.add_argument("--category",  default=None, help="Filter to one category")
    parser.add_argument("--dimension", default=None, help="Filter to questions testing one dimension")
    args = parser.parse_args()

    print("=" * 65)
    print("  Financial RAG — 50-Question Evaluation | 5 Accuracy Dimensions")
    print(f"  Model: {args.model}")
    print("=" * 65)

    with open(GOLDEN_SET, encoding="utf-8") as f:
        golden_set = json.load(f)

    vector_store = VectorStore(persist_directory="data/chroma_db")
    generator    = RAGGenerator(model_name=args.model)
    groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))

    results = run_eval(
        golden_set, generator, groq_client, vector_store,
        limit=args.limit,
        category_filter=args.category,
        dimension_filter=args.dimension,
    )

    summary = compute_summary(results)
    print_summary(summary)

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"eval50_{timestamp}.json"
    payload   = {
        "run_timestamp": datetime.now().isoformat(),
        "model":         args.model,
        "total_cases":   len(results),
        "summary":       summary,
        "results":       results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
