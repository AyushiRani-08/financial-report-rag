# utils/logger.py
"""Structured logging utility for the Financial Report RAG system.

Logs are written to logs/rag_queries.jsonl as line-delimited JSON records.
Each record captures query, retrieval quality, and LLM latency for analysis.
"""

import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

# Create logs directory if it doesn t exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

QUERY_LOG_FILE = LOGS_DIR / "rag_queries.jsonl"

# Standard Python logger for startup/error events (written to logs/app.log)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("rag")


def log_query(
    query: str,
    query_type: str,
    paper_id,
    top_k: int,
    num_chunks_retrieved: int,
    similarity_scores: list,
    retrieval_latency_ms: float,
    llm_latency_ms: float,
    total_latency_ms: float,
    model_name: str,
    answer_length: int,
):
    """Append a structured log record to logs/rag_queries.jsonl."""
    avg_score = round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0.0
    min_score = round(min(similarity_scores), 4) if similarity_scores else 0.0
    max_score = round(max(similarity_scores), 4) if similarity_scores else 0.0

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_type": query_type,
        "paper_id": paper_id or "all",
        "top_k": top_k,
        "query_preview": query[:120],
        "num_chunks_retrieved": num_chunks_retrieved,
        "similarity_scores": {
            "avg": avg_score,
            "min": min_score,
            "max": max_score,
            "all": [round(s, 4) for s in similarity_scores],
        },
        "latency_ms": {
            "retrieval": round(retrieval_latency_ms, 1),
            "llm": round(llm_latency_ms, 1),
            "total": round(total_latency_ms, 1),
        },
        "model": model_name,
        "answer_length_chars": answer_length,
    }

    with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        f"[{query_type.upper()}] chunks={num_chunks_retrieved} "
        f"avg_score={avg_score} "
        f"retrieval={retrieval_latency_ms:.0f}ms "
        f"llm={llm_latency_ms:.0f}ms "
        f"total={total_latency_ms:.0f}ms"
    )


class Timer:
    """Simple context manager for measuring elapsed time in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
