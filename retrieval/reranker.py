# retrieval/reranker.py
"""
LocalReranker — 100% free, ultra-lightweight cross-encoder re-ranking.
Uses FlashRank (quantized ONNX models) for blazing-fast CPU inference (< 35ms)
with minimal memory footprint (< 40MB RAM), safe for Streamlit Cloud and low-spec containers.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LocalReranker:
    """
    Reranks vector search candidates using a cross-encoder model.
    Falls back gracefully to original vector order if FlashRank is unavailable.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._ranker = None
        self._available = False

        try:
            from flashrank import Ranker
            # Ranker caches the ~4-30MB ONNX model locally
            self._ranker = Ranker(model_name=self.model_name)
            self._available = True
            logger.info("[LocalReranker] FlashRank initialized with model: %s", self.model_name)
        except Exception as e:
            logger.warning("[LocalReranker] FlashRank initialization failed (%s). Will use vector order.", e)

    @property
    def is_available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of retrieved chunks against the query.

        Args:
            query: The user's query string.
            chunks: List of chunk dicts (must contain 'text').
            top_k: Number of highest-ranked chunks to return.

        Returns:
            List of top_k chunk dicts sorted by relevance score.
        """
        if not chunks:
            return []
        if len(chunks) <= top_k or not self._available:
            return chunks[:top_k]

        try:
            from flashrank import RerankRequest

            # FlashRank expects a list of dicts with 'id' and 'text'
            passages = [
                {"id": i, "text": chunk.get("text", ""), "chunk_ref": chunk}
                for i, chunk in enumerate(chunks)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(rerank_request)

            reranked_chunks: List[Dict[str, Any]] = []
            for item in results[:top_k]:
                orig_chunk = item["chunk_ref"]
                # Attach the cross-encoder relevance score (0.0 to 1.0)
                orig_chunk["rerank_score"] = round(float(item.get("score", 0.0)), 4)
                reranked_chunks.append(orig_chunk)

            logger.info(
                "[LocalReranker] Successfully reranked %d candidates to top %d",
                len(chunks), len(reranked_chunks)
            )
            return reranked_chunks

        except Exception as e:
            logger.warning("[LocalReranker] Reranking failed (%s). Falling back to vector ranking.", e)
            return chunks[:top_k]
