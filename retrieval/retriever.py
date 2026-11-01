# retrieval/retriever.py
"""
Retriever — works with both VectorStore backends (pgvector + ChromaDB).
The underlying VectorStore.query() method handles backend selection transparently.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from indexing.embedder import Embedder
from indexing.vector_store import VectorStore
from retrieval.reranker import LocalReranker


class Retriever:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
        user_id: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[LocalReranker] = None,
    ):
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name,
                user_id=user_id,
            )
        self.embedder = self.vector_store.embedder
        self.reranker = reranker or LocalReranker()
        # Expose collection for legacy callers
        self.collection = self.vector_store.collection

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        paper_id: str | None = None,
        ticker: str | None = None,
        fiscal_period: str | None = None,
        statement_scope: str | None = None,
        form_type: str | None = None,
        session_id: str | None = None,
        where_override: dict | None = None,
        prefer_tables: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top_k most relevant chunks for the query using two-stage retrieval:
        1. Fetch candidate pool (up to 15 chunks) from VectorStore via cosine similarity.
        2. Rerank candidates with FlashRank Cross-Encoder to select the top_k.

        prefer_tables=True guarantees table chunks reach the candidate pool: numeric
        questions otherwise drown in prose chunks that merely mention the metric name.
        """
        # 1. Embed query
        query_vector = self.embedder.embed_query(query).tolist()

        # Candidate pool: fetch more candidates so re-ranker has high-quality selection
        candidate_k = max(top_k * 4, 15)

        scope_args = dict(
            query_vector=query_vector,
            paper_id=paper_id,
            ticker=ticker,
            fiscal_period=fiscal_period,
            statement_scope=statement_scope,
        )

        chunks: List[Dict[str, Any]] = []
        if prefer_tables:
            table_chunks = self._query_with_fallback(
                scope_args, top_k=candidate_k, is_table=True
            )
            text_chunks = self._query_with_fallback(
                scope_args, top_k=top_k, is_table=False
            )
            chunks = _merge_unique(table_chunks, text_chunks)
        else:
            chunks = self._query_with_fallback(scope_args, top_k=candidate_k)

        # 4. Re-rank candidate chunks down to top_k using FlashRank cross-encoder
        if chunks:
            chunks = self.reranker.rerank(query=query, chunks=chunks, top_k=top_k)

        return chunks

    def _query_with_fallback(
        self, scope_args: Dict[str, Any], top_k: int, is_table: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Vector search, progressively relaxing scope and period before giving up."""
        chunks = self.vector_store.query(top_k=top_k, is_table=is_table, **scope_args)

        if not chunks and scope_args.get("statement_scope"):
            relaxed = {**scope_args, "statement_scope": None}
            chunks = self.vector_store.query(top_k=top_k, is_table=is_table, **relaxed)

        if not chunks and scope_args.get("fiscal_period") and scope_args.get("ticker") \
                and not scope_args.get("paper_id"):
            relaxed = {**scope_args, "fiscal_period": None, "statement_scope": None}
            chunks = self.vector_store.query(top_k=top_k, is_table=is_table, **relaxed)

        return chunks


def _merge_unique(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Concatenates result groups, keeping the first occurrence of each chunk id."""
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for chunk in group:
            chunk_id = chunk.get("id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(chunk)
    return merged


if __name__ == "__main__":
    retriever = Retriever()
    test_query = "What is the revenue growth trend?"
    print(f"Query: '{test_query}'\n")
    results = retriever.retrieve(query=test_query, top_k=2)
    for rank, item in enumerate(results, start=1):
        print(f"--- Rank {rank} (Vector Cosine: {item['similarity_score']} | Rerank Score: {item.get('rerank_score', 'N/A')}) ---")
        print(f"Page: {item['metadata'].get('page_number', 'N/A')}")
        print(f"Text: {item['text'][:200]}...\n")