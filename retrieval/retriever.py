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


class Retriever:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
        user_id: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
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
    ) -> List[Dict[str, Any]]:
        """
        Returns the top_k most relevant chunks for the query.

        Supports statement_scope ('consolidated' vs 'standalone').
        """
        # 1. Embed query
        query_vector = self.embedder.embed_query(query).tolist()

        # 2. Primary search with all filters
        chunks = self.vector_store.query(
            query_vector=query_vector,
            top_k=top_k,
            paper_id=paper_id,
            ticker=ticker,
            fiscal_period=fiscal_period,
            statement_scope=statement_scope,
        )

        # 3. Graceful fallback: relax statement_scope or fiscal_period if no results
        if not chunks and statement_scope:
            chunks = self.vector_store.query(
                query_vector=query_vector,
                top_k=top_k,
                paper_id=paper_id,
                ticker=ticker,
                fiscal_period=fiscal_period,
                statement_scope=None,
            )

        if not chunks and fiscal_period and ticker and not paper_id:
            chunks = self.vector_store.query(
                query_vector=query_vector,
                top_k=top_k,
                paper_id=None,
                ticker=ticker,
                fiscal_period=None,
                statement_scope=None,
            )

        return chunks


if __name__ == "__main__":
    retriever = Retriever()
    test_query = "What is the revenue growth trend?"
    print(f"Query: '{test_query}'\n")
    results = retriever.retrieve(query=test_query, top_k=2)
    for rank, item in enumerate(results, start=1):
        print(f"--- Rank {rank} (Score: {item['similarity_score']}) ---")
        print(f"Page: {item['metadata']['page_number']}")
        print(f"Text: {item['text'][:200]}...\n")