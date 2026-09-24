#ChromaDB computes cosine distance and returns the top $K$ chunks with their text, page numbers, and similarity distances.
# retrieval/retriever.py
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from indexing.embedder import Embedder
from indexing.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
    ):
        self.vector_store = VectorStore(
            persist_directory=persist_directory, collection_name=collection_name
        )
        self.collection = self.vector_store.collection
        self.embedder = self.vector_store.embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        paper_id: str | None = None,
        ticker: str | None = None,
        fiscal_period: str | None = None,
        form_type: str | None = None,
        session_id: str | None = None,
        where_override: dict | None = None,
    ) -> List[Dict[str, Any]]:
        """Searches ChromaDB for the top_k most relevant chunks with strict isolation boundaries.

        Args:
            query: The user question.
            top_k: Number of relevant chunks to return.
            paper_id: Restrict search to a specific document.
            ticker: Restrict search to a specific company ticker (e.g. 'AAPL').
            fiscal_period: Restrict search to a specific period (e.g. 'FY2025').
            form_type: Restrict search to a specific SEC form (e.g. '10-K', '10-Q').
            session_id: Restrict to session-scoped documents or shared global documents.
            where_override: Explicit ChromaDB where filter dict.
        """
        # 1. Embed query with BGE prompt formatting
        query_vector = self.embedder.embed_query(query).tolist()

        # 2. Build compound where filter for Retrieval Isolation
        def _build_filter(p_id, t, fp, ft, s_id):
            conds = []
            if p_id:
                conds.append({"paper_id": p_id})
            if t:
                conds.append({"ticker": t.upper()})
            if fp:
                conds.append({"fiscal_period": fp})
            if ft:
                conds.append({"form_type": ft})
            if s_id and s_id != "global":
                conds.append({"$or": [{"session_id": s_id}, {"session_id": "global"}]})
            if len(conds) == 1:
                return conds[0]
            elif len(conds) > 1:
                return {"$and": conds}
            return None

        where_filter = where_override or _build_filter(
            paper_id, ticker, fiscal_period, form_type, session_id
        )

        # 3. Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Fallback: if period was specified but returned 0 results, fall back to ticker-level isolation
        # to ensure user still receives company results without cross-company bleed.
        if (
            (not results["documents"] or not results["documents"][0])
            and fiscal_period
            and ticker
            and not paper_id
        ):
            fallback_filter = _build_filter(paper_id, ticker, None, form_type, session_id)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=fallback_filter,
                include=["documents", "metadatas", "distances"],
            )

        # 4. Format outputs into a clean list of dictionaries
        retrieved_chunks = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i in range(len(docs)):
                retrieved_chunks.append(
                    {
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        # Cosine distance: 0 = identical, 2 = opposite
                        # Cosine similarity approx = 1 - distance
                        "distance": distances[i],
                        "similarity_score": round(1.0 - distances[i], 4),
                    }
                )

        return retrieved_chunks


if __name__ == "__main__":
    retriever = Retriever()

    # Test Query against "Attention Is All You Need"
    test_query = "What is the architecture of Multi-Head Attention?"
    print(f"Query: '{test_query}'\n")

    results = retriever.retrieve(query=test_query, top_k=2)

    for rank, item in enumerate(results, start=1):
        print(
            f"--- Rank {rank} (Similarity Score: {item['similarity_score']}) ---"
        )
        print(f"Chunk ID : {item['id']}")
        print(f"Page     : {item['metadata']['page_number']}")
        print(f"Excerpt  : {item['text'][:250]}...\n")