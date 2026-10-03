# indexing/vector_store.py
"""
VectorStore — dual-backend implementation.

Backend is selected automatically based on environment:
  - SUPABASE_URL set  → pgvector on Supabase (production)
  - SUPABASE_URL not set → ChromaDB on local disk (local dev)

Public API is identical in both cases so Retriever / app.py need no changes.
"""
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from indexing.embedder import Embedder
from ingestion.pdf_parser import chunk_pdf
from ingestion.html_parser import chunk_html
from utils.metadata_parser import parse_filing_metadata

# Privacy redaction (graceful no-op if unavailable)
try:
    from privacy import redact_chunk
    _PRIVACY_AVAILABLE = True
except ImportError:
    _PRIVACY_AVAILABLE = False
    def redact_chunk(text):          # type: ignore
        return text, []


# ─────────────────────────────────────────────────────────────────
# BACKEND DETECTION
# ─────────────────────────────────────────────────────────────────

def _use_supabase() -> bool:
    return bool(os.environ.get("SUPABASE_URL"))


def _is_valid_uuid(val: str) -> bool:
    """Returns True only if val is a valid UUID string."""
    import re
    return bool(re.fullmatch(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        str(val or "").lower()
    ))


# ─────────────────────────────────────────────────────────────────
# SUPABASE / pgvector BACKEND
# ─────────────────────────────────────────────────────────────────

class _SupabaseBackend:
    """pgvector backend using Supabase Postgres."""

    def __init__(self, user_id: Optional[str] = None):
        import psycopg2
        import psycopg2.extras
        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self.dsn = (
            os.environ.get("POSTGRES_DSN")
            or os.environ.get("POSTGRES_DIRECT_DSN")
            or ""
        )
        self.user_id = user_id
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self.dsn)
        elif self._conn.get_transaction_status() == self._psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            self._conn.rollback()
        return self._conn

    def upsert(self, chunks: list[dict], user_id: str, paper_id: str,
               ticker: str, fiscal_period: str, form_type: str, ext: str):
        if not _is_valid_uuid(user_id):
            print(f"[VectorStore] skipping upsert — invalid user_id: '{user_id}'")
            return
        sql = """
            INSERT INTO public.embeddings
                (id, user_id, paper_id, ticker, fiscal_period, form_type,
                 page_number, word_count, file_type, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                content       = EXCLUDED.content,
                embedding     = EXCLUDED.embedding,
                page_number   = EXCLUDED.page_number,
                word_count    = EXCLUDED.word_count
        """
        try:
            with self.conn:
                with self.conn.cursor() as cur:
                    for c in chunks:
                        cur.execute(sql, (
                            f"{paper_id}_chunk_{c['chunk_id']}",
                            user_id,
                            paper_id,
                            ticker,
                            fiscal_period,
                            form_type,
                            c["page_number"],
                            c["word_count"],
                            ext.lstrip("."),
                            c["text"],
                            str(c["embedding"].tolist()),
                        ))
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

    def delete(self, paper_id: str, user_id: str) -> int:
        if not _is_valid_uuid(user_id):
            return 0
        try:
            with self.conn:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM public.embeddings WHERE paper_id=%s AND user_id=%s",
                        (paper_id, user_id),
                    )
                    return cur.rowcount
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

    def list_documents(self, user_id: str) -> list[str]:
        if not _is_valid_uuid(user_id):
            return []
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT paper_id FROM public.embeddings WHERE user_id=%s ORDER BY paper_id",
                    (user_id,),
                )
                return [r[0] for r in cur.fetchall()]
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

    def count(self, user_id: str) -> int:
        if not _is_valid_uuid(user_id):
            return 0
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.embeddings WHERE user_id=%s",
                    (user_id,),
                )
                return cur.fetchone()[0]
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

    def query(self, query_vector: list[float], top_k: int, user_id: str,
              paper_id: Optional[str] = None, ticker: Optional[str] = None,
              fiscal_period: Optional[str] = None) -> list[dict]:
        """Cosine similarity search with optional metadata filters."""
        if not _is_valid_uuid(user_id):
            print(f"[VectorStore] skipping query — invalid user_id: '{user_id}' (not authenticated)")
            return []
        conditions = ["user_id = %s"]
        params: list = [user_id]

        if paper_id:
            conditions.append("paper_id = %s")
            params.append(paper_id)
        if ticker:
            conditions.append("ticker = %s")
            params.append(ticker.upper())
        if fiscal_period:
            conditions.append("fiscal_period = %s")
            params.append(fiscal_period)

        where = " AND ".join(conditions)
        vec_str = str(query_vector)

        sql = f"""
            SELECT
                id,
                paper_id,
                ticker,
                fiscal_period,
                form_type,
                page_number,
                word_count,
                file_type,
                content,
                1 - (embedding <=> %s::vector) AS similarity
            FROM public.embeddings
            WHERE {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params_full = [vec_str] + params + [vec_str, top_k]

        try:
            with self.conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
                cur.execute(sql, params_full)
                rows = cur.fetchall()
        except Exception:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
            raise

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "text": r["content"],
                "similarity_score": round(float(r["similarity"]), 4),
                "distance": round(1.0 - float(r["similarity"]), 4),
                "metadata": {
                    "paper_id":      r["paper_id"],
                    "ticker":        r["ticker"],
                    "fiscal_period": r["fiscal_period"],
                    "form_type":     r["form_type"],
                    "page_number":   r["page_number"],
                    "word_count":    r["word_count"],
                    "file_type":     r["file_type"],
                },
            })
        return results


# ─────────────────────────────────────────────────────────────────
# CHROMADB BACKEND (local dev fallback)
# ─────────────────────────────────────────────────────────────────

class _ChromaBackend:
    """ChromaDB backend for local development."""

    def __init__(self, persist_directory: str = "data/chroma_db",
                 collection_name: str = "papers"):
        import chromadb
        persist = Path(persist_directory)
        persist.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[dict], user_id: str, paper_id: str,
               ticker: str, fiscal_period: str, form_type: str, ext: str):
        ids        = [f"{paper_id}_chunk_{c['chunk_id']}" for c in chunks]
        documents  = [c["text"] for c in chunks]
        embeddings = [c["embedding"].tolist() for c in chunks]
        metadatas  = [
            {
                "paper_id":      paper_id,
                "ticker":        ticker,
                "fiscal_period": fiscal_period,
                "form_type":     form_type,
                "statement_scope": "consolidated",
                "is_consolidated": True,
                "session_id":    user_id or "global",
                "page_number":   c["page_number"],
                "word_count":    c["word_count"],
                "file_type":     ext.lstrip("."),
            }
            for c in chunks
        ]
        self.collection.upsert(ids=ids, documents=documents,
                               embeddings=embeddings, metadatas=metadatas)

    def delete(self, paper_id: str, user_id: str) -> int:
        existing = self.collection.get(where={"paper_id": paper_id})
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
            return len(existing["ids"])
        return 0

    def list_documents(self, user_id: str) -> list[str]:
        try:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            return sorted(list({m["paper_id"] for m in metas if m and "paper_id" in m}))
        except Exception:
            return []

    def count(self, user_id: str) -> int:
        return self.collection.count()

    def query(self, query_vector: list[float], top_k: int, user_id: str,
              paper_id: Optional[str] = None, ticker: Optional[str] = None,
              fiscal_period: Optional[str] = None,
              statement_scope: Optional[str] = None) -> list[dict]:
        conds = []
        if paper_id:
            conds.append({"paper_id": paper_id})
        if ticker:
            conds.append({"ticker": ticker.upper()})
        if fiscal_period:
            conds.append({"fiscal_period": fiscal_period})
        if statement_scope:
            conds.append({"statement_scope": statement_scope.lower()})
        where = ({"$and": conds} if len(conds) > 1
                 else conds[0] if conds else None)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity_score": round(1.0 - results["distances"][0][i], 4),
                })
        return chunks


# ─────────────────────────────────────────────────────────────────
# PUBLIC VectorStore CLASS
# ─────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Unified VectorStore with automatic backend selection.
      - SUPABASE_URL set     → pgvector on Supabase
      - SUPABASE_URL not set → ChromaDB on local disk

    The public API (add_document, delete_document, query, list_documents,
    chunk_count) is identical regardless of backend.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
        user_id: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.user_id = user_id or "global"
        self.embedder = Embedder()

        env_backend = os.environ.get("VECTOR_BACKEND", "").lower()
        if backend == "chroma" or env_backend == "chroma":
            use_supa = False
        elif backend == "supabase" or env_backend == "supabase":
            use_supa = True
        elif _use_supabase() and _is_valid_uuid(self.user_id):
            use_supa = True
        else:
            use_supa = False

        if use_supa:
            self._backend = _SupabaseBackend(user_id=self.user_id)
            # Expose a .collection shim for legacy callers in app.py
            self.collection = _CollectionShim(self._backend, self.user_id)
        else:
            self._backend = _ChromaBackend(persist_directory, collection_name)
            self.collection = self._backend.collection

    # ── Document ingestion ──────────────────────────────────────

    def add_document(
        self,
        file_path: str,
        paper_id: Optional[str] = None,
        ticker: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Chunks, embeds, redacts PII, and stores a financial document."""
        path_obj = Path(file_path)
        paper_id = paper_id or path_obj.stem
        ext = path_obj.suffix.lower()

        # 1. Parse & chunk
        print(f"1. Chunking {file_path} ({ext})...")
        if ext in (".html", ".htm"):
            chunks = chunk_html(file_path, chunk_size=500, overlap_pct=0.10)
        elif ext == ".pdf":
            chunks = chunk_pdf(file_path, chunk_size=500, overlap_pct=0.10)
        elif ext in (".txt", ".md", ".csv", ".json", ".xml"):
            chunks = self._chunk_generic_text(file_path)
        else:
            raise ValueError(f"Unsupported format: '{ext}'")

        if not chunks:
            return 0

        # 2. PII redaction
        if _PRIVACY_AVAILABLE:
            print(f"2. Redacting PII across {len(chunks)} chunks...")
            for chunk in chunks:
                clean, _ = redact_chunk(chunk["text"])
                chunk["text"] = clean
        else:
            print("2. [Privacy] skipping — redact_chunk unavailable.")

        # 3. Embed
        print("3. Generating embeddings...")
        chunks = self.embedder.embed_chunks(chunks)

        # 4. Resolve metadata
        inferred = parse_filing_metadata(paper_id)
        final_ticker  = ticker or inferred.get("ticker") or ""
        final_period  = fiscal_period or inferred.get("fiscal_period") or ""
        final_form    = inferred.get("form_type") or ""
        effective_uid = session_id or self.user_id

        # 5. Store
        print("4. Storing vectors...")
        self._backend.upsert(
            chunks=chunks,
            user_id=effective_uid,
            paper_id=paper_id,
            ticker=final_ticker,
            fiscal_period=final_period,
            form_type=final_form,
            ext=ext,
        )
        print(f"[OK] Stored {len(chunks)} chunks for '{paper_id}'.")
        return len(chunks)

    # ── Document management ─────────────────────────────────────

    def delete_document(self, paper_id: str) -> int:
        return self._backend.delete(paper_id, self.user_id)

    def list_documents(self) -> list[str]:
        return self._backend.list_documents(self.user_id)

    @property
    def chunk_count(self) -> int:
        return self._backend.count(self.user_id)

    # ── Vector search (called by Retriever) ─────────────────────

    def query(
        self,
        query_vector: list[float],
        top_k: int,
        paper_id: Optional[str] = None,
        ticker: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        statement_scope: Optional[str] = None,
    ) -> list[dict]:
        return self._backend.query(
            query_vector=query_vector,
            top_k=top_k,
            user_id=self.user_id,
            paper_id=paper_id,
            ticker=ticker,
            fiscal_period=fiscal_period,
            statement_scope=statement_scope,
        )

    # ── Generic text chunker ─────────────────────────────────────

    def _chunk_generic_text(self, file_path: str,
                             chunk_size: int = 500, overlap_pct: float = 0.10) -> list:
        path_obj = Path(file_path)
        ext = path_obj.suffix.lower()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if ext == ".csv":
            lines = content.splitlines()
            if lines:
                content = f"Financial Table (Columns: {lines[0]}):\n" + "\n".join(lines[1:])
        elif ext in (".json", ".xml"):
            try:
                import json
                content = json.dumps(json.loads(content), indent=2)
            except Exception:
                pass

        words = content.split()
        if not words:
            return []
        step = max(1, int(chunk_size * (1.0 - overlap_pct)))
        chunks, chunk_id = [], 1
        for i in range(0, len(words), step):
            w = words[i:i + chunk_size]
            if not w:
                break
            chunks.append({
                "chunk_id":    chunk_id,
                "page_number": (i // chunk_size) + 1,
                "text":        " ".join(w),
                "word_count":  len(w),
                "file_type":   ext.lstrip("."),
            })
            chunk_id += 1
            if i + chunk_size >= len(words):
                break
        return chunks

    # ── Legacy compat ─────────────────────────────────────────────

    def add_pdf(self, pdf_path: str) -> int:
        return self.add_document(pdf_path)

    def backfill_isolation_metadata(self) -> int:
        return 0  # No-op for pgvector backend


# ─────────────────────────────────────────────────────────────────
# Shim: makes app.py's get_indexed_documents(_collection) work
# ─────────────────────────────────────────────────────────────────

class _CollectionShim:
    """
    Mimics the ChromaDB collection interface used in app.py:
      - .get(include=["metadatas"])
      - .count()
    """
    def __init__(self, backend: _SupabaseBackend, user_id: str):
        self._backend = backend
        self._user_id = user_id

    def get(self, include=None) -> dict:
        docs = self._backend.list_documents(self._user_id)
        return {"metadatas": [{"paper_id": d} for d in docs]}

    def count(self) -> int:
        return self._backend.count(self._user_id)

    def query(self, query_embeddings, n_results, where=None, include=None):
        # Delegate to backend — where filter is ignored at this level
        # (user isolation is handled by backend automatically)
        results = self._backend.query(
            query_vector=query_embeddings[0],
            top_k=n_results,
            user_id=self._user_id,
        )
        return {
            "ids":       [[r["id"] for r in results]],
            "documents": [[r["text"] for r in results]],
            "metadatas": [[r["metadata"] for r in results]],
            "distances": [[r["distance"] for r in results]],
        }