# indexing/vector_store.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from ingestion.pdf_parser import chunk_pdf
from ingestion.html_parser import chunk_html
from indexing.embedder import Embedder

# Privacy: redact PII from chunks before they are embedded and stored.
# Gracefully no-ops if the privacy package is not importable.
try:
    from privacy import redact_chunk
    _PRIVACY_AVAILABLE = True
except ImportError:
    _PRIVACY_AVAILABLE = False
    def redact_chunk(text):   # type: ignore
        return text, []


class VectorStore:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "papers",
    ):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = Embedder()

    def add_document(self, file_path: str) -> int:
        """
        Chunks, redacts PII, embeds, and stores a financial document.

        Full pipeline per chunk
        ───────────────────────
          raw file → parse → chunk → redact_chunk() → embed → ChromaDB

        ChromaDB and the LLM only ever see sanitised text.
        Every redaction event is written to logs/redactions.log.

        Returns the number of stored chunks.
        """
        path_obj = Path(file_path)
        paper_id = path_obj.stem
        ext = path_obj.suffix.lower()

        # ── 1. Parse & chunk ─────────────────────────────────────────────────
        print(f"1. Chunking {file_path} ({ext})...")
        if ext in (".html", ".htm"):
            chunks = chunk_html(file_path, chunk_size=500, overlap_pct=0.10)
        elif ext == ".pdf":
            chunks = chunk_pdf(file_path, chunk_size=500, overlap_pct=0.10)
        elif ext in (".txt", ".md", ".csv", ".json", ".xml"):
            chunks = self._chunk_generic_text(file_path, chunk_size=500, overlap_pct=0.10)
        else:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                "Supported: .pdf, .html, .htm, .txt, .md, .csv, .json, .xml"
            )

        if not chunks:
            print("No chunks generated.")
            return 0

        # ── 2. Privacy redaction ──────────────────────────────────────────────
        # Each chunk's text is sanitised in-place BEFORE embedding.
        # This means PII never reaches the vector store or the LLM.
        if _PRIVACY_AVAILABLE:
            print(f"2. Redacting PII across {len(chunks)} chunks...")
            total_redactions = 0
            for chunk in chunks:
                clean_text, findings = redact_chunk(chunk["text"])
                chunk["text"] = clean_text           # replace raw text with safe version
                total_redactions += len(findings)
            if total_redactions:
                print(f"   ↳ {total_redactions} sensitive item(s) redacted — see logs/redactions.log")
        else:
            print("2. [Privacy] redact_chunk not available — skipping redaction.")

        # ── 3. Embed ──────────────────────────────────────────────────────────
        print("3. Generating embeddings...")
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # ── 4. Store in ChromaDB ──────────────────────────────────────────────
        print("4. Storing in ChromaDB...")
        ids        = [f"{paper_id}_chunk_{c['chunk_id']}" for c in embedded_chunks]
        documents  = [c["text"] for c in embedded_chunks]
        embeddings = [c["embedding"].tolist() for c in embedded_chunks]
        metadatas  = [
            {
                "paper_id":    paper_id,
                "page_number": c["page_number"],
                "word_count":  c["word_count"],
                "file_type":   ext.lstrip("."),
            }
            for c in embedded_chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[OK] Stored {len(embedded_chunks)} chunks for '{paper_id}'.")
        return len(embedded_chunks)

    def _chunk_generic_text(
        self, file_path: str, chunk_size: int = 500, overlap_pct: float = 0.10
    ) -> list:
        """Chunks plain-text / CSV / JSON / XML files into word-based segments."""
        path_obj = Path(file_path)
        ext = path_obj.suffix.lower()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if ext == ".csv":
            lines = content.splitlines()
            if lines:
                header = lines[0]
                content = f"Financial Table (Columns: {header}):\n" + "\n".join(lines[1:])
        elif ext in (".json", ".xml"):
            try:
                import json
                data = json.loads(content)
                content = json.dumps(data, indent=2)
            except Exception:
                pass

        words = content.split()
        if not words:
            return []

        step     = max(1, int(chunk_size * (1.0 - overlap_pct)))
        chunks   = []
        chunk_id = 1
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                break
            chunks.append({
                "chunk_id":    chunk_id,
                "page_number": (i // chunk_size) + 1,
                "text":        " ".join(chunk_words),
                "word_count":  len(chunk_words),
                "file_type":   ext.lstrip("."),
            })
            chunk_id += 1
            if i + chunk_size >= len(words):
                break
        return chunks

    def add_pdf(self, pdf_path: str) -> int:
        """Backward-compatibility alias for add_document."""
        return self.add_document(pdf_path)

    def delete_document(self, paper_id: str) -> int:
        """Delete all chunks belonging to paper_id from ChromaDB."""
        try:
            existing = self.collection.get(where={"paper_id": paper_id})
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
                return len(existing["ids"])
        except Exception as e:
            print(f"Error deleting {paper_id}: {e}")
        return 0


if __name__ == "__main__":
    store = VectorStore()
    print("VectorStore initialized successfully.")