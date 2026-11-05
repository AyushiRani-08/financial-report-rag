"""
scripts/backfill_is_table.py

Backfills the `is_table` metadata flag on chunks ingested before the flag
existed.

Chroma treats a filter on a missing metadata key as a non-match, so
`where={"is_table": False}` also returned nothing. Numeric questions request
the table branch first and the text branch second; both came back empty and
the answer degraded to "No relevant financial documents found".

Table chunks are written as Markdown (see table_to_markdown_chunks), so the
separator row `| --- | --- |` is a reliable signature for the backfill.

Usage:
    python scripts/backfill_is_table.py --dry-run
    python scripts/backfill_is_table.py
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb

_TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+", re.MULTILINE)


def looks_like_table(text: str) -> bool:
    return bool(_TABLE_SEPARATOR.search(text or ""))


def backfill(persist: str, collection_name: str, dry_run: bool = False) -> int:
    client = chromadb.PersistentClient(path=persist)
    collection = client.get_collection(collection_name)
    existing = collection.get(include=["documents", "metadatas"])

    ids = existing["ids"]
    documents = existing["documents"]
    metadatas = existing["metadatas"]

    updates = []
    table_count = 0
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        if "is_table" in (metadata or {}):
            continue
        is_table = looks_like_table(document)
        table_count += int(is_table)
        merged = dict(metadata or {})
        merged["is_table"] = is_table
        updates.append((chunk_id, merged))

    print(
        f"{len(ids)} chunks in '{collection_name}' | "
        f"{len(updates)} missing is_table | {table_count} look like tables"
    )
    if dry_run or not updates:
        return 0

    collection.update(
        ids=[chunk_id for chunk_id, _ in updates],
        metadatas=[metadata for _, metadata in updates],
    )
    print(f"backfilled {len(updates)} chunks")
    return len(updates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill is_table chunk metadata")
    parser.add_argument("--persist", default="data/chroma_db")
    parser.add_argument("--collection", default="papers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(args.persist, args.collection, args.dry_run)


if __name__ == "__main__":
    main()
