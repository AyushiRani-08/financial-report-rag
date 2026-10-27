#!/usr/bin/env python3
"""
scripts/print_vector_backend.py
--------------------------------
Utility to inspect which vector-store backend is active and confirm
connection-pool configuration.

Usage:
    python scripts/print_vector_backend.py [--list-docs] [--user-id UUID]

Requires environment variables:
    POSTGRES_DSN or POSTGRES_DIRECT_DSN  -> activates pgvector / Supabase backend
    (neither set)                        -> activates ChromaDB local backend
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the active vector-store backend.")
    parser.add_argument("--list-docs", action="store_true", help="List indexed document IDs for --user-id.")
    parser.add_argument("--user-id", default=None, help="User UUID to query documents for (required with --list-docs).")
    args = parser.parse_args()

    postgres_dsn = os.environ.get("POSTGRES_DSN") or os.environ.get("POSTGRES_DIRECT_DSN")

    print("=" * 60)
    print("  Financial RAG - Vector Store Backend Inspector")
    print("=" * 60)

    if postgres_dsn:
        print(f"  DSN detected   : {'*' * 10}{postgres_dsn[-20:]}")
        print("  Expected backend: _SupabaseBackend  (pgvector on Postgres)")
    else:
        print("  No POSTGRES_DSN detected.")
        print("  Expected backend: _ChromaBackend  (local ChromaDB)")

    print()

    try:
        from indexing.vector_store import VectorStore

        dummy_uid = args.user_id or "00000000-0000-0000-0000-000000000000"
        vs = VectorStore(user_id=dummy_uid)
        backend = vs._backend
        backend_name = type(backend).__name__

        print(f"  Active backend  : {backend_name}")

        if hasattr(backend, "pool"):
            pool = backend.pool
            try:
                print(f"  Pool min-conn   : {pool.minconn}")
                print(f"  Pool max-conn   : {pool.maxconn}")
                print(f"  Pool closed?    : {pool.closed}")
            except Exception as pe:
                print(f"  Pool info error : {pe}")
        else:
            print("  Connection pool : not present (single-connection mode)")

        if args.list_docs:
            if not args.user_id:
                print("\n  WARNING: --user-id is required to list documents. Use --user-id <UUID>.")
                sys.exit(1)
            print(f"\n  Documents indexed for user {args.user_id}:")
            try:
                docs = backend.list_documents(args.user_id)
                if docs:
                    for doc in docs:
                        print(f"    * {doc}")
                else:
                    print("    (none found)")
            except Exception as e:
                print(f"    Error listing docs: {e}")

    except Exception as e:
        print(f"\n  Could not instantiate VectorStore: {e}")
        print("     This is expected if no DB is reachable. The env-var check above is still valid.")

    print()
    print("=" * 60)
    print("  Env vars checked:")
    print(f"    POSTGRES_DSN        = {'(set)' if os.environ.get('POSTGRES_DSN') else '(not set)'}")
    print(f"    POSTGRES_DIRECT_DSN = {'(set)' if os.environ.get('POSTGRES_DIRECT_DSN') else '(not set)'}")
    print(f"    SUPABASE_URL        = {'(set)' if os.environ.get('SUPABASE_URL') else '(not set)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
