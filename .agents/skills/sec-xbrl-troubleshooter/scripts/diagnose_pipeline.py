#!/usr/bin/env python3
"""
Diagnostic script for SEC XBRL & pgvector pipeline health.
Zero token cost: executes locally and outputs pass/fail status.
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

def check_postgres_connection():
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        print("[FAIL] POSTGRES_DSN environment variable missing.")
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(dsn, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        conn.close()
        print("[PASS] PostgreSQL / Supabase connection active.")
        return True
    except Exception as e:
        print(f"[FAIL] PostgreSQL connection failed: {e}")
        return False

def check_vector_table(ticker: str):
    import psycopg2
    try:
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur = conn.cursor()
        cur.execute(
            "SELECT count(*), pg_column_size(embedding) FROM public.embeddings "
            "WHERE ticker = %s GROUP BY pg_column_size(embedding) LIMIT 1;",
            (ticker.upper(),)
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            print(f"[WARN] No vector embeddings found for ticker: {ticker}")
            return False
        count, size = row
        print(f"[PASS] Found {count} embedding rows for {ticker} (byte size: {size}).")
        return True
    except Exception as e:
        print(f"[FAIL] Vector table query error: {e}")
        return False

def check_xbrl_facts(ticker: str):
    import psycopg2
    try:
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur = conn.cursor()
        cur.execute(
            """
            SELECT count(f.id)
            FROM financial_facts f
            JOIN filings fi ON f.filing_id = fi.id
            JOIN companies c ON fi.company_id = c.id
            WHERE c.ticker = %s;
            """,
            (ticker.upper(),)
        )
        count = cur.fetchone()[0]
        conn.close()
        if count == 0:
            print(f"[WARN] Zero XBRL financial facts found in DB for ticker: {ticker}")
            return False
        print(f"[PASS] Found {count} structured XBRL financial facts for {ticker}.")
        return True
    except Exception as e:
        print(f"[FAIL] XBRL facts check error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SEC/XBRL Ingestion Diagnostic Tool")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to test")
    args = parser.parse_args()

    print(f"=== SEC/XBRL & Vector Pipeline Diagnostics for {args.ticker.upper()} ===")
    checks = [
        check_postgres_connection(),
        check_vector_table(args.ticker),
        check_xbrl_facts(args.ticker)
    ]

    if all(checks):
        print("\n>>> ALL CHECKS PASSED: Pipeline is healthy.")
        sys.exit(0)
    else:
        print("\n>>> WARNING: One or more checks failed. Review output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
