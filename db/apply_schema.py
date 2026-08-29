"""
db/apply_schema.py
Run this once to create all tables and indexes in your PostgreSQL database.
Usage: python db/apply_schema.py
"""

import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import psycopg2

load_dotenv()

SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def apply_schema(dsn: str | None = None):
    dsn = dsn or os.environ.get("POSTGRES_DSN")
    if not dsn:
        print("❌ POSTGRES_DSN not set in environment or .env file.")
        sys.exit(1)

    print(f"[*] Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(dsn)
    except psycopg2.OperationalError as e:
        print(f"[ERROR] Could not connect: {e}")
        print("   Is your PostgreSQL container running? Try: docker ps")
        sys.exit(1)

    sql = SCHEMA_FILE.read_text()

    with conn:
        with conn.cursor() as cur:
            print(f"[*] Applying schema from {SCHEMA_FILE.name}...")
            cur.execute(sql)

    conn.close()
    print("[OK] Schema applied successfully!")
    print("   Tables created: companies, filings, financial_facts")
    print("   Indexes created: idx_facts_lookup, idx_filings_company, idx_facts_period, idx_companies_market")


if __name__ == "__main__":
    apply_schema()
