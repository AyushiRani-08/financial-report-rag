"""
scripts/ingest_apple_reports.py
================================
One-shot script to ingest Apple's 3-year annual reports (10-K) into
both ChromaDB (vector search) and PostgreSQL (XBRL structured facts).

Covers:
  - AAPL FY2022  (10-K filed Oct 2022, period ended Sep 24 2022)
  - AAPL FY2023  (10-K filed Nov 2023, period ended Sep 30 2023)
  - AAPL FY2024  (10-K filed Nov 2024, period ended Sep 28 2024)

Each report goes through two parallel pipelines:
  A) SEC EDGAR API  → XBRL structured facts → PostgreSQL
     (needed for quantitative ground-truth queries in run_eval_50.py)

  B) SEC EDGAR HTML → chunked text → BGE embeddings → ChromaDB
     (needed for qualitative / semantic retrieval by the RAG pipeline)

Usage:
    python scripts/ingest_apple_reports.py
    python scripts/ingest_apple_reports.py --years 2024
    python scripts/ingest_apple_reports.py --skip-xbrl      # ChromaDB only
    python scripts/ingest_apple_reports.py --skip-chroma    # PostgreSQL only
    python scripts/ingest_apple_reports.py --dry-run        # Preview only
"""

import os
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    try: _sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
import sys
import argparse
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Target filings ──────────────────────────────────────────────────────────
# Apple's fiscal year ends in late September. FY label = calendar year of end.
APPLE_FILINGS = [
    {
        "ticker":        "AAPL",
        "fiscal_period": "FY2022",
        "fiscal_year":   2022,
        "form_type":     "10-K",
        "paper_id":      "aapl-20220924",      # matches SEC accession date
        "description":   "Apple FY2022 Annual Report (ended Sep 24, 2022)",
    },
    {
        "ticker":        "AAPL",
        "fiscal_period": "FY2023",
        "fiscal_year":   2023,
        "form_type":     "10-K",
        "paper_id":      "aapl-20230930",
        "description":   "Apple FY2023 Annual Report (ended Sep 30, 2023)",
    },
    {
        "ticker":        "AAPL",
        "fiscal_period": "FY2024",
        "fiscal_year":   2024,
        "form_type":     "10-K",
        "paper_id":      "aapl-20240928",
        "description":   "Apple FY2024 Annual Report (ended Sep 28, 2024)",
    },
]

DOWNLOAD_DIR = Path("data/raw_pdfs")


def banner(text: str):
    print(f"\n{'─'*65}")
    print(f"  {text}")
    print(f"{'─'*65}")


def ingest_xbrl(filing: dict) -> dict:
    """
    Pull XBRL structured facts from SEC EDGAR API → PostgreSQL.
    Returns a summary dict.
    """
    from ingestion.sec_fetcher import fetch_and_store_xbrl

    banner(f"[XBRL] {filing['description']}")
    print(f"  -> Fetching facts for {filing['ticker']} {filing['fiscal_period']} from SEC EDGAR...")

    try:
        result = fetch_and_store_xbrl(
            ticker=filing["ticker"],
            fiscal_period=filing["fiscal_period"],
            form_type=filing["form_type"],
            progress_callback=lambda msg: print(f"     {msg}"),
        )
        print(f"  [OK] XBRL: {result['facts_inserted']:,} facts | periods: {result['periods_found']}")
        return {"status": "ok", "facts_inserted": result["facts_inserted"], "periods": result["periods_found"]}
    except Exception as exc:
        print(f"  [FAIL] XBRL: {exc}")
        return {"status": "error", "error": str(exc)}


def ingest_chroma(filing: dict, download_dir: Path) -> dict:
    """
    Download the HTML 10-K from SEC EDGAR → chunk → embed → ChromaDB.
    Returns a summary dict.
    """
    from ingestion.sec_fetcher import download_sec_filing
    from indexing.vector_store import VectorStore

    banner(f"[CHROMA] {filing['description']}")

    # ── Step 1: Download HTML if not already cached ──────────────────────
    cached_files = list(download_dir.glob(f"{filing['ticker'].lower()}*{filing['fiscal_year']}*10k*.htm")) + \
                   list(download_dir.glob(f"{filing['ticker'].lower()}*{filing['fiscal_year']}*10-k*.htm"))

    if cached_files:
        local_path = str(cached_files[0])
        print(f"  [CACHE] Using cached file: {cached_files[0].name}")
    else:
        print(f"  [DL] Downloading 10-K HTML from SEC EDGAR...")
        try:
            dl = download_sec_filing(
                ticker=filing["ticker"],
                form_type=filing["form_type"],
                save_dir=download_dir,
                fiscal_year=filing["fiscal_year"],
                progress_callback=lambda msg: print(f"     {msg}"),
            )
            local_path = dl["file_path"]
            print(f"  [OK] Downloaded: {dl['file_name']} ({dl['size_mb']} MB)")
        except Exception as exc:
            print(f"  [FAIL] Download: {exc}")
            return {"status": "error", "error": str(exc)}

    # ── Step 2: Chunk → Embed → Store in ChromaDB ────────────────────────
    print(f"  🔢 Embedding and storing in ChromaDB...")
    try:
        vs = VectorStore(persist_directory="data/chroma_db", backend="chroma")
        n_chunks = vs.add_document(
            file_path=local_path,
            paper_id=filing["paper_id"],
            ticker=filing["ticker"],
            fiscal_period=filing["fiscal_period"],
        )
        print(f"  [OK] ChromaDB: {n_chunks:,} chunks stored (paper_id={filing['paper_id']})")
        return {"status": "ok", "chunks_stored": n_chunks}
    except Exception as exc:
        print(f"  [FAIL] ChromaDB: {exc}")
        return {"status": "error", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Ingest Apple 3-year 10-K reports")
    parser.add_argument("--years",        nargs="*", type=int, default=None,
                        help="Fiscal years to ingest e.g. --years 2023 2024 (default: all 3)")
    parser.add_argument("--skip-xbrl",   action="store_true", help="Skip PostgreSQL XBRL ingestion")
    parser.add_argument("--skip-chroma", action="store_true", help="Skip ChromaDB vector ingestion")
    parser.add_argument("--dry-run",     action="store_true", help="List what would be done without executing")
    args = parser.parse_args()

    filings = APPLE_FILINGS
    if args.years:
        filings = [f for f in filings if f["fiscal_year"] in args.years]

    if not filings:
        print("No filings matched the given --years filter.")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  Apple 3-Year Annual Report Ingestion Pipeline")
    print("="*65)
    print(f"  Filings to process : {len(filings)}")
    print(f"  XBRL -> PostgreSQL : {'SKIP' if args.skip_xbrl  else 'YES'}")
    print(f"  HTML -> ChromaDB   : {'SKIP' if args.skip_chroma else 'YES'}")
    print(f"  Dry run            : {'YES' if args.dry_run else 'NO'}")
    print("="*65)

    if args.dry_run:
        print("\n[DRY RUN] Would process these filings:")
        for f in filings:
            print(f"  • {f['description']}")
            print(f"    paper_id={f['paper_id']}  ticker={f['ticker']}  period={f['fiscal_period']}")
        return

    # Check required env vars
    missing = []
    if not args.skip_xbrl   and not os.getenv("POSTGRES_DSN"):
        missing.append("POSTGRES_DSN")
    if not args.skip_chroma and not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY  (for embedder)")
    if missing:
        print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
        print("   Set them in your .env file or as GitHub Secrets.")
        sys.exit(1)

    # ── Run ingestion for each filing ────────────────────────────────────
    report = []
    for i, filing in enumerate(filings, 1):
        print(f"\n[{i}/{len(filings)}] Processing: {filing['description']}")
        row = {"filing": filing["description"], "fiscal_period": filing["fiscal_period"]}

        if not args.skip_xbrl:
            xbrl_res = ingest_xbrl(filing)
            row["xbrl"] = xbrl_res
            time.sleep(2)  # Be polite to SEC EDGAR

        if not args.skip_chroma:
            chroma_res = ingest_chroma(filing, DOWNLOAD_DIR)
            row["chroma"] = chroma_res
            time.sleep(2)

        report.append(row)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  INGESTION SUMMARY")
    print("="*65)
    all_ok = True
    for row in report:
        xbrl_s   = row.get("xbrl",   {}).get("status", "skipped")
        chroma_s = row.get("chroma", {}).get("status", "skipped")
        xbrl_icon   = "[OK]" if xbrl_s   == "ok" else ("[SKIP]" if xbrl_s   == "skipped" else "[FAIL]")
        chroma_icon = "[OK]" if chroma_s == "ok" else ("[SKIP]" if chroma_s == "skipped" else "[FAIL]")
        if xbrl_s == "error" or chroma_s == "error":
            all_ok = False
        xbrl_detail   = f"({row['xbrl'].get('facts_inserted', 0):,} facts)"   if xbrl_s == "ok"   else row.get("xbrl", {}).get("error", "")
        chroma_detail = f"({row['chroma'].get('chunks_stored', 0):,} chunks)" if chroma_s == "ok" else row.get("chroma", {}).get("error", "")
        print(f"\n  {row['fiscal_period']}")
        print(f"    XBRL   {xbrl_icon}  {xbrl_detail}")
        print(f"    Chroma {chroma_icon}  {chroma_detail}")

    print("\n" + ("[OK] All ingestions completed successfully!" if all_ok else "[FAIL] Some ingestions failed -- check logs above."))
    print("="*65)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
