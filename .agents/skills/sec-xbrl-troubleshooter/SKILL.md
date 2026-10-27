---
name: sec-xbrl-troubleshooter
description: >-
  Use this skill when diagnosing SEC EDGAR XBRL ingestion issues, missing financial facts, 
  10-K/10-Q parsing failures, or vector dimension mismatches in Supabase pgvector.
---

# SEC & XBRL Ingestion Troubleshooter

Follow this operational runbook to diagnose and repair missing financial facts, unindexed filings, or vector embedding mismatches.

## Step 1: Run Diagnostic Script
Execute the deterministic health check to inspect database connectivity, embedding dimensions, and XBRL facts:
```bash
python .agents/skills/sec-xbrl-troubleshooter/scripts/diagnose_pipeline.py --ticker AAPL
```
- **If script reports zero vector rows:** Jump to [Step 2: Vector Store Recovery](#step-2-vector-store-recovery).
- **If script reports dimension mismatch:** See [references/schemas.md](./references/schemas.md).
- **If script reports missing XBRL facts:** Jump to [Step 3: XBRL Facts Audit](#step-3-xbrl-facts-audit).

## Step 2: Vector Store Recovery
1. Open and inspect [indexing/vector_store.py](file:///c:/Users/ayush/dev/financial-report-rag/indexing/vector_store.py).
2. Verify the embedding model dimension matches the Supabase column definition:
   - `BAAI/bge-small-en-v1.5` → **384 dimensions**
   - `text-embedding-3-small` → **1536 dimensions**
3. Verify connection pool acquisition in `_SupabaseBackend`:
   - Connection must be checked out via `self._pool.getconn()` and returned in a `finally` block via `self._pool.putconn(conn)`.
4. Run an isolated search test:
   ```bash
   python -c "from indexing.vector_store import VectorStore; vs = VectorStore(); print(vs.search('total revenue', ticker='AAPL', top_k=2))"
   ```

## Step 3: XBRL Facts Audit
1. Open [retrieval/xbrl_retriever.py](file:///c:/Users/ayush/dev/financial-report-rag/retrieval/xbrl_retriever.py).
2. Check if the requested metric matches the canonical taxonomy in [utils/canonical_map.py](file:///c:/Users/ayush/dev/financial-report-rag/utils/canonical_map.py).
3. If an EDGAR company reports under non-standard tags (e.g., `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`), query raw facts directly:
   ```bash
   python -c "from retrieval.xbrl_retriever import XBRLRetriever; r = XBRLRetriever(); print(r.get_key_financials('AAPL'))"
   ```

## Step 4: Verification Checklist
- [ ] `diagnose_pipeline.py` exits with status `0` (ALL CHECKS PASSED).
- [ ] Vector similarity query returns non-zero results with similarity > `0.50`.
- [ ] For known errors, consult [references/error_codes.md](./references/error_codes.md).
