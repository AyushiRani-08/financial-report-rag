-- =============================================================
-- Financial Report RAG — PostgreSQL Schema
-- Supports: US (SEC/XBRL/US-GAAP) + India (MCA/AOC-4/Ind-AS)
-- Apply with: psql -U <user> -d financial_rag -f db/schema.sql
--             (or: python db/apply_schema.py)
--
-- This file is idempotent: it creates tables if missing, adds new
-- columns to existing tables, removes exact duplicates and then
-- creates the indexes.
-- =============================================================

CREATE TABLE IF NOT EXISTS companies (
    company_id      SERIAL PRIMARY KEY,
    ticker          TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    cik             TEXT,              -- SEC CIK (US) or CIN (India)
    market          TEXT NOT NULL,     -- 'US' or 'IN'
    sector          TEXT
);

CREATE TABLE IF NOT EXISTS filings (
    filing_id       SERIAL PRIMARY KEY,
    company_id      INTEGER REFERENCES companies(company_id) ON DELETE CASCADE,
    form_type       TEXT NOT NULL,     -- '10-K', '10-Q', 'AOC-4'
    fiscal_period   TEXT NOT NULL,     -- 'FY2025', 'Q2FY2025'
    filed_date      DATE,
    source_url      TEXT
);

CREATE TABLE IF NOT EXISTS financial_facts (
    fact_id         SERIAL PRIMARY KEY,
    filing_id       INTEGER REFERENCES filings(filing_id) ON DELETE CASCADE,
    tag             TEXT NOT NULL,     -- raw tag: 'us-gaap:Revenues'
    canonical_field TEXT NOT NULL,     -- normalized field: 'revenue', 'net_income'
    value           NUMERIC,
    unit            TEXT,              -- 'USD', 'USD/share', 'shares'
    period_start    DATE,              -- NULL for point-in-time (balance sheet) facts
    period_end      DATE,
    duration_days   INTEGER,           -- (period_end - period_start) in days; NULL for instants
    is_instant      BOOLEAN,           -- TRUE for balance-sheet facts
    source_page     TEXT               -- page/section citation for RAG attribution
);

-- ── Forward-compatible column additions for existing databases ───
ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS duration_days INTEGER;
ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS is_instant BOOLEAN;

-- ── Deduplicate before enforcing the natural key ─────────────────
-- A filing reports the same tag for several overlapping periods, and
-- re-running ingestion used to duplicate every row. Keep one row per
-- (filing, tag, unit, period) — the earliest inserted wins.
DELETE FROM financial_facts a
USING financial_facts b
WHERE a.fact_id > b.fact_id
  AND a.filing_id = b.filing_id
  AND a.tag = b.tag
  AND COALESCE(a.unit, '') = COALESCE(b.unit, '')
  AND COALESCE(a.period_start, DATE '0001-01-01') = COALESCE(b.period_start, DATE '0001-01-01')
  AND COALESCE(a.period_end, DATE '0001-01-01') = COALESCE(b.period_end, DATE '0001-01-01');

-- Re-ingestion is idempotent from now on.
CREATE UNIQUE INDEX IF NOT EXISTS uq_facts_natural_key
    ON financial_facts (
        filing_id,
        tag,
        COALESCE(unit, ''),
        COALESCE(period_start, DATE '0001-01-01'),
        COALESCE(period_end, DATE '0001-01-01')
    );

-- Primary RAG lookup: find facts by normalized field name for a filing
CREATE INDEX IF NOT EXISTS idx_facts_lookup
    ON financial_facts (canonical_field, filing_id);

-- Support filtering filings by company
CREATE INDEX IF NOT EXISTS idx_filings_company
    ON filings (company_id);

-- Support time-series queries (most recent period first). Ordered by real
-- dates, never by the fiscal_period label ('Q4FY2024' sorts above 'FY2024').
CREATE INDEX IF NOT EXISTS idx_facts_period
    ON financial_facts (period_end DESC, duration_days DESC);

-- Support market-level filtering (US vs IN)
CREATE INDEX IF NOT EXISTS idx_companies_market
    ON companies (market, sector);
