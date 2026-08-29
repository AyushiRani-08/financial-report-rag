-- =============================================================
-- Financial Report RAG — PostgreSQL Schema
-- Supports: US (SEC/XBRL/US-GAAP) + India (MCA/AOC-4/Ind-AS)
-- Apply with: psql -U <user> -d financial_rag -f db/schema.sql
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
    fiscal_period   TEXT NOT NULL,     -- 'FY2025', 'Q3-2025'
    filed_date      DATE,
    source_url      TEXT
);

CREATE TABLE IF NOT EXISTS financial_facts (
    fact_id         SERIAL PRIMARY KEY,
    filing_id       INTEGER REFERENCES filings(filing_id) ON DELETE CASCADE,
    tag             TEXT NOT NULL,     -- raw tag: 'us-gaap:Revenues' or 'in-gaap:Revenue'
    canonical_field TEXT NOT NULL,     -- normalized field: 'revenue', 'net_income', 'eps', etc.
    value           NUMERIC,
    unit            TEXT,              -- 'USD', 'INR', 'shares'
    period_start    DATE,
    period_end      DATE,
    source_page     TEXT               -- page/section citation for RAG attribution
);

-- Primary RAG lookup: find facts by normalized field name for a filing
CREATE INDEX IF NOT EXISTS idx_facts_lookup
    ON financial_facts (canonical_field, filing_id);

-- Support filtering filings by company
CREATE INDEX IF NOT EXISTS idx_filings_company
    ON filings (company_id);

-- Support time-series queries (most recent period first)
CREATE INDEX IF NOT EXISTS idx_facts_period
    ON financial_facts (period_end DESC);

-- Support market-level filtering (US vs IN)
CREATE INDEX IF NOT EXISTS idx_companies_market
    ON companies (market, sector);
