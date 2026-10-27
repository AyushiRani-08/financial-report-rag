# Database Schemas & Vector Config

### 1. `embeddings` Table (Vector Store)
```sql
CREATE TABLE public.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT,
    ticker VARCHAR(10) NOT NULL,
    fiscal_period VARCHAR(10),
    form_type VARCHAR(10),
    page_number INT,
    word_count INT,
    file_type VARCHAR(20),
    content TEXT,
    embedding vector(1536)  -- Or 384 for bge-small
);

CREATE INDEX ON public.embeddings USING ivfflat (embedding vector_cosine_ops);
```

### 2. Normalized XBRL Facts Hierarchy
- `companies`: `(id, ticker, name, cik)`
- `filings`: `(id, company_id, form, filing_date, period_end)`
- `financial_facts`: `(id, filing_id, canonical_name, raw_tag, value, unit, decimals)`
