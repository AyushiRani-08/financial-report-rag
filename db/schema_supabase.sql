-- =============================================================
-- FinSight — Supabase Migration Schema
-- Run this in Supabase SQL Editor (project: isrmgpuopuzpmufmhsbe)
-- =============================================================

-- ── 0. Enable pgvector extension ─────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── 1. User profiles (auto-linked to auth.users on signup) ───
CREATE TABLE IF NOT EXISTS public.user_profiles (
    user_id     UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email       TEXT NOT NULL,
    full_name   TEXT,
    avatar_url  TEXT,
    role        TEXT NOT NULL DEFAULT 'analyst',  -- 'viewer' | 'analyst' | 'admin'
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── 2. Add user_id to existing tables ────────────────────────
ALTER TABLE public.companies
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE public.filings
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- ── 3. Embeddings table (replaces ChromaDB) ──────────────────
CREATE TABLE IF NOT EXISTS public.embeddings (
    id              TEXT PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    paper_id        TEXT NOT NULL,
    ticker          TEXT DEFAULT '',
    fiscal_period   TEXT DEFAULT '',
    form_type       TEXT DEFAULT '',
    page_number     INT  DEFAULT 1,
    word_count      INT  DEFAULT 0,
    file_type       TEXT DEFAULT '',
    content         TEXT NOT NULL,
    embedding       VECTOR(384),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── 4. Indexes ────────────────────────────────────────────────
-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON public.embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_embeddings_user_paper
    ON public.embeddings (user_id, paper_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_ticker
    ON public.embeddings (user_id, ticker);

-- ── 5. Enable Row Level Security ─────────────────────────────
ALTER TABLE public.user_profiles   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.embeddings      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.companies       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.filings         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.financial_facts ENABLE ROW LEVEL SECURITY;

-- ── 6. RLS Policies ──────────────────────────────────────────

-- user_profiles: users can only read/update their own profile
CREATE POLICY "profiles_select_own" ON public.user_profiles
    FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "profiles_update_own" ON public.user_profiles
    FOR UPDATE USING (user_id = auth.uid());

-- embeddings: fully isolated per user
CREATE POLICY "embeddings_all_own" ON public.embeddings
    FOR ALL USING (user_id = auth.uid());

-- companies: users see their own + shared global rows (user_id IS NULL)
CREATE POLICY "companies_own_or_global" ON public.companies
    FOR ALL USING (user_id = auth.uid() OR user_id IS NULL);

-- filings: same pattern as companies
CREATE POLICY "filings_own_or_global" ON public.filings
    FOR ALL USING (user_id = auth.uid() OR user_id IS NULL);

-- financial_facts: accessible if the parent filing is accessible
CREATE POLICY "facts_via_filings" ON public.financial_facts
    FOR ALL USING (
        filing_id IN (
            SELECT filing_id FROM public.filings
            WHERE user_id = auth.uid() OR user_id IS NULL
        )
    );

-- ── 7. Auto-create user profile on Google signup ─────────────
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.user_profiles (user_id, email, full_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name'),
        NEW.raw_user_meta_data->>'avatar_url'
    )
    ON CONFLICT (user_id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ── 8. Helper: list documents for a user ─────────────────────
CREATE OR REPLACE FUNCTION public.get_user_documents(p_user_id UUID)
RETURNS TABLE(paper_id TEXT, ticker TEXT, fiscal_period TEXT, chunk_count BIGINT)
LANGUAGE sql
SECURITY DEFINER
AS $$
    SELECT
        paper_id,
        ticker,
        fiscal_period,
        COUNT(*) AS chunk_count
    FROM public.embeddings
    WHERE user_id = p_user_id
    GROUP BY paper_id, ticker, fiscal_period
    ORDER BY paper_id;
$$;
