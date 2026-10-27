-- Run this once in Supabase Dashboard → SQL Editor
-- Creates a tiny table to store PKCE verifiers server-side.
-- Survives container restarts, sleep cycles, and multi-instance deployments.

CREATE TABLE IF NOT EXISTS pkce_sessions (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    verifier    TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index so expiry queries are fast
CREATE INDEX IF NOT EXISTS pkce_sessions_ts_idx ON pkce_sessions (created_at);

-- Disable RLS — only holds short-lived random tokens, no PII
ALTER TABLE pkce_sessions DISABLE ROW LEVEL SECURITY;
