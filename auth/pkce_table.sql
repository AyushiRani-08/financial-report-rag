-- Run this ONCE in Supabase Dashboard → SQL Editor
-- Creates the pkce_sessions table so PKCE verifiers survive container restarts.
-- Without this, the app falls back to /tmp which breaks on container sleep/restart.

CREATE TABLE IF NOT EXISTS pkce_sessions (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    verifier    TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index so expiry queries are fast
CREATE INDEX IF NOT EXISTS pkce_sessions_ts_idx ON pkce_sessions (created_at);

-- Disable RLS — only holds short-lived random tokens, no PII
ALTER TABLE pkce_sessions DISABLE ROW LEVEL SECURITY;

-- Grant access to the anon and service_role
GRANT SELECT, INSERT, DELETE ON pkce_sessions TO anon;
GRANT SELECT, INSERT, DELETE ON pkce_sessions TO service_role;
