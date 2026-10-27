#!/usr/bin/env python3
"""
scripts/debug_pkce.py
----------------------
PKCE Authentication Diagnostic Tool.

Run this script to diagnose "code challenge does not match previously saved
code verifier" errors in your Streamlit/Supabase OAuth flow.

Usage:
    python scripts/debug_pkce.py

What it checks:
1. SUPABASE_URL and SUPABASE_ANON_KEY are set
2. Supabase pkce_sessions table is reachable and shows recent rows
3. /tmp PKCE verifier file exists and has not expired
4. Confirms redirect URL configuration advice
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

PKCE_FILE = os.path.join(tempfile.gettempdir(), "finsight_pkce_verifier.json")
PKCE_TTL  = 600  # 10 minutes (must match auth/session.py)

SEP = "=" * 65


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")


def check_env_vars():
    section("1. Environment Variables")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if url:
        ok(f"SUPABASE_URL is set ({url[:40]}...)")
    else:
        fail("SUPABASE_URL is NOT set — auth will fail entirely")

    if key:
        ok(f"SUPABASE_ANON_KEY is set ({key[:20]}...)")
    else:
        fail("SUPABASE_ANON_KEY is NOT set — auth will fail entirely")

    redirect = os.environ.get("SITE_URL") or os.environ.get("STREAMLIT_SITE_URL") or ""
    if redirect:
        ok(f"SITE_URL / STREAMLIT_SITE_URL = {redirect}")
    else:
        warn("SITE_URL not set. auth/session.py infers redirect URL from headers — verify it matches Supabase allowed redirect URLs")


def check_pkce_table():
    section("2. Supabase pkce_sessions Table")
    try:
        from auth.session import get_supabase
        sb = get_supabase()
        res = (
            sb.table("pkce_sessions")
            .select("id, created_at")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        rows = res.data or []
        ok(f"pkce_sessions table reachable. Found {len(rows)} recent row(s).")
        if rows:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            for row in rows:
                created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                age = (now - created).total_seconds()
                status = "VALID" if age < PKCE_TTL else "EXPIRED"
                info(f"  id={row['id'][:8]}... created {age:.0f}s ago [{status}]")
        else:
            warn("No rows in pkce_sessions. This means no OAuth flow has been initiated yet, OR rows are being deleted immediately after use (normal).")
    except Exception as e:
        fail(f"Could not query pkce_sessions: {e}")
        warn("Possible causes:")
        warn("  - Table does not exist. Run the migration:")
        warn("      CREATE TABLE IF NOT EXISTS public.pkce_sessions (")
        warn("        id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),")
        warn("        verifier   TEXT NOT NULL,")
        warn("        created_at TIMESTAMPTZ NOT NULL DEFAULT now()")
        warn("      );")
        warn("  - Row-Level Security blocks SELECT. Disable RLS or add a policy.")


def check_tmp_file():
    section("3. /tmp PKCE Verifier File")
    if os.path.exists(PKCE_FILE):
        try:
            with open(PKCE_FILE) as f:
                data = json.load(f)
            age = time.time() - data.get("ts", 0)
            if age < PKCE_TTL:
                ok(f"File exists and is VALID (age={age:.0f}s, path={PKCE_FILE})")
            else:
                warn(f"File exists but is EXPIRED (age={age:.0f}s > TTL={PKCE_TTL}s)")
        except Exception as e:
            fail(f"File exists but could not be read: {e}")
    else:
        info(f"No /tmp verifier file found at {PKCE_FILE}")
        info("This is normal if no flow is in progress or the file was already consumed.")


def check_redirect_urls():
    section("4. Redirect URL Configuration Advice")
    info("The PKCE 'code challenge does not match' error almost always means")
    info("the verifier was saved in one container/tab and read in another.")
    info("")
    info("Ensure these URLs are in Supabase Dashboard -> Auth -> Redirect URLs:")
    info("  http://localhost:8501/**")
    info("  https://<your-app>.streamlit.app/**")
    info("")
    info("The ?cv= URL param is the PRIMARY verifier carrier.")
    info("If Supabase strips or truncates the redirect_to URL, ?cv= is lost.")
    info("Fix: Keep the verifier short. Current impl uses 43-char base64url (fine).")
    info("")
    info("If deploying on Streamlit Community Cloud with multiple instances:")
    info("  - Use Supabase pkce_sessions table as the only persistent store")
    info("  - The /tmp file WILL fail across pods (each pod has its own /tmp)")


def main():
    print(SEP)
    print("  PKCE Authentication Diagnostic — Financial RAG")
    print(SEP)

    check_env_vars()
    check_pkce_table()
    check_tmp_file()
    check_redirect_urls()

    print(f"\n{SEP}")
    print("  Diagnosis complete.")
    print(SEP)


if __name__ == "__main__":
    main()
