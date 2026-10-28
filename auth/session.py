# auth/session.py
"""
Session helpers for FinSight.

PKCE strategy for Streamlit — how the verifier survives the redirect:

  PRIMARY: ?cv=verifier encoded in redirect_to URL
  ─────────────────────────────────────────────────
  Supabase DOES pass through query params appended to redirect_to.
  e.g. redirect_to = "http://localhost:8501?cv=abc123"
  After OAuth, Supabase redirects to:  http://localhost:8501?code=XYZ&cv=abc123
  Works as long as the redirect URL pattern is whitelisted in Supabase Dashboard.

  REQUIREMENT: Add these to Supabase Dashboard → Auth → Redirect URLs:
    http://localhost:8501/**
    https://your-app.streamlit.app/**

  FALLBACKS (in order):
    1. Supabase pkce_sessions table  — survives container restarts
    2. /tmp file                     — same-container fallback
    3. st.session_state              — same-tab fallback
"""
import os
import sys
import json
import time
import secrets
import hashlib
import base64
import tempfile
from urllib.parse import urlencode

import streamlit as st
from auth.supabase_client import get_supabase

# ─────────────────────────────────────────────
# SET TO True TO SEE FULL PKCE DEBUG PANEL
# (disable before committing to production)
DEBUG_AUTH = False
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# PKCE HELPERS
# ─────────────────────────────────────────────

def _pkce_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()

def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# COOKIE HELPERS FOR PERSISTENT LOGIN
# ─────────────────────────────────────────────

def _save_session_cookie(refresh_token: str):
    """Sets a 30-day session cookie in the user's browser."""
    components.html(
        f"""
        <script>
            document.cookie = "finsight_rt={refresh_token}; path=/; max-age=2592000; SameSite=Lax";
        </script>
        """,
        height=0,
        width=0,
    )

def _clear_session_cookie():
    """Clears the persistent session cookie."""
    components.html(
        """
        <script>
            document.cookie = "finsight_rt=; path=/; max-age=0; SameSite=Lax";
        </script>
        """,
        height=0,
        width=0,
    )

# ─────────────────────────────────────────────
# PKCE VERIFIER STORE  (DB primary, /tmp fallback)
#
# WHY NOT temp file alone:
#   - Streamlit Cloud sleeps idle apps → container wiped → /tmp gone
#   - Multi-instance deployments: login hits container A, callback hits B
#
# WHY "most recent" pattern works:
#   Only one OAuth flow can be in progress at a time per user session.
#   We store the latest verifier and always read the freshest one.
# ─────────────────────────────────────────────
_PKCE_FILE = os.path.join(tempfile.gettempdir(), "finsight_pkce_verifier.json")
_PKCE_TTL  = 600  # 10 minutes


def _save_verifier(verifier: str) -> None:
    """
    Persist the PKCE verifier to:
      1. Supabase pkce_sessions table  (survives container restarts)
      2. /tmp file                      (instant local fallback)
      3. st.session_state              (works if tab never navigated away)
    """
    # — Layer 1: Supabase DB ———————————————————————————
    try:
        get_supabase().table("pkce_sessions").insert(
            {"verifier": verifier}
        ).execute()
        # Clean up expired rows while we're here
        get_supabase().table("pkce_sessions").delete().lt(
            "created_at",
            f"now() - interval '{_PKCE_TTL} seconds'"
        ).execute()
        print("[PKCE] verifier saved → Supabase DB", file=sys.stderr)
    except Exception as e:
        print(f"[PKCE] DB save failed (will use /tmp): {e}", file=sys.stderr)

    # — Layer 2: /tmp file ———————————————————————————
    try:
        with open(_PKCE_FILE, "w") as f:
            json.dump({"verifier": verifier, "ts": time.time()}, f)
        print(f"[PKCE] verifier saved → {_PKCE_FILE}", file=sys.stderr)
    except Exception as e:
        print(f"[PKCE] /tmp save failed: {e}", file=sys.stderr)

    # — Layer 3: session_state (best-effort) —————————————————
    st.session_state["pkce_verifier"] = verifier


def _load_verifier() -> str:
    """
    Recover the PKCE verifier from the first available source:
      1. Supabase pkce_sessions  (most recent row within TTL)
      2. /tmp file               (same-container fallback)
      3. st.session_state        (same-tab fallback)
    Cleans up consumed records automatically.
    """
    # — Layer 1: Supabase DB ———————————————————————————
    try:
        res = (
            get_supabase()
            .table("pkce_sessions")
            .select("id, verifier, created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            row = res.data[0]
            from datetime import datetime, timezone
            created = datetime.fromisoformat(
                row["created_at"].replace("Z", "+00:00")
            )
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age < _PKCE_TTL:
                # Delete consumed row
                get_supabase().table("pkce_sessions").delete().eq(
                    "id", row["id"]
                ).execute()
                print(f"[PKCE] verifier loaded from DB (age={age:.1f}s)", file=sys.stderr)
                return row["verifier"]
            print(f"[PKCE] DB row expired (age={age:.1f}s)", file=sys.stderr)
    except Exception as e:
        print(f"[PKCE] DB load failed: {e}", file=sys.stderr)

    # — Layer 2: /tmp file ———————————————————————————
    try:
        with open(_PKCE_FILE) as f:
            data = json.load(f)
        os.unlink(_PKCE_FILE)
        age = time.time() - data.get("ts", 0)
        if age < _PKCE_TTL:
            print(f"[PKCE] verifier loaded from /tmp (age={age:.1f}s)", file=sys.stderr)
            return data["verifier"]
        print(f"[PKCE] /tmp file expired (age={age:.1f}s)", file=sys.stderr)
    except FileNotFoundError:
        print("[PKCE] /tmp file not found", file=sys.stderr)
    except Exception as e:
        print(f"[PKCE] /tmp load error: {e}", file=sys.stderr)

    # — Layer 3: session_state ——————————————————————————
    cv = st.session_state.get("pkce_verifier", "")
    if cv:
        print("[PKCE] verifier loaded from session_state", file=sys.stderr)
    return cv


# ─────────────────────────────────────────────
# CORE AUTH HELPERS
# ─────────────────────────────────────────────

def get_current_user() -> dict | None:
    """Returns the logged-in user dict, or None if not authenticated."""
    return st.session_state.get("auth_user")

def is_authenticated() -> bool:
    return get_current_user() is not None

def require_auth() -> dict:
    """
    Enforces authentication.
    1. Returns in-memory session if active.
    2. Handles OAuth callback if ?code is in URL.
    3. Attempts silent auto-login via persistent browser cookie.
    4. Renders centered login page and stops execution if not authenticated.
    """
    if is_authenticated():
        return get_current_user()

    params = st.query_params.to_dict()

    # ── OAuth callback ──────────────────────────────────────────────────
    if "code" in params:
        # Layer 1 (primary): ?cv= in the callback URL — Supabase passes it through
        code_verifier = params.get("cv", "")
        if code_verifier:
            print("[PKCE] verifier recovered from ?cv= URL param", file=sys.stderr)

        # Layer 2: Supabase DB (survives container restarts)
        if not code_verifier:
            code_verifier = _load_verifier()

        if DEBUG_AUTH:
            with st.expander("PKCE Debug Info", expanded=True):
                st.markdown("**All query params returned by Supabase:**")
                st.json(params)
                st.markdown(f"**`?cv=` in URL:** `{repr(params.get('cv', ''))}`")
                st.markdown(f"**Final code_verifier:** `{repr(code_verifier[:20] + '...' if len(code_verifier) > 20 else code_verifier)}`  "
                            f"(len={len(code_verifier)})")
                st.markdown(f"**auth_code (first 20 chars):** `{params['code'][:20]}...`")
            print(f"[DEBUG-PKCE] params keys={list(params.keys())}", file=sys.stderr)
            print(f"[DEBUG-PKCE] cv={repr(code_verifier[:20])}...", file=sys.stderr)

        _handle_oauth_callback(params["code"], code_verifier)
        if is_authenticated():
            return get_current_user()

    # Silent cookie restore (survives tab close & server restart)
    try:
        cookie_rt = st.context.cookies.get("finsight_rt")
        if cookie_rt:
            sb = get_supabase()
            res = sb.auth.refresh_session(cookie_rt)
            if res and res.user:
                _store_user(res.user)
                if res.session and res.session.refresh_token:
                    _save_session_cookie(res.session.refresh_token)
                return get_current_user()
    except Exception:
        _clear_session_cookie()

    if not is_authenticated():
        _render_login_page()
        st.stop()

    return get_current_user()


def get_google_auth_url() -> str:
    """
    Generates a fresh PKCE verifier, saves it to fallback layers,
    and returns the Supabase OAuth authorization URL.
    """
    verifier  = _pkce_verifier()
    challenge = _pkce_challenge(verifier)

    base_redirect = _get_redirect_url()
    redirect_with_cv = f"{base_redirect}?cv={verifier}"

    _save_verifier(verifier)

    try:
        supabase_url = st.secrets["supabase"]["url"]
    except (KeyError, FileNotFoundError):
        supabase_url = os.environ.get("SUPABASE_URL", "")

    params = {
        "provider": "google",
        "redirect_to": redirect_with_cv,
        "code_challenge": challenge,
        "code_challenge_method": "s256",
    }
    return f"{supabase_url}/auth/v1/authorize?{urlencode(params)}"


def login_with_google():
    """
    Initiates Google OAuth with PKCE.
    Uses target=_top link navigation to prevent Google from blocking
    OAuth inside Streamlit Cloud iframes.
    """
    auth_url = get_google_auth_url()
    st.markdown(
        f"""
        <script>window.top.location.href = "{auth_url}";</script>
        <meta http-equiv="refresh" content="0; url={auth_url}">
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def logout():
    """Clears the session, clears browser cookies, and signs out from Supabase."""
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass
    _clear_session_cookie()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.query_params.clear()
    st.rerun()


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _handle_oauth_callback(code: str, code_verifier: str):
    """Exchanges the OAuth code + PKCE verifier for a Supabase session."""
    if not code:
        return
    print(f"[Auth] exchange attempt: code={code[:15]}..., "
          f"cv_present={bool(code_verifier)}, cv_len={len(code_verifier)}",
          file=sys.stderr)
    try:
        supabase = get_supabase()
        exchange_params = {"auth_code": code}
        if code_verifier:
            exchange_params["code_verifier"] = code_verifier
        res = supabase.auth.exchange_code_for_session(exchange_params)
        if res.user:
            _store_user(res.user)
            if res.session and res.session.refresh_token:
                _save_session_cookie(res.session.refresh_token)
            st.query_params.clear()
            if DEBUG_AUTH:
                print("[Auth] exchange SUCCESS", file=sys.stderr)
    except Exception as e:
        err_str = str(e)
        st.error(f"Authentication failed: {err_str}")
        print(f"[Auth] exchange FAILED: {err_str}", file=sys.stderr)
        if DEBUG_AUTH:
            st.expander("Exchange Error Detail", expanded=True).code(err_str)


def _store_user(user):
    """Normalises Supabase user and stores in session_state."""
    meta = user.user_metadata or {}
    st.session_state["auth_user"] = {
        "id":         user.id,
        "email":      user.email,
        "name":       meta.get("full_name") or meta.get("name") or user.email.split("@")[0],
        "avatar_url": meta.get("avatar_url") or meta.get("picture") or "",
        "role":       "analyst",
    }
    try:
        row = (
            get_supabase()
            .table("user_profiles")
            .select("role")
            .eq("user_id", user.id)
            .single()
            .execute()
        )
        if row.data:
            st.session_state["auth_user"]["role"] = row.data.get("role", "analyst")
    except Exception:
        pass


def _get_redirect_url() -> str:
    """Returns the base OAuth redirect URL (without query params)."""
    # 1. Dynamically inspect request host header so localhost stays on localhost
    try:
        host = st.context.headers.get("host", "")
        if "localhost" in host or "127.0.0.1" in host:
            return f"http://{host}"
        if "streamlit.app" in host:
            return f"https://{host}"
    except Exception:
        pass

    # 2. Check local environment override
    override = os.environ.get("OAUTH_REDIRECT_URL", "")
    if override:
        return override

    # 3. Check secrets override
    try:
        sec_override = st.secrets["app"]["oauth_redirect_url"]
        if sec_override:
            return sec_override
    except (KeyError, FileNotFoundError):
        pass

    return "http://localhost:8501"


# ─────────────────────────────────────────────
# LOGIN PAGE UI (Centered, No Sidebar)
# ─────────────────────────────────────────────

def _render_login_page():
    st.markdown("""
    <style>
    /* 1. Fully eliminate sidebar and navigation on login screen */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* 2. Full viewport centering */
    .stMain, [data-testid="stMain"], section.main {
        margin-left: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    .stMainBlockContainer, [data-testid="stMainBlockContainer"], .block-container {
        max-width: 460px !important;
        margin: 14vh auto 0 auto !important;
        padding: 2.5rem 2rem 2rem 2rem !important;
        background: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
    }

    /* 3. Typography & Styling */
    .login-badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }
    .login-logo {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.025em;
        text-align: center;
        margin-bottom: 0.35rem;
    }
    .login-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 1.75rem;
        line-height: 1.45;
    }
    .login-divider {
        font-size: 0.7rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        text-align: center;
        margin-bottom: 1.25rem;
        font-weight: 600;
    }
    .login-footer {
        text-align: center;
        margin-top: 1.5rem;
        font-size: 0.72rem;
        color: #475569;
        line-height: 1.5;
    }
    </style>

    <div style="text-align: center;">
        <span class="login-badge">FINANCIAL RESEARCH PLATFORM</span>
        <div class="login-logo">FinSight</div>
        <div class="login-sub">
            Institutional Financial Research Engine<br>
            <span style="font-size: 0.76rem; color: #64748b;">
                Vectorized SEC EDGAR Retrieval &amp; Verified XBRL Grounding
            </span>
        </div>
        <div class="login-divider">Sign in with your Google account</div>
    </div>
    """, unsafe_allow_html=True)

    auth_url = get_google_auth_url()

    st.markdown(f"""
    <a href="{auth_url}" target="_top" style="text-decoration: none;">
        <div style="
            width: 100%;
            background-color: #2563eb;
            color: #ffffff;
            font-weight: 600;
            padding: 0.65rem 1rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            cursor: pointer;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            transition: background-color 0.15s ease;
            box-sizing: border-box;
        " onmouseover="this.style.backgroundColor='#1d4ed8'" onmouseout="this.style.backgroundColor='#2563eb'">
            <svg width="18" height="18" viewBox="0 0 24 24">
                <path fill="#ffffff" d="M21.35 11.1h-9.17v2.98h5.27c-.23 1.25-.94 2.3-2 3.01v2.5h3.24c1.89-1.74 2.98-4.3 2.98-7.39 0-.7-.06-1.37-.18-2.1z"/>
                <path fill="#ffffff" d="M12.18 21c2.7 0 4.96-.89 6.62-2.42l-3.24-2.5c-.9.6-2.05.96-3.38.96-2.6 0-4.8-1.76-5.59-4.12H3.25v2.58A9.99 9.99 0 0 0 12.18 21z"/>
                <path fill="#ffffff" d="M6.59 12.92c-.2-.6-.31-1.24-.31-1.92s.11-1.32.31-1.92V6.5H3.25A9.99 9.99 0 0 0 2.18 11c0 1.61.39 3.14 1.07 4.5l3.34-2.58z"/>
                <path fill="#ffffff" d="M12.18 4.96c1.47 0 2.79.51 3.83 1.5l2.87-2.87C17.13 1.95 14.87 1 12.18 1 7.63 1 3.73 3.58 2.18 7.5l3.34 2.58c.79-2.36 2.99-4.12 5.59-4.12z"/>
            </svg>
            Continue with Google
        </div>
    </a>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-footer">
        Private &amp; Auditable Research Environment<br>
        Documents, chats, and audit logs are strictly isolated to your user ID.
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_user(current_user: dict, page_key: str = "main"):
    """Renders the standard FinSight user identity card in the sidebar."""
    from ui.components import render_user_profile
    render_user_profile(current_user)

