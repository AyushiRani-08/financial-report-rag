# auth/session.py
"""
Session helpers for FinSight.

PKCE strategy for Streamlit (stateless):
  1. Generate code_verifier + code_challenge ourselves before redirecting.
  2. Encode the code_verifier directly into the redirect_to URL as ?cv=...
     so it survives the external Google redirect.
  3. On callback, read ?code and ?cv from query params, inject cv into
     the gotrue client, then call exchange_code_for_session.
"""
import os
import secrets
import hashlib
import base64
from urllib.parse import urlencode

import streamlit as st
from auth.supabase_client import get_supabase


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

    # OAuth callback
    if "code" in params:
        code_verifier = params.get("cv", "")
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


def login_with_google():
    """Initiates Google OAuth with PKCE, encoding the verifier in the redirect URL."""
    verifier  = _pkce_verifier()
    challenge = _pkce_challenge(verifier)

    base_redirect = _get_redirect_url()
    # Encode verifier in redirect URL so it survives the external redirect
    redirect_with_cv = f"{base_redirect}?cv={verifier}"

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
    auth_url = f"{supabase_url}/auth/v1/authorize?{urlencode(params)}"

    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
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
    import sys
    print(f"[Auth] callback: code={code[:15]}..., cv_present={bool(code_verifier)}, cv_len={len(code_verifier)}", file=sys.stderr)
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
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        print(f"[Auth] exchange error: {e}", file=sys.stderr)


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
    try:
        override = st.secrets["app"]["oauth_redirect_url"]
        if override:
            return override
    except (KeyError, FileNotFoundError):
        pass
    override = os.environ.get("OAUTH_REDIRECT_URL", "")
    if override:
        return override
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
        <div class="login-logo">📊 FinSight</div>
        <div class="login-sub">
            Institutional Financial Research Engine<br>
            <span style="font-size: 0.76rem; color: #64748b;">
                Vectorized SEC EDGAR Retrieval &amp; Verified XBRL Grounding
            </span>
        </div>
        <div class="login-divider">Sign in with your institutional or Google account</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔐  Continue with Google",
                 use_container_width=True,
                 key="btn_google_login",
                 type="primary"):
        login_with_google()

    st.markdown("""
    <div class="login-footer">
        Private &amp; Auditable Research Environment<br>
        Documents, chats, and audit logs are strictly isolated to your user ID.
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_user(current_user: dict, page_key: str = "main"):
    """Renders the standard FinSight user identity card & logout button in the sidebar."""
    avatar = current_user.get("avatar_url", "")
    avatar_html = (
        f'<img src="{avatar}" style="width:28px;height:28px;border-radius:50%;vertical-align:middle;margin-right:8px;">'
        if avatar else
        '<span style="font-size:1.1rem;margin-right:6px;">👤</span>'
    )
    user_name = current_user.get("name", "User")
    user_email = current_user.get("email", "")
    st.sidebar.markdown(f"""
    <div style="padding: 0.4rem 0 0.8rem 0;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">
            FINSIGHT
        </div>
        <div style="font-size: 0.76rem; color: #64748b; margin-top: 1px;">
            Institutional Financial Research Engine
        </div>
        <div style="
            margin-top: 0.85rem;
            padding: 0.55rem 0.75rem;
            background: #0f1624;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 6px;
            display: flex;
            align-items: center;
        ">
            {avatar_html}
            <div>
                <div style="font-size:0.8rem;font-weight:600;color:#e2e8f0;line-height:1.2;">{user_name}</div>
                <div style="font-size:0.7rem;color:#64748b;">{user_email}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("Sign Out", key=f"btn_logout_{page_key}", use_container_width=True):
        logout()
