# auth/supabase_client.py
"""
Singleton Supabase client for FinSight.
Uses SUPABASE_URL and SUPABASE_KEY from environment / .streamlit/secrets.toml.
"""
import os
import streamlit as st
from supabase import create_client, Client


def get_supabase() -> Client:
    """
    Returns a cached Supabase client.
    Reads credentials from st.secrets (production) with fallback to env vars (local dev).
    """
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]
    except (KeyError, FileNotFoundError):
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not found. "
            "Add [supabase] url and anon_key to .streamlit/secrets.toml "
            "or set SUPABASE_URL / SUPABASE_ANON_KEY environment variables."
        )
    return create_client(url, key)


def get_supabase_admin() -> Client:
    """
    Returns a Supabase client with the service role key (bypasses RLS).
    Use ONLY for server-side operations like storage uploads and admin tasks.
    Never expose this key to the browser.
    """
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_role_key"]
    except (KeyError, FileNotFoundError):
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Supabase service role key not found. "
            "Add service_role_key to .streamlit/secrets.toml "
            "or set SUPABASE_SERVICE_ROLE_KEY environment variable."
        )
    return create_client(url, key)
