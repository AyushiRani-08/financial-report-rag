"""
ui/components.py — Reusable, modular UI components for FinSight.
Uses monochrome SVG icons from ui.icons and standard styling from ui.theme.
"""

from typing import List, Optional, Callable
import streamlit as st
from ui.icons import get_icon
from ui.theme import render_theme_toggle


def render_user_profile(current_user: dict):
    """
    Renders a clean, minimalist user profile card without emojis.
    Sign out is moved into the Settings menu.
    """
    avatar = current_user.get("avatar_url", "")
    user_name = current_user.get("name", "User")
    user_email = current_user.get("email", "")

    if avatar:
        avatar_html = (
            f'<img src="{avatar}" style="width:28px;height:28px;border-radius:50%;'
            f'vertical-align:middle;margin-right:8px;object-fit:cover;">'
        )
    else:
        # Minimal circular initial avatar
        initial = (user_name[:1] or "U").upper()
        avatar_html = (
            f'<div style="width:28px;height:28px;border-radius:50%;'
            f'background:rgba(128,128,128,0.2);display:inline-flex;align-items:center;'
            f'justify-content:center;font-size:0.78rem;font-weight:600;'
            f'margin-right:8px;flex-shrink:0;">{initial}</div>'
        )

    st.sidebar.markdown(f"""
    <div style="padding: 0.2rem 0 0.4rem 0;">
        <div style="font-size: 1.05rem; font-weight: 700; letter-spacing: -0.015em; margin-bottom: 2px;">
            FinSight
        </div>
        <div style="font-size: 0.75rem; opacity: 0.7; margin-bottom: 0.6rem;">
            Financial Research &amp; Fact Verification
        </div>
        <div class="user-card" style="display: flex; align-items: center;">
            {avatar_html}
            <div style="overflow: hidden; min-width: 0;">
                <div class="user-card-name" style="text-overflow: ellipsis; white-space: nowrap; overflow: hidden;">{user_name}</div>
                <div class="user-card-email" style="text-overflow: ellipsis; white-space: nowrap; overflow: hidden;">{user_email}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_settings_section(
    available_docs: Optional[List[str]] = None,
    on_clear_chat: Optional[Callable] = None,
    on_reset_cache: Optional[Callable] = None,
    on_delete_doc: Optional[Callable] = None,
    on_logout: Optional[Callable] = None,
):
    """
    Renders a unified, clean 'Settings' expander containing:
    1. Theme toggle (Light / Dark)
    2. Search retrieval depth (Top-K)
    3. Session actions (Clear chat, reset cache)
    4. Document deletion (if documents exist)
    5. Sign out button
    """
    settings_label = f"Settings"
    with st.expander(settings_label, expanded=False):
        # 1. Theme
        st.markdown(f"**Appearance**")
        render_theme_toggle()

        # 2. Search Depth (only if on main research workspace)
        if "slider_top_k" in st.session_state or on_clear_chat is not None:
            st.markdown(f"**Search Depth**")
            st.slider(
                "Context chunks (Top-K)",
                min_value=1,
                max_value=8,
                value=4,
                key="slider_top_k",
            )

        # 3. Session Actions
        if on_clear_chat or on_reset_cache:
            st.markdown(f"**Data & Session**")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("Clear conversation", use_container_width=True, key="btn_clear_chat", type="secondary"):
                    if on_clear_chat:
                        on_clear_chat()
                    st.rerun()
            with col_s2:
                if st.button("Reset cache", use_container_width=True, key="btn_reset_cache", type="secondary"):
                    if on_reset_cache:
                        on_reset_cache()
                    st.rerun()

        # 4. Document cleanup
        if available_docs and on_delete_doc:
            st.markdown(f"**Document Cleanup**")
            doc_to_del = st.selectbox("Select document", available_docs, key="sel_doc_del")
            if st.button("Remove document", use_container_width=True, key="btn_del_doc", type="secondary"):
                on_delete_doc(doc_to_del)
                st.rerun()

        # 5. Sign Out (Clean, standard button at the bottom of settings)
        if on_logout:
            st.markdown("---")
            if st.button("Sign out", key="btn_settings_logout", use_container_width=True, type="secondary"):
                on_logout()


def render_scope_bar(scope_label: str, grounding_label: str):
    """Renders the active document scope bar without emojis."""
    st.markdown(
        f'<div class="app-scope-bar">'
        f'<span>Scope: <b>{scope_label}</b></span>'
        f'<span style="font-size: 0.78rem; opacity: 0.85;">{grounding_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
