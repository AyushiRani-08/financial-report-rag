"""
ui/theme.py — Modular design token system and theme manager for FinSight.
Provides unified, clean styling for Light and Dark modes without emojis or neon effects.
"""

from typing import Dict, Any
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS (Easily customize or extend themes here)
# ─────────────────────────────────────────────────────────────────────────────
THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "Light": {
        "bg": "#f9fafb",
        "surface": "#ffffff",
        "sidebar_bg": "#ffffff",
        "border": "#e5e7eb",
        "border_subtle": "#f3f4f6",
        "text": "#111827",
        "text_secondary": "#4b5563",
        "text_muted": "#6b7280",
        "primary": "#2563eb",
        "primary_hover": "#1d4ed8",
        "primary_text": "#ffffff",
        "secondary_bg": "#ffffff",
        "secondary_border": "#d1d5db",
        "secondary_text": "#374151",
        "secondary_hover": "#f3f4f6",
        "chat_user_bg": "#f3f4f6",
        "chat_assistant_bg": "#ffffff",
        "input_bg": "#ffffff",
        "input_border": "#d1d5db",
        "input_focus": "#2563eb",
        "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.05)",
    },
    "Dark": {
        "bg": "#0f172a",
        "surface": "#1e293b",
        "sidebar_bg": "#1e293b",
        "border": "#334155",
        "border_subtle": "#1e293b",
        "text": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "primary": "#2563eb",
        "primary_hover": "#1d4ed8",
        "primary_text": "#ffffff",
        "secondary_bg": "#1e293b",
        "secondary_border": "#475569",
        "secondary_text": "#e2e8f0",
        "secondary_hover": "#334155",
        "chat_user_bg": "#1e293b",
        "chat_assistant_bg": "#0f172a",
        "input_bg": "#0f172a",
        "input_border": "#334155",
        "input_focus": "#3b82f6",
        "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.2)",
    },
}


def get_current_theme() -> str:
    """Returns the current theme: 'Light' or 'Dark' (defaults to 'Dark')."""
    if "app_theme" not in st.session_state:
        st.session_state["app_theme"] = "Dark"
    return st.session_state["app_theme"]


def set_theme(theme_name: str):
    """Sets the active theme and triggers a rerun."""
    if theme_name in THEME_TOKENS and st.session_state.get("app_theme") != theme_name:
        st.session_state["app_theme"] = theme_name
        st.rerun()


def render_theme_toggle():
    """Renders a simple theme toggle radio."""
    current = get_current_theme()
    selected = st.radio(
        "Theme",
        options=["Dark", "Light"],
        horizontal=True,
        index=0 if current == "Dark" else 1,
        key="ui_theme_radio",
        label_visibility="collapsed",
    )
    if selected != current:
        st.session_state["app_theme"] = selected
        st.rerun()


def apply_theme():
    """
    Applies clean, standard styling using the active design tokens.
    Guarantees consistent typography, inputs, containers, and buttons across themes.
    """
    theme = get_current_theme()
    t = THEME_TOKENS.get(theme, THEME_TOKENS["Dark"])

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* Global Base */
    .stApp {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}

    header[data-testid="stHeader"] {{
        background: {t['bg']} !important;
        border-bottom: 1px solid {t['border']} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']} !important;
        border-right: 1px solid {t['border']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {t['text_muted']} !important;
    }}

    /* Metric Cards */
    [data-testid="stMetric"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: {t['shadow_sm']} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {t['text']} !important;
        font-weight: 600 !important;
        font-size: 1.35rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {t['text_muted']} !important;
        font-size: 0.8rem !important;
    }}

    /* Text Inputs, Selectboxes, File Uploaders */
    .stTextInput input, .stSelectbox select, [data-baseweb="select"] > div {{
        background-color: {t['input_bg']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['input_border']} !important;
        border-radius: 6px !important;
    }}
    .stTextInput input:focus {{
        border-color: {t['input_focus']} !important;
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
        color: {t['text']} !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: {t['shadow_sm']} !important;
    }}
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{
        color: {t['text_secondary']} !important;
        line-height: 1.6 !important;
    }}

    /* Chat Input */
    [data-testid="stChatInput"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['input_border']} !important;
        border-radius: 8px !important;
        color: {t['text']} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {t['primary']} !important;
        color: {t['primary_text']} !important;
        border: 1px solid {t['primary']} !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.45rem 1rem !important;
    }}
    .stButton > button:hover {{
        background-color: {t['primary_hover']} !important;
        border-color: {t['primary_hover']} !important;
    }}
    .stButton > button[kind="secondary"] {{
        background-color: {t['secondary_bg']} !important;
        color: {t['secondary_text']} !important;
        border: 1px solid {t['secondary_border']} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background-color: {t['secondary_hover']} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 6px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {t['text']} !important;
        font-weight: 500 !important;
    }}

    /* Custom Scope Bar */
    .app-scope-bar {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: {t['text_secondary']};
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* Header styling */
    .app-main-header {{
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid {t['border']};
    }}
    .app-main-title {{
        font-size: 1.35rem;
        font-weight: 600;
        color: {t['text']};
        margin: 0;
        letter-spacing: -0.015em;
    }}
    .app-main-subtitle {{
        font-size: 0.85rem;
        color: {t['text_muted']};
        margin-top: 0.2rem;
    }}

    /* User Profile Card */
    .user-card {{
        padding: 0.55rem 0.75rem;
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }}
    .user-card-name {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {t['text']};
    }}
    .user-card-email {{
        font-size: 0.75rem;
        color: {t['text_muted']};
    }}

    /* Dividers */
    hr {{
        border-color: {t['border']} !important;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
