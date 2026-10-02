# app.py — FinSight navigation shell
# Defines page titles and order for the sidebar nav.
# All page content lives in separate files.
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="FinSight — Financial Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("research_workspace.py", title="Research Workspace"),
    st.Page("pages/1_Dashboard.py",  title="Dashboard"),
    st.Page("pages/2_Evaluation.py", title="Evaluation"),
])
pg.run()