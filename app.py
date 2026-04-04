import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils.style import inject_css, SHIELD_SVG

st.set_page_config(
    page_title="Aegis",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Define pages ───────────────────────────────────────────────────────────────
overview = st.Page("pages/overview.py", title="Overview",   icon="🏠", default=True)
risk     = st.Page("pages/risk.py",     title="Risk Score", icon="⚠️")
insights = st.Page("pages/insights.py", title="Insights",   icon="💡")

pg = st.navigation([overview, risk, insights], position="hidden")

# ── Custom sidebar ─────────────────────────────────────────────────────────────
facility = st.session_state.get("facility", "UCI Medical Center")

with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-logo">
            {SHIELD_SVG}
            <span class="sidebar-logo-text">Aegis</span>
        </div>
        <div class="sidebar-tagline">ED Early Warning</div>
        <hr class="sidebar-divider">
        <div class="sidebar-section-label">Monitor</div>
    """, unsafe_allow_html=True)

    st.page_link(overview, label="Overview",   icon="🏠")
    st.page_link(risk,     label="Risk Score", icon="⚠️")
    st.page_link(insights, label="Insights",   icon="💡")

    st.markdown(f"""
        <div class="sidebar-footer">
            <div class="facility-label">Active facility</div>
            <div class="facility-name">{facility}</div>
        </div>
    """, unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────────
headers = {
    overview: ("🏠", "Overview",   "Emergency Department capacity monitoring at a glance."),
    risk:     ("⚠️", "Risk Score", "Predicted overload risk scores and contributing factors."),
    insights: ("💡", "Insights",   "Trends, patterns, and anomaly flags across facilities."),
}
icon, title, caption = headers[pg]
st.markdown(f"""
    <div class="page-header">
        <span class="page-header-icon">{icon}</span>
        <span class="page-header-title">{title}</span>
    </div>
    <div class="page-caption">{caption}</div>
""", unsafe_allow_html=True)

# ── Run current page ───────────────────────────────────────────────────────────
pg.run()
