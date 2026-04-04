import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils.style import inject_css, SHIELD_SVG
from utils.data import load_clean_data

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
facility = st.session_state.get("facility", "UCI Medical Center")

# Compute burden alert for selected facility
df = load_clean_data()
fac_rows = df[df["facility_name"] == facility].sort_values("year")
if not fac_rows.empty:
    latest_burden_pct = fac_rows.iloc[-1]["burden_score"] * 100
else:
    latest_burden_pct = 0.0

if latest_burden_pct >= 85:
    alert_html = """
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div class="pulse-dot" style="width:8px;height:8px;border-radius:50%;
                 background:#ef4444;flex-shrink:0;"></div>
            <span style="font-size:11px;color:#ef4444;font-weight:600;">High Burden Alert</span>
        </div>
    """
else:
    alert_html = """
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div style="width:8px;height:8px;border-radius:50%;
                 background:#22c55e;flex-shrink:0;"></div>
            <span style="font-size:11px;color:#22c55e;font-weight:500;">Burden Normal</span>
        </div>
    """

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
            {alert_html}
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
