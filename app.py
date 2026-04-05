import numpy as np
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils.style import inject_css, SHIELD_SVG
from utils.data import (
    load_clean_data, load_features,
    load_logistic_model, load_feature_list,
)

st.set_page_config(
    page_title="Aegis",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Define pages ───────────────────────────────────────────────────────────────
overview = st.Page("pages/overview.py", title="Overview",   default=True)
risk     = st.Page("pages/risk.py",     title="Risk Score")
insights = st.Page("pages/insights.py", title="Insights")

pg = st.navigation([overview, risk, insights], position="hidden")

# ── Sidebar data ───────────────────────────────────────────────────────────────
facility  = st.session_state.get("facility", "UCI Medical Center")
df        = load_clean_data()
df_feat   = load_features()
lr_model  = load_logistic_model()
feat_cols = load_feature_list()

fac_rows = df[df["facility_name"] == facility].sort_values("year")
latest_burden_pct = fac_rows.iloc[-1]["burden_score"] * 100 if not fac_rows.empty else 0.0

# Compute sidebar risk score
fac_feat = df_feat[df_feat["facility_name"] == facility].sort_values("year")
sidebar_risk = None
if not fac_feat.empty:
    row = fac_feat.iloc[-1]
    fv  = [float(row.get(c, 0) or 0) for c in feat_cols]
    try:
        sidebar_risk = float(lr_model.predict_proba(np.array(fv).reshape(1, -1))[0][1])
    except Exception:
        sidebar_risk = None

st.session_state["risk_score"] = sidebar_risk if sidebar_risk is not None else 0.0

# Pulse dot — based on risk score from session state
if st.session_state.get("risk_score", 0) >= 0.7:
    alert_html = """
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div class="pulse-dot" style="width:7px;height:7px;border-radius:50%;
                 background:#ef4444;flex-shrink:0;"></div>
            <span style="font-family:'DM Sans',sans-serif;font-size:11px;
                  color:#ef4444;font-weight:500;">High Burden Alert</span>
        </div>
    """
else:
    alert_html = """
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div style="width:7px;height:7px;border-radius:50%;
                 background:#22c55e;flex-shrink:0;"></div>
            <span style="font-family:'DM Sans',sans-serif;font-size:11px;
                  color:#22c55e;font-weight:400;">Burden Normal</span>
        </div>
    """

# Risk score badge
if sidebar_risk is not None:
    if sidebar_risk >= 0.70:
        rs_color, rs_label = "#ef4444", "High"
    elif sidebar_risk >= 0.40:
        rs_color, rs_label = "#f59e0b", "Medium"
    else:
        rs_color, rs_label = "#22c55e", "Low"
    risk_badge_html = f"""
        <div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
            <div style="font-family:'DM Sans',sans-serif;font-size:10px;color:#9ca3af;
                  text-transform:uppercase;letter-spacing:0.06em;">Risk Score</div>
            <div style="font-family:'DM Serif Display',serif;font-size:15px;
                  font-weight:normal;color:{rs_color};">{sidebar_risk:.2f}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:10px;font-weight:500;
                  color:{rs_color};background:rgba(0,0,0,0.04);
                  padding:2px 8px;border-radius:10px;">{rs_label}</div>
        </div>
    """
else:
    risk_badge_html = ""

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

    st.page_link(overview, label="Overview")
    st.page_link(risk,     label="Risk Score")
    st.page_link(insights, label="Insights")

    st.markdown(f"""
        <div class="sidebar-footer">
            <div class="facility-label">Active facility</div>
            <div class="facility-name">{facility}</div>
            {risk_badge_html}
            {alert_html}
        </div>
    """, unsafe_allow_html=True)

# ── Page header with timestamp ─────────────────────────────────────────────────
headers = {
    overview: ("Overview",   "Emergency Department capacity monitoring at a glance."),
    risk:     ("Risk Score", "Predicted overload risk scores and contributing factors."),
    insights: ("Insights",   "Trends, patterns, and anomaly flags across facilities."),
}
title, caption = headers[pg]
st.markdown(f"""
    <div class="page-header" style="justify-content:space-between;">
        <span class="page-header-title">{title}</span>
        <span style="font-family:'DM Sans',sans-serif;font-size:11px;color:#9ca3af;align-self:center;">
            Data as of 2024 &nbsp;·&nbsp; HCAI California
        </span>
    </div>
    <div class="page-caption">{caption}</div>
""", unsafe_allow_html=True)

# ── Run current page ───────────────────────────────────────────────────────────
pg.run()
