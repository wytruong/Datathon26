"""Shared styling helpers for Aegis — light theme, Streamlit 1.36+."""
import streamlit as st
import textwrap

# ── Color tokens ───────────────────────────────────────────────────────────────
BG          = "#f7f5f1"
CARD_BG     = "#ffffff"
BORDER      = "#e2ded8"
ACCENT      = "#14352a"
TEXT        = "#1a1a1a"
MUTED       = "#9ca3af"
GRID        = "#f0ede8"
SUCCESS     = "#22c55e"
WARNING_COL = "#f59e0b"
DANGER      = "#ef4444"
AMBER_TEXT  = "#d97706"

# ── Plotly chart base ──────────────────────────────────────────────────────────
CHART_BASE = dict(
    plot_bgcolor=CARD_BG,
    paper_bgcolor=CARD_BG,
    font=dict(color="#374151", family="Inter, sans-serif", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)

# Recommended fixed chart width for consistent layouts (pixels)
CHART_WIDTH = 860
# ── Shield SVG ─────────────────────────────────────────────────────────────────
SHIELD_SVG = '<span style="font-size:18px;line-height:1;color:#14352a">🛡️</span>'

# ── Global CSS (injected via st.html so it reaches the live DOM) ───────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, *, *::before, *::after {
    font-family: 'Inter', sans-serif;
}

/* App background and container spacing */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #f7f5f1;
}
[data-testid="stMain"] .block-container {
    padding-top: 1.4rem;
    max-width: 1100px;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar basics */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 0.5px solid #ece9e3;
}

/* Page header */
.page-header { display: flex; align-items: center; gap: 10px; padding: 4px 0 2px; }
.page-header-icon { font-size: 22px; }
.page-header-title { font-size: 22px; font-weight: 700; color: #1a1a1a; }
.page-caption { font-size: 13px; color: #6b7280; margin-bottom: 6px; }

/* Metric cards */
.metric-card { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; padding: 14px 16px; }
.metric-label { font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 26px; font-weight: 600; color: #1a1a1a; }
.metric-delta { font-size: 11px; margin-top: 6px; margin-bottom: 10px; }
.metric-progress-track { height: 4px; border-radius: 3px; background: #f0ede8; overflow: hidden; }
.metric-progress-fill { height: 100%; border-radius: 3px; }

/* Plotly wrapper */
[data-testid="stPlotlyChart"] { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; padding: 8px; overflow: hidden; }

/* Risk card, drivers, anomaly table kept simple and consistent */
.risk-score-card, .drivers-card, .ai-card, .anomaly-wrapper { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; padding: 18px; }

/* Keep Streamlit chrome minimal but visible for debugging; hide only menu/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Sidebar custom HTML */
.sidebar-logo { display: flex; align-items: center; gap: 8px; padding: 18px 12px 6px; }
.sidebar-logo-text { font-size: 16px; font-weight: 600; color: #14352a; }
.sidebar-tagline { font-size: 10px; color: #9ca3af; padding: 2px 12px 10px; }
.sidebar-section-label { font-size: 10px; font-weight: 600; color: #9ca3af; text-transform: uppercase; padding: 8px 12px 4px; }
.sidebar-divider { border: none; border-top: 0.5px solid #ece9e3; margin: 6px 12px; }
.sidebar-footer { padding: 12px; border-top: 0.5px solid #ece9e3; margin-top: 6px; }
.facility-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; margin-bottom: 3px; }
.facility-name  { font-size: 12px; font-weight: 500; color: #374151; }

/* st.page_link nav styling */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    font-size: 13px; font-weight: 400; color: #374151; border-radius: 7px; padding: 8px 12px; margin: 1px 0; text-decoration: none; display: flex; align-items: center; gap: 8px;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: #f3f4f6; color: #14352a; }
[data-testid="stSidebar"] [data-testid="stPageLink-active"] a { background: #edf4f1; color: #14352a; font-weight: 500; }

/* Metric delta colors */
.metric-delta-up { color: #ef4444; }
.metric-delta-down { color: #22c55e; }

/* Risk score card */
.risk-score-card { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; padding: 20px; height: 100%; }
.risk-score-label { font-size: 11px; font-weight: 500; color: #9ca3af; text-transform: uppercase; margin-bottom: 8px; }
.risk-score-value { font-size: 48px; font-weight: 600; color: #d97706; line-height: 1; margin-bottom: 10px; }
.risk-pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px; text-transform: uppercase; margin-bottom: 16px; }
.pill-high { background: rgba(239,68,68,0.1); color: #ef4444; }
.pill-medium { background: rgba(245,158,11,0.1); color: #f59e0b; }
.pill-low { background: rgba(34,197,94,0.1); color: #22c55e; }
.gauge-track { height: 8px; border-radius: 4px; background: linear-gradient(to right, #22c55e 0%, #f59e0b 50%, #ef4444 100%); position: relative; margin-bottom: 8px; }
.gauge-marker { position: absolute; top: -5px; width: 16px; height: 16px; border-radius: 50%; background: #ffffff; border: 2.5px solid #14352a; transform: translateX(-50%); box-shadow: 0 1px 4px rgba(0,0,0,0.12); }
.gauge-labels { display: flex; justify-content: space-between; font-size: 10px; color: #9ca3af; }

/* Drivers */
.drivers-card { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; padding: 18px; }
.drivers-title { font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; margin-bottom: 12px; }
.driver-item { margin-bottom: 12px; }
.driver-header { display:flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.driver-name { font-size: 12px; color: #374151; }
.driver-value { font-size: 12px; font-weight: 600; color: #1a1a1a; }
.driver-bar-track { height: 6px; border-radius: 4px; background: #f0ede8; overflow: hidden; }
.driver-bar-fill { height: 100%; border-radius: 4px; background: rgba(20,53,42,0.8); }

/* Anomaly table */
.anomaly-wrapper { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; overflow: hidden; }
.anomaly-table { width: 100%; border-collapse: collapse; }
.anomaly-table th { font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; padding: 10px 12px; border-bottom: 0.5px solid #e2ded8; text-align: left; }
.anomaly-table td { font-size: 12px; color: #374151; padding: 10px 12px; border-bottom: 0.5px solid #f5f3ef; }
.status-critical { color: #ef4444; font-weight: 600; }
.status-warning { color: #f59e0b; font-weight: 600; }
.status-resolved { color: #22c55e; font-weight: 600; }

/* Section headings + AI card */
.section-header { font-size: 13px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px; }
.section-sub { font-size: 12px; color: #6b7280; margin-bottom: 10px; }
.ai-card { background: #ffffff; border: 0.5px solid #e2ded8; border-radius: 10px; padding: 16px; }
.ai-card-header { font-size: 10px; font-weight: 600; color: #9ca3af; text-transform: uppercase; margin-bottom: 8px; }
.ai-card-body { font-size: 13px; color: #374151; line-height: 1.5; }

/* Pulse animation for burden alert dot */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.pulse-dot { animation: pulse 1.5s ease-in-out infinite; }

</style>
"""


def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def render_sidebar(active_page: str = "overview"):
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
        st.page_link("app.py",            label="Overview",   icon="🏠")
        st.page_link("pages/risk.py",     label="Risk Score", icon="⚠️")
        st.page_link("pages/insights.py", label="Insights",   icon="💡")
        st.markdown(f"""
            <div class="sidebar-footer">
                <div class="facility-label">Active facility</div>
                <div class="facility-name">{facility}</div>
            </div>
        """, unsafe_allow_html=True)


def render_page_header(icon: str, title: str, caption: str):
    st.markdown(f"""
        <div class="page-header">
            <span class="page-header-icon">{icon}</span>
            <span class="page-header-title">{title}</span>
        </div>
        <div class="page-caption">{caption}</div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta_text: str, delta_up: bool, progress_pct: float) -> str:
    delta_class  = "metric-delta-up" if delta_up else "metric-delta-down"
    delta_symbol = "▲" if delta_up else "▼"
    bar_color = "#ef4444" if progress_pct >= 85 else ("#f59e0b" if progress_pct >= 60 else "#22c55e")
    bar_width = min(progress_pct, 100)
    return textwrap.dedent(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta_symbol} {delta_text}</div>
            <div class="metric-progress-track">
                <div class="metric-progress-fill"
                     style="width:{bar_width:.0f}%; background:{bar_color}"></div>
            </div>
        </div>
    """).strip()


def driver_bars_html(features: list) -> str:
    max_val = max(v for _, v in features)
    items = ""
    for name, val in features:
        bar_pct = val / max_val * 100
        items += f"""
            <div class="driver-item">
                <div class="driver-header">
                    <span class="driver-name">{name}</span>
                    <span class="driver-value">{val:.0%}</span>
                </div>
                <div class="driver-bar-track">
                    <div class="driver-bar-fill" style="width:{bar_pct:.0f}%"></div>
                </div>
            </div>
        """
    return textwrap.dedent(f"""
        <div class="drivers-card">
            <div class="drivers-title">Top Risk Drivers</div>
            {items}
        </div>
    """).strip()


def risk_score_card_html(score: float) -> str:
    if score >= 0.70:
        level, pill_class = "High", "pill-high"
    elif score >= 0.40:
        level, pill_class = "Medium", "pill-medium"
    else:
        level, pill_class = "Low", "pill-low"
    marker_pct = score * 100
    return textwrap.dedent(f"""
        <div class="risk-score-card">
            <div class="risk-score-label">Overload Risk Score</div>
            <div class="risk-score-value">{score:.2f}</div>
            <div><span class="risk-pill {pill_class}">{level} Risk</span></div>
            <div class="gauge-track">
                <div class="gauge-marker" style="left:{marker_pct:.0f}%"></div>
            </div>
            <div class="gauge-labels">
                <span>Low</span><span>Medium</span><span>High</span>
            </div>
            <div class="risk-sublabel">Overload probability — next 4 hours</div>
        </div>
    """).strip()
