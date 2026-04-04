import textwrap
import sys
import os
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.style import CHART_BASE, CHART_WIDTH
from utils.data import (
    load_facility_ci, load_county_summary,
    load_stats_results,
)

facility_ci    = load_facility_ci()
county_summary = load_county_summary()
stats          = load_stats_results()

st.divider()

# ── Top Facilities by ED Burden (with 95% CI) ──────────────────────────────────
st.markdown('<div class="section-header">Top Facilities by ED Burden (with 95% CI)</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Average burden score with 95% confidence interval across all available years</div>', unsafe_allow_html=True)

chart_col, spacer = st.columns([3, 1])
with chart_col:
    top15 = facility_ci.nlargest(15, "mean_burden").sort_values("mean_burden")
    error_minus = (top15["mean_burden"] - top15["ci_lower"]).clip(lower=0).tolist()
    error_plus  = (top15["ci_upper"] - top15["mean_burden"]).clip(lower=0).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top15["mean_burden"].tolist(),
        y=top15["facility_name"].tolist(),
        orientation="h",
        marker_color="rgba(20,53,42,0.75)",
        error_x=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color="#9ca3af",
            thickness=1.5,
            width=4,
        ),
        hovertemplate="%{y}<br>Mean burden: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**{
        **CHART_BASE,
        "title": dict(text="Top 15 Facilities by Average ED Burden", font=dict(size=13, color="#1a1a1a"), x=0),
        "xaxis": dict(showgrid=True, gridcolor="#f0ede8", color="#9ca3af", title="Mean Burden Score"),
        "yaxis": dict(showgrid=False, color="#374151", tickfont=dict(size=10)),
        "margin": dict(l=10, r=30, t=45, b=10),
    })
    st.plotly_chart(fig, width=CHART_WIDTH)

# ── T-Test Result Card ─────────────────────────────────────────────────────────
ttest     = stats.get("ttest_covid", {})
p_val     = ttest.get("p_value", 1.0)
t_stat    = ttest.get("t_stat", 0.0)
interp    = ttest.get("interpretation", "")
mean_pre  = ttest.get("mean_pre", 0.0)
mean_post = ttest.get("mean_post", 0.0)
sig_color = "#16a34a" if p_val < 0.05 else "#9ca3af"
sig_bg    = "rgba(34,197,94,0.07)" if p_val < 0.05 else "rgba(156,163,175,0.07)"

st.html(textwrap.dedent(f"""
    <div style="border:0.5px solid #e2ded8;border-radius:10px;padding:16px;background:#ffffff;margin:12px 0;">
        <div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;margin-bottom:8px;">
            Pre vs Post-COVID Burden — Paired T-Test (per facility)
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div><span style="font-size:11px;color:#6b7280;">t-statistic</span><br>
                 <span style="font-size:18px;font-weight:600;color:#1a1a1a;">{t_stat:.4f}</span></div>
            <div><span style="font-size:11px;color:#6b7280;">p-value</span><br>
                 <span style="font-size:18px;font-weight:600;color:#1a1a1a;">{p_val:.4f}</span></div>
            <div><span style="font-size:11px;color:#6b7280;">Mean pre-COVID</span><br>
                 <span style="font-size:18px;font-weight:600;color:#1a1a1a;">{mean_pre:.4f}</span></div>
            <div><span style="font-size:11px;color:#6b7280;">Mean post-COVID</span><br>
                 <span style="font-size:18px;font-weight:600;color:#1a1a1a;">{mean_post:.4f}</span></div>
            <div style="display:flex;align-items:center;">
                <span style="background:{sig_bg};color:{sig_color};font-size:11px;font-weight:600;
                      padding:4px 12px;border-radius:20px;text-transform:uppercase;">{interp}</span>
            </div>
        </div>
    </div>
""").strip())

st.divider()

# ── Anomaly Flags ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Anomaly Flags</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Periods where burden exceeded expected thresholds (z-score &gt; 2.0)</div>', unsafe_allow_html=True)

try:
    import pandas as pd
    import os
    anomaly_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "anomalies.csv")
    anomalies = pd.read_csv(anomaly_path)
except Exception:
    anomalies = None

table_col, _ = st.columns([3, 1])
with table_col:
    if anomalies is None or anomalies.empty:
        st.html(textwrap.dedent("""
            <div class="anomaly-wrapper" style="padding:20px;">
                <div style="color:#6b7280;font-size:13px;text-align:center;">
                    No anomalies detected at z &gt; 2.0 threshold.
                    Data shows stable burden patterns across facilities.
                </div>
            </div>
        """).strip())
    else:
        status_cls = {"Critical": "status-critical", "Warning": "status-warning", "Resolved": "status-resolved"}
        rows_html = ""
        for _, r in anomalies.iterrows():
            burden_disp = f"{r['burden_score']:.3f}"
            z_disp      = f"{r['z_score']:.2f}"
            rows_html += f"""
            <tr>
                <td>{r['year']}</td>
                <td>{r['facility_name']}</td>
                <td>{burden_disp}</td>
                <td>{z_disp}</td>
            </tr>"""
        html = textwrap.dedent(f"""
            <div class="anomaly-wrapper">
                <table class="anomaly-table">
                    <thead><tr>
                        <th>Year</th><th>Facility</th>
                        <th>Burden Score</th><th>Z-Score</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        """).strip()
        st.html(html)

st.divider()

# ── Top Counties by ED Burden ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Top Counties by ED Burden</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Average burden score across all years, top 10 counties</div>', unsafe_allow_html=True)

county_avg = (
    county_summary
    .groupby("county_name")["mean_burden_score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values(ascending=True)   # ascending for horizontal bar readability
    .reset_index()
)

chart_col2, _ = st.columns([3, 1])
with chart_col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=county_avg["mean_burden_score"].tolist(),
        y=county_avg["county_name"].tolist(),
        orientation="h",
        marker_color="rgba(20,53,42,0.75)",
        text=[f"{v:.3f}" for v in county_avg["mean_burden_score"]],
        textposition="outside",
        textfont=dict(size=10, color="#374151"),
        hovertemplate="%{y}<br>Avg burden: %{x:.3f}<extra></extra>",
    ))
    fig2.update_layout(**{
        **CHART_BASE,
        "title": dict(text="Top 10 Counties — Average ED Burden Score", font=dict(size=13, color="#1a1a1a"), x=0),
        "xaxis": dict(showgrid=True, gridcolor="#f0ede8", color="#9ca3af", title="Avg Burden Score"),
        "yaxis": dict(showgrid=False, color="#374151", tickfont=dict(size=11)),
        "margin": dict(l=10, r=60, t=45, b=10),
    })
    st.plotly_chart(fig2, width=CHART_WIDTH)
