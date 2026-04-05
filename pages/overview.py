import textwrap
import sys
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.style import CHART_BASE, CHART_WIDTH
from utils.data import load_clean_data, load_arima_forecast

df    = load_clean_data()
arima = load_arima_forecast()

facilities = sorted(df["facility_name"].dropna().unique().tolist())


def _search_input(label, key_input, key_select, default):
    query = st.text_input(label, value="", placeholder="Type to search…", key=key_input,
                          help="Type any part of the facility name to filter the list below.")
    matches = [f for f in facilities if query.lower() in f.lower()] if query else facilities
    if not matches:
        st.warning("No matching facilities.")
        return default
    idx = matches.index(default) if default in matches else 0
    return st.selectbox(
        "Matching facilities", matches, index=idx,
        label_visibility="collapsed", key=key_select,
    )


def _sparkline_svg(values, width=80, height=24):
    """Generate an inline SVG polyline from a list of floats."""
    if not values or len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    n   = len(values)
    pts = []
    for i, v in enumerate(values):
        x = round(i / (n - 1) * width, 2)
        y = round(height - ((v - mn) / rng) * (height - 4) - 2, 2)
        pts.append(f"{x},{y}")
    trend_color = "#ef4444" if values[-1] > values[0] else "#22c55e"
    points_str  = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;margin-top:8px;">'
        f'<polyline points="{points_str}" fill="none" stroke="{trend_color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def _card(label, value, delta_text, delta_up, progress_pct, show_bar=True, sparkline_vals=None):
    delta_class  = "metric-delta-up" if delta_up else "metric-delta-down"
    delta_symbol = "▲" if delta_up else "▼"
    bar_color    = "#ef4444" if progress_pct >= 85 else ("#f59e0b" if progress_pct >= 60 else "#22c55e")
    bar_width    = min(progress_pct, 100)

    if sparkline_vals and len(sparkline_vals) >= 2:
        bottom_html = _sparkline_svg(sparkline_vals)
    elif show_bar:
        bottom_html = f"""
        <div class="metric-progress-track">
            <div class="metric-progress-fill" style="width:{bar_width:.0f}%;background:{bar_color}"></div>
        </div>"""
    else:
        bottom_html = ""

    return textwrap.dedent(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta_symbol} {delta_text}</div>
            {bottom_html}
        </div>
    """).strip()


# ── Primary facility selector ──────────────────────────────────────────────────
col_f, col_d, _ = st.columns([2, 2, 3])
with col_f:
    default_facility = st.session_state.get("facility", facilities[0]) \
                       if st.session_state.get("facility") in facilities else facilities[0]
    facility = _search_input("Search Facility", "fac_query", "fac_select", default_facility)
    st.session_state["facility"] = facility

with col_d:
    st.markdown("<div style='font-size:13px;color:#6b7280;margin-bottom:4px;'>Year Range</div>",
                unsafe_allow_html=True)
    yr_col1, yr_col2 = st.columns(2)
    with yr_col1:
        from_year = st.number_input("From", min_value=2012, max_value=2024, value=2012,
                                    step=1, key="yr_from",
                                    help="Start year for the chart and metrics.")
    with yr_col2:
        to_year = st.number_input("To", min_value=2012, max_value=2024, value=2024,
                                  step=1, key="yr_to",
                                  help="End year for the chart and metrics.")
    if from_year > to_year:
        st.warning("'From' year must be ≤ 'To' year.")
        from_year, to_year = to_year, from_year
    year_range = (int(from_year), int(to_year))

# ── Comparison toggle ──────────────────────────────────────────────────────────
compare_mode = st.toggle("Compare with another facility", key="compare_mode")

facility2 = None
if compare_mode:
    default2 = st.session_state.get("facility2", facilities[1] if len(facilities) > 1 else facilities[0])
    if default2 not in facilities:
        default2 = facilities[1] if len(facilities) > 1 else facilities[0]
    facility2 = _search_input("Search Second Facility", "fac2_query", "fac2_select", default2)
    st.session_state["facility2"] = facility2

st.divider()

# ── Filter data ────────────────────────────────────────────────────────────────
fac_df = df[df["facility_name"] == facility].sort_values("year")
fac_df = fac_df[(fac_df["year"] >= year_range[0]) & (fac_df["year"] <= year_range[1])]

if fac_df.empty:
    st.warning(f"No data for **{facility}** in the selected year range.")
    st.stop()

# ── Metrics ────────────────────────────────────────────────────────────────────
latest   = fac_df.sort_values("year").iloc[-1]
prev_row = fac_df.sort_values("year").iloc[-2] if len(fac_df) >= 2 else None

current_burden_pct = round(latest["burden_score"] * 100, 1)
avg_encounters     = int(fac_df["ed_visit"].mean())

if prev_row is not None:
    prev_burden_pct = round(prev_row["burden_score"] * 100, 1)
    burden_delta    = round(current_burden_pct - prev_burden_pct, 1)
    burden_up       = burden_delta >= 0
    delta_text      = f"{abs(burden_delta)}% vs previous year"
else:
    burden_delta = 0.0
    burden_up    = False
    delta_text   = "no prior year"

# Sparkline data: last 5 years of burden (all years, not just filtered range)
all_fac_df      = df[df["facility_name"] == facility].sort_values("year")
spark_burden     = (all_fac_df["burden_score"].dropna().tail(6) * 100).tolist()
spark_encounters = all_fac_df["ed_visit"].dropna().tail(6).tolist()

# ── Context info bar ───────────────────────────────────────────────────────────
svc_level  = latest.get("er_service_level_desc", "—")
data_years = f"{int(fac_df['year'].min())}–{int(fac_df['year'].max())}"
n_years    = df[df["facility_name"] == facility]["year"].nunique()

st.markdown(f"""
<div class="info-bar">
    <div class="info-bar-item">
        <span class="info-bar-label">Facility</span>
        <span class="info-bar-value">{facility}</span>
    </div>
    <div class="info-bar-divider"></div>
    <div class="info-bar-item">
        <span class="info-bar-label">Year Range</span>
        <span class="info-bar-value">{data_years}</span>
    </div>
    <div class="info-bar-divider"></div>
    <div class="info-bar-item">
        <span class="info-bar-label">Service Level</span>
        <span class="info-bar-value">{svc_level}</span>
    </div>
    <div class="info-bar-divider"></div>
    <div class="info-bar-item">
        <span class="info-bar-label">Years of Data</span>
        <span class="info-bar-value">{n_years}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Burden trend chart ─────────────────────────────────────────────────────────
years   = fac_df["year"].tolist()
burdens = (fac_df["burden_score"] * 100).tolist()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years, y=burdens,
    mode="lines+markers",
    line=dict(color="rgba(20,53,42,0.8)", width=2),
    marker=dict(size=4, color="rgba(20,53,42,0.8)"),
    fill="tozeroy", fillcolor="rgba(20,53,42,0.07)",
    name=facility,
    hovertemplate="Year %{x}<br>Burden: %{y:.1f}%<extra></extra>",
))

if compare_mode and facility2:
    fac2_df = df[df["facility_name"] == facility2].sort_values("year")
    fac2_df = fac2_df[(fac2_df["year"] >= year_range[0]) & (fac2_df["year"] <= year_range[1])]
    if not fac2_df.empty:
        fig.add_trace(go.Scatter(
            x=fac2_df["year"].tolist(),
            y=(fac2_df["burden_score"] * 100).tolist(),
            mode="lines+markers",
            line=dict(color="rgba(245,158,11,0.8)", width=2),
            marker=dict(size=4, color="rgba(245,158,11,0.8)"),
            name=facility2,
            hovertemplate="Year %{x}<br>Burden: %{y:.1f}%<extra></extra>",
        ))
    else:
        st.warning(f"No data for **{facility2}** in the selected year range.")

if not arima.empty:
    last_year   = fac_df["year"].max()
    last_burden = fac_df[fac_df["year"] == last_year]["burden_score"].values[0] * 100
    fc_years    = [last_year] + arima["year"].tolist()
    fc_values   = [last_burden] + (arima["forecast"] * 100).tolist()
    fig.add_trace(go.Scatter(
        x=fc_years, y=fc_values,
        mode="lines",
        line=dict(color="rgba(20,53,42,0.4)", width=2, dash="dot"),
        name="Forecast",
        hovertemplate="Year %{x}<br>Forecast: %{y:.1f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=fc_years[-1], y=fc_values[-1],
        text="Forecast", showarrow=False,
        font=dict(size=10, color="rgba(20,53,42,0.6)"),
        xanchor="left", yanchor="middle", xshift=6,
    )

fig.add_hline(
    y=85, line_dash="dot", line_color="#ef4444", line_width=1.5,
    annotation_text="Overload threshold (85%)", annotation_position="top left",
    annotation_font_color="#ef4444", annotation_font_size=10,
)
fig.update_layout(**{
    **CHART_BASE,
    "title": dict(text=f"ED Burden Trend — {facility}",
                  font=dict(size=13, color="#1a1a18"), x=0, y=0.98),
    "xaxis": dict(showgrid=False, tickmode="linear", dtick=1, color="#9ca3af", linecolor="#e2ded8", range=[2011, 2028]),
    "yaxis": dict(showgrid=True, gridcolor="#f0ede8", ticksuffix="%", range=[0, 125], color="#9ca3af"),
    "hovermode": "x unified",
    "legend": dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
                   font=dict(size=11, color="#374151")),
    "margin": dict(l=10, r=10, t=60, b=60),
})
st.plotly_chart(fig, width=CHART_WIDTH)

# ── Metric cards with sparklines ───────────────────────────────────────────────
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        _card(
            "Current Burden",
            f"{current_burden_pct}%",
            delta_text,
            burden_up,
            current_burden_pct,
            sparkline_vals=spark_burden,
        ),
        unsafe_allow_html=True,
    )
with m2:
    enc_pct = min(avg_encounters / 3.2, 100)
    st.markdown(
        _card(
            "Avg Encounters (per year)",
            f"{avg_encounters:,}",
            "per year",
            False,
            enc_pct,
            sparkline_vals=spark_encounters,
        ),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        _card("Treatment Stations", "N/A", "see data notebook", False, 0, show_bar=False),
        unsafe_allow_html=True,
    )

# ── Export button ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
csv_data = fac_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Export facility data as CSV",
    data=csv_data,
    file_name=f"{facility.lower().replace(' ', '_')}_{year_range[0]}_{year_range[1]}.csv",
    mime="text/csv",
    help="Download the filtered facility data for this year range.",
)
