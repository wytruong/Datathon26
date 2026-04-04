import textwrap
import sys
import os
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.style import CHART_BASE, CHART_WIDTH
from utils.data import load_clean_data, load_arima_forecast

df      = load_clean_data()
arima   = load_arima_forecast()

# ── Facility selector ──────────────────────────────────────────────────────────
facilities = sorted(df["facility_name"].dropna().unique().tolist())

col_f, col_d, _ = st.columns([2, 2, 3])
with col_f:
    default_facility = st.session_state.get("facility", facilities[0]) \
                       if st.session_state.get("facility") in facilities else facilities[0]
    query = st.text_input("Search Facility", value=default_facility, placeholder="Type to search…")
    matches = [f for f in facilities if query.lower() in f.lower()] if query else facilities
    if not matches:
        st.warning("No matching facilities.")
        facility = default_facility
    elif len(matches) == 1:
        facility = matches[0]
        st.caption(f"Selected: **{facility}**")
    else:
        facility = st.selectbox("Matching facilities", matches, label_visibility="collapsed")
    st.session_state["facility"] = facility
with col_d:
    year_range = st.slider("Year Range", 2012, 2024, (2012, 2024))

st.divider()

# ── Filter facility data ───────────────────────────────────────────────────────
fac_df = df[df["facility_name"] == facility].sort_values("year")

if fac_df.empty:
    st.warning(f"No data available for **{facility}**.")
    st.stop()

fac_df = fac_df[
    (fac_df["year"] >= year_range[0]) &
    (fac_df["year"] <= year_range[1])
]

if fac_df.empty:
    st.warning(f"No data for **{facility}** in the selected year range.")
    st.stop()

# ── Metrics ────────────────────────────────────────────────────────────────────
latest   = fac_df.sort_values("year").iloc[-1]
prev_row = fac_df.sort_values("year").iloc[-2] if len(fac_df) >= 2 else None

current_burden_raw = latest["burden_score"]
current_burden_pct = round(current_burden_raw * 100, 1)

if prev_row is not None:
    prev_burden_pct = round(prev_row["burden_score"] * 100, 1)
    burden_delta    = round(current_burden_pct - prev_burden_pct, 1)
    burden_up       = burden_delta >= 0
    delta_text      = f"{abs(burden_delta)}% vs previous year"
else:
    burden_delta = 0.0
    burden_up    = False
    delta_text   = "no prior year"

avg_encounters = int(fac_df["ed_visit"].mean())


def _card(label, value, delta_text, delta_up, progress_pct, show_bar=True):
    delta_class  = "metric-delta-up" if delta_up else "metric-delta-down"
    delta_symbol = "▲" if delta_up else "▼"
    bar_color    = "#ef4444" if progress_pct >= 85 else ("#f59e0b" if progress_pct >= 60 else "#22c55e")
    bar_width    = min(progress_pct, 100)
    bar_html = f"""
        <div class="metric-progress-track">
            <div class="metric-progress-fill" style="width:{bar_width:.0f}%;background:{bar_color}"></div>
        </div>
    """ if show_bar else ""
    return textwrap.dedent(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta_symbol} {delta_text}</div>
            {bar_html}
        </div>
    """).strip()


# ── Area chart ─────────────────────────────────────────────────────────────────
years   = fac_df["year"].tolist()
burdens = (fac_df["burden_score"] * 100).tolist()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years, y=burdens,
    mode="lines+markers",
    line=dict(color="rgba(20,53,42,0.8)", width=2),
    marker=dict(size=4, color="rgba(20,53,42,0.8)"),
    fill="tozeroy", fillcolor="rgba(20,53,42,0.07)",
    name="Burden",
    hovertemplate="Year %{x}<br>Burden: %{y:.1f}%<extra></extra>",
))

# ARIMA forecast dotted line
if not arima.empty:
    # Connect from last real data point into forecast
    last_year    = fac_df["year"].max()
    last_burden  = (fac_df[fac_df["year"] == last_year]["burden_score"].values[0]) * 100
    fc_years     = [last_year] + arima["year"].tolist()
    # Scale forecast to % (arima was trained on raw burden_score)
    fc_values    = [last_burden] + (arima["forecast"] * 100).tolist()
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
    "title": dict(text=f"ED Burden Trend — {facility}", font=dict(size=13, color="#1a1a1a"), x=0),
    "xaxis": dict(showgrid=False, tickmode="linear", dtick=1, color="#9ca3af", linecolor="#e2ded8"),
    "yaxis": dict(showgrid=True, gridcolor="#f0ede8", ticksuffix="%", range=[0, 125], color="#9ca3af"),
    "hovermode": "x unified",
    "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                   font=dict(size=11, color="#374151")),
    "margin": dict(l=10, r=10, t=45, b=10),
})
st.plotly_chart(fig, width=CHART_WIDTH)

# ── Metric cards ───────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        _card("Current Burden", f"{current_burden_pct}%", delta_text, burden_up, current_burden_pct),
        unsafe_allow_html=True,
    )
with m2:
    enc_pct = min(avg_encounters / 3.2, 100)
    st.markdown(
        _card("Avg Encounters (per year)", f"{avg_encounters:,}", "per year", False, enc_pct),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        _card("Treatment Stations", "N/A", "see data notebook", False, 0, show_bar=False),
        unsafe_allow_html=True,
    )
