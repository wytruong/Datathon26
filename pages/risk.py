import sys
import os
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.style import risk_score_card_html, driver_bars_html
from utils.data import (
    load_clean_data, load_features, load_logistic_model,
    load_logistic_coefficients, load_feature_list,
)

df        = load_clean_data()
df_feat   = load_features()
lr_model  = load_logistic_model()
coef_df   = load_logistic_coefficients()
feat_cols = load_feature_list()

# ── Facility from session state ────────────────────────────────────────────────
facility = st.session_state.get("facility", sorted(df["facility_name"].unique())[0])

# ── Build baseline feature vector ─────────────────────────────────────────────
fac_feat = df_feat[df_feat["facility_name"] == facility].sort_values("year")

RISK_SCORE  = 0.5
feature_values = []

if fac_feat.empty:
    st.warning(f"No feature data available for **{facility}**.")
else:
    latest_row = fac_feat.iloc[-1]
    for col in feat_cols:
        val = latest_row.get(col, 0)
        feature_values.append(0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val))

    X_input = np.array(feature_values).reshape(1, -1)
    try:
        RISK_SCORE = float(lr_model.predict_proba(X_input)[0][1])
    except Exception:
        RISK_SCORE = 0.5

# ── Top 5 features by absolute coefficient ────────────────────────────────────
top5 = (
    coef_df
    .assign(abs_coef=coef_df["coefficient"].abs())
    .nlargest(5, "abs_coef")
    [["feature", "coefficient"]]
)
coef_max  = top5["coefficient"].abs().max()
FEATURES  = [
    (row["feature"].replace("_", " ").title(), abs(row["coefficient"]) / coef_max)
    for _, row in top5.iterrows()
]
top_driver_name = top5.iloc[0]["feature"].replace("_", " ").title()

# ── Risk level label ───────────────────────────────────────────────────────────
def _risk_level(score):
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"

# ── Layout: baseline risk + feature drivers ───────────────────────────────────
st.divider()

left, right = st.columns([1, 2], gap="large")

with left:
    st.html(risk_score_card_html(RISK_SCORE))

with right:
    st.html(driver_bars_html(FEATURES))

# ── What-If Scenario ──────────────────────────────────────────────────────────
st.html("<div style='height:10px'></div>")
st.markdown(
    "<div style='font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:4px;'>What-If Scenario</div>",
    unsafe_allow_html=True,
)

ed_visits_input = st.slider(
    "Adjust ED Visit Volume",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.05,
    help="Multiplier on current ED visit volume. 1.0 = current level.",
)

# Recompute risk with modified feature vector
if feature_values:
    modified = feature_values.copy()
    ed_visit_keys = {"ed_visits_per_year"}
    for i, col in enumerate(feat_cols):
        if col in ed_visit_keys:
            modified[i] = feature_values[i] * ed_visits_input

    X_mod = np.array(modified).reshape(1, -1)
    try:
        WHAT_IF_SCORE = float(lr_model.predict_proba(X_mod)[0][1])
    except Exception:
        WHAT_IF_SCORE = RISK_SCORE
else:
    WHAT_IF_SCORE = RISK_SCORE

delta      = round(WHAT_IF_SCORE - RISK_SCORE, 4)
delta_sign = "▲" if delta >= 0 else "▼"
delta_col  = "#ef4444" if delta >= 0 else "#22c55e"
delta_str  = f"{delta_sign} {abs(delta):+.4f}".replace("+-", "+")

wi_level      = _risk_level(WHAT_IF_SCORE)
wi_pill_class = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}[wi_level]

st.html(f"""
<div style="border:0.5px solid #e2ded8;border-radius:10px;padding:16px;background:#ffffff;
            display:flex;align-items:center;gap:32px;flex-wrap:wrap;">
    <div>
        <div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;
                    margin-bottom:4px;">Baseline Risk</div>
        <div style="font-size:22px;font-weight:600;color:#9ca3af;">{RISK_SCORE:.3f}</div>
    </div>
    <div style="font-size:22px;color:#d1d5db;">→</div>
    <div>
        <div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;
                    margin-bottom:4px;">Adjusted Risk</div>
        <div style="font-size:28px;font-weight:700;color:#d97706;">{WHAT_IF_SCORE:.3f}</div>
    </div>
    <div>
        <div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;
                    margin-bottom:4px;">Delta</div>
        <div style="font-size:18px;font-weight:600;color:{delta_col};">{delta_str}</div>
    </div>
    <div>
        <span class="risk-pill {wi_pill_class}">{wi_level} Risk</span>
    </div>
</div>
""")

# ── Expandable AI Summary ─────────────────────────────────────────────────────
st.html("<div style='height:10px'></div>")

with st.expander("View AI Operational Summary", expanded=False):
    risk_level = _risk_level(RISK_SCORE)
    st.html(f"""
    <div class="ai-card">
        <div class="ai-card-header">AI Summary</div>
        <div style="margin-bottom:10px;display:flex;flex-direction:column;gap:4px;">
            <div style="font-size:12px;color:#6b7280;">
                <span style="font-weight:600;color:#374151;">Facility:</span> {facility}
            </div>
            <div style="font-size:12px;color:#6b7280;">
                <span style="font-weight:600;color:#374151;">Risk Level:</span> {risk_level}
            </div>
            <div style="font-size:12px;color:#6b7280;">
                <span style="font-weight:600;color:#374151;">Top Driver:</span> {top_driver_name}
            </div>
        </div>
        <div class="ai-card-body">
            Full AI summary will appear here once connected to language model in Phase 7.
        </div>
    </div>
    """)
