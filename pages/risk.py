import textwrap
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

df       = load_clean_data()
df_feat  = load_features()
lr_model = load_logistic_model()
coef_df  = load_logistic_coefficients()
feat_cols = load_feature_list()

# ── Facility from session state (set by overview page selector) ────────────────
facility = st.session_state.get("facility", sorted(df["facility_name"].unique())[0])

# ── Build feature row for selected facility (latest year) ─────────────────────
fac_feat = df_feat[df_feat["facility_name"] == facility].sort_values("year")

if fac_feat.empty:
    st.warning(f"No feature data available for **{facility}**.")
    RISK_SCORE = 0.5
else:
    latest_row = fac_feat.iloc[-1]
    # Build ordered feature vector; fill missing with 0
    feature_values = []
    for col in feat_cols:
        val = latest_row.get(col, 0)
        feature_values.append(0 if (val is None or (isinstance(val, float) and np.isnan(val))) else val)

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
# driver_bars_html expects list of (name, value) with value in [0,1] range
coef_max = top5["coefficient"].abs().max()
FEATURES = [
    (row["feature"].replace("_", " ").title(), abs(row["coefficient"]) / coef_max)
    for _, row in top5.iterrows()
]

# ── Layout ─────────────────────────────────────────────────────────────────────
st.divider()

left, right = st.columns([1, 2], gap="large")

with left:
    st.html(risk_score_card_html(RISK_SCORE))

with right:
    st.html(driver_bars_html(FEATURES))

st.html("<div style='height:14px'></div>")

ai_html = textwrap.dedent("""
    <div class="ai-card">
        <div class="ai-card-header">AI Summary</div>
        <div class="ai-card-body">
            AI-generated narrative summary will appear here. This section will describe
            the primary drivers of today's risk score, flag any unusual patterns, and
            suggest potential interventions — powered by a language model connected to
            live facility data.
        </div>
    </div>
""").strip()
st.html(ai_html)
