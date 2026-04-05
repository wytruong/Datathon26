import re
import sys
import os
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.style import risk_score_card_html, driver_bars_html
from utils.data import (
    load_clean_data, load_features, load_logistic_model,
    load_logistic_coefficients, load_feature_list,
    load_facility_context,
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
    "<div style='font-family:DM Serif Display,serif;font-size:15px;font-weight:normal;color:#1a1a18;margin-bottom:4px;'>What-If Scenario</div>",
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

WHAT_IF_SCORE = min(1.0, RISK_SCORE * ed_visits_input)

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
        <div style="font-family:'DM Sans',sans-serif;font-size:10px;font-weight:500;color:#9ca3af;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Baseline Risk</div>
        <div style="font-family:'DM Serif Display',serif;font-size:22px;font-weight:normal;color:#9ca3af;">{RISK_SCORE:.3f}</div>
    </div>
    <div style="font-size:22px;color:#d1d5db;">→</div>
    <div>
        <div style="font-family:'DM Sans',sans-serif;font-size:10px;font-weight:500;color:#9ca3af;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Adjusted Risk</div>
        <div style="font-family:'DM Serif Display',serif;font-size:28px;font-weight:normal;color:#d97706;">{WHAT_IF_SCORE:.3f}</div>
    </div>
    <div>
        <div style="font-family:'DM Sans',sans-serif;font-size:10px;font-weight:500;color:#9ca3af;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Delta</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:18px;font-weight:500;color:{delta_col};">{delta_str}</div>
    </div>
    <div>
        <span class="risk-pill {wi_pill_class}">{wi_level} Risk</span>
    </div>
</div>
""")

# ── Context-aware recommendations ─────────────────────────────────────────────
st.html("<div style='height:10px'></div>")

# Resolve oshpd_id for the selected facility
_fac_rows = df[df["facility_name"] == facility]
_oshpd_id = int(_fac_rows["oshpd_id"].iloc[0]) if not _fac_rows.empty else None

# Look up facility context
_ctx = None
if _oshpd_id is not None:
    _ctx_df = load_facility_context()
    _ctx_row = _ctx_df[_ctx_df["oshpd_id"] == _oshpd_id]
    if not _ctx_row.empty:
        _ctx = _ctx_row.iloc[0]

print(f"[risk] Full context: {_ctx.to_dict() if _ctx is not None else 'NOT FOUND'}")

# Parse lower numeric bound from strings like "100-149" or "500+"
def _bed_lower(s):
    if not s or str(s) == "nan":
        return None
    m = re.match(r"^(\d+)", str(s).strip())
    return int(m.group(1)) if m else None

# Extract context fields (fall back to empty strings gracefully)
_bed_str   = str(_ctx["LICENSED_BED_SIZE"])   if _ctx is not None else ""
_ownership = str(_ctx["HospitalOwnership"])    if _ctx is not None else ""
_urban     = str(_ctx["UrbanRuralDesi"])       if _ctx is not None else ""
_teaching  = str(_ctx["TEACHINGDesignation"])  if _ctx is not None else ""
_pc_short  = str(_ctx["PrimaryCareShortageArea"])  if _ctx is not None else ""
_mh_short  = str(_ctx["MentalHealthShortageArea"]) if _ctx is not None else ""

_bl = _bed_lower(_bed_str)
if _bl is not None:
    _bed_cat = "Small (<100 beds)" if _bl < 100 else ("Medium (100–300 beds)" if _bl < 300 else "Large (>300 beds)")
else:
    _bed_cat = "Unknown bed size"

# Risk border color (based on baseline risk score)
_border = "#ef4444" if RISK_SCORE >= 0.7 else ("#f59e0b" if RISK_SCORE >= 0.4 else "#22c55e")

# ── Low risk: simple green card ───────────────────────────────────────────────
if RISK_SCORE < 0.4:
    st.html(f"""
<div style="border:0.5px solid #e2ded8;border-left:4px solid #22c55e;border-radius:10px;
            padding:18px 20px;background:#ffffff;margin-top:4px;">
    <div style="font-family:'DM Sans',sans-serif;font-size:10px;font-weight:600;
                color:#22c55e;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;">Status</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#374151;">
        Burden within normal range. Continue monitoring quarterly.
    </div>
</div>
""")

# ── Elevated / high risk: full contextual card ────────────────────────────────
else:
    # Section 1 — pills
    _pill = "display:inline-block;padding:3px 10px;border-radius:20px;background:#f7f5f1;border:1px solid #e2ded8;font-family:'DM Sans',sans-serif;font-size:11px;color:#374151;margin:2px 2px;"
    _pills_html = "".join(
        f'<span style="{_pill}">{p}</span>'
        for p in [
            _bed_cat,
            _ownership or "—",
            _urban or "—",
            f"Teaching: {'Yes' if _teaching == 'Teaching' else 'No'}",
            f"Primary Care Shortage: {_pc_short or '—'}",
            f"Mental Health Shortage: {_mh_short or '—'}",
        ]
    )

    # Section 2 — root causes (all that apply)
    _causes = []
    _steps  = []

    if _pc_short == "Yes":
        _causes.append("Primary care shortage likely driving ED overload — patients using ED as primary care substitute")
        _steps.append("Coordinate with county health to expand federally qualified health centers in catchment area")

    if _mh_short == "Yes":
        _causes.append("Mental health shortage contributing to ED burden — psychiatric patients boarding in ED")
        _steps.append("Establish psychiatric fast-track lane, coordinate with county behavioral health")

    if _urban in ("Rural", "Frontier"):
        _causes.append("Rural facility with limited transfer options — higher burden per available station")
        _steps.append("Review mutual aid transfer agreements with regional trauma centers")

    if _bl is not None and _bl < 100:
        _causes.append("Small facility capacity constraints — few stations absorbing high relative volume")
        _steps.append("Apply for HCAI capacity expansion grant, review station utilization by shift")

    if _ownership == "Government":
        _causes.append("Public hospital serving uninsured population — higher ED utilization expected")
        _steps.append("Review Medi-Cal reimbursement optimization, coordinate with county on diversion protocols")

    if not _causes:
        _causes.append("No specific structural risk factors identified — monitor for volume trends")
        _steps.append("Continue standard monitoring and quarterly review")

    _rc_html = "".join(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:#374151;'
        f'padding:8px 12px;background:#fafaf9;border-radius:6px;margin-bottom:6px;'
        f'border-left:3px solid {_border};">{c}</div>'
        for c in _causes
    )
    _step_html = "".join(
        f'<li style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:#374151;margin-bottom:6px;">{s}</li>'
        for s in _steps
    )

    _sh = "font-family:'DM Serif Display',serif;font-size:14px;font-weight:normal;color:#1a1a18;"

    st.html(f"""
<div style="border:0.5px solid #e2ded8;border-left:4px solid {_border};border-radius:10px;
            padding:18px 20px;background:#ffffff;margin-top:4px;">
    <div style="font-family:'DM Sans',sans-serif;font-size:10px;font-weight:600;
                color:{_border};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:14px;">
        Contextual Risk Assessment
    </div>

    <div style="{_sh}margin-bottom:8px;">Facility Context</div>
    <div style="margin-bottom:14px;">{_pills_html}</div>

    <div style="{_sh}margin-bottom:8px;">Root Cause Analysis</div>
    <div style="margin-bottom:14px;">{_rc_html}</div>

    <div style="{_sh}margin-bottom:8px;">Actionable Next Steps</div>
    <ul style="margin:0;padding-left:18px;">{_step_html}</ul>
</div>
""")

# ── Expandable AI Summary ─────────────────────────────────────────────────────
st.html("<div style='height:10px'></div>")

# Compute burden trend from clean data (last two available years)
_fac_burden = (
    df[df["facility_name"] == facility]
    .sort_values("year")[["year", "true_burden_score"]]
    .dropna(subset=["true_burden_score"])
)
if len(_fac_burden) >= 2:
    _b_prev, _b_last = _fac_burden["true_burden_score"].iloc[-2], _fac_burden["true_burden_score"].iloc[-1]
    _trend = "Increasing" if _b_last > _b_prev * 1.03 else ("Decreasing" if _b_last < _b_prev * 0.97 else "Stable")
else:
    _trend = "Insufficient data"

with st.expander("View AI Operational Summary", expanded=False):
    _risk_label   = _risk_level(RISK_SCORE)
    _pc_note      = "in a primary care shortage area" if _pc_short == "Yes" else "with standard primary care access"
    _action       = (
        "review transfer protocols and surge staffing" if RISK_SCORE >= 0.7
        else "monitor burden trends closely"          if RISK_SCORE >= 0.4
        else "continue standard quarterly review"
    )
    _summary = (
        f"{facility} presents a {_risk_label.lower()} overload risk score of {RISK_SCORE:.2f} "
        f"for the next period. The primary driver is {top_driver_name}, reflecting "
        f"{_trend.lower()} burden pressure over the past reporting cycle. "
        f"Based on facility context — {_urban or 'unknown location'}, {_ownership or 'unknown ownership'}, "
        f"serving a population {_pc_note} — operational teams should {_action}."
    )
    st.html(f"""
<div class="ai-card">
    <div class="ai-card-header">Operational Summary</div>
    <div class="ai-card-body" style="font-family:'DM Sans',sans-serif;font-size:13px;
                                     color:#374151;line-height:1.6;">{_summary}</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:10px;color:#9ca3af;
                margin-top:10px;">Rule-based summary — AI layer pending</div>
</div>
""")
