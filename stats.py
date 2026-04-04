"""
stats.py — Statistical inference for Aegis ED Early Warning.
Run: python stats.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

CLEAN_PATH    = "data/ca_ed_clean.csv"
FEATURES_PATH = "data/ca_ed_features.csv"
MODELS_DIR    = "models"

FACILITY_COL  = "facility_name"
COUNTY_COL    = "county_name"
YEAR_COL      = "year"
BURDEN_COL    = "burden_score"
SERVICE_COL   = "er_service_level_desc"

COVID_YEAR    = 2020
Z_THRESHOLD   = 2.0
CI_ALPHA      = 0.05


def _round(val):
    """Round a float to 4 decimal places; handle non-finite gracefully."""
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _interpret(p, alpha=0.05):
    return "significant" if p < alpha else "not significant"


def main():
    df       = pd.read_csv(CLEAN_PATH)
    df_feat  = pd.read_csv(FEATURES_PATH)
    print(f"Loaded clean: {df.shape}  |  features: {df_feat.shape}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — PAIRED T-TEST
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1 — PAIRED T-TEST")
    print("=" * 60)

    ttest_results = {}

    # --- 1a: even vs odd years ---
    even_mask = (df[YEAR_COL] % 2 == 0)
    even_vals = df.loc[even_mask,  BURDEN_COL].dropna().values
    odd_vals  = df.loc[~even_mask, BURDEN_COL].dropna().values
    # independent test (different facilities/years in each group)
    min_n = min(len(even_vals), len(odd_vals))
    t_eo, p_eo = stats.ttest_ind(even_vals[:min_n], odd_vals[:min_n], equal_var=False)
    print(f"\nEven vs Odd years (independent, n={min_n} each):")
    print(f"  t = {t_eo:.4f},  p = {p_eo:.4f}  → {_interpret(p_eo)}")
    ttest_results["even_vs_odd"] = {
        "t_stat": _round(t_eo), "p_value": _round(p_eo),
        "interpretation": _interpret(p_eo), "n": min_n,
    }

    # --- 1b: pre vs post COVID (paired per facility, using facility means) ---
    pre  = df[df[YEAR_COL] <  COVID_YEAR].groupby(FACILITY_COL)[BURDEN_COL].mean()
    post = df[df[YEAR_COL] >= COVID_YEAR].groupby(FACILITY_COL)[BURDEN_COL].mean()
    paired = pd.DataFrame({"pre": pre, "post": post}).dropna()
    t_covid, p_covid = stats.ttest_rel(paired["pre"].values, paired["post"].values)
    print(f"\nPre-COVID vs Post-COVID burden_score (paired per facility, n={len(paired)}):")
    print(f"  t = {t_covid:.4f},  p = {p_covid:.4f}  → {_interpret(p_covid)}")
    print(f"  Mean pre-COVID : {paired['pre'].mean():.4f}")
    print(f"  Mean post-COVID: {paired['post'].mean():.4f}")
    ttest_results["pre_vs_post_covid"] = {
        "t_stat": _round(t_covid), "p_value": _round(p_covid),
        "interpretation": _interpret(p_covid),
        "mean_pre": _round(paired["pre"].mean()),
        "mean_post": _round(paired["post"].mean()),
        "n_facilities": len(paired),
    }

    with open(f"{MODELS_DIR}/ttest_results.json", "w") as f:
        json.dump(ttest_results, f, indent=2)
    print(f"\nSaved → {MODELS_DIR}/ttest_results.json")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — MANN-WHITNEY U TEST
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2 — MANN-WHITNEY U TEST")
    print("=" * 60)

    print("\nUnique service level values:")
    svc_counts = df[SERVICE_COL].value_counts()
    print(svc_counts.to_string())

    highest_level = svc_counts.index[0]   # most frequent = BASIC usually; we want highest tier
    # "highest" = COMPREHENSIVE (most advanced), all others = rest
    # Determine by known ED tier hierarchy
    tier_order = {"COMPREHENSIVE": 3, "BASIC": 2, "STANDBY": 1}
    ranked = sorted(svc_counts.index, key=lambda x: tier_order.get(x, 0), reverse=True)
    highest_level = ranked[0]   # COMPREHENSIVE
    print(f"\nHighest service level (group A): {highest_level}")
    print(f"All others           (group B): {[v for v in ranked[1:]]}")

    group_a = df[df[SERVICE_COL] == highest_level][BURDEN_COL].dropna().values
    group_b = df[df[SERVICE_COL] != highest_level][BURDEN_COL].dropna().values

    u_stat, p_mw = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    print(f"\nMann-Whitney U: U = {u_stat:.4f},  p = {p_mw:.4f}  → {_interpret(p_mw)}")
    print(f"  Median {highest_level}   : {np.median(group_a):.4f}  (n={len(group_a)})")
    print(f"  Median others       : {np.median(group_b):.4f}  (n={len(group_b)})")

    mw_results = {
        "highest_level": highest_level,
        "u_stat": _round(u_stat),
        "p_value": _round(p_mw),
        "interpretation": _interpret(p_mw),
        "median_highest": _round(np.median(group_a)),
        "median_others":  _round(np.median(group_b)),
        "n_highest": int(len(group_a)),
        "n_others":  int(len(group_b)),
    }
    with open(f"{MODELS_DIR}/mannwhitney_results.json", "w") as f:
        json.dump(mw_results, f, indent=2)
    print(f"Saved → {MODELS_DIR}/mannwhitney_results.json")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3 — CONFIDENCE INTERVALS")
    print("=" * 60)

    ci_rows = []
    for facility, grp in df.groupby(FACILITY_COL):
        vals = grp[BURDEN_COL].dropna().values
        n = len(vals)
        if n < 2:
            continue
        mean  = np.mean(vals)
        se    = stats.sem(vals)
        lo, hi = stats.t.interval(1 - CI_ALPHA, df=n - 1, loc=mean, scale=se)
        ci_rows.append({
            "facility_name": facility,
            "mean_burden":   round(mean, 4),
            "ci_lower":      round(lo,   4),
            "ci_upper":      round(hi,   4),
            "ci_width":      round(hi - lo, 4),
            "n_years":       n,
        })

    ci_df = pd.DataFrame(ci_rows)

    # Top 20 by mean burden
    top20 = ci_df.nlargest(20, "mean_burden")[
        ["facility_name", "mean_burden", "ci_lower", "ci_upper", "n_years"]
    ]
    top20.to_csv(f"{MODELS_DIR}/facility_ci.csv", index=False)
    print(f"Saved top-20 by mean burden → {MODELS_DIR}/facility_ci.csv")

    # Top 5 widest CIs
    top5_wide = ci_df.nlargest(5, "ci_width")[
        ["facility_name", "mean_burden", "ci_lower", "ci_upper", "ci_width", "n_years"]
    ]
    print("\nTop 5 facilities with widest 95% CIs:")
    print(top5_wide.to_string(index=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — ROLLING Z-SCORE ANOMALIES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4 — ROLLING Z-SCORE ANOMALIES")
    print("=" * 60)

    df_sorted = df.sort_values([FACILITY_COL, YEAR_COL]).copy()

    grp = df_sorted.groupby(FACILITY_COL)[BURDEN_COL]
    df_sorted["roll_mean"] = grp.transform(lambda s: s.rolling(3, min_periods=2).mean())
    df_sorted["roll_std"]  = grp.transform(lambda s: s.rolling(3, min_periods=2).std())

    # Avoid division by zero
    df_sorted["z_score"] = np.where(
        df_sorted["roll_std"] > 0,
        (df_sorted[BURDEN_COL] - df_sorted["roll_mean"]) / df_sorted["roll_std"],
        0.0,
    )
    df_sorted["anomaly"] = (df_sorted["z_score"].abs() > Z_THRESHOLD).astype(int)

    anomaly_df = df_sorted[df_sorted["anomaly"] == 1][
        [YEAR_COL, FACILITY_COL, BURDEN_COL, "z_score", "anomaly"]
    ].copy()
    anomaly_df["z_score"] = anomaly_df["z_score"].round(4)
    anomaly_df.to_csv(f"{MODELS_DIR}/anomalies.csv", index=False)

    total_anomalies = len(anomaly_df)
    print(f"Total anomaly rows: {total_anomalies}")

    top5_anom = (
        anomaly_df.groupby(FACILITY_COL)["anomaly"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nTop 5 most anomalous facilities:")
    print(top5_anom.to_string())
    print(f"\nSaved → {MODELS_DIR}/anomalies.csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 — COUNTY LEVEL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 5 — COUNTY LEVEL SUMMARY")
    print("=" * 60)

    county_year = (
        df.groupby([COUNTY_COL, YEAR_COL])[BURDEN_COL]
        .mean()
        .reset_index()
        .rename(columns={BURDEN_COL: "mean_burden_score"})
    )
    county_year["mean_burden_score"] = county_year["mean_burden_score"].round(4)

    top10_counties = (
        county_year.groupby(COUNTY_COL)["mean_burden_score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"mean_burden_score": "avg_burden_all_years"})
    )
    top10_counties["avg_burden_all_years"] = top10_counties["avg_burden_all_years"].round(4)

    county_year.to_csv(f"{MODELS_DIR}/county_summary.csv", index=False)
    print(f"Saved → {MODELS_DIR}/county_summary.csv")
    print("\nTop 10 counties by average burden_score:")
    print(top10_counties.to_string(index=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6 — SAVE ALL RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 6 — SAVE ALL RESULTS")
    print("=" * 60)

    combined = {
        "ttest_covid": {
            "t_stat":         _round(t_covid),
            "p_value":        _round(p_covid),
            "interpretation": _interpret(p_covid),
            "mean_pre":       _round(paired["pre"].mean()),
            "mean_post":      _round(paired["post"].mean()),
        },
        "mannwhitney": {
            "u_stat":         _round(u_stat),
            "p_value":        _round(p_mw),
            "interpretation": _interpret(p_mw),
            "highest_level":  highest_level,
        },
        "anomaly_count": int(total_anomalies),
        "top_counties":  top10_counties.to_dict(orient="records"),
    }

    with open(f"{MODELS_DIR}/stats_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("Stats complete. All results saved to models/")


if __name__ == "__main__":
    main()
