"""
feature_engineering.py — Build ML-ready features from ca_ed_clean.csv.
Run: python feature_engineering.py
"""
import pandas as pd
import numpy as np

INPUT_PATH  = "data/ca_ed_final.csv"
OUTPUT_PATH = "data/ca_ed_features_v2.csv"

FACILITY_COL = "facility_id"
YEAR_COL     = "year"
BURDEN_COL   = "burden_score"
VISIT_COL    = "ed_visit"
BASE_YEAR    = 2012
COVID_YEAR   = 2020


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}\n")

    # ── STEP 1 — SORT ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — SORT")
    print("=" * 60)

    df = df.sort_values([FACILITY_COL, YEAR_COL], ascending=True).reset_index(drop=True)
    print(f"Sorted by [{FACILITY_COL}, {YEAR_COL}] ascending. Shape: {df.shape}")

    # ── STEP 2 — LAG FEATURES ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — LAG FEATURES")
    print("=" * 60)

    grp = df.groupby(FACILITY_COL)[BURDEN_COL]
    df["burden_lag_1"] = grp.shift(1)
    df["burden_lag_2"] = grp.shift(2)
    df["burden_lag_3"] = grp.shift(3)

    lag_nulls = df[["burden_lag_1", "burden_lag_2", "burden_lag_3"]].isnull().sum()
    print("Nulls created by lagging:")
    print(lag_nulls.to_string())

    # New lag: visits_per_station_normalized (if present)
    VPS_NORM_COL = "visits_per_station_normalized"
    try:
        if VPS_NORM_COL in df.columns:
            df["visits_per_station_lag1"] = (
                df.groupby(FACILITY_COL)[VPS_NORM_COL].shift(1)
            )
            print(f"\nvisits_per_station_lag1 created from {VPS_NORM_COL}.")
            print(f"  Nulls: {df['visits_per_station_lag1'].isnull().sum()}")
        else:
            print(f"\n{VPS_NORM_COL} not found — visits_per_station_lag1 skipped.")
    except Exception as exc:
        print(f"WARNING: Could not create visits_per_station_lag1: {exc}")

    # ── STEP 3 — ROLLING FEATURES ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — ROLLING FEATURES")
    print("=" * 60)

    def rolling_mean(series, window):
        return series.rolling(window, min_periods=1).mean()

    def rolling_std(series, window):
        return series.rolling(window, min_periods=1).std().fillna(0)

    df["rolling_mean_3"] = grp.transform(lambda s: rolling_mean(s, 3))
    df["rolling_mean_7"] = grp.transform(lambda s: rolling_mean(s, 7))
    df["rolling_std_7"]  = grp.transform(lambda s: rolling_std(s, 7))

    sample_facility = df[FACILITY_COL].iloc[0]
    sample = df[df[FACILITY_COL] == sample_facility][
        [YEAR_COL, BURDEN_COL, "rolling_mean_3", "rolling_mean_7", "rolling_std_7"]
    ]
    print(f"Rolling features for facility: {sample_facility}")
    print(sample.to_string(index=False))

    # ── STEP 4 — MOMENTUM ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — MOMENTUM")
    print("=" * 60)

    df["pct_change_recent"] = (
        (df[BURDEN_COL] - df["burden_lag_1"]) / df["burden_lag_1"]
    ).fillna(0).clip(-1, 1)

    df["ed_visits_per_year"] = df[VISIT_COL]

    print("pct_change_recent — min: {:.4f}  max: {:.4f}  mean: {:.4f}".format(
        df["pct_change_recent"].min(),
        df["pct_change_recent"].max(),
        df["pct_change_recent"].mean(),
    ))
    print(f"ed_visits_per_year column added (copy of {VISIT_COL}).")

    # ── STEP 5 — TIME FEATURES ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — TIME FEATURES")
    print("=" * 60)

    df["month"]          = 1
    df["is_post_covid"]  = (df[YEAR_COL] >= COVID_YEAR).astype(int)
    df["year_index"]     = df[YEAR_COL] - BASE_YEAR

    print(f"month         : always 1 (annual data placeholder)")
    print(f"is_post_covid : 1 if year >= {COVID_YEAR}")
    print(f"year_index    : year - {BASE_YEAR}  (range {df['year_index'].min()}–{df['year_index'].max()})")

    # ── STEP 6 — TARGET VARIABLE ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — TARGET VARIABLE")
    print("=" * 60)

    # Per-facility 75th-percentile threshold
    facility_p75 = (
        df.groupby(FACILITY_COL)[BURDEN_COL]
        .transform(lambda s: s.quantile(0.75))
    )

    next_year_burden = df.groupby(FACILITY_COL)[BURDEN_COL].shift(-1)
    df["high_burden_next"] = (next_year_burden > facility_p75).astype("Int64")

    null_target = df["high_burden_next"].isnull().sum()
    df = df.dropna(subset=["high_burden_next"]).copy()
    df["high_burden_next"] = df["high_burden_next"].astype(int)

    print(f"Dropped {null_target} rows where high_burden_next was null (last year per facility).")
    balance = df["high_burden_next"].value_counts().sort_index()
    print("Class balance:")
    print(f"  0 (not high burden next year): {balance.get(0, 0)}")
    print(f"  1 (high burden next year)     : {balance.get(1, 0)}")
    print(f"  Positive rate: {balance.get(1, 0) / len(df):.1%}")

    # ── STEP 7 — DROP NULLS ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — DROP NULLS")
    print("=" * 60)

    lag_cols = ["burden_lag_1", "burden_lag_2", "burden_lag_3"]
    rows_before = len(df)
    df = df.dropna(subset=lag_cols).copy()
    rows_dropped = rows_before - len(df)
    print(f"Dropped {rows_dropped} rows with nulls in lag columns.")
    print(f"Final row count: {len(df)}")

    # ── STEP 8 — SAVE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — SAVE")
    print("=" * 60)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")

    final_balance = df["high_burden_next"].value_counts().sort_index()
    feature_cols = [c for c in df.columns if c not in [
        "year", "oshpd_id", "facility_name", "county_name",
        "er_service_level_desc", "ed_admit", "ed_visit",
        "burden_score", "facility_id", "high_burden_next",
        "true_burden_score", "burden_score_normalized",
        "visits_per_station", "visits_per_station_normalized",
    ]]
    print("\n── Final Summary ──")
    print(f"Total rows          : {len(df)}")
    print(f"Total feature count : {len(feature_cols)}")
    print(f"Features            : {feature_cols}")
    print(f"Class balance       : 0 = {final_balance.get(0, 0)}, 1 = {final_balance.get(1, 0)}")
    print(f"Positive rate       : {final_balance.get(1, 0) / len(df):.1%}")
    print(f"Year range          : {df[YEAR_COL].min()} – {df[YEAR_COL].max()}")
    print(f"Unique facilities   : {df[FACILITY_COL].nunique()}")


if __name__ == "__main__":
    main()
