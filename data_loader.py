"""
data_loader.py — Load, clean, and reshape ca_ed_data.csv for Aegis.
Run: python data_loader.py
"""
import re
import pandas as pd
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_name(name: str) -> str:
    """Lowercase, strip, replace spaces with underscores."""
    return name.strip().lower().replace(" ", "_")


def _make_facility_id(name: str) -> str:
    """Lowercase, spaces → hyphens, strip non-alphanumeric (except hyphens)."""
    slug = name.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9\-]", "", slug)


def _find_col(columns, *keywords) -> str | None:
    """Return first column whose name exactly equals or ends with any keyword."""
    # Prefer exact matches first
    for kw in keywords:
        for col in columns:
            if col == kw:
                return col
    # Fall back to suffix / word-boundary match (e.g. "ed_visit" ends with "visit")
    for kw in keywords:
        for col in columns:
            parts = col.split("_")
            if parts[-1] == kw or col.endswith(f"_{kw}") or col.startswith(f"{kw}_"):
                return col
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    RAW_PATH   = "data/ca_ed_data.csv"
    CLEAN_PATH = "data/ca_ed_clean.csv"

    # ── STEP 1 — INSPECT ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — INSPECT")
    print("=" * 60)

    try:
        df = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        print(f"ERROR: File not found → {RAW_PATH}")
        return
    except Exception as exc:
        print(f"ERROR loading file: {exc}")
        return

    print("Column names and dtypes:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nShape: {df.shape}")

    # ── STEP 2 — CLEAN COLUMN NAMES ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — CLEAN COLUMN NAMES")
    print("=" * 60)

    df.columns = [_clean_name(c) for c in df.columns]
    print("Cleaned column names:", list(df.columns))

    # Identify key columns by inspection (no hardcoding)
    col_year     = _find_col(df.columns, "year")
    col_quarter  = _find_col(df.columns, "quarter", "qtr")
    col_facility = _find_col(df.columns, "facility_name", "facility")
    col_type     = _find_col(df.columns, "type")
    col_count    = _find_col(df.columns, "count", "value", "visits")

    print(f"\nIdentified key columns:")
    print(f"  year       → {col_year}")
    print(f"  quarter    → {col_quarter}")
    print(f"  facility   → {col_facility}")
    print(f"  type       → {col_type}")
    print(f"  count      → {col_count}")

    # ── STEP 3 — HANDLE MISSING DATA ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — HANDLE MISSING DATA")
    print("=" * 60)

    print("Null counts per column:")
    print(df.isnull().sum().to_string())

    rows_before = len(df)

    # Drop rows missing the count column (closest analog to encounters)
    # treatment_stations is not present in raw data; handled in Step 6
    if col_count:
        df = df.dropna(subset=[col_count])
    if col_facility:
        df = df.dropna(subset=[col_facility])

    rows_dropped = rows_before - len(df)
    print(f"\nRows dropped due to nulls: {rows_dropped}")
    print(f"Rows remaining: {len(df)}")

    # ── STEP 4 — PARSE DATES ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — PARSE DATES")
    print("=" * 60)

    if col_year:
        df[col_year] = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
        year_min = df[col_year].min()
        year_max = df[col_year].max()
        print(f"Year range: {year_min} – {year_max}")
    else:
        print("No year column found.")
        year_min = year_max = None

    if col_quarter:
        df[col_quarter] = pd.to_numeric(df[col_quarter], errors="coerce").astype("Int64")
        print(f"Quarters present: {sorted(df[col_quarter].dropna().unique().tolist())}")

    # ── STEP 5 — DROP SPARSE FACILITIES ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — DROP SPARSE FACILITIES")
    print("=" * 60)

    MIN_PERIODS = 3

    if col_facility and col_year:
        # Count distinct time periods per facility
        periods_per_facility = (
            df.groupby(col_facility)[col_year]
            .nunique()
            .rename("n_periods")
        )
        facilities_before = periods_per_facility.shape[0]
        valid_facilities   = periods_per_facility[periods_per_facility >= MIN_PERIODS].index
        df = df[df[col_facility].isin(valid_facilities)]
        facilities_after   = df[col_facility].nunique()
        facilities_dropped = facilities_before - facilities_after
        print(f"Facilities before filter : {facilities_before}")
        print(f"Facilities dropped (<{MIN_PERIODS} periods): {facilities_dropped}")
        print(f"Facilities remaining     : {facilities_after}")
    else:
        print("Cannot filter sparse facilities — facility or year column missing.")

    # ── PIVOT: long → wide ────────────────────────────────────────────────────
    # The raw data is long-format: one row per (facility, year, type).
    # Pivot so each 'type' value (e.g. ED_Visit, ED_Admit) becomes its own column.

    if col_type and col_count and col_facility and col_year:
        id_cols = [c for c in df.columns if c not in [col_type, col_count]]
        df = (
            df
            .pivot_table(
                index=id_cols,
                columns=col_type,
                values=col_count,
                aggfunc="sum",
            )
            .reset_index()
        )
        df.columns = [_clean_name(str(c)) for c in df.columns]
        print(f"\nAfter pivot, columns: {list(df.columns)}")

        # Re-identify column names post-pivot
        col_facility = _find_col(df.columns, "facility_name", "facility")
        col_year     = _find_col(df.columns, "year")

    # ── STEP 6 — CREATE BURDEN SCORE ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — CREATE BURDEN SCORE")
    print("=" * 60)

    col_encounters  = _find_col(df.columns, "ed_visit", "encounter", "visit")
    col_stations    = _find_col(df.columns, "treatment_station", "station", "bed")

    print(f"Encounters column found  : {col_encounters}")
    print(f"Treatment stations column: {col_stations}")

    if col_encounters and col_stations:
        df[col_encounters] = pd.to_numeric(df[col_encounters], errors="coerce")
        df[col_stations]   = pd.to_numeric(df[col_stations],   errors="coerce")
        df["burden_score"] = df[col_encounters] / df[col_stations]
    elif col_encounters:
        print(
            "WARNING: No treatment_stations column found. "
            "Computing burden_score as ED_Visit normalised by its 90th-percentile "
            "(proxy for relative load)."
        )
        df[col_encounters] = pd.to_numeric(df[col_encounters], errors="coerce")
        p90 = df[col_encounters].quantile(0.90)
        if p90 > 0:
            df["burden_score"] = df[col_encounters] / p90
        else:
            df["burden_score"] = np.nan
            print("WARNING: 90th-percentile is zero — burden_score set to NaN.")
    else:
        print("WARNING: Cannot compute burden_score — required columns missing.")
        df["burden_score"] = np.nan

    df["burden_score"] = df["burden_score"].clip(0, 2.0)

    bs = df["burden_score"].dropna()
    print(f"\nburden_score — min: {bs.min():.4f}  max: {bs.max():.4f}  mean: {bs.mean():.4f}")

    # ── STEP 7 — FACILITY ID ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — FACILITY ID")
    print("=" * 60)

    if col_facility:
        df["facility_id"] = df[col_facility].astype(str).apply(_make_facility_id)
        print("Sample facility_id values:")
        print(df[["facility_id", col_facility]].drop_duplicates().head(5).to_string(index=False))
    else:
        print("WARNING: No facility column found — skipping facility_id.")

    # ── STEP 8 — SAVE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — SAVE")
    print("=" * 60)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved → {CLEAN_PATH}")

    print("\n── Final Summary ──")
    print(f"Total rows          : {len(df)}")
    if col_facility:
        print(f"Unique facilities   : {df[col_facility].nunique()}")
    if col_year:
        print(f"Date range          : {df[col_year].min()} – {df[col_year].max()}")
    print(f"Columns in output   : {list(df.columns)}")

    if col_facility and "burden_score" in df.columns:
        top5 = (
            df.groupby(col_facility)["burden_score"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top5.columns = [col_facility, "avg_burden_score"]
        print("\nTop 5 facilities by avg burden_score:")
        print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
