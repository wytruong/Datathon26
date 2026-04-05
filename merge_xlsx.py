import pandas as pd
import os
import sys

# ── STEP 1 — LOAD XLSX ────────────────────────────────────────────────────────
SEARCH_PATHS = [
    "data/emergency-department-volume-and-capacity-2021-2023.xlsx",
    "emergency-department-volume-and-capacity-2021-2023.xlsx",
]

xlsx_path = next((p for p in SEARCH_PATHS if os.path.exists(p)), None)
if xlsx_path is None:
    print("ERROR: XLSX file not found. Place it in the project root or data/ folder.")
    sys.exit(1)

print(f"Found XLSX at: {xlsx_path}")
xl = pd.ExcelFile(xlsx_path)
print(f"\nSheet names: {xl.sheet_names}")

# Load first sheet (adjust index if needed)
sheet = xl.sheet_names[0]
print(f"\nLoading sheet: '{sheet}'")
df_xl = xl.parse(sheet)

print(f"\nColumns:\n{list(df_xl.columns)}")
print(f"\nFirst 5 rows:\n{df_xl.head()}")
print(f"\nShape: {df_xl.shape}")

# ── STEP 2 — IDENTIFY KEY COLUMNS ─────────────────────────────────────────────
cols_lower = {c: c.lower() for c in df_xl.columns}

def find_col(keywords):
    for orig, lower in cols_lower.items():
        if any(k in lower for k in keywords):
            return orig
    return None

station_col  = find_col(["station", "treatment", "capacity", "beds"])
facility_col = find_col(["facility", "hospital", "name"])
year_col     = find_col(["year", "yr", "period"])

print(f"\nIdentified columns:")
print(f"  Treatment stations : {station_col}")
print(f"  Facility name      : {facility_col}")
print(f"  Year               : {year_col}")

missing = [name for name, col in [("station", station_col), ("facility", facility_col), ("year", year_col)] if col is None]
if missing:
    print(f"\nWARNING: Could not auto-detect columns: {missing}")
    print("Edit the find_col calls above with the exact column names and re-run.")
    sys.exit(1)

# ── STEP 3 — CLEAN XLSX ───────────────────────────────────────────────────────
vps_col = "Visits_Per_Station"
keep_cols = [facility_col, year_col, station_col]
if vps_col in df_xl.columns:
    keep_cols.append(vps_col)

df_xl = df_xl[keep_cols].copy()
rename_map = {facility_col: "facility_name", year_col: "year", station_col: "treatment_stations"}
if vps_col in df_xl.columns:
    rename_map[vps_col] = "visits_per_station"
df_xl.rename(columns=rename_map, inplace=True)

df_xl["facility_name"]      = df_xl["facility_name"].astype(str).str.strip()
df_xl["year"]               = pd.to_numeric(df_xl["year"], errors="coerce")
df_xl["treatment_stations"] = pd.to_numeric(df_xl["treatment_stations"], errors="coerce")

df_xl.dropna(subset=["treatment_stations"], inplace=True)
print(f"\nRows after dropping null treatment_stations: {len(df_xl)}")

# ── STEP 4 — MERGE ────────────────────────────────────────────────────────────
df = pd.read_csv("data/ca_ed_clean.csv")
print(f"\nca_ed_clean.csv shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Normalise join keys
df["_fac_key"]    = df["facility_name"].astype(str).str.strip().str.lower()
df_xl["_fac_key"] = df_xl["facility_name"].str.lower()
df["_year_key"]   = pd.to_numeric(df["year"], errors="coerce")
df_xl["_year_key"] = df_xl["year"]

xl_merge_cols = ["_fac_key", "_year_key", "treatment_stations"]
if "visits_per_station" in df_xl.columns:
    xl_merge_cols.append("visits_per_station")

df = df.merge(
    df_xl[xl_merge_cols],
    on=["_fac_key", "_year_key"],
    how="left",
)
df.drop(columns=["_fac_key", "_year_key"], inplace=True)

# Recompute burden where real station data exists
ed_visit_col = next((c for c in df.columns if "visit" in c.lower() or "ed_visit" in c.lower()), None)
if ed_visit_col is None:
    print("\nWARNING: Could not find ED visit column — skipping true_burden_score computation.")
    df["true_burden_score"]   = df.get("burden_score", float("nan"))
    df["has_treatment_data"]  = 0
else:
    mask = df["treatment_stations"].notna() & (df["treatment_stations"] > 0)
    df["true_burden_score"]  = df.get("burden_score", float("nan"))
    df.loc[mask, "true_burden_score"] = (
        df.loc[mask, ed_visit_col] / df.loc[mask, "treatment_stations"]
    )
    df["has_treatment_data"] = mask.astype(int)

    # ── Step 4a — Raw stats before normalizing ────────────────────────────────
    raw = df.loc[mask, "true_burden_score"]
    print(f"\ntrue_burden_score (raw, before normalization):")
    print(f"  min  : {raw.min():.2f}")
    print(f"  max  : {raw.max():.2f}")
    print(f"  mean : {raw.mean():.2f}")

    # ── Step 4b — Normalize true_burden_score (0–2, facility 90th pct) ───────
    p90_burden = (
        df[mask]
        .groupby("facility_name")["true_burden_score"]
        .transform(lambda x: x.quantile(0.90))
    )
    df["burden_score_normalized"] = float("nan")
    df.loc[mask, "burden_score_normalized"] = (
        df.loc[mask, "true_burden_score"] / p90_burden
    ).clip(0, 2.0)

    # ── Step 4c — Normalize Visits_Per_Station (0–2, facility 90th pct) ──────
    if "visits_per_station" in df.columns:
        vps_mask = mask & df["visits_per_station"].notna()
        p90_vps = (
            df[vps_mask]
            .groupby("facility_name")["visits_per_station"]
            .transform(lambda x: x.quantile(0.90))
        )
        df["visits_per_station_normalized"] = float("nan")
        df.loc[vps_mask, "visits_per_station_normalized"] = (
            df.loc[vps_mask, "visits_per_station"] / p90_vps
        ).clip(0, 2.0)

real_count = df["has_treatment_data"].sum()
print(f"\nRows with real treatment station data: {real_count}")

# ── STEP 5 — SAVE ─────────────────────────────────────────────────────────────
out_path = "data/ca_ed_merged.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

fallback = len(df) - real_count
top5 = (
    df[df["has_treatment_data"] == 1]
    .groupby("facility_name")["burden_score_normalized"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

print(f"\n── Final Summary ──────────────────────────────────────────")
print(f"  Total rows                        : {len(df)}")
print(f"  Rows with real treatment data     : {real_count}")
print(f"  Rows using fallback burden_score  : {fallback}")
print(f"\n  Columns in final file:")
for c in df.columns:
    print(f"    {c}")
print(f"\n  Top 5 facilities by burden_score_normalized:")
print(top5.to_string())
