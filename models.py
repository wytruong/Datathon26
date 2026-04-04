"""
models.py — Train and save ML models for Aegis ED Early Warning.
Run: python models.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_PATH   = "data/ca_ed_features.csv"
MODELS_DIR   = "models"

# ── Column config ──────────────────────────────────────────────────────────────
TARGET_COL   = "high_burden_next"
BURDEN_COL   = "burden_score"
YEAR_COL     = "year"
RANDOM_STATE = 42

DROP_COLS = [
    "year", "oshpd_id", "facility_name", "county_name",
    "facility_id", "er_service_level_desc",
    "ed_visit", "ed_admit", "burden_score", "high_burden_next",
]

TRAIN_CUTOFF = 2021  # train on year <= this
TEST_START   = 2022  # test  on year >= this


def _save(obj, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    print(f"  Saved → {path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}\n")

    # ── Time-aware split (before dropping year) ────────────────────────────────
    train_df = df[df[YEAR_COL] <= TRAIN_CUTOFF].copy()
    test_df  = df[df[YEAR_COL] >= TEST_START].copy()
    print(f"Train size : {len(train_df)} rows (year <= {TRAIN_CUTOFF})")
    print(f"Test size  : {len(test_df)} rows  (year >= {TEST_START})\n")

    # ── Build X / y for classification models ─────────────────────────────────
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[TARGET_COL].copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — LOGISTIC REGRESSION
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1 — LOGISTIC REGRESSION")
    print("=" * 60)

    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred  = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, lr_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

    # Coefficients
    lr_model = lr_pipeline.named_steps["model"]
    coef_df = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": lr_model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\nTop 10 coefficients (by magnitude):")
    print(coef_df.head(10).to_string(index=False))

    _save(lr_pipeline, "logistic_model.pkl")
    coef_df.to_csv(os.path.join(MODELS_DIR, "logistic_coefficients.csv"), index=False)
    print(f"  Saved → {MODELS_DIR}/logistic_coefficients.csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — XGBOOST
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2 — XGBOOST")
    print("=" * 60)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=3,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred  = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, xgb_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

    importances_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 10 feature importances:")
    print(importances_df.head(10).to_string(index=False))

    _save(xgb_model, "xgb_model.pkl")
    importances_df.to_csv(os.path.join(MODELS_DIR, "xgb_importances.csv"), index=False)
    print(f"  Saved → {MODELS_DIR}/xgb_importances.csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — LINEAR REGRESSION (predict burden_score)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3 — LINEAR REGRESSION")
    print("=" * 60)

    train_lr = train_df.dropna(subset=[BURDEN_COL])
    test_lr  = test_df.dropna(subset=[BURDEN_COL])
    y_train_lr = train_lr[BURDEN_COL].copy()
    y_test_lr  = test_lr[BURDEN_COL].copy()
    X_train_lr = train_lr[feature_cols].copy()
    X_test_lr  = test_lr[feature_cols].copy()

    lin_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LinearRegression()),
    ])
    lin_model.fit(X_train_lr, y_train_lr)
    lin_pred = lin_model.predict(X_test_lr)

    rmse = np.sqrt(mean_squared_error(y_test_lr, lin_pred))
    r2   = r2_score(y_test_lr, lin_pred)
    print(f"RMSE      : {rmse:.4f}")
    print(f"R-squared : {r2:.4f}")

    _save(lin_model, "linear_model.pkl")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — ARIMA (statewide average burden_score forecast)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4 — ARIMA")
    print("=" * 60)

    state_avg = (
        df.groupby(YEAR_COL)[BURDEN_COL]
        .mean()
        .sort_index()
    )
    print("Statewide average burden_score per year:")
    print(state_avg.to_string())

    arima_model = ARIMA(state_avg.values, order=(1, 1, 1))
    arima_fit   = arima_model.fit()
    print(f"\nAIC: {arima_fit.aic:.4f}")

    n_forecast   = 3
    last_year    = int(state_avg.index.max())
    forecast_obj = arima_fit.get_forecast(steps=n_forecast)
    forecast_mean = forecast_obj.predicted_mean
    conf_int      = forecast_obj.conf_int(alpha=0.05)

    forecast_years = list(range(last_year + 1, last_year + 1 + n_forecast))
    forecast_df = pd.DataFrame({
        "year":       forecast_years,
        "forecast":   forecast_mean,
        "lower_95":   conf_int[:, 0],
        "upper_95":   conf_int[:, 1],
    })
    print("\nForecast (3 years ahead):")
    print(forecast_df.to_string(index=False))

    _save(arima_fit, "arima_model.pkl")
    forecast_df.to_csv(os.path.join(MODELS_DIR, "arima_forecast.csv"), index=False)
    print(f"  Saved → {MODELS_DIR}/arima_forecast.csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 — SAVE FEATURE LIST
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 5 — SAVE FEATURE LIST")
    print("=" * 60)

    feature_list_path = os.path.join(MODELS_DIR, "feature_list.txt")
    with open(feature_list_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print(f"Saved {len(feature_cols)} features → {feature_list_path}")
    print("Features:", feature_cols)


if __name__ == "__main__":
    main()
