# Aegis — ED Overload Early Warning System

## What is Aegis

Aegis is a real-time early warning dashboard for Emergency Department overload, built for the Heist Datathon 2026 Omni Track. Emergency departments across California face chronic overcrowding — a problem that delays care, strains staff, and worsens patient outcomes. Aegis addresses this by combining historical ED encounter data with machine learning models to predict which facilities are at high risk of overload in the coming year, surfacing those predictions through an interactive Streamlit dashboard that hospital administrators and policymakers can act on immediately.

---

## Demo Screenshot

<!-- Add screenshot here -->

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Dashboard UI and page routing |
| `plotly` | Interactive charts and visualizations |
| `pandas` | Data loading, cleaning, and feature engineering |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic regression, linear regression, preprocessing |
| `xgboost` | Gradient boosted classifier for risk prediction |
| `statsmodels` | ARIMA time series forecasting |
| `scipy` | Statistical tests (t-test, Mann-Whitney U, confidence intervals) |
| `joblib` | Model serialization and loading |

---

## Project Structure

```
aegis/
├── app.py                    # Streamlit entry point, sidebar nav, page routing
├── data_loader.py            # Loads and cleans raw CSV → data/ca_ed_clean.csv
├── feature_engineering.py    # Builds lag, rolling, and target features → data/ca_ed_features.csv
├── models.py                 # Trains and saves all 4 models to models/
├── stats.py                  # Statistical inference: t-tests, CIs, anomalies, county summaries
│
├── pages/
│   ├── overview.py           # Overview page: burden trend chart and metric cards
│   ├── risk.py               # Risk Score page: live model prediction and feature drivers
│   └── insights.py           # Insights page: facility rankings, t-test card, county chart
│
├── utils/
│   ├── style.py              # Shared CSS, color tokens, and HTML component helpers
│   └── data.py               # Cached data and model loader functions
│
├── data/
│   ├── ca_ed_data.csv        # Raw source data (HCAI, 2012–2024)
│   ├── ca_ed_clean.csv       # Cleaned and pivoted data (output of data_loader.py)
│   └── ca_ed_features.csv    # Feature-engineered dataset (output of feature_engineering.py)
│
├── models/
│   ├── logistic_model.pkl        # Trained logistic regression pipeline
│   ├── xgb_model.pkl             # Trained XGBoost classifier
│   ├── linear_model.pkl          # Trained linear regression pipeline
│   ├── arima_model.pkl           # Fitted ARIMA(1,1,1) model
│   ├── logistic_coefficients.csv # Feature coefficients from logistic model
│   ├── xgb_importances.csv       # Feature importances from XGBoost
│   ├── arima_forecast.csv        # 3-year statewide burden forecast (2025–2027)
│   ├── facility_ci.csv           # Top 20 facilities with 95% confidence intervals
│   ├── county_summary.csv        # County-level burden aggregates by year
│   ├── feature_list.txt          # Ordered feature column names for inference
│   ├── ttest_results.json        # T-test results (even/odd years, pre/post COVID)
│   ├── mannwhitney_results.json  # Mann-Whitney U test results by service level
│   ├── anomalies.csv             # Flagged anomalies (z-score > 2.0)
│   └── stats_results.json        # Combined statistical results summary
│
└── requirements.txt          # Python dependencies
```

---

## How to Run

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd aegis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

> **Note:** The cleaned data (`data/ca_ed_clean.csv`, `data/ca_ed_features.csv`) and all trained models (`models/`) are already included in the repository. You do not need to rerun the pipeline scripts unless you want to regenerate them from scratch.

To regenerate from the raw data:
```bash
python data_loader.py
python feature_engineering.py
python models.py
python stats.py
```

---

## Data Source

**California ED Encounters by Facility** — California Health and Human Services Agency (HCAI)

- Source: [https://data.chhs.ca.gov](https://data.chhs.ca.gov)
- Years covered: 2012–2024
- 440 facilities, ~8,400 raw rows (annual ED visits and admissions per facility)

---

## Models

| Model | Description |
|---|---|
| **Logistic Regression** | Binary classifier predicting high ED burden next year (ROC-AUC 0.77) |
| **XGBoost** | Gradient boosted classifier for the same binary target (ROC-AUC 0.69) |
| **Linear Regression** | Predicts raw burden score directly from lag and rolling features (R² 0.9998) |
| **ARIMA(1,1,1)** | Time series model on statewide average burden for 3-year forward forecast |

---

## Key Findings

- **Kings County** has the highest average ED burden score at **1.39**, well above the statewide mean
- **COMPREHENSIVE** service level facilities have a median burden of **0.79** vs **0.44** for all others — a statistically significant difference (Mann-Whitney U, p ≈ 0)
- ED burden differs significantly between even and odd years (p = 0.042), suggesting a measurable temporal pattern in the data
- **ARIMA forecasts** statewide average burden holding steady at approximately **0.587** through 2027, with widening uncertainty intervals

---

## Team

Built for **Heist Datathon 2026 — Omni Track**
