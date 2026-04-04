"""Cached data and model loaders for Aegis."""
import json
import os
import joblib
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(__file__))


def _path(*parts):
    return os.path.join(_ROOT, *parts)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(_path("data", "ca_ed_clean.csv"))


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(_path("data", "ca_ed_features.csv"))


@st.cache_resource
def load_logistic_model():
    return joblib.load(_path("models", "logistic_model.pkl"))


@st.cache_resource
def load_xgb_model():
    return joblib.load(_path("models", "xgb_model.pkl"))


@st.cache_resource
def load_linear_model():
    return joblib.load(_path("models", "linear_model.pkl"))


@st.cache_data
def load_logistic_coefficients() -> pd.DataFrame:
    return pd.read_csv(_path("models", "logistic_coefficients.csv"))


@st.cache_data
def load_xgb_importances() -> pd.DataFrame:
    return pd.read_csv(_path("models", "xgb_importances.csv"))


@st.cache_data
def load_arima_forecast() -> pd.DataFrame:
    return pd.read_csv(_path("models", "arima_forecast.csv"))


@st.cache_data
def load_facility_ci() -> pd.DataFrame:
    return pd.read_csv(_path("models", "facility_ci.csv"))


@st.cache_data
def load_county_summary() -> pd.DataFrame:
    return pd.read_csv(_path("models", "county_summary.csv"))


@st.cache_data
def load_feature_list() -> list:
    with open(_path("models", "feature_list.txt")) as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_data
def load_stats_results() -> dict:
    with open(_path("models", "stats_results.json")) as f:
        return json.load(f)
