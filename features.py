"""
Shared data-preparation helpers.

Both the training script (train_models.py) and the Streamlit app (app.py)
import from here, so the exact same cleaning + feature engineering is applied
at training time and at inference time.
"""

import numpy as np
import pandas as pd

TARGET_COL = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]


def tidy_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """The raw UCI file has a BOM, stray tabs and a typo ('Nacionality')."""
    df = df.copy()
    df.columns = (
        df.columns.str.replace("﻿", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.strip()
    )
    df = df.rename(columns={"Nacionality": "Nationality"})
    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, returning 0 where den == 0 (student enrolled in no units)."""
    return np.where(den > 0, num / den.replace(0, np.nan), 0.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven features describing academic progress.
    The raw data only has counts, so we turn them into rates and trends.
    """
    df = df.copy()
    s1_enr = df["Curricular units 1st sem (enrolled)"]
    s1_app = df["Curricular units 1st sem (approved)"]
    s2_enr = df["Curricular units 2nd sem (enrolled)"]
    s2_app = df["Curricular units 2nd sem (approved)"]

    df["pass_rate_sem1"] = _safe_ratio(s1_app, s1_enr)
    df["pass_rate_sem2"] = _safe_ratio(s2_app, s2_enr)
    df["pass_rate_trend"] = df["pass_rate_sem2"] - df["pass_rate_sem1"]
    df["grade_trend"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )
    df["total_units_approved"] = s1_app + s2_app
    # Financial stress flag: owes money OR tuition not up to date
    df["financial_risk"] = (
        (df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0)
    ).astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    """Return X (engineered features) and y (string labels)."""
    df = add_engineered_features(tidy_column_names(df))
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    return X, y
