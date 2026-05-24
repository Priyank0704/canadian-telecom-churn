"""
Feature engineering for the churn model.

Produces a clean modelling matrix from the segmented dataset:
  - One-hot encodes categoricals
  - Keeps engineered Canadian-context features
  - Drops identifiers and leakage-prone columns
  - Returns X, y, feature_names

This module is imported by train.py — no side effects when imported.
"""
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR

SEGMENTED_PATH = PROCESSED_DIR / "customers_segmented.parquet"

# Columns to drop before modelling
DROP_COLS = [
    "customerID", "Churn",          # raw target + ID
    "cluster", "segment_name",      # M2 outputs, not model inputs
    "province_name",                # redundant with province_code
    "provider",                     # already encoded as is_big3 + complaint rate
    "segment",                      # CRTC market segment
    "TotalCharges",                 # = tenure × MonthlyCharges → multicollinearity
]

TARGET_COL = "churn_flag"


def load_segmented() -> pd.DataFrame:
    df = pd.read_parquet(SEGMENTED_PATH)
    # Coerce TotalCharges (had blanks for tenure=0 customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def build_modelling_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Returns:
        X: feature matrix (all-numeric, one-hot encoded)
        y: target series (0/1)
        feature_names: list of column names in X
    """
    y = df[TARGET_COL].astype(int)

    # Drop unwanted cols defensively (some might not exist depending on M2 outputs)
    to_drop = [c for c in DROP_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=to_drop)

    # Also drop any tenure_bucket / affordability_band columns left over from EDA
    leftovers = [c for c in X.columns if X[c].dtype.name == "category"]
    if leftovers:
        X = X.drop(columns=leftovers)

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    # Ensure all columns are numeric float for downstream models
    X_encoded = X_encoded.astype(float)

    feature_names = X_encoded.columns.tolist()
    return X_encoded, y, feature_names


if __name__ == "__main__":
    df = load_segmented()
    X, y, features = build_modelling_matrix(df)
    print(f"[ok] X shape: {X.shape}")
    print(f"[ok] y shape: {y.shape}")
    print(f"[ok] positive class rate: {y.mean():.1%}")
    print(f"\nFeatures ({len(features)}):")
    for f in features:
        print(f"  - {f}")
