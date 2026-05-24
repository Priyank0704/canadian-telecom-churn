"""
Generate a tiny synthetic Telco CSV for CI.

Why this exists: CI runners shouldn't depend on GitHub being reachable
(or on the IBM repo continuing to exist). The synthetic file matches the
real schema exactly so the downstream pipeline can be exercised end-to-end.

This script is NEVER run in production — only by .github/workflows/ci.yml
as a substitute for src/collection/download_telco.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TELCO_RAW


def make_synthetic_telco(n: int = 7043, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"{i:04d}-CI" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n,
                                p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
              "Bank transfer (automatic)", "Credit card (automatic)"], n,
        ),
        "MonthlyCharges": rng.uniform(18, 120, n).round(2),
    })
    total = (df["tenure"] * df["MonthlyCharges"] + rng.normal(0, 50, n)).clip(0).round(2)
    df["TotalCharges"] = total.astype(object)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["Churn"] = rng.choice(["Yes", "No"], n, p=[0.265, 0.735])

    TELCO_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TELCO_RAW, index=False)
    print(f"[ci] Wrote synthetic Telco CSV with {n} rows to {TELCO_RAW}")
    return TELCO_RAW


if __name__ == "__main__":
    make_synthetic_telco()
