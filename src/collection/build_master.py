"""
Master dataset builder.

Joins the three sources into a single analytical table:
  1. IBM Telco Churn (customer-level)            — base behavioural data
  2. CRTC complaint rates (provider-level)        — service-quality context
  3. StatCan household spend (province-level)     — affordability context

The IBM dataset has no provider or province columns, so we synthesise them
using realistic Canadian market-share distributions. This is documented
transparently in the project README — every interview question about
"why did you synthesize that?" has a clean answer: real Canadian customer-
level churn data is not publicly available due to PIPEDA, so we apply
realistic Canadian context to the public IBM benchmark.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    CRTC_RAW,
    MASTER_DATASET,
    RANDOM_STATE,
    STATCAN_RAW,
    TELCO_RAW,
)


def assign_provider(monthly_charges: float, contract: str, rng: np.random.Generator) -> str:
    """
    Heuristic: high spenders on premium contracts skew Big-3.
    Low spenders skew toward flanker/value brands.
    """
    if monthly_charges > 80 and contract != "Month-to-month":
        return rng.choice(["Rogers", "Bell", "Telus"], p=[0.40, 0.32, 0.28])
    if monthly_charges < 40:
        return rng.choice(
            ["Koodo", "Fido", "Freedom", "Videotron"], p=[0.30, 0.25, 0.25, 0.20]
        )
    return rng.choice(
        ["Rogers", "Bell", "Telus", "Freedom", "Videotron", "Koodo", "Fido"],
        p=[0.22, 0.20, 0.18, 0.12, 0.10, 0.10, 0.08],
    )


def assign_province(rng: np.random.Generator) -> str:
    """Weights from StatCan 2024 population estimates."""
    provinces = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]
    weights   = [0.387, 0.226, 0.140, 0.117, 0.036, 0.030, 0.026, 0.021, 0.013, 0.004]
    return rng.choice(provinces, p=weights)


def build_master() -> Path:
    print(f"[load] {TELCO_RAW}")
    telco = pd.read_csv(TELCO_RAW)

    # TotalCharges has blanks for new customers (tenure=0). Coerce to numeric.
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
    telco["TotalCharges"] = telco["TotalCharges"].fillna(0)

    # Binary target
    telco["churn_flag"] = (telco["Churn"] == "Yes").astype(int)

    # ---- Layer 1: assign provider + province ----
    rng = np.random.default_rng(RANDOM_STATE)
    telco["provider"] = [
        assign_provider(mc, c, rng)
        for mc, c in zip(telco["MonthlyCharges"], telco["Contract"])
    ]
    telco["province_code"] = [assign_province(rng) for _ in range(len(telco))]

    # ---- Layer 2: join CRTC provider context ----
    print(f"[load] {CRTC_RAW}")
    crtc = pd.read_csv(CRTC_RAW)
    telco = telco.merge(crtc, on="provider", how="left")

    # ---- Layer 3: join StatCan province context ----
    print(f"[load] {STATCAN_RAW}")
    statcan = pd.read_csv(STATCAN_RAW)
    telco = telco.merge(statcan, on="province_code", how="left")

    # ---- Engineered Canadian-context features ----
    telco["spend_vs_province_avg"] = (
        telco["MonthlyCharges"] / telco["monthly_spend_cad"]
    ).round(3)
    telco["is_big3"] = telco["provider"].isin(["Rogers", "Bell", "Telus"]).astype(int)
    telco["high_complaint_provider"] = (
        telco["complaints_per_10k"] > telco["complaints_per_10k"].median()
    ).astype(int)

    # ---- Persist ----
    MASTER_DATASET.parent.mkdir(parents=True, exist_ok=True)
    telco.to_parquet(MASTER_DATASET, index=False)
    print(f"\n[ok] Master dataset → {MASTER_DATASET}")
    print(f"     Rows: {len(telco):,}")
    print(f"     Columns: {len(telco.columns)}")
    print(f"     Churn rate: {telco['churn_flag'].mean():.1%}")
    print("\n     Provider distribution:")
    print(telco["provider"].value_counts().to_string())
    print("\n     Province distribution:")
    print(telco["province_code"].value_counts().to_string())
    return MASTER_DATASET


if __name__ == "__main__":
    build_master()
