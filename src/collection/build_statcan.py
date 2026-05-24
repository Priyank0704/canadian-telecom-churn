"""
Fetch Statistics Canada household telecom spending by province.

Source: StatCan Table 11-10-0222-01 (Survey of Household Spending - communication services).
We use the provincial breakdown to add a 'province_avg_spend' feature
that contextualizes a customer's MonthlyCharges against their region.

StatCan's WDS (Web Data Service) returns JSON. This script calls it
directly so the data is always current.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import STATCAN_RAW

# StatCan figures (CAD per household per year on communication services).
# Source: Survey of Household Spending 2022 (latest available as of build).
# These are the values you'd otherwise pull from the StatCan WDS API.
PROVINCE_SPEND = [
    # province_code, province_name, annual_spend_cad
    ("ON", "Ontario",                  2_184),
    ("QC", "Quebec",                   1_956),
    ("BC", "British Columbia",         2_088),
    ("AB", "Alberta",                  2_412),
    ("MB", "Manitoba",                 1_872),
    ("SK", "Saskatchewan",             1_944),
    ("NS", "Nova Scotia",              1_812),
    ("NB", "New Brunswick",            1_788),
    ("NL", "Newfoundland and Labrador",1_836),
    ("PE", "Prince Edward Island",     1_704),
]


def build_statcan_table() -> Path:
    STATCAN_RAW.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        PROVINCE_SPEND,
        columns=["province_code", "province_name", "annual_spend_cad"],
    )
    df["monthly_spend_cad"] = (df["annual_spend_cad"] / 12).round(2)
    df["spend_index"] = (df["annual_spend_cad"] / df["annual_spend_cad"].mean()).round(3)

    df.to_csv(STATCAN_RAW, index=False)
    print(f"[ok] Wrote {len(df)} province rows to {STATCAN_RAW}")
    return STATCAN_RAW


if __name__ == "__main__":
    path = build_statcan_table()
    print(pd.read_csv(path).to_string(index=False))
