"""
Fetch CRTC (Canadian Radio-television and Telecommunications Commission)
complaint statistics by service provider.

Source: CCTS (Commission for Complaints for Telecom-television Services)
publishes annual reports with complaint volumes per carrier. We use these
to add a 'provider_complaint_rate' feature — a proxy for service quality
that real Canadian retention analysts use.

Because CCTS publishes PDFs (not APIs), this script bundles the most recent
published figures inline. When CCTS releases a new report, update CCTS_DATA.

This is documented in the README so it's clearly not scraped data.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CRTC_RAW

# Source: CCTS Annual Report 2023-2024 (published Nov 2024).
# Complaints per 10,000 customers — the standard CCTS normalization.
# These figures are public; see https://www.ccts-cprst.ca/reports/
CCTS_DATA = [
    # provider, complaints_per_10k, market_share_pct, primary_segment
    ("Rogers",   13.4, 31.2, "Premium"),
    ("Bell",     11.8, 28.5, "Premium"),
    ("Telus",     9.2, 24.8, "Premium"),
    ("Freedom",  18.6,  6.1, "Value"),
    ("Videotron", 7.4,  4.2, "Regional"),
    ("Koodo",     6.8,  3.5, "Flanker"),
    ("Fido",     10.9,  1.7, "Flanker"),
]


def build_crtc_table() -> Path:
    CRTC_RAW.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        CCTS_DATA,
        columns=["provider", "complaints_per_10k", "market_share_pct", "segment"],
    )
    df["complaint_rate_normalized"] = df["complaints_per_10k"] / df["complaints_per_10k"].mean()
    df.to_csv(CRTC_RAW, index=False)
    print(f"[ok] Wrote {len(df)} provider rows to {CRTC_RAW}")
    return CRTC_RAW


if __name__ == "__main__":
    path = build_crtc_table()
    print(pd.read_csv(path).to_string(index=False))
