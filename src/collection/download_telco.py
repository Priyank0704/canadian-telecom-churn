"""
Download the IBM Telco Customer Churn dataset.

This is the canonical benchmark dataset for churn modelling:
7,043 customers, 21 features, ~26.5% churn rate.

We reframe it as Canadian telecom (Rogers/Bell/Telus context) in the
feature engineering layer — the underlying customer behaviour patterns
(tenure, contract type, payment method, service bundle) are the same
regardless of geography.
"""
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import TELCO_RAW

# Primary + fallback mirrors. Both serve the identical IBM-published file.
MIRROR_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/KimathiNewton/Telco-Customer-Churn/master/Datasets/telco_churn.csv",
]


def download_telco() -> Path:
    TELCO_RAW.parent.mkdir(parents=True, exist_ok=True)

    if TELCO_RAW.exists():
        print(f"[skip] Telco dataset already at {TELCO_RAW}")
        return TELCO_RAW

    last_err = None
    for url in MIRROR_URLS:
        try:
            print(f"[download] {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            # Sanity: the file must contain the expected header
            if b"customerID" not in response.content[:200]:
                raise ValueError("Unexpected file contents — wrong mirror?")
            TELCO_RAW.write_bytes(response.content)
            size_kb = TELCO_RAW.stat().st_size / 1024
            print(f"[ok] Saved {size_kb:.1f} KB to {TELCO_RAW}")
            return TELCO_RAW
        except Exception as e:
            print(f"[fail] {e}")
            last_err = e
            continue

    raise RuntimeError(f"All mirrors failed. Last error: {last_err}")


if __name__ == "__main__":
    path = download_telco()
    # Quick sanity check
    import pandas as pd
    df = pd.read_csv(path)
    print(f"\nShape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(f"Columns: {list(df.columns)}")
