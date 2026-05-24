"""Central configuration: paths, constants, business assumptions."""
import random
from pathlib import Path

import numpy as np

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TELCO_RAW = RAW_DIR / "telco" / "telco_churn.csv"
CRTC_RAW = RAW_DIR / "crtc" / "crtc_complaints.csv"
STATCAN_RAW = RAW_DIR / "statcan" / "household_telecom_spending.csv"

MASTER_DATASET = PROCESSED_DIR / "master_churn.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_DIR = REPORTS_DIR / "shap"

# ---- Business assumptions (used in ROI calculator) ----
# Source: CRTC 2024 Communications Monitoring Report — avg monthly wireless ARPU
AVG_MONTHLY_ARPU_CAD = 64.0
# Industry rule of thumb: retention offer typically 1-3 months of margin
DEFAULT_RETENTION_OFFER_CAD = 50.0
# Gross margin on wireless service in Canada (Big 3 carriers, approx)
GROSS_MARGIN = 0.45
# Customer acquisition cost — Canadian telecom benchmark
CAC_CAD = 350.0
# Average customer lifetime in months (post-retention)
AVG_LIFETIME_MONTHS = 36

# ---- Modelling ----
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_COL = "Churn"

# ---- Reproducibility ----
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
