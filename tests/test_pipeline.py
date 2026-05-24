"""
Smoke tests for the data pipeline + feature engineering.

These run in CI on every push. They don't validate model accuracy
(too brittle for CI), but they catch the bugs that break a build:
  - Imports work
  - Config values are sane
  - Feature builder produces an all-numeric, no-NaN matrix
  - Master dataset has expected columns and row count
"""
import sys
from pathlib import Path

import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_config_imports_cleanly():
    from src import config
    assert config.RANDOM_STATE == 42
    assert 0 < config.TEST_SIZE < 1
    assert config.AVG_MONTHLY_ARPU_CAD > 0
    assert 0 < config.GROSS_MARGIN < 1


def test_crtc_builder_produces_expected_columns():
    from src.collection.build_crtc import build_crtc_table
    path = build_crtc_table()
    import pandas as pd
    df = pd.read_csv(path)
    expected = {"provider", "complaints_per_10k", "market_share_pct",
                "segment", "complaint_rate_normalized"}
    assert expected.issubset(df.columns)
    assert len(df) >= 5
    # The Big 3 must be present
    for big3 in ("Rogers", "Bell", "Telus"):
        assert big3 in df["provider"].values


def test_statcan_builder_covers_all_provinces():
    from src.collection.build_statcan import build_statcan_table
    path = build_statcan_table()
    import pandas as pd
    df = pd.read_csv(path)
    expected_provinces = {"ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"}
    assert set(df["province_code"]) == expected_provinces
    # Spending must be positive
    assert (df["annual_spend_cad"] > 0).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "master_churn.parquet").exists(),
    reason="Master dataset not built yet (M1 hasn't been run)",
)
def test_master_dataset_shape():
    import pandas as pd

    from src.config import MASTER_DATASET
    df = pd.read_parquet(MASTER_DATASET)
    # The IBM benchmark is 7,043 rows exactly
    assert len(df) == 7043
    # Churn rate must land in the published 25-28% band
    assert 0.24 < df["churn_flag"].mean() < 0.30
    # The Canadian context columns must exist
    for col in ("province_code", "provider", "complaints_per_10k",
                "monthly_spend_cad", "spend_vs_province_avg"):
        assert col in df.columns, f"Missing expected column: {col}"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "customers_segmented.parquet").exists(),
    reason="Segmented dataset not built yet (M2 hasn't been run)",
)
def test_feature_builder_produces_clean_matrix():
    from src.features.build_features import build_modelling_matrix, load_segmented
    df = load_segmented()
    X, y, feature_names = build_modelling_matrix(df)
    # All-numeric
    assert X.select_dtypes(exclude="number").shape[1] == 0
    # No NaN
    assert not X.isna().any().any()
    # Target is binary
    assert set(y.unique()).issubset({0, 1})
    # Feature names match column order
    assert list(X.columns) == feature_names
    # Reasonable number of features (after one-hot)
    assert 30 < len(feature_names) < 100


def test_config_paths_are_under_project_root():
    """Catch the bug where someone hard-codes /tmp or absolute paths."""
    from src import config
    project_root = config.PROJECT_ROOT
    assert str(config.DATA_DIR).startswith(str(project_root))
    assert str(config.MODELS_DIR).startswith(str(project_root))
    assert str(config.REPORTS_DIR).startswith(str(project_root))
