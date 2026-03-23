import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data, audit_data
from src.features.preprocessing import create_features, prepare_data, build_preprocessor
from src.utils.config_loader import load_config

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "TV": [10.0, 50.0, 80.0, None, 30.0],
        "Radio": [5.0, 15.0, 25.0, 10.0, None],
        "Social Media": [1.0, 3.0, 5.0, 2.0, 4.0],
        "Influencer": ["Mega", "Micro", "Nano", "Macro", "Mega"],
        "Sales": [50.0, 180.0, 290.0, 120.0, None],
    })

def test_load_raw_data():
    df = load_raw_data()
    assert df.shape[0] > 0
    assert "Sales" in df.columns
    assert "TV" in df.columns

def test_audit_data(sample_df):
    audit = audit_data(sample_df)
    assert audit["shape"] == (5, 5)
    assert audit["duplicates"] == 0
    assert "TV" in audit["missing_count"]

def test_create_features(sample_df):
    df_clean = sample_df.dropna().copy()
    df_feat = create_features(df_clean)
    assert "total_budget" in df_feat.columns
    assert "tv_share" in df_feat.columns
    assert "tv_social_interaction" in df_feat.columns
    # Verification que total_budget = somme des 3 canaux
    expected = df_clean["TV"] + df_clean["Radio"] + df_clean["Social Media"]
    pd.testing.assert_series_equal(df_feat["total_budget"].reset_index(drop=True),
                                   expected.reset_index(drop=True))

def test_prepare_data_shape():
    config = load_config()
    df = load_raw_data()
    X, y, preprocessor, numeric_feats, cat_feats = prepare_data(df, config)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10
    assert len(y) > 4000

def test_preprocessor_no_nan():
    config = load_config()
    df = load_raw_data()
    X, y, preprocessor, numeric_feats, cat_feats = prepare_data(df, config)
    X_transformed = preprocessor.fit_transform(X)
    assert not np.isnan(X_transformed).any()

def test_target_range():
    config = load_config()
    df = load_raw_data()
    X, y, _, _, _ = prepare_data(df, config)
    assert y.min() > 0
    assert y.max() < 1000
