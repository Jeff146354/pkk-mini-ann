"""
Shared fixtures and configuration for the Dengue Fever ANN Predictor test suite.

All tests are runnable with `pytest` + `hypothesis` from the project root.
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_train_features():
    """Minimal sample training features DataFrame matching the real schema."""
    return pd.DataFrame({
        'city': ['sj', 'sj', 'iq', 'iq'],
        'year': [1994, 1994, 2000, 2000],
        'weekofyear': [1, 2, 1, 2],
        'week_start_date': ['1994-01-03', '1994-01-10', '2000-01-03', '2000-01-10'],
        'ndvi_ne': [0.1, 0.2, 0.3, 0.4],
        'ndvi_nw': [0.1, 0.2, 0.3, 0.4],
        'ndvi_se': [0.1, 0.2, 0.3, 0.4],
        'ndvi_sw': [0.1, 0.2, 0.3, 0.4],
        'precipitation_amt_mm': [10.0, 20.0, 30.0, 40.0],
        'reanalysis_air_temp_k': [300.0, 301.0, 295.0, 296.0],
        'reanalysis_avg_temp_k': [299.0, 300.0, 294.0, 295.0],
        'reanalysis_dew_point_temp_k': [293.0, 294.0, 290.0, 291.0],
        'reanalysis_max_air_temp_k': [305.0, 306.0, 300.0, 301.0],
        'reanalysis_min_air_temp_k': [295.0, 296.0, 290.0, 291.0],
        'reanalysis_precip_amt_kg_per_m2': [15.0, 25.0, 35.0, 45.0],
        'reanalysis_relative_humidity_percent': [70.0, 72.0, 80.0, 82.0],
        'reanalysis_sat_precip_amt_mm': [12.0, 22.0, 32.0, 42.0],
        'reanalysis_specific_humidity_g_per_kg': [15.0, 16.0, 14.0, 15.0],
        'reanalysis_tdtr_k': [10.0, 10.5, 9.0, 9.5],
        'station_avg_temp_c': [27.0, 28.0, 22.0, 23.0],
        'station_diur_temp_rng_c': [8.0, 8.5, 7.0, 7.5],
        'station_max_temp_c': [32.0, 33.0, 28.0, 29.0],
        'station_min_temp_c': [22.0, 23.0, 18.0, 19.0],
        'station_precip_mm': [20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def sample_train_labels():
    """Minimal sample training labels DataFrame."""
    return pd.DataFrame({
        'city': ['sj', 'sj', 'iq', 'iq'],
        'year': [1994, 1994, 2000, 2000],
        'weekofyear': [1, 2, 1, 2],
        'total_cases': [5, 10, 3, 7],
    })


@pytest.fixture
def sample_merged_df(sample_train_features, sample_train_labels):
    """Merged training DataFrame (features + labels)."""
    return pd.merge(
        sample_train_features,
        sample_train_labels,
        on=['city', 'year', 'weekofyear'],
        how='left'
    )
