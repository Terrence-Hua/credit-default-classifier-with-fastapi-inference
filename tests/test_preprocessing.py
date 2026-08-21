"""Tests for the preprocessing pipeline and compare_models utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PAY_STATUS_FEATURES,
    build_preprocessor,
    feature_names_out,
    load_data,
)
from src.compare_models import select_best, print_comparison_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample_df(n: int = 50) -> pd.DataFrame:
    """Return a minimal DataFrame with all required feature columns."""
    rng = np.random.default_rng(0)
    data: dict = {}
    for col in NUMERIC_FEATURES:
        data[col] = rng.uniform(0, 100_000, n)
    for col in PAY_STATUS_FEATURES:
        data[col] = rng.integers(-2, 3, n)
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.integers(0, 4, n)
    data["DEFAULT"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return _make_sample_df()


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------


def test_preprocessor_output_shape(sample_df):
    """Transform should produce a 2-D array with expected column count."""
    X = sample_df[ALL_FEATURES]
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.shape == (len(sample_df), len(ALL_FEATURES))


def test_preprocessor_no_nan_output(sample_df):
    """No NaNs should survive the pipeline."""
    # Introduce missing values
    sample_df.loc[0:5, NUMERIC_FEATURES[0]] = float("nan")
    sample_df.loc[6:10, CATEGORICAL_FEATURES[0]] = float("nan")
    X = sample_df[ALL_FEATURES]
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()


def test_preprocessor_unknown_category(sample_df):
    """Unknown categories at inference time should not raise."""
    X_train = sample_df[ALL_FEATURES]
    pre = build_preprocessor()
    pre.fit(X_train)

    # Introduce a category never seen in training
    X_test = sample_df[ALL_FEATURES].copy()
    X_test.loc[0, CATEGORICAL_FEATURES[0]] = 999
    out = pre.transform(X_test)
    assert out.shape[0] == len(X_test)


def test_feature_names_out(sample_df):
    """feature_names_out returns a list of strings with length == n_features."""
    X = sample_df[ALL_FEATURES]
    pre = build_preprocessor()
    pre.fit(X)
    names = feature_names_out(pre)
    assert isinstance(names, list)
    assert len(names) == len(ALL_FEATURES)
    assert all(isinstance(n, str) for n in names)


def test_all_features_list():
    """ALL_FEATURES should be the union of the three feature groups."""
    assert set(ALL_FEATURES) == set(NUMERIC_FEATURES + PAY_STATUS_FEATURES + CATEGORICAL_FEATURES)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


def test_load_data_returns_correct_shapes():
    """load_data should return (X, y) where X has exactly ALL_FEATURES columns."""
    X, y = load_data()
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


def test_select_best_picks_highest():
    cv = {"lr": 0.75, "rf": 0.88, "gb": 0.85}
    assert select_best(cv) == "rf"


def test_print_comparison_table_runs(capsys):
    cv = {"lr": 0.75, "rf": 0.88}
    print_comparison_table(cv)
    captured = capsys.readouterr()
    assert "rf" in captured.out
    assert "0.8800" in captured.out
