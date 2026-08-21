"""
Preprocessing pipeline for the credit-default dataset.

Exports:
    NUMERIC_FEATURES   — list of continuous feature names
    CATEGORICAL_FEATURES — list of ordinal categorical feature names
    build_preprocessor() — returns a fitted-ready ColumnTransformer
    load_data()        — read raw CSV and return (X, y)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "credit_default.csv"

TARGET = "DEFAULT"

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

# Treated as ordinal: payment status columns follow a natural order
# (-2=no use, -1=pay duly, 0=revolving credit, 1-9=months late)
PAY_STATUS_FEATURES = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

# Nominal categoricals that get ordinal encoding with unknown handling
CATEGORICAL_FEATURES = [
    "EDUCATION",
    "MARRIAGE",
]

ALL_FEATURES = NUMERIC_FEATURES + PAY_STATUS_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
    - Imputes missing numeric values with the median, then standard-scales.
    - Imputes missing pay-status values with the most frequent, encodes as ordinal int.
    - Imputes missing categoricals with the most frequent, encodes as ordinal int.
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pay_status_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[sorted([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
                    * len(PAY_STATUS_FEATURES),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("pay", pay_status_pipe, PAY_STATUS_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_data(path: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Read the credit-default CSV.

    Returns (X, y) where X contains all feature columns and y is the binary
    DEFAULT target.
    """
    csv_path = Path(path) if path is not None else DATA_PATH
    df = pd.read_csv(csv_path)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return the feature names after transform, in column order."""
    return list(preprocessor.get_feature_names_out())
