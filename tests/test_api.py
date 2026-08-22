"""API contract tests for the credit-default FastAPI service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# A valid request payload matching the CreditFeatures schema
VALID_PAYLOAD = {
    "LIMIT_BAL": 50000,
    "AGE": 35,
    "BILL_AMT1": 15000,
    "BILL_AMT2": 14000,
    "BILL_AMT3": 13000,
    "BILL_AMT4": 12000,
    "BILL_AMT5": 11000,
    "BILL_AMT6": 10000,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 1500,
    "PAY_AMT3": 1500,
    "PAY_AMT4": 1500,
    "PAY_AMT5": 1500,
    "PAY_AMT6": 1500,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "EDUCATION": 2,
    "MARRIAGE": 1,
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_schema():
    r = client.get("/health")
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------


def test_model_info_returns_200():
    r = client.get("/model-info")
    assert r.status_code == 200


def test_model_info_schema():
    r = client.get("/model-info")
    body = r.json()
    assert "model_name" in body
    assert "holdout_roc_auc" in body
    assert "features" in body
    assert isinstance(body["features"], list)
    assert body["holdout_roc_auc"] > 0.80


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_returns_200():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200


def test_predict_schema():
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert "default_probability" in body
    assert "prediction" in body
    assert "model" in body
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_predict_probability_is_float():
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert isinstance(body["default_probability"], float)


def test_predict_high_risk():
    """A delinquent applicant should get a higher default probability."""
    high_risk = dict(VALID_PAYLOAD)
    high_risk.update(
        {
            "PAY_0": 3,
            "PAY_2": 3,
            "PAY_3": 3,
            "PAY_4": 3,
            "PAY_5": 3,
            "PAY_6": 3,
            "PAY_AMT1": 0,
            "PAY_AMT2": 0,
            "PAY_AMT3": 0,
        }
    )
    low_risk = dict(VALID_PAYLOAD)
    low_risk.update(
        {
            "PAY_0": -2,
            "PAY_2": -2,
            "PAY_3": -2,
            "PAY_4": -2,
            "PAY_5": -2,
            "PAY_6": -2,
            "LIMIT_BAL": 200000,
            "PAY_AMT1": 20000,
            "PAY_AMT2": 20000,
            "PAY_AMT3": 20000,
        }
    )
    r_high = client.post("/predict", json=high_risk)
    r_low = client.post("/predict", json=low_risk)
    assert r_high.json()["default_probability"] > r_low.json()["default_probability"]


def test_predict_invalid_age():
    """AGE below 18 should fail schema validation (422)."""
    bad = dict(VALID_PAYLOAD)
    bad["AGE"] = 10
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field():
    """Missing required field should return 422."""
    bad = dict(VALID_PAYLOAD)
    del bad["LIMIT_BAL"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_shap_flag():
    """explain=true should return shap_values dict, not null."""
    r = client.post("/predict?explain=true", json=VALID_PAYLOAD)
    body = r.json()
    assert body["shap_values"] is not None
    assert isinstance(body["shap_values"], dict)
    assert len(body["shap_values"]) > 0
