"""
FastAPI inference service for the credit-default classifier.

Endpoints:
    GET  /health        — liveness check
    GET  /model-info    — model metadata and holdout metrics
    POST /predict       — single-record inference with optional SHAP values
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sklearn.pipeline import Pipeline

from src.schemas import (
    CreditFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# App init and model load
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit default classifier",
    description="Predicts probability of credit default from payment history.",
    version="1.0.0",
)

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("models/metrics.json")

_pipeline: Optional[Pipeline] = None
_metrics: Optional[dict] = None


def _load_model() -> tuple[Pipeline, dict]:
    """Load the model pipeline and metrics from disk. Cached after first call."""
    global _pipeline, _metrics
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    if _metrics is None:
        if not METRICS_PATH.exists():
            raise RuntimeError(f"Metrics file not found: {METRICS_PATH}")
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _pipeline, _metrics


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load the model at startup to avoid cold-start latency on first request."""
    try:
        _load_model()
    except RuntimeError:
        pass  # Model may not exist in test environments; /health will report it


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check. Returns 200 with model_loaded=true when ready."""
    loaded = MODEL_PATH.exists() and METRICS_PATH.exists()
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return model name, holdout metrics, and feature list."""
    try:
        _, metrics = _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelInfoResponse(
        model_name=metrics["best_model"],
        holdout_roc_auc=metrics["holdout_roc_auc"],
        cv_roc_auc=metrics["cv_results"][metrics["best_model"]],
        features=metrics["features"],
        train_size=metrics["train_size"],
        test_size=metrics["test_size"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    features: CreditFeatures,
    explain: bool = Query(False, description="Include SHAP feature contributions"),
) -> PredictionResponse:
    """
    Predict credit-default probability for a single applicant.

    Set `explain=true` to include top-10 SHAP feature contributions in the
    response. Explanation adds ~50-100ms latency.
    """
    t0 = time.perf_counter()
    try:
        pipeline, metrics = _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([features.model_dump()])
    prob = float(pipeline.predict_proba(row)[0, 1])
    pred = int(prob >= 0.5)

    shap_vals: Optional[dict[str, float]] = None
    if explain:
        from src.shap_analysis import explain_single
        shap_vals = explain_single(pipeline, row)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    # Log latency for observability; not returned in response
    _ = elapsed_ms

    return PredictionResponse(
        default_probability=round(prob, 6),
        prediction=pred,
        model=metrics["best_model"],
        shap_values=shap_vals,
    )
