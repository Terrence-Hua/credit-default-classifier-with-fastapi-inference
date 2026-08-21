"""
FastAPI application for credit-default inference.

Endpoints:
    GET  /health       — liveness check
    GET  /model-info   — metadata about the loaded model
    POST /predict      — run inference on a CreditApplication
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from src.preprocessing import ALL_FEATURES
from src.schemas import CreditApplication, PredictionResponse, SHAPContribution

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("models/metrics.json")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and SHAP explainer on startup; release on shutdown."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run src/train.py first."
        )
    pipeline: Pipeline = joblib.load(MODEL_PATH)
    _state["pipeline"] = pipeline

    # SHAP TreeExplainer on the underlying GBM / RF model
    model_step = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model_step, feature_perturbation="interventional")
    _state["explainer"] = explainer

    # preprocessor — used to transform a single row before SHAP
    _state["preprocessor"] = pipeline.named_steps["preprocessor"]

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _state["metrics"] = json.load(f)
    else:
        _state["metrics"] = {}

    yield

    _state.clear()


app = FastAPI(
    title="Credit default classifier",
    description="Binary credit-default prediction with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    model_loaded = "pipeline" in _state
    return {
        "status": "ok" if model_loaded else "model_not_loaded",
        "model_loaded": str(model_loaded),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Return metadata about the loaded model."""
    metrics = _state.get("metrics", {})
    return {
        "model_type": metrics.get("best_model", "unknown"),
        "holdout_roc_auc": metrics.get("holdout_roc_auc"),
        "cv_results": metrics.get("cv_results", {}),
        "features": ALL_FEATURES,
        "train_size": metrics.get("train_size"),
        "test_size": metrics.get("test_size"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    """
    Run credit-default inference.

    Returns the binary prediction, default probability, and top-10 SHAP
    feature contributions.
    """
    pipeline: Pipeline = _state.get("pipeline")  # type: ignore[assignment]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    feature_dict = application.to_feature_dict()
    row = pd.DataFrame([feature_dict])[ALL_FEATURES]

    # Predict
    prob = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(prob >= 0.5)

    # SHAP explanations
    preprocessor = _state["preprocessor"]
    explainer = _state["explainer"]
    row_transformed = preprocessor.transform(row)

    shap_values = explainer.shap_values(row_transformed)
    # For binary classifiers shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    feature_names = list(preprocessor.get_feature_names_out())
    # Map back to short names (strip transformer prefix)
    short_names = [n.split("__", 1)[-1] for n in feature_names]

    contributions = [
        SHAPContribution(
            feature=short_names[i],
            value=float(row_transformed[0, i]),
            shap_value=float(sv[i]),
        )
        for i in range(len(short_names))
    ]
    # Sort by absolute SHAP value, return top 10
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    top_contributions = contributions[:10]

    return PredictionResponse(
        prediction=prediction,
        default_probability=round(prob, 6),
        shap_contributions=top_contributions,
    )
