"""
Evaluation plot helpers.

Saves a confusion matrix and ROC curve PNG for a fitted sklearn pipeline
evaluated on a held-out test set.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import pandas as pd


def save_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: str | Path = "plots/confusion_matrix.png",
    dpi: int = 120,
) -> Path:
    """
    Plot and save the confusion matrix for the fitted pipeline on (X_test, y_test).

    Returns the resolved output path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    y_pred = pipeline.predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, colorbar=False, cmap="Blues",
        display_labels=["no default", "default"],
    )
    ax.set_title("Confusion matrix (holdout)")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def save_roc_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: str | Path = "plots/roc_curve.png",
    model_name: str = "Best model",
    dpi: int = 120,
) -> tuple[Path, float]:
    """
    Plot and save the ROC curve for the fitted pipeline on (X_test, y_test).

    Returns (resolved path, holdout ROC-AUC).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)
    ax.set_title(f"ROC curve (holdout) — AUC = {auc:.4f}")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out, auc
