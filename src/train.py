"""
Model training script.

Trains logistic regression, random forest, and gradient boosting on the
credit-default dataset. Compares models by 5-fold cross-validated ROC-AUC,
saves the best pipeline to models/best_model.joblib, and writes evaluation
plots to plots/.

Usage:
    python src/train.py [--data PATH] [--models-dir DIR] [--plots-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import ALL_FEATURES, build_preprocessor, load_data

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
METRICS_PATH = Path("models") / "metrics.json"


def build_candidates() -> dict[str, Pipeline]:
    """Return named pipelines for each candidate model."""
    pre = build_preprocessor()

    candidates: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        C=0.1,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    del pre  # unused; each pipeline builds its own
    return candidates


def cross_validate_all(
    candidates: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Return mean cross-validated ROC-AUC for each candidate."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results: dict[str, float] = {}
    for name, pipeline in candidates.items():
        t0 = time.perf_counter()
        scores = cross_val_score(
            pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        elapsed = time.perf_counter() - t0
        mean_auc = float(scores.mean())
        std_auc = float(scores.std())
        results[name] = mean_auc
        print(
            f"  {name:<25}  ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}"
            f"  ({elapsed:.1f}s)"
        )
    return results


def save_plots(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
) -> None:
    """Save confusion matrix and ROC curve for the best model on the test set."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion matrix (holdout)")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="Best model")
    ax.set_title("ROC curve (holdout)")
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=120)
    plt.close()

    holdout_auc = roc_auc_score(y_test, y_prob)
    print(f"\nHoldout ROC-AUC: {holdout_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["no default", "default"]))
    return holdout_auc


def main(
    data_path: str | None = None,
    models_dir: str = "models",
    plots_dir: str = "plots",
) -> None:
    MODELS = Path(models_dir)
    PLOTS = Path(plots_dir)
    MODELS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    print("Loading data...")
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Default rate (train): {y_train.mean():.1%}")

    print("\nCross-validating candidates (5-fold, ROC-AUC)...")
    candidates = build_candidates()
    cv_results = cross_validate_all(candidates, X_train, y_train, cv=5)

    best_name = max(cv_results, key=cv_results.__getitem__)
    best_cv_auc = cv_results[best_name]
    print(f"\nBest model: {best_name}  (CV AUC = {best_cv_auc:.4f})")

    print("\nFitting best model on full training set...")
    best_pipeline = candidates[best_name]
    best_pipeline.fit(X_train, y_train)

    holdout_auc = save_plots(best_pipeline, X_test, y_test, PLOTS)

    model_path = MODELS / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved to {model_path}")

    metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "holdout_roc_auc": holdout_auc,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": ALL_FEATURES,
    }
    METRICS_PATH.parent.mkdir(exist_ok=True)
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {MODELS / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to CSV")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()
    main(data_path=args.data, models_dir=args.models_dir, plots_dir=args.plots_dir)
