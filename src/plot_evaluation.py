"""
Re-generate evaluation plots from the saved model.

Produces:
    plots/confusion_matrix.png
    plots/roc_curve.png
    plots/calibration_curve.png

Usage:
    python src/plot_evaluation.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.preprocessing import load_data
from sklearn.model_selection import train_test_split


PLOTS_DIR = Path("plots")
MODEL_PATH = Path("models/best_model.joblib")


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded: {MODEL_PATH}")

    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion matrix (holdout, 20%)")
    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="Gradient boosting")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC curve (holdout, 20%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / "roc_curve.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")

    # calibration curve
    fig, ax = plt.subplots(figsize=(5, 4))
    CalibrationDisplay.from_predictions(
        y_test, y_prob, n_bins=10, ax=ax, name="Gradient boosting"
    )
    ax.set_title("Calibration curve (holdout)")
    fig.tight_layout()
    out = PLOTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
