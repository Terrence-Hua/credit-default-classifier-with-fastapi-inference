"""
SHAP summary analysis on the holdout set.

Generates plots/shap_summary.png — a beeswarm plot showing global feature
impact across the test set.

Usage:
    python src/shap_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.preprocessing import ALL_FEATURES, load_data

MODEL_PATH = Path("models/best_model.joblib")
PLOTS_DIR = Path("plots")


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"No model at {MODEL_PATH}. Run src/train.py first.")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X, y = load_data()
    _, X_test, _, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    # Use a sample for speed
    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    X_transformed = preprocessor.transform(X_sample)

    feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classifier that returns list [neg, pos]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Beeswarm / summary plot
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        sv,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
        plot_type="dot",
    )
    plt.title("SHAP feature impact (holdout sample, n=500)")
    plt.tight_layout()
    out = PLOTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Mean absolute SHAP per feature
    mean_abs = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP|:")
    print(importance.head(10).to_string())


if __name__ == "__main__":
    main()
