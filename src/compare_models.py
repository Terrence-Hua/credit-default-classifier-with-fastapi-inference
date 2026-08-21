"""
Model comparison utilities.

Prints a ranked table of cross-validated ROC-AUC scores and returns the
name of the best-performing model.
"""
from __future__ import annotations

import json
from pathlib import Path


def print_comparison_table(cv_results: dict[str, float]) -> None:
    """Print a sorted model comparison table to stdout."""
    sorted_results = sorted(cv_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Model':<28} {'CV ROC-AUC':>12}")
    print("-" * 42)
    for name, auc in sorted_results:
        print(f"  {name:<26} {auc:>12.4f}")
    print("-" * 42)
    best_name, best_auc = sorted_results[0]
    print(f"  Best: {best_name}  ({best_auc:.4f})\n")


def select_best(cv_results: dict[str, float]) -> str:
    """Return the model name with the highest mean CV ROC-AUC."""
    return max(cv_results, key=cv_results.__getitem__)


def load_metrics(metrics_path: str | Path = "models/metrics.json") -> dict:
    """Load saved metrics from a JSON file."""
    with open(metrics_path) as f:
        return json.load(f)


def summary_from_metrics(metrics_path: str | Path = "models/metrics.json") -> str:
    """Return a one-line summary of the best model and holdout AUC."""
    m = load_metrics(metrics_path)
    return (
        f"Best model: {m['best_model']}  "
        f"| CV AUC: {m['cv_results'][m['best_model']]:.4f}  "
        f"| Holdout AUC: {m['holdout_roc_auc']:.4f}"
    )
