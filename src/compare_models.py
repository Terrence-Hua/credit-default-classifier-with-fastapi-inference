"""
Print model comparison table from models/metrics.json.

Usage:
    python src/compare_models.py
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        print("No metrics.json found — run src/train.py first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    cv_results: dict[str, float] = metrics["cv_results"]
    best_name: str = metrics["best_model"]
    holdout_auc: float = metrics["holdout_roc_auc"]

    print("\nModel comparison — 5-fold cross-validated ROC-AUC")
    print("-" * 48)
    print(f"{'Model':<28} {'CV ROC-AUC':>12}  {'Selected':>8}")
    print("-" * 48)
    for name, auc in sorted(cv_results.items(), key=lambda kv: kv[1], reverse=True):
        marker = "<-- best" if name == best_name else ""
        print(f"  {name:<26} {auc:>12.4f}  {marker}")
    print("-" * 48)
    print(f"  Holdout ROC-AUC ({best_name}): {holdout_auc:.4f}")
    print(f"  Train / test split: {metrics['train_size']:,} / {metrics['test_size']:,}")
    print()


if __name__ == "__main__":
    main()
