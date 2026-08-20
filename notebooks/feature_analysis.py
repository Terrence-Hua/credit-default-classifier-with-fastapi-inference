"""
Feature importance and target correlation analysis.

Runs mutual information, point-biserial correlation, and a quick
decision-tree-based importance ranking before any full model training.
Saves plots/feature_importance_pretrain.png.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

PLOTS_DIR = "plots"

df = pd.read_csv("data/credit_default.csv")
X = df.drop(columns="DEFAULT")
y = df["DEFAULT"]

# --- point-biserial correlation -------------------------------------------
pb_corrs = {f: pointbiserialr(y, X[f]).statistic for f in X.columns}
pb_series = pd.Series(pb_corrs).sort_values(key=abs, ascending=True)

# --- mutual information ---------------------------------------------------
mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=True)

# --- shallow decision tree importances ------------------------------------
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X, y)
dt_series = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=True)

# --- plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, series, title, color in zip(
    axes,
    [pb_series, mi_series, dt_series],
    ["Point-biserial |r| with DEFAULT", "Mutual information", "Decision tree importance"],
    ["steelblue", "seagreen", "tomato"],
):
    vals = series.abs() if "biserial" in title else series
    ax.barh(vals.index, vals.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("")

fig.suptitle("Feature importance — pre-training analysis", fontsize=13)
fig.tight_layout()
out = f"{PLOTS_DIR}/feature_importance_pretrain.png"
fig.savefig(out, dpi=120)
plt.close()
print(f"Saved {out}")

# --- payment delay breakdown ----------------------------------------------
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
dr_by_delay = {}
for col in pay_cols:
    dr_by_delay[col] = df.groupby(col)["DEFAULT"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
for col in pay_cols:
    s = dr_by_delay[col]
    ax.plot(s.index, s.values, marker="o", label=col)
ax.set_xlabel("Payment status (-1=paid duly, 0=revolving, 1-6=months late)")
ax.set_ylabel("Default rate")
ax.set_title("Default rate by payment status per month")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=8)
fig.tight_layout()
out2 = f"{PLOTS_DIR}/default_by_pay_status.png"
fig.savefig(out2, dpi=120)
plt.close()
print(f"Saved {out2}")

print("\nTop 5 features by mutual information:")
print(mi_series.sort_values(ascending=False).head())
