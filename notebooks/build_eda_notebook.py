"""Build eda.ipynb programmatically so it can be committed without running Jupyter."""
import json
from pathlib import Path


def cell(source: str, cell_type: str = "code") -> dict:
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


cells = [
    cell(
        "# Credit Default — Exploratory Data Analysis\n\n"
        "Dataset: synthetic (mirrors UCI Taiwan Credit Card Default structure, n=10 000).",
        "markdown",
    ),
    cell(
        """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
PLOTS = Path("../plots")
PLOTS.mkdir(exist_ok=True)

df = pd.read_csv("../data/credit_default.csv")
print(df.shape)
df.head()"""
    ),
    cell(
        """\
# --- basic info ---------------------------------------------------------------
print(df.dtypes)
print("\\nMissing values:")
print(df.isnull().sum())"""
    ),
    cell("## Class distribution", "markdown"),
    cell(
        """\
fig, ax = plt.subplots(figsize=(5, 3))
counts = df["DEFAULT"].value_counts().sort_index()
ax.bar(["No default", "Default"], counts.values, color=["steelblue", "tomato"])
ax.set_ylabel("Count")
ax.set_title("Class distribution")
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v:,} ({v/len(df):.1%})", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(PLOTS / "class_distribution.png", dpi=120)
plt.show()"""
    ),
    cell("## Numeric feature distributions", "markdown"),
    cell(
        """\
numeric_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, col in zip(axes, numeric_cols):
    df[col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("")
fig.suptitle("Numeric feature distributions", y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "numeric_distributions.png", dpi=120)
plt.show()"""
    ),
    cell("## Payment delay distribution (PAY_0 is most recent month)", "markdown"),
    cell(
        """\
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for ax, col in zip(axes.flat, pay_cols):
    vc = df[col].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="steelblue")
    ax.set_title(col)
    ax.set_xlabel("Status")
fig.suptitle("Payment status distribution (-1=paid, 0=revolving, 1-6=months late)")
fig.tight_layout()
fig.savefig(PLOTS / "payment_status_dist.png", dpi=120)
plt.show()"""
    ),
    cell("## Correlation heatmap", "markdown"),
    cell(
        """\
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax)
ax.set_title("Feature correlation matrix")
fig.tight_layout()
fig.savefig(PLOTS / "correlation_heatmap.png", dpi=120)
plt.show()"""
    ),
    cell("## Default rate by categorical features", "markdown"),
    cell(
        """\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, col, labels in zip(
    axes,
    ["EDUCATION", "MARRIAGE"],
    [
        {1: "Grad school", 2: "University", 3: "High school", 4: "Other"},
        {1: "Married", 2: "Single", 3: "Other"},
    ],
):
    dr = df.groupby(col)["DEFAULT"].mean().reset_index()
    dr[col] = dr[col].map(labels)
    ax.bar(dr[col], dr["DEFAULT"], color="tomato")
    ax.set_ylabel("Default rate")
    ax.set_title(f"Default rate by {col}")
    ax.set_ylim(0, 0.4)
    for _, row in dr.iterrows():
        ax.text(row[col], row["DEFAULT"] + 0.005, f"{row['DEFAULT']:.1%}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(PLOTS / "default_by_category.png", dpi=120)
plt.show()"""
    ),
    cell("## Point-biserial correlation with DEFAULT", "markdown"),
    cell(
        """\
from scipy.stats import pointbiserialr

features = [c for c in df.columns if c != "DEFAULT"]
corrs = []
for f in features:
    r, p = pointbiserialr(df["DEFAULT"], df[f])
    corrs.append({"feature": f, "r": r, "p": p})

corr_df = pd.DataFrame(corrs).sort_values("r", ascending=False)
print(corr_df.to_string(index=False))"""
    ),
    cell("## Credit limit vs default rate", "markdown"),
    cell(
        """\
df["LIMIT_BAL_bin"] = pd.qcut(df["LIMIT_BAL"], q=10, labels=False)
dr = df.groupby("LIMIT_BAL_bin")["DEFAULT"].agg(["mean", "count"]).reset_index()
midpoints = df.groupby("LIMIT_BAL_bin")["LIMIT_BAL"].median().values

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(midpoints, dr["mean"], marker="o", color="tomato")
ax.set_xlabel("LIMIT_BAL (decile midpoint)")
ax.set_ylabel("Default rate")
ax.set_title("Default rate by credit limit decile")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
fig.tight_layout()
fig.savefig(PLOTS / "default_by_limit.png", dpi=120)
plt.show()

df.drop(columns="LIMIT_BAL_bin", inplace=True)"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

out = Path("notebooks/eda.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out}")
