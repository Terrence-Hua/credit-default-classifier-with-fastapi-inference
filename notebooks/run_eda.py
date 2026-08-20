"""Run EDA and save plots to plots/ directory."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr

sys.path.insert(0, str(Path(__file__).parent.parent))

sns.set_theme(style="whitegrid", palette="muted")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

df = pd.read_csv("data/credit_default.csv")
print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Default rate: {df['DEFAULT'].mean():.1%}")
print(f"Missing: {df.isnull().sum().sum()}")

# class distribution
fig, ax = plt.subplots(figsize=(5, 3))
counts = df["DEFAULT"].value_counts().sort_index()
ax.bar(["No default", "Default"], counts.values, color=["steelblue", "tomato"])
ax.set_ylabel("Count")
ax.set_title("Class distribution")
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v:,} ({v/len(df):.1%})", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(PLOTS / "class_distribution.png", dpi=120)
plt.close()

# numeric distributions
numeric_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, col in zip(axes, numeric_cols):
    df[col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(col)
fig.tight_layout()
fig.savefig(PLOTS / "numeric_distributions.png", dpi=120)
plt.close()

# payment status
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for ax, col in zip(axes.flat, pay_cols):
    vc = df[col].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="steelblue")
    ax.set_title(col)
    ax.set_xlabel("Status")
fig.suptitle("Payment status distribution")
fig.tight_layout()
fig.savefig(PLOTS / "payment_status_dist.png", dpi=120)
plt.close()

# correlation heatmap
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
            linewidths=0.4, ax=ax)
ax.set_title("Feature correlation matrix")
fig.tight_layout()
fig.savefig(PLOTS / "correlation_heatmap.png", dpi=120)
plt.close()

# default by category
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
plt.close()

# credit limit vs default rate
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
plt.close()
df.drop(columns="LIMIT_BAL_bin", inplace=True)

# point-biserial correlations
features = [c for c in df.columns if c != "DEFAULT"]
corrs = []
for f in features:
    r, p = pointbiserialr(df["DEFAULT"], df[f])
    corrs.append({"feature": f, "r": r, "p": p})
corr_df = pd.DataFrame(corrs).sort_values("r", key=abs, ascending=False)
print("\nTop correlations with DEFAULT:")
print(corr_df.head(10).to_string(index=False))

print("\nPlots saved to plots/")
