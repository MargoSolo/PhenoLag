"""Diagnostic-debt heatmap: % of each disorder class falling in each lag bucket,
for the TOP-10 classes by sample size (current data). Replaces the stale heatmap
whose classes were not the 10 largest.

Run: PYTHONPATH=. python -m pipeline.fig_debt_heatmap
Saves figures/fig_debt_heatmap.png
"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.config import DB_PATH, FIGURES

BUCKETS = [0, 5, 10, 20, 30, 50, np.inf]
BLABELS = ["<5", "5-10", "10-20", "20-30", "30-50", ">50"]


def short(name: str) -> str:
    # explicit overrides (kept verbatim, not passed through .capitalize())
    overrides = {
        "Rare systemic or rheumatologic disease": "Systemic/autoinflammatory",
        "Rare inborn errors of metabolism": "Metabolic (IEM)",
        "Rare developmental defect during embryogenesis": "Developmental",
    }
    if name in overrides:
        return overrides[name]
    return name.replace("Rare ", "").replace(" disease", "").replace(" disorder", "").capitalize()


def main():
    con = sqlite3.connect(str(DB_PATH))
    d = pd.read_sql("""SELECT disorder_class, lag_years FROM phenolag
        WHERE lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946
          AND disorder_class IS NOT NULL""", con)
    con.close()

    top = d["disorder_class"].value_counts().head(10)
    d = d[d["disorder_class"].isin(top.index)].copy()
    d["bucket"] = pd.cut(d["lag_years"], bins=BUCKETS, labels=BLABELS, right=False)

    # % of class in each bucket (rows sum to 100)
    mat = (d.groupby("disorder_class", observed=True)["bucket"]
             .value_counts(normalize=True).mul(100).unstack().reindex(columns=BLABELS).fillna(0))
    mat = mat.reindex(top.index)  # order by N desc
    rowlabels = [f"{short(c)} (n={top[c]})" for c in mat.index]

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    im = ax.imshow(mat.values, aspect="auto", cmap="Reds", vmin=0, vmax=max(40, mat.values.max()))
    ax.set_xticks(range(len(BLABELS))); ax.set_xticklabels(BLABELS, fontsize=12)
    ax.set_yticks(range(len(rowlabels))); ax.set_yticklabels(rowlabels, fontsize=11)
    ax.set_xlabel("Lag bucket (years)", fontsize=13)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if v > 25 else "#333")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("% of class", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_debt_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # print the headline cells for the caption
    print(mat.round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
