"""Lag pre- vs post-NGS — clean horizontal box + jittered strip (raincloud style),
with summary statistics labelled. Handles the post-NGS 0-spike gracefully.

Run: PYTHONPATH=. python -m pipeline.fig_lag_violin
Saves figures/fig_lag_violin.png and export_figures/lag_violin_pre_post_NGS.png
"""
from __future__ import annotations
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.config import DB_PATH

NGS = 2010
PRE, POST = "#B85042", "#065A82"


def main():
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute("SELECT lag_years, first_clinical_year FROM phenolag "
                       "WHERE lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946").fetchall()
    con.close()
    lag = np.array([r[0] for r in rows], float)
    clin = np.array([r[1] for r in rows], float)
    groups = [lag[clin >= NGS], lag[clin < NGS]]          # bottom -> top
    cols = [POST, PRE]
    names = ["NGS era\n(described ≥2010)", "Pre-NGS\n(described <2010)"]
    ypos = [1, 2]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    rng = np.random.default_rng(7)
    for g, c, y in zip(groups, cols, ypos):
        # jittered points (raincloud) slightly below the box
        jx = g + 0  # x is the lag value
        jy = y - 0.22 + rng.uniform(-0.11, 0.11, len(g))
        ax.scatter(jx, jy, s=8, color=c, alpha=0.28, edgecolors="none", zorder=2)
        # box on top
        bp = ax.boxplot(g, positions=[y + 0.12], vert=False, widths=0.28,
                        patch_artist=True, showfliers=False, zorder=4,
                        medianprops=dict(color="black", lw=2.2),
                        whiskerprops=dict(color=c, lw=1.5), capprops=dict(color=c, lw=1.5))
        for box in bp["boxes"]:
            box.set(facecolor=c, alpha=0.55, edgecolor=c)
        med, q1, q3 = np.median(g), np.percentile(g, 25), np.percentile(g, 75)
        ax.text(72, y, f"median {med:.0f} y\nIQR [{q1:.0f}-{q3:.0f}] · n={len(g)}",
                va="center", ha="right", fontsize=11, color=c, fontweight="bold")

    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=12)
    ax.set_ylim(0.4, 2.7)
    ax.set_xlim(-2, 73)
    ax.set_xlabel("Lag (years): gene verification − clinical description", fontsize=12)
    ax.set_title("Diagnostic lag collapses in the NGS era", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    for d in (DB_PATH.parent / "figures", DB_PATH.parent / "export_figures"):
        d.mkdir(exist_ok=True)
    fig.savefig(DB_PATH.parent / "figures" / "fig_lag_violin.png", dpi=200, bbox_inches="tight")
    fig.savefig(DB_PATH.parent / "export_figures" / "lag_violin_pre_post_NGS.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("pre-NGS n", len(groups[1]), "median", np.median(groups[1]),
          "| NGS-era n", len(groups[0]), "median", np.median(groups[0]))


if __name__ == "__main__":
    main()
