"""Wide (14x6) clinical-vs-genetic timeline scatter for the poster.

Rebuilt on current data (clean set: suspect_dating=0, lag>=0, clin>=1946).
Markers are clipped to the gen>=clin triangle so lag=0 points (which sit ON
the y=x line) never appear to spill into the impossible negative-lag region.

Run: PYTHONPATH=. python -m pipeline.fig_timeline_wide
Saves: figures/fig3_timeline_scatter_wide.{png,pdf}
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "phenolag.db"
FIG = ROOT / "figures"


def main():
    con = sqlite3.connect(str(DB))
    d = pd.read_sql("""
        SELECT first_clinical_year AS clin, first_genetic_year AS gen, lag_years AS lag
        FROM phenolag
        WHERE suspect_dating = 0 AND lag_years IS NOT NULL
          AND first_clinical_year >= 1946 AND first_genetic_year IS NOT NULL
    """, con)
    con.close()
    n = len(d)
    neg = int((d["gen"] < d["clin"]).sum())
    print(f"N={n}, points below diagonal (gen<clin)={neg}")

    fig, ax = plt.subplots(figsize=(14, 6))
    lo, hi = 1946, 2026
    ylo = 1980  # genetic verifications start ~1982; crop the empty pre-1980 band

    vmax = max(1, d["lag"].quantile(0.95))
    sc = ax.scatter(d["clin"], d["gen"], c=d["lag"], cmap="RdYlGn_r",
                    s=20, alpha=0.7, edgecolors="none", vmin=0, vmax=vmax, zorder=3)
    # clip markers to gen >= clin so none spill below the diagonal
    sc.set_clip_path(Polygon([[lo, lo], [hi, hi], [lo, hi]], transform=ax.transData))

    ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=1.4,
            alpha=0.6, label="lag = 0", zorder=4)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Lag (years)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlim(lo, hi); ax.set_ylim(ylo, hi)
    ax.set_xlabel("Year of first clinical description", fontsize=16)
    ax.set_ylabel("Year of genetic verification", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig3_timeline_scatter_wide.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig3_timeline_scatter_wide.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/fig3_timeline_scatter_wide.{png,pdf}")


if __name__ == "__main__":
    main()
