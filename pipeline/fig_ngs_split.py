"""The 19-year lag is a pre-NGS legacy: diseases described AFTER clinical exome
sequencing (~2010) are molecularly solved essentially immediately.

Splits the clean sample by era of clinical description (before vs from 2010) and
shows the lag distribution of each. Saves figures/fig_ngs_split.png (+ .jpg).

Run: PYTHONPATH=. python -m pipeline.fig_ngs_split
"""
from __future__ import annotations
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.config import DB_PATH, FIGURES

NGS = 2010
PRE, POST = "#B85042", "#065A82"


def main():
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT lag_years, first_clinical_year FROM phenolag "
        "WHERE lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946").fetchall()
    con.close()
    lag = np.array([r[0] for r in rows], float)
    clin = np.array([r[1] for r in rows], float)
    pre = lag[clin < NGS]
    post = lag[clin >= NGS]
    bins = np.arange(0, 75, 3)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.hist(pre, bins=bins, density=True, color=PRE, alpha=0.65, edgecolor="white",
            label=f"Described before NGS, <2010  (n={len(pre)}, median {np.median(pre):.0f} y)")
    ax.hist(post, bins=bins, density=True, color=POST, alpha=0.75, edgecolor="white",
            label=f"Described in NGS era, ≥2010  (n={len(post)}, median {np.median(post):.0f} y)")
    ax.axvline(np.median(pre), color=PRE, ls="--", lw=2)
    ax.axvline(np.median(post), color=POST, ls="--", lw=2)
    ax.annotate(f"{100*(post<=2).mean():.0f}% solved within 2 years",
                xy=(1.5, ax.get_ylim()[1]*0.0), xytext=(18, 0.16),
                fontsize=12, color=POST, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=POST, lw=1.5))
    ax.set_xlabel("Lag (years): gene verification − clinical description", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("The lag is a pre-NGS legacy\n(diseases described from 2010 are solved immediately)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(0, 72)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_ngs_split.png", dpi=200, bbox_inches="tight")
    from PIL import Image
    Image.open(FIGURES / "fig_ngs_split.png").convert("RGB").save(FIGURES / "fig_ngs_split.jpg", "JPEG", quality=92)
    plt.close(fig)
    print(f"pre-NGS n={len(pre)} median={np.median(pre):.0f}; NGS-era n={len(post)} median={np.median(post):.0f}; "
          f"post<=2y={100*(post<=2).mean():.0f}%")
    print("saved", FIGURES / "fig_ngs_split.png")


if __name__ == "__main__":
    main()
