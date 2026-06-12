"""Time-to-genetic-verification: empirical CDFs (poster figure).
Left  — by decade of clinical description (avoids right-truncation bias).
Right — by inheritance pattern. Mitochondrial is EXCLUDED: the cohort is built on
nuclear-gene sources (G2P/ClinGen/HPO/gnomAD), so mtDNA-encoded disease is
under-represented (n too small to plot honestly).

Run: PYTHONPATH=. python -m pipeline.fig_ttv
Saves figures/fig_ttv_cdf.png
"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.config import DB_PATH, FIGURES

CLEAN = "lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946 AND first_genetic_year IS NOT NULL"


def cdf(ax, vals, label, color):
    v = np.sort(np.asarray(vals, float))
    if len(v) == 0:
        return
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
            color=color, lw=2, label=label)


def inh_group(s):
    if not isinstance(s, str):
        return "Other"
    if "Mitochondrial" in s and "Autosomal" not in s and "X-linked" not in s:
        return "Mitochondrial"
    if "X-linked" in s and "Autosomal dominant" not in s and "Autosomal recessive" not in s:
        return "X-linked"
    if "Autosomal recessive" in s and "Autosomal dominant" not in s:
        return "AR"
    if "Autosomal dominant" in s and "Autosomal recessive" not in s:
        return "AD"
    return "Other"


def main():
    con = sqlite3.connect(str(DB_PATH))
    d = pd.read_sql(f"SELECT lag_years, first_clinical_year, inheritance FROM phenolag WHERE {CLEAN}", con)
    con.close()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: by clinical decade
    d["dec"] = (d["first_clinical_year"] // 10 * 10).astype(int)
    decs = sorted(x for x in d["dec"].unique() if x >= 1950)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(decs)))
    for c, dc in zip(cmap, decs):
        sub = d[d["dec"] == dc]["lag_years"]
        if len(sub) < 10:
            continue
        cdf(axL, sub, f"{dc}s (n={len(sub)}, med={int(np.median(sub))})", c)
    axL.axvline(20, color="grey", ls=":", lw=1)
    axL.set_title("Empirical CDF by decade of clinical description\n(no right-truncation bias)", fontsize=13, fontweight="bold")
    axL.set_xlabel("Time-to-genetic-verification (years from clinical description)", fontsize=11)
    axL.set_ylabel("Cumulative fraction verified", fontsize=11)
    axL.set_xlim(0, 72); axL.set_ylim(0, 1.02); axL.legend(fontsize=8, loc="lower right")
    axL.spines[["top", "right"]].set_visible(False)

    # Right: by inheritance (AR/AD/X-linked only; Mitochondrial excluded)
    d["inh"] = d["inheritance"].map(inh_group)
    colors = {"AR": "#2C6FB5", "AD": "#C0392B", "X-linked": "#8E44AD"}
    for g, col in colors.items():
        sub = d[d["inh"] == g]["lag_years"]
        if len(sub) == 0:
            continue
        cdf(axR, sub, f"{g} (n={len(sub)}, med={int(np.median(sub))})", col)
    axR.axvline(20, color="grey", ls=":", lw=1)
    axR.set_title("Empirical CDF by inheritance pattern\n(nuclear-gene disorders; mtDNA excluded)", fontsize=13, fontweight="bold")
    axR.set_xlabel("Time-to-genetic-verification (years)", fontsize=11)
    axR.set_ylabel("Cumulative fraction verified", fontsize=11)
    axR.set_xlim(0, 72); axR.set_ylim(0, 1.02); axR.legend(fontsize=9, loc="lower right")
    axR.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig_ttv_cdf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("inheritance groups:", d["inh"].value_counts().to_dict())
    print("saved", FIGURES / "fig_ttv_cdf.png")


if __name__ == "__main__":
    main()
