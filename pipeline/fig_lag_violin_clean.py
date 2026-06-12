"""Two variants of the pre/post-NGS lag violin on the CLEANED sample (tail artifacts
flagged out). v1 = plain; v2 = top long-lag outliers point-labelled (all verified
against PubMed as genuine primary gene-disease discoveries).

Run: PYTHONPATH=. python -m pipeline.fig_lag_violin_clean
-> export_figures/lag_violin_clean.png and lag_violin_clean_labeled.png
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
OUT = DB_PATH.parent / "export_figures"

# top verified long-lag outliers (disease short label, gene, lag) — confirmed primary
# gene-disease discovery papers via PubMed
OUTLIERS = [
    ("Acrokeratoelastoidosis", "CCDC91", 71),
    ("Striopallidodentate calcinosis", "PDGFRB", 66),
    ("Reticular retinal dystrophy", "RCBTB1", 66),
    ("Cerebelloparenchymal disorder", "PMPCA", 65),
    ("Pyle disease", "SFRP4", 63),
    ("Multiple self-healing epithelioma", "TGFBR1", 63),
]


def base(ax, groups, stats_y=75.5, show_n=True, stat="median"):
    use_mean = stat == "mean"
    parts = ax.violinplot(groups, positions=[1, 2], widths=0.55,
                          showmedians=not use_mean, showmeans=use_mean, showextrema=False)
    for b, c in zip(parts["bodies"], [PRE, POST]):
        b.set_facecolor(c); b.set_alpha(0.40); b.set_edgecolor(c); b.set_linewidth(1.2)
    line = parts["cmeans"] if use_mean else parts["cmedians"]
    line.set_color("black"); line.set_linewidth(2)
    rng = np.random.default_rng(7)
    for i, (g, c) in enumerate(zip(groups, [PRE, POST])):
        x = (i + 1) + rng.uniform(-0.07, 0.07, len(g))
        ax.scatter(x, g, s=7, color=c, alpha=0.30, edgecolors="none", zorder=3)
        yy = stats_y if i == 0 else (stats_y if stats_y < 60 else 40)
        if use_mean:
            m, sd = np.mean(g), np.std(g, ddof=1)
            se = sd / np.sqrt(len(g)); lo, hi = m - 1.96 * se, m + 1.96 * se
            ax.errorbar(i + 1, m, yerr=1.96 * se, color="black", lw=2,
                        capsize=5, capthick=2, zorder=4)
            txt = f"mean {m:.1f} y\n95% CI [{lo:.1f}-{hi:.1f}]\nSD {sd:.1f}"
        else:
            txt = f"median {np.median(g):.0f} y\nIQR [{np.percentile(g,25):.0f}-{np.percentile(g,75):.0f}]"
        txt += (f"\nn={len(g)}" if show_n else "")
        ax.text(i + 1, yy, txt,
                ha="center", va="top", fontsize=10, color="black", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Pre-NGS\n(<2010)", "NGS era\n(≥2010)"], fontsize=12)
    ax.set_ylim(-3, 78)
    ax.set_ylabel("Lag (years): gene verification − clinical description", fontsize=12)
    ax.tick_params(labelsize=11); ax.spines[["top", "right"]].set_visible(False)


def main():
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute("SELECT lag_years, first_clinical_year FROM phenolag "
                       "WHERE lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946").fetchall()
    con.close()
    lag = np.array([r[0] for r in rows], float); clin = np.array([r[1] for r in rows], float)
    groups = [lag[clin < NGS], lag[clin >= NGS]]

    # v1 plain (a touch lower and wider than narrow); no n
    fig, ax = plt.subplots(figsize=(6.8, 6.2)); base(ax, groups, show_n=False, stat="mean")
    ax.set_xlim(0.5, 2.5)
    ax.set_title("Diagnostic lag collapses\nin the NGS era", fontsize=13.5, fontweight="bold")
    plt.tight_layout(); fig.savefig(OUT / "lag_violin_clean.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # v2 labelled outliers — labels parked in clear space far right
    fig, ax = plt.subplots(figsize=(10.5, 7.4)); base(ax, groups, stats_y=40)
    ax.set_xlim(0.5, 3.4)
    ax.set_title("Diagnostic lag collapses in the NGS era — longest lags are real, late-discovered genes",
                 fontsize=12.5, fontweight="bold")
    for k, (name, gene, lg) in enumerate(OUTLIERS):
        ty = 74 - k * 6.0
        ax.annotate(f"{name} ({gene}, {lg} y)", xy=(1.12, lg), xytext=(2.55, ty),
                    fontsize=10, va="center", color="#333",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    plt.tight_layout(); fig.savefig(OUT / "lag_violin_clean_labeled.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("pre-NGS n", len(groups[0]), "median", np.median(groups[0]),
          "max", int(groups[0].max()), "| saved v1 + v2 to", OUT)


if __name__ == "__main__":
    main()
