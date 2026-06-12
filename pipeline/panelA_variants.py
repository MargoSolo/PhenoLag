"""Panel-A alternatives for 'time-to-verification by era' that avoid the
'recent decades look worse' artifact (which is really short follow-up).

Saves three PNGs into ./panelA_variants/ :
  v1_within10y_bars.png        fraction solved within 10 y of description, by decade
  v2_curves_truncated15y.png   KM cumulative incidence, x-axis cut at the common 15-y horizon
  v3_within10_20y.png          fraction solved within 10 y and 20 y, by decade

Run: PYTHONPATH=. python -m pipeline.panelA_variants
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pipeline.fig_acceleration import load_at_risk

OUT = Path(__file__).resolve().parent.parent / "panelA_variants"
NAVY, TEAL = "#065A82", "#2E8B8B"


def km_by_decade(d, decs):
    fits = {}
    for dec in decs:
        sub = d[d.dec == dec]
        fits[dec] = (KaplanMeierFitter().fit(sub.dur, sub.solved), len(sub))
    return fits


def main():
    OUT.mkdir(exist_ok=True)
    d = load_at_risk()
    decs10 = [x for x in sorted(d.dec.unique()) if 1950 <= x <= 2010 and (d.dec == x).sum() >= 20]
    decs20 = [x for x in decs10 if x <= 2000]
    fits = km_by_decade(d, decs10)

    # ---- V1: % solved within 10 y, by decade (fixed window, apples-to-apples) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    s10 = [100 * (1 - float(fits[x][0].predict(10))) for x in decs10]
    ax.bar(range(len(decs10)), s10, color=NAVY, width=0.62, edgecolor="white")
    for i, v in enumerate(s10):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(decs10))); ax.set_xticklabels([f"{x}s" for x in decs10], fontsize=12)
    ax.set_ylim(0, max(s10) + 8)
    ax.set_ylabel("Solved within 10 years of description (%)", fontsize=12)
    ax.set_xlabel("Decade of clinical description", fontsize=12)
    ax.set_title("Acceleration: chance of solving a disease within a decade\nrose from ~0% to ~36%",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=11)
    plt.tight_layout(); fig.savefig(OUT / "v1_within10y_bars.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- V2: KM curves truncated at the common 15-year horizon ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(decs10)))
    t = np.linspace(0, 15, 120)
    for c, dec in zip(cmap, decs10):
        kmf, n = fits[dec]
        ax.plot(t, 1 - kmf.predict(t).values, color=c, lw=2.6, label=f"{dec}s (n={n})")
    ax.set_xlim(0, 15); ax.set_ylim(0, 0.6)
    ax.set_xlabel("Years since clinical description (common 15-y window)", fontsize=12)
    ax.set_ylabel("Fraction molecularly solved (KM)", fontsize=12)
    ax.set_title("Within the first 15 years, recent cohorts are solved fastest", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", title="Described in")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=11)
    plt.tight_layout(); fig.savefig(OUT / "v2_curves_truncated15y.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- V3: % solved within 10 y AND 20 y, by decade ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(decs10)); w = 0.4
    s10b = [100 * (1 - float(fits[x2][0].predict(10))) for x2 in decs10]
    s20b = [100 * (1 - float(fits[x2][0].predict(20))) if x2 in decs20 else np.nan for x2 in decs10]
    ax.bar(x - w/2, s10b, w, color=NAVY, label="within 10 y")
    ax.bar(x + w/2, s20b, w, color=TEAL, label="within 20 y")
    ax.set_xticks(x); ax.set_xticklabels([f"{x2}s" for x2 in decs10], fontsize=12)
    ax.set_ylabel("Solved within window (%)", fontsize=12)
    ax.set_xlabel("Decade of clinical description", fontsize=12)
    ax.set_title("Solved within 10 and 20 years, by era\n(20-y window not estimable for 2010s)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=11)
    plt.tight_layout(); fig.savefig(OUT / "v3_within10_20y.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print("saved 3 variants to", OUT)
    print("within-10y:", {f"{x2}s": round(v) for x2, v in zip(decs10, s10b)})


if __name__ == "__main__":
    main()
