"""RMST within 10 years vs decade of clinical description, one line per disorder class,
with 95% CI. Blue-tone palette. 10-year window = the maximal follow-up common to all
cohorts (the 2010s have <15 y follow-up, so 10 y is the apples-to-apples horizon).

At-risk denominator (solved + right-censored unsolved); KM per (class x decade) cell;
RMST(10)=∫₀¹⁰ S(t)dt; bootstrap 95% CI; cells need >=13 diseases.

Run: PYTHONPATH=. python -m pipeline.fig_rmst_lines_by_class
-> export_figures/rmst10_lines_by_class.png
"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time as rmst_fn
from pipeline.fig_acceleration import load_at_risk
from pipeline.fig_debt_heatmap import short
from pipeline.config import DB_PATH

TAU = 10
MIN_N = 13
# blue-tone palette + distinct markers so 5 lines stay legible
CLASSES = [
    ("Rare neurologic disease", "#08306B", "o"),
    ("Rare inborn errors of metabolism", "#2171B5", "s"),
    ("Rare developmental defect during embryogenesis", "#4292C6", "^"),
    ("Rare skin disease", "#6BAED6", "D"),
    ("Rare immune disease", "#41B6C4", "v"),
]


def rmst_with_ci(dur, ev, tau, B=400, seed=3):
    # bootstrap CI (stable near the censoring boundary, unlike the analytic RMST
    # variance which blows up when almost everything is censored)
    point = rmst_fn(KaplanMeierFitter().fit(dur, ev), t=tau)
    rng = np.random.default_rng(seed); dur = np.asarray(dur); ev = np.asarray(ev); n = len(dur)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            bs.append(rmst_fn(KaplanMeierFitter().fit(dur[idx], ev[idx]), t=tau))
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (point, point)
    return float(point), float(lo), float(hi)


def main():
    con = sqlite3.connect(str(DB_PATH))
    cls = pd.read_sql("SELECT orpha_code, parent_name FROM classifications", con).drop_duplicates("orpha_code")
    con.close()
    d = load_at_risk().merge(cls, on="parent_name" if False else "orpha_code", how="left")
    # start at 1980s: before that every class sits at RMST=10 (0% solved <=10y,
    # pre-genetics era) and the flat overlapping lines are uninformative
    decs = list(range(1980, 2020, 10))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for cname, color, mk in CLASSES:
        xs, ys, los, his = [], [], [], []
        for dec in decs:
            sub = d[(d.parent_name == cname) & (d.dec == dec)]
            if len(sub) < MIN_N:
                continue
            v, lo, hi = rmst_with_ci(sub.dur.values, sub.solved.values, TAU)
            xs.append(dec); ys.append(v); los.append(lo); his.append(hi)
        if not xs:
            continue
        xs = np.array(xs)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, marker=mk, color=color, lw=2.3, ms=6, label=short(cname))
    ax.set_xticks(decs); ax.set_xticklabels([f"{x}s" for x in decs], fontsize=11)
    ax.set_ylim(0, 10.4)
    ax.set_ylabel("Restricted mean years UNRESOLVED within 10 y\n(lower = faster molecular resolution)", fontsize=12)
    ax.set_xlabel("Decade of first clinical description", fontsize=12)
    ax.set_title("Molecular resolution accelerates across eras, by disorder class\n"
                 "(RMST, 10-year window; shaded = 95% CI)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left", title="Disorder class", ncol=2, framealpha=0.9)
    ax.tick_params(labelsize=11); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = DB_PATH.parent / "export_figures" / "rmst10_lines_by_class.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
