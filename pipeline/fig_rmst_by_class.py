"""RMST within 10 years, BY disorder class (the same 10 classes as the heatmap).
Restricted mean years a disease stays UNRESOLVED in the first 10 years after clinical
description, per class, on the at-risk denominator (solved + still-unsolved/censored).
Lower = faster molecular resolution. Bootstrap 95% CI.

Run: PYTHONPATH=. python -m pipeline.fig_rmst_by_class
-> export_figures/rmst10_by_class.png
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
# same 10 classes as the diagnostic-debt heatmap (top-10 by solved N)
HEATMAP_CLASSES = [
    "Rare neurologic disease", "Rare inborn errors of metabolism", "Rare skin disease",
    "Rare developmental defect during embryogenesis", "Rare bone disease",
    "Rare endocrine disease", "Rare ophthalmic disorder", "Rare immune disease",
    "Rare hematologic disease", "Rare systemic or rheumatologic disease",
]


def rmst_ci(dur, ev, tau, B=400, seed=11):
    k = KaplanMeierFitter().fit(dur, ev)
    point = rmst_fn(k, t=tau)
    s10 = 1 - float(k.predict(tau))
    rng = np.random.default_rng(seed); n = len(dur); dur = np.asarray(dur); ev = np.asarray(ev)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            bs.append(rmst_fn(KaplanMeierFitter().fit(dur[idx], ev[idx]), t=tau))
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return point, lo, hi, s10


def main():
    con = sqlite3.connect(str(DB_PATH))
    cls = pd.read_sql("SELECT orpha_code, parent_name FROM classifications", con).drop_duplicates("orpha_code")
    con.close()
    d = load_at_risk().merge(cls, on="orpha_code", how="left")

    rows = []
    for c in HEATMAP_CLASSES:
        sub = d[d.parent_name == c]
        if len(sub) < 15:
            continue
        p, lo, hi, s10 = rmst_ci(sub.dur.values, sub.solved.values, TAU)
        rows.append(dict(cls=short(c), n=len(sub), rmst10=p, lo=lo, hi=hi, s10=s10))
    df = pd.DataFrame(rows).sort_values("rmst10")  # fastest (lowest RMST) on top

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(df))[::-1]
    xerr = [df.rmst10 - df.lo, df.hi - df.rmst10]
    ax.barh(y, df.rmst10, xerr=xerr, color="#065A82", alpha=0.85, height=0.6,
            error_kw=dict(ecolor="#9AA0A6", capsize=4, lw=1))
    for yi, r in zip(y, df.itertuples()):
        ax.text(r.rmst10 + 0.15, yi, f"{r.rmst10:.1f} y  ({100*r.s10:.0f}% solved ≤10y · n={r.n})",
                va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(df.cls, fontsize=11)
    ax.set_xlim(0, 11.5)
    ax.set_xlabel("Restricted mean years UNRESOLVED within first 10 years\n(lower = faster molecular resolution)", fontsize=12)
    ax.set_title("Time-to-resolution by disorder class (10-year window)", fontsize=13.5, fontweight="bold")
    ax.tick_params(labelsize=11); ax.spines[["top", "right"]].set_visible(False)
    ax.invert_xaxis() if False else None
    plt.tight_layout()
    out = DB_PATH.parent / "export_figures" / "rmst10_by_class.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(df.round(2).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
