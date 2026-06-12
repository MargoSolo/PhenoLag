"""ESHG top-3 figures — survival-correct view of the discovery lag.
Main inferences use a fixed post-description window (no comparing final plateaus
between cohorts with unequal follow-up).

Outputs -> eshg_top3_figures/ :
  1_probability_solved_10y_by_decade.png/pdf      P(solved <=10y) = 1-S(10), KM, 95% CI
  2_rmst_10_15y_by_decade.png/pdf                 restricted mean years unresolved within 10/15 y
  3_description_vs_solution_year_scatter.png/pdf  solved diseases only, lag-coloured
  km_probability_by_decade.csv / rmst_by_decade.csv / description_vs_solution_year.csv
  short_methods_note.md

Run: PYTHONPATH=. python -m pipeline.eshg_top3
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time as rmst_fn
from pipeline.fig_acceleration import load_at_risk
from pipeline.config import DB_PATH

OUT = DB_PATH.parent / "eshg_top3_figures"
NAVY, TEAL, RUST = "#065A82", "#2E8B8B", "#B85042"


def s_at(kmf, t):
    """Survival and 95% CI at time t from a fitted KM."""
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    idx = sf.index[sf.index <= t]
    i = idx.max() if len(idx) else sf.index.min()
    s = float(sf.loc[i].iloc[0])
    lo = float(ci.loc[i].iloc[0]); hi = float(ci.loc[i].iloc[1])
    return s, lo, hi


def rmst_ci(durations, events, tau, B=400, seed=12345):
    kmf = KaplanMeierFitter().fit(durations, events)
    point = rmst_fn(kmf, t=tau)
    rng = np.random.default_rng(seed)
    n = len(durations); d = np.asarray(durations); e = np.asarray(events)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            k = KaplanMeierFitter().fit(d[idx], e[idx])
            boots.append(rmst_fn(k, t=tau))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, lo, hi


def main():
    OUT.mkdir(exist_ok=True)
    d = load_at_risk()
    decs = [x for x in sorted(d.dec.unique()) if 1950 <= x <= 2010 and (d.dec == x).sum() >= 20]
    labels = [f"{x}s" for x in decs]

    # ---------- Figure 1: P(solved within 10 y) ----------
    rows = []
    for dec in decs:
        sub = d[d.dec == dec]
        kmf = KaplanMeierFitter().fit(sub.dur, sub.solved)
        s, lo, hi = s_at(kmf, 10)
        rows.append(dict(decade=f"{dec}s", n=len(sub), n_solved=int(sub.solved.sum()),
                         p_solved_10y=1 - s, ci_low=1 - hi, ci_high=1 - lo))
    km = pd.DataFrame(rows)
    km.to_csv(OUT / "km_probability_by_decade.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(decs))
    yerr = [km.p_solved_10y - km.ci_low, km.ci_high - km.p_solved_10y]
    ax.errorbar(x, km.p_solved_10y * 100, yerr=np.array(yerr) * 100, fmt="o", color=NAVY,
                ms=9, lw=2, capsize=5, ecolor="#9AA0A6")
    for xi, r in zip(x, rows):
        ax.annotate(f"n={r['n']}", (xi, -4), ha="center", fontsize=10, color="#555", annotation_clip=False)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(-6, 75)
    ax.set_ylabel("P(molecularly solved within 10 years) %", fontsize=12)
    ax.set_xlabel("Decade of first clinical description", fontsize=12)
    ax.set_title("Probability of molecular resolution within 10 years\n(Kaplan-Meier, fixed window; 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "1_probability_solved_10y_by_decade.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "1_probability_solved_10y_by_decade.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: RMST unresolved within 10 / 15 y ----------
    rr = []
    for dec in decs:
        sub = d[d.dec == dec]
        p10, l10, h10 = rmst_ci(sub.dur.values, sub.solved.values, 10)
        p15, l15, h15 = rmst_ci(sub.dur.values, sub.solved.values, 15)
        rr.append(dict(decade=f"{dec}s", n=len(sub),
                       rmst10=p10, rmst10_lo=l10, rmst10_hi=h10,
                       rmst15=p15, rmst15_lo=l15, rmst15_hi=h15))
    rdf = pd.DataFrame(rr)
    rdf.to_csv(OUT / "rmst_by_decade.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for col, lo, hi, tau, color, off in [("rmst10", "rmst10_lo", "rmst10_hi", "10 y", NAVY, -0.08),
                                         ("rmst15", "rmst15_lo", "rmst15_hi", "15 y", TEAL, 0.08)]:
        ye = [rdf[col] - rdf[lo], rdf[hi] - rdf[col]]
        ax.errorbar(x + off, rdf[col], yerr=ye, fmt="o-", color=color, ms=7, lw=2,
                    capsize=4, ecolor="#9AA0A6", label=f"within {tau}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Restricted mean years UNRESOLVED\nwithin the window", fontsize=12)
    ax.set_xlabel("Decade of first clinical description", fontsize=12)
    ax.set_title("Restricted mean years unresolved (RMST)\nlower = faster molecular resolution; 95% CI",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, title="Window after description"); ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "2_rmst_10_15y_by_decade.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "2_rmst_10_15y_by_decade.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 3: description vs solution year (solved only) ----------
    sv = d[d.solved & d.gen.notna() & d.clin_year.notna()].copy()
    sv["lag"] = sv.gen - sv.clin_year
    bucket = pd.cut(sv.lag, [-1, 5, 10, 20, np.inf], labels=["0-5 y", "6-10 y", "11-20 y", ">20 y"])
    sv["bucket"] = bucket
    cols = {"0-5 y": "#1A9850", "6-10 y": "#A6D96A", "11-20 y": "#FDAE61", ">20 y": "#D73027"}
    sv[["clin_year", "gen", "lag", "bucket"]].to_csv(OUT / "description_vs_solution_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    lo_, hi_ = 1946, 2026
    for b, c in cols.items():
        m = sv.bucket == b
        ax.scatter(sv.clin_year[m], sv.gen[m], s=14, alpha=0.6, color=c, edgecolors="none", label=b)
    ax.plot([lo_, hi_], [lo_, hi_], "k--", lw=1.3, label="solved same year")
    ax.plot([lo_, hi_], [lo_ + 10, hi_ + 10], color="grey", ls=":", lw=1, label="10-year lag")
    ax.plot([lo_, hi_], [lo_ + 20, hi_ + 20], color="grey", ls="-.", lw=1, label="20-year lag")
    ax.set_xlim(lo_, hi_); ax.set_ylim(1980, hi_)
    ax.set_xlabel("Year of first clinical description", fontsize=12)
    ax.set_ylabel("Year of molecular solution", fontsize=12)
    ax.set_title("Description vs solution year (solved diseases): recent solved diseases hug the diagonal = lag compression",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", title="Lag", ncol=2)
    ax.tick_params(labelsize=11); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "Solved diseases only; censoring and unsolved diseases are handled in the KM / RMST analyses.",
             ha="center", fontsize=9, color="#666")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(OUT / "3_description_vs_solution_year_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "3_description_vs_solution_year_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    (OUT / "short_methods_note.md").write_text(METHODS)
    print("KM P(solved<=10y):")
    print(km.round(3).to_string(index=False))
    print("\nRMST:")
    print(rdf.round(1).to_string(index=False))
    print("\nsaved ->", OUT)


METHODS = """# Methods note — discovery-lag, survival-correct analysis

**Unit.** One rare disease (Orphanet 'Disease' with a clinical-description year ≥ 1946).

**Event.** Molecular solution = first datable gene-validity publication.
`event_observed = 1` if a molecular-solution year exists (clean, non-suspect),
else 0 (still unresolved, right-censored).
`observed_time = lag_years` if solved, else `snapshot_year − clinical_description_year`.

**Why a fixed window.** Recent cohorts have shorter follow-up, so their final
cumulative solved fraction is censored low. We therefore do NOT compare final
plateaus. All inference uses a fixed post-description window (10 and 15 years).

**Figure 1.** Kaplan–Meier per decade of clinical description;
P(solved within 10 y) = 1 − S(10), with KM 95% CI.

**Figure 2.** Restricted mean survival time (RMST) of the *unresolved* function,
RMST(τ) = ∫₀^τ S(t) dt for τ = 10 and 15 y = mean years a disease stays unresolved
within the window. Lower = faster resolution. 95% CI by bootstrap (B = 400).

**Figure 3 (illustrative only).** Year of clinical description vs year of molecular
solution, solved diseases only, coloured by lag. Diagonal = solved same year;
dotted/dashed = 10-/20-year lag. Censoring and unsolved diseases are handled by the
KM/RMST analyses, not here.

**Headline rule.** The main conclusion is drawn from the KM 10-year probability and
the RMST, NOT from the median lag among solved diseases.
"""


if __name__ == "__main__":
    main()
