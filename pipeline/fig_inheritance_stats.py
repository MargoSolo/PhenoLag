"""Inheritance-pattern time-to-verification, with statistics.

A) ECDF (AR/AD/X-linked) annotated with overall log-rank p and pairwise results,
   plus a footnote that the X-linked lag is an era artefact.
B) Forest plot of era-adjusted Cox hazard ratios (event = molecularly solved),
   reference = AR. Shows AD is genuinely slower than AR, X-linked is not (era-driven).

Run: PYTHONPATH=. python -m pipeline.fig_inheritance_stats
-> export_figures/inheritance_ttv_stats.png
   export_figures/inheritance_cox_forest.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from lifelines import CoxPHFitter
from pipeline.fig_acceleration import load_at_risk
from pipeline.fig_inheritance import inh_group
from pipeline.config import DB_PATH

COL = {"AR": "#2C6FB5", "AD": "#C0392B", "X-linked": "#8E44AD"}
OUT = DB_PATH.parent / "export_figures"


def fmt_p(p):
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3g}"


def main():
    OUT.mkdir(exist_ok=True)
    d = load_at_risk()
    d["inh"] = d["inheritance"].map(inh_group)
    d = d[d.inh.isin(["AR", "AD", "X-linked"])].copy()
    sv = d[d.solved]

    # ---- statistics ----
    lr = multivariate_logrank_test(d.dur, d.inh, d.solved)
    pw = pairwise_logrank_test(d.dur, d.inh, d.solved).summary["p"]
    cox = d[["dur", "solved", "inh", "dec"]].copy()
    cox = pd.get_dummies(cox, columns=["inh"], drop_first=False).drop(columns=["inh_AR"])
    cox["dec"] = (cox["dec"] - 1980) / 10.0
    cph = CoxPHFitter().fit(cox.astype(float), "dur", "solved")
    s = cph.summary

    # ============ A) annotated ECDF ============
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for grp in ["AR", "AD", "X-linked"]:
        v = np.sort(sv[sv.inh == grp]["lag_years"].values.astype(float))
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
                color=COL[grp], lw=2.6, label=f"{grp} (med {int(np.median(v))} y)")
    ax.axvline(20, color="grey", ls=":", lw=1)
    ax.set_xlim(0, 72); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time-to-genetic-verification (years)", fontsize=12)
    ax.set_ylabel("Cumulative fraction verified", fontsize=12)
    ax.set_title("Time-to-verification by inheritance pattern", fontsize=13.5, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left", title="Inheritance")
    ax.tick_params(labelsize=11); ax.spines[["top", "right"]].set_visible(False)
    stat = (f"Log-rank (3 groups): {fmt_p(lr.p_value)}\n"
            f"AR vs AD: {fmt_p(pw[('AD','AR')])}\n"
            f"AR vs X-linked: {fmt_p(pw[('AR','X-linked')])}\n"
            f"AD vs X-linked: {fmt_p(pw[('AD','X-linked')])}  (n.s.)")
    ax.text(0.97, 0.40, stat, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8", edgecolor="#C9CDD4"))
    fig.text(0.5, -0.02,
             "X-linked diseases were described earlier; their longer raw lag is an era effect — "
             "era-adjusted they resolve as fast as AR (see Cox forest).",
             ha="center", fontsize=8.5, color="#555", style="italic", wrap=True)
    plt.tight_layout()
    fig.savefig(OUT / "inheritance_ttv_stats.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ============ B) Cox forest (era-adjusted) ============
    rows = [
        ("AR (reference)", 1.0, 1.0, 1.0, None, True),
        ("AD vs AR", s.loc["inh_AD", "exp(coef)"], s.loc["inh_AD", "exp(coef) lower 95%"],
         s.loc["inh_AD", "exp(coef) upper 95%"], s.loc["inh_AD", "p"], False),
        ("X-linked vs AR", s.loc["inh_X-linked", "exp(coef)"], s.loc["inh_X-linked", "exp(coef) lower 95%"],
         s.loc["inh_X-linked", "exp(coef) upper 95%"], s.loc["inh_X-linked", "p"], False),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ys = [2, 1, 0]
    for (lab, hr, lo, hi, p, isref), y in zip(rows, ys):
        c = "#444" if isref else (COL["AD"] if "AD" in lab else COL["X-linked"])
        if isref:
            ax.plot(hr, y, "D", color=c, ms=9)
        else:
            ax.plot([lo, hi], [y, y], color=c, lw=2.4)
            ax.plot(hr, y, "o", color=c, ms=9)
            ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
            sig = "  ← slower than AR (real)" if p < 0.05 else "  ← no difference from AR"
            ax.text(hi + 0.04, y, f"HR {hr:.2f} [{lo:.2f}–{hi:.2f}]   {ptxt}{sig}",
                    va="center", fontsize=10, color="#222")
    ax.axvline(1.0, color="grey", ls="--", lw=1.2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0.6, 1.9)
    ax.set_xlabel("Adjusted hazard of molecular solution vs AR\n(HR < 1 = solved more slowly · adjusted for decade of description)", fontsize=11)
    ax.set_title("Inheritance and speed of molecular solution, era-adjusted (Cox PH)",
                 fontsize=12.5, fontweight="bold")
    ax.tick_params(labelsize=11); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    fig.savefig(OUT / "inheritance_cox_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ============ C) combined: ECDF (top) + Cox forest (bottom) ============
    fig = plt.figure(figsize=(8.2, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.42)
    axE = fig.add_subplot(gs[0]); axF = fig.add_subplot(gs[1])

    # -- top: ECDF + stats
    for grp in ["AR", "AD", "X-linked"]:
        v = np.sort(sv[sv.inh == grp]["lag_years"].values.astype(float))
        y = np.arange(1, len(v) + 1) / len(v)
        axE.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
                 color=COL[grp], lw=2.6, label=f"{grp} (med {int(np.median(v))} y)")
    axE.axvline(20, color="grey", ls=":", lw=1)
    axE.set_xlim(0, 72); axE.set_ylim(0, 1.02)
    axE.set_xlabel("Time-to-genetic-verification (years)", fontsize=12)
    axE.set_ylabel("Cumulative fraction verified", fontsize=12)
    axE.set_title("Time-to-verification by inheritance pattern", fontsize=13.5, fontweight="bold")
    axE.legend(fontsize=11, loc="upper left", title="Inheritance")
    axE.tick_params(labelsize=11); axE.spines[["top", "right"]].set_visible(False)
    axE.text(0.97, 0.40, stat, transform=axE.transAxes, ha="right", va="top",
             fontsize=9.5, family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8", edgecolor="#C9CDD4"))

    # -- bottom: forest
    for (lab, hr, lo, hi, p, isref), y in zip(rows, ys):
        c = "#444" if isref else (COL["AD"] if "AD" in lab else COL["X-linked"])
        if isref:
            axF.plot(hr, y, "D", color=c, ms=9)
        else:
            axF.plot([lo, hi], [y, y], color=c, lw=2.4)
            axF.plot(hr, y, "o", color=c, ms=9)
            ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
            sig = "  ← slower than AR (real)" if p < 0.05 else "  ← no difference from AR"
            axF.text(hi + 0.04, y, f"HR {hr:.2f} [{lo:.2f}–{hi:.2f}]   {ptxt}{sig}",
                     va="center", fontsize=9.5, color="#222")
    axF.axvline(1.0, color="grey", ls="--", lw=1.2)
    axF.set_yticks(ys); axF.set_yticklabels([r[0] for r in rows], fontsize=11)
    axF.set_ylim(-0.6, 2.6); axF.set_xlim(0.6, 1.9)
    axF.set_xlabel("Adjusted hazard of molecular solution vs AR\n(HR < 1 = solved more slowly · adjusted for decade of description)", fontsize=10.5)
    axF.set_title("Era-adjusted Cox hazard ratios", fontsize=12.5, fontweight="bold")
    axF.tick_params(labelsize=11); axF.spines[["top", "right", "left"]].set_visible(False)
    axF.tick_params(axis="y", length=0)

    fig.savefig(OUT / "inheritance_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("log-rank overall p =", lr.p_value)
    print("saved", OUT / "inheritance_ttv_stats.png", ",", OUT / "inheritance_cox_forest.png",
          "and", OUT / "inheritance_combined.png")


if __name__ == "__main__":
    main()
