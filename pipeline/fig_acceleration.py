"""Censoring-honest 'acceleration' figure (replaces the survivorship-biased
solved-only CDF). Uses the full at-risk denominator (solved events + still-unsolved
censored) so recent decades are not flattered by counting only the already-solved.

Left  — Kaplan-Meier cumulative incidence of molecular solution, by decade described.
Right — fraction solved within 10 years of description, by decade (well-defined under
        censoring; the honest 'acceleration' metric).

Run: PYTHONPATH=. python -m pipeline.fig_acceleration
Saves figures/fig_acceleration.png (+ .jpg for the JPEG poster slot)
"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pipeline.config import DB_PATH, SNAPSHOT_YEAR, FIGURES


def load_at_risk():
    con = sqlite3.connect(str(DB_PATH))
    dis = pd.read_sql("SELECT orpha_code, omim_ids FROM disorders WHERE disorder_type='Disease'", con)
    ph = pd.read_sql("SELECT orpha_code, first_clinical_year clin, lag_years, suspect_dating, first_genetic_year gen, inheritance FROM phenolag", con)
    om = pd.read_sql("SELECT omim_id, earliest_year FROM omim_clinical_cache WHERE earliest_year IS NOT NULL", con)
    con.close()
    omap = dict(zip(om.omim_id.astype(str), om.earliest_year))
    def oc(ids):
        if not isinstance(ids, str) or not ids:
            return np.nan
        ys = [omap[m.strip()] for m in ids.split(",") if m.strip() in omap]
        return min(ys) if ys else np.nan
    d = dis.merge(ph, on="orpha_code", how="left")
    d["clin_year"] = d["clin"].fillna(d["omim_ids"].map(oc))
    d["solved"] = (d.gen.notna()) & (d.lag_years >= 0) & (d.suspect_dating == 0)
    d = d[d.clin_year >= 1946].copy()
    d["dur"] = np.where(d.solved, d.lag_years, SNAPSHOT_YEAR - d.clin_year)
    d = d[d.dur >= 0].copy()
    d["dec"] = (d.clin_year // 10 * 10).astype(int)
    return d


def main():
    d = load_at_risk()
    # 1950-2010 only: 2020s have <10 y follow-up, so a 10-year solve rate is undefined
    decs = [x for x in sorted(d.dec.unique()) if 1950 <= x <= 2010 and (d.dec == x).sum() >= 20]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(decs)))
    s10 = []
    for c, dec in zip(cmap, decs):
        sub = d[d.dec == dec]
        kmf = KaplanMeierFitter().fit(sub.dur, sub.solved)
        t = np.linspace(0, 60, 240)
        ci = 1 - kmf.predict(t).values
        axL.plot(t, ci, color=c, lw=2.4, label=f"{dec}s (n={len(sub)})")
        s10.append(1 - float(kmf.predict(10)))
    axL.axvline(10, color="grey", ls=":", lw=1)
    axL.set_xlim(0, 60); axL.set_ylim(0, 1)
    axL.set_xlabel("Years since clinical description", fontsize=12)
    axL.set_ylabel("Fraction molecularly solved (KM, incl. unsolved)", fontsize=12)
    axL.set_title("Cumulative incidence of molecular solution, by era", fontsize=13, fontweight="bold")
    axL.legend(fontsize=9, loc="upper left", title="Described in")
    axL.tick_params(labelsize=11)
    axL.spines[["top", "right"]].set_visible(False)

    # Right: time-to-verification by inheritance pattern (mitochondrial excluded)
    def inh_group(s):
        if not isinstance(s, str):
            return "Other"
        if "X-linked" in s:
            return "X-linked"
        if "Autosomal recessive" in s and "Autosomal dominant" not in s:
            return "AR"
        if "Autosomal dominant" in s and "Autosomal recessive" not in s:
            return "AD"
        return "Other"
    sv = d[d.solved].copy()
    sv["inh"] = sv["inheritance"].map(inh_group)
    for grp, col in [("AR", "#2C6FB5"), ("AD", "#C0392B"), ("X-linked", "#8E44AD")]:
        v = np.sort(sv[sv.inh == grp]["lag_years"].values.astype(float))
        if len(v) == 0:
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        axR.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
                 color=col, lw=2.6, label=f"{grp} (n={len(v)}, med={int(np.median(v))})")
    axR.axvline(20, color="grey", ls=":", lw=1)
    axR.set_xlim(0, 72); axR.set_ylim(0, 1.02)
    axR.set_xlabel("Time-to-genetic-verification (years)", fontsize=12)
    axR.set_ylabel("Cumulative fraction verified", fontsize=12)
    axR.set_title("By inheritance pattern", fontsize=13, fontweight="bold")
    axR.legend(fontsize=10, loc="lower right")
    axR.tick_params(labelsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig_acceleration.png", dpi=200, bbox_inches="tight")
    from PIL import Image
    Image.open(FIGURES / "fig_acceleration.png").convert("RGB").save(FIGURES / "fig_acceleration.jpg", "JPEG", quality=92)
    plt.close(fig)
    print("s10 by decade:", {f"{dd}s": round(100*v) for dd, v in zip(decs, s10)})
    print("saved", FIGURES / "fig_acceleration.png")


if __name__ == "__main__":
    main()
