"""Validate phenolag dates against the human-curated gold standard (119 diseases).

Gold: figures/ai_validation_gold_standard_updated_v19.xlsx (all_120 sheet), where
MANUAL_clin_correct / MANUAL_gen_correct (YES/NO) and MANUAL_*_correct_pmid record
the expert-verified correct PMID for each disease.

Truth PMID:
  MANUAL_*_correct_pmid if present,
  else consensus_*_pmid when MANUAL_*_correct == YES,
  else unknown (excluded).

Metrics (vs phenolag.first_clinical_* / first_genetic_*):
  - PMID-selection accuracy (exact match)
  - year mean absolute error and within-2-year rate
The gold over-samples hard agreement strata, so we also report population-reweighted
accuracy using agreement_gen / agreement_clin fractions from ai_consensus_omim (n=1140).

Run: PYTHONPATH=. python -m pipeline.validate_gold
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from pipeline.config import DB_PATH
from pipeline.pubmed_dates import get_years

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "figures" / "ai_validation_gold_standard_updated_v19.xlsx"


def _pmid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s.isdigit() else None


def main():
    gold = pd.read_excel(GOLD, sheet_name="all_120")
    gold["orpha_code"] = gold["orpha_code"].astype(str).str.replace(r"\.0$", "", regex=True)

    # truth PMIDs
    def truth(row, side):
        man = _pmid(row[f"MANUAL_{side}_correct_pmid"])
        if man:
            return man
        if str(row[f"MANUAL_{side}_correct"]).strip().upper() == "YES":
            return _pmid(row[f"consensus_{side}_pmid"])
        return None
    gold["truth_clin_pmid"] = gold.apply(lambda r: truth(r, "clin"), axis=1)
    gold["truth_gen_pmid"] = gold.apply(lambda r: truth(r, "gen"), axis=1)

    # years for all truth PMIDs
    allp = [p for p in pd.concat([gold["truth_clin_pmid"], gold["truth_gen_pmid"]]).dropna().unique()]
    yrs = get_years(allp)
    gold["truth_clin_year"] = gold["truth_clin_pmid"].map(lambda p: yrs.get(p) if p else None)
    gold["truth_gen_year"] = gold["truth_gen_pmid"].map(lambda p: yrs.get(p) if p else None)

    # phenolag stored values + stratum
    con = sqlite3.connect(str(DB_PATH))
    ph = pd.read_sql("""SELECT orpha_code, first_clinical_pmid clin_pmid, first_clinical_year clin_year,
                               first_genetic_pmid gen_pmid, first_genetic_year gen_year FROM phenolag""", con)
    ph["orpha_code"] = ph["orpha_code"].astype(str)
    strat = pd.read_sql("SELECT orpha_code, agreement_clin, agreement_gen FROM ai_consensus_omim", con)
    strat["orpha_code"] = strat["orpha_code"].astype(str)
    # population weights per stratum
    pop = {
        "clin": strat["agreement_clin"].replace("", np.nan).value_counts(dropna=True).to_dict(),
        "gen": strat["agreement_gen"].replace("", np.nan).value_counts(dropna=True).to_dict(),
    }
    con.close()

    # gold already carries agreement_clin/agreement_gen (from consensus at gold-build
    # time); use those for per-stratum grouping. strat only supplies population weights.
    g = gold.merge(ph, on="orpha_code", how="left")

    def report(side):
        tp, yr = f"truth_{side}_pmid", f"truth_{side}_year"
        pp, py = f"{side}_pmid", f"{side}_year"
        agcol = f"agreement_{side}"
        d = g[g[tp].notna()].copy()
        d[pp] = d[pp].map(_pmid)
        d["pmid_hit"] = d[pp] == d[tp]
        d["yhit2"] = (d[py] - d[yr]).abs() <= 2
        d["yerr"] = (d[py] - d[yr]).abs()
        n = len(d)
        print(f"\n=== {side.upper()} ===  (truth available: {n}/119)")
        print(f"  raw PMID-match accuracy : {d['pmid_hit'].mean():.1%}")
        print(f"  raw year within ±2y     : {d['yhit2'].mean():.1%}")
        print(f"  raw year MAE            : {d['yerr'].mean():.2f} y  (median {d['yerr'].median():.0f})")
        # population-reweighted PMID accuracy. Only trust strata with >=3 gold
        # examples (a single mislabelled example otherwise dominates a heavy stratum).
        weights = pop[side]
        tot = sum(weights.values())
        grp = d.groupby(agcol)["pmid_hit"].agg(["mean", "size"])
        num = covered = 0.0
        for strat_name, r in grp.iterrows():
            if r["size"] < 3:
                continue
            w = weights.get(strat_name, 0)
            num += r["mean"] * w
            covered += w
        if covered:
            print(f"  reweighted PMID accuracy: {num/covered:.1%}  "
                  f"(strata with n>=3 gold, covering {covered/tot:.0%} of population)")
        # per-stratum
        per = d.groupby(agcol).agg(n=("pmid_hit", "size"), pmid_acc=("pmid_hit", "mean"),
                                   mae=("yerr", "mean"))
        print(per.to_string())
        err = d[~d["pmid_hit"]][["orpha_code", "name", agcol, pp, py, tp, yr, "yerr"]].copy()
        err.insert(2, "side", side)
        err.columns = ["orpha_code", "name", "side", "stratum",
                       "phenolag_pmid", "phenolag_year", "truth_pmid", "truth_year", "year_err"]
        return err

    errs = pd.concat([report("clin"), report("gen")], ignore_index=True)
    out = ROOT / "figures" / "gold_validation_errors.csv"
    errs.sort_values(["side", "stratum", "year_err"], ascending=[True, True, False]).to_csv(out, index=False)
    print(f"\n{len(errs)} disagreements written to {out.name} (worklist for dating fixes)")


if __name__ == "__main__":
    main()
