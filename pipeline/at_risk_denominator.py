from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from pipeline.config import DB_PATH, SNAPSHOT_YEAR


def main():
    con = sqlite3.connect(str(DB_PATH))
    dis = pd.read_sql("SELECT orpha_code, omim_ids FROM disorders WHERE disorder_type='Disease'", con)
    ph = pd.read_sql("""SELECT orpha_code, first_clinical_year clin, lag_years, suspect_dating,
                               first_genetic_year gen FROM phenolag""", con)
    omim = pd.read_sql("SELECT omim_id, earliest_year FROM omim_clinical_cache WHERE earliest_year IS NOT NULL", con)
    con.close()
    omap = dict(zip(omim["omim_id"].astype(str), omim["earliest_year"]))

    def omim_clin(ids):
        if not isinstance(ids, str) or not ids:
            return np.nan
        ys = [omap[m.strip()] for m in ids.split(",") if m.strip() in omap]
        return min(ys) if ys else np.nan

    d = dis.merge(ph, on="orpha_code", how="left")
    # clinical year: phenolag value, else OMIM-cache earliest
    d["clin_year"] = d["clin"].fillna(d["omim_ids"].map(omim_clin))
    d["solved"] = (d["gen"].notna()) & (d["lag_years"] >= 0) & (d["suspect_dating"] == 0)
    d = d[d["clin_year"] >= 1946].copy()
    d["duration"] = np.where(d["solved"], d["lag_years"], SNAPSHOT_YEAR - d["clin_year"])
    d = d[d["duration"] >= 0]

    n = len(d); nsolved = int(d["solved"].sum()); ncens = n - nsolved
    print(f"At-risk denominator (Orphanet 'Disease', clinical year >=1946 datable): {n}")
    print(f"  solved (event)      : {nsolved} ({nsolved/n:.0%})")
    print(f"  unsolved (censored) : {ncens} ({ncens/n:.0%})")
    undatable = dis.shape[0] - n
    print(f"  NB: {undatable} 'Disease' disorders have no datable clinical year (no OMIM/MEDLINE) — excluded")

    kmf = KaplanMeierFitter().fit(d["duration"], event_observed=d["solved"])
    med = kmf.median_survival_time_
    print(f"\nKaplan-Meier time-to-molecular-solution (true denominator):")
    print(f"  median time to solution : {med:.0f} y")
    for t in (10, 20, 30, 40):
        print(f"  solved within {t:2d} y     : {1-float(kmf.predict(t)):.0%}")


if __name__ == "__main__":
    main()
