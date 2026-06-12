"""Fetch clinical-description years for ALL Orphanet 'Disease' OMIMs (gene or not),
so the survival / time-to-solution analyses have a true at-risk denominator instead
of only the gene-bearing cohort. Incremental: only fetches OMIMs missing from
omim_clinical_cache. Free NCBI E-utilities (elink + efetch). No Gemini.

Run: PYTHONPATH=. python -m pipeline.fetch_denominator
"""
from __future__ import annotations
import time
import sqlite3
from pipeline.config import DB_PATH
from pipeline.omim_dates import _elink_batch
from pipeline.pubmed_dates import get_years


def main():
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT omim_ids FROM disorders WHERE disorder_type='Disease' "
        "AND omim_ids IS NOT NULL AND omim_ids != ''").fetchall()
    allo = set()
    for (s,) in rows:
        for m in s.split(","):
            m = m.strip()
            if m:
                allo.add(m)
    cached = {r[0] for r in con.execute("SELECT omim_id FROM omim_clinical_cache")}
    missing = sorted(allo - cached)
    con.close()
    print(f"[DENOM] {len(allo)} Disease OMIMs, {len(missing)} missing -> fetching")
    if not missing:
        return

    omim_pmids, all_pmids = {}, set()
    B = 20
    for i in range(0, len(missing), B):
        batch = missing[i:i + B]
        res = _elink_batch(batch)
        for m in batch:
            pm = res.get(m, [])
            omim_pmids[m] = pm
            all_pmids.update(pm)
        if (i // B) % 20 == 0:
            print(f"  ... elink {min(i+B,len(missing))}/{len(missing)}")
        time.sleep(0.25)

    print(f"[DENOM] resolving years for {len(all_pmids)} PMIDs ...")
    yrs = get_years(list(all_pmids))

    con = sqlite3.connect(str(DB_PATH))
    n = 0
    for m, pm in omim_pmids.items():
        valid = [(p, yrs.get(p)) for p in pm if yrs.get(p) is not None]
        if valid:
            ep, ey = min(valid, key=lambda x: x[1])
            con.execute("INSERT OR REPLACE INTO omim_clinical_cache "
                        "(omim_id, earliest_pmid, earliest_year, n_omim_refs) VALUES (?,?,?,?)",
                        (m, ep, ey, len(pm)))
            n += 1
        else:
            con.execute("INSERT OR REPLACE INTO omim_clinical_cache (omim_id, n_omim_refs) VALUES (?,?)",
                        (m, len(pm)))
    con.commit()
    con.close()
    print(f"[DENOM] done. clinical year set for {n}/{len(missing)} new OMIMs.")


if __name__ == "__main__":
    main()
