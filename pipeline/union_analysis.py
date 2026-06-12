"""One-off analysis (experiments branch): quantify the PMID-union idea.

Question: for diseases that currently have a clinical date but NO genetic
date (mostly HPO-tier, no G2P publication list), can we recover a genetic
date from HPO/ORPHA-cited PMIDs already in the local DB, using the same
molecular-genetics MeSH filter the pipeline uses?

Also: review-bias test. For diseases that DO have a G2P genetic year,
compare it against the HPO-union genetic estimate to see if non-G2P
sources are systematically later/earlier.

Pure local DB — no network calls.
"""
from __future__ import annotations
from statistics import median
from pipeline.db import get_conn
from pipeline.pubmed_dates import _has_genetic_evidence_mesh, GENE_VERIFICATION_FLOOR_YEAR


from pipeline.lag_calculator import collect_hpo_pmids as hpo_pmids_for


def year_mesh(cur, pmids):
    """Local {pmid: (year, mesh)} from pubmed_cache."""
    if not pmids:
        return {}
    ph = ",".join("?" * len(pmids))
    cur.execute(f"SELECT pmid, year, mesh_terms FROM pubmed_cache WHERE pmid IN ({ph})", list(pmids))
    return {r[0]: (r[1], r[2]) for r in cur.fetchall()}


def earliest_genetic(ym):
    """Apply the pipeline's molecular-genetics MeSH filter to a {pmid:(year,mesh)} map."""
    valid = [(p, y) for p, (y, _) in ym.items() if y is not None]
    if not valid:
        return None
    genetic = [(p, y) for p, (y, m) in ym.items()
               if y is not None and y >= GENE_VERIFICATION_FLOOR_YEAR and _has_genetic_evidence_mesh(m)]
    if genetic:
        return min(genetic, key=lambda x: x[1])[1]
    modern = [(p, y) for p, y in valid if y >= GENE_VERIFICATION_FLOOR_YEAR]
    if modern:
        return min(modern, key=lambda x: x[1])[1]
    return min(valid, key=lambda x: x[1])[1]


def main():
    conn = get_conn()
    cur = conn.cursor()

    # --- Part A: recovery potential for the 418 clinical-but-no-genetic ---
    cur.execute("""
        SELECT orpha_code, omim_id, first_clinical_year
        FROM phenolag
        WHERE first_clinical_year IS NOT NULL AND first_genetic_year IS NULL
    """)
    missing = cur.fetchall()
    # omim_id column on phenolag may be a single id; pull full omim_ids from disorders
    print(f"[A] clinical-but-no-genetic diseases: {len(missing)}")

    recovered = 0
    rec_pos_lag = 0
    no_hpo_pmids = 0
    no_year = 0
    for orpha, _omim, clin_year in missing:
        cur.execute("SELECT omim_ids FROM disorders WHERE orpha_code = ?", (orpha,))
        row = cur.fetchone()
        omim_ids_str = row[0] if row else None
        pmids = hpo_pmids_for(cur, orpha, omim_ids_str)
        if not pmids:
            no_hpo_pmids += 1
            continue
        ym = year_mesh(cur, pmids)
        gy = earliest_genetic(ym)
        if gy is None:
            no_year += 1
            continue
        recovered += 1
        if clin_year is not None and gy - clin_year >= 0:
            rec_pos_lag += 1

    print(f"[A] no HPO/ORPHA PMIDs at all:        {no_hpo_pmids}")
    print(f"[A] PMIDs but none with local year:   {no_year}")
    print(f"[A] genetic year RECOVERABLE:         {recovered}")
    print(f"[A]   of which lag>=0 (usable):       {rec_pos_lag}")

    # --- Part B: review-bias test on diseases that HAVE a G2P genetic year ---
    cur.execute("""
        SELECT orpha_code, omim_id, first_genetic_year
        FROM phenolag
        WHERE first_genetic_year IS NOT NULL AND selection_tier IN ('G2P','ClinGen')
    """)
    have = cur.fetchall()
    diffs = []
    both = 0
    for orpha, _omim, g2p_gen_year in have:
        cur.execute("SELECT omim_ids FROM disorders WHERE orpha_code = ?", (orpha,))
        row = cur.fetchone()
        omim_ids_str = row[0] if row else None
        pmids = hpo_pmids_for(cur, orpha, omim_ids_str)
        if not pmids:
            continue
        ym = year_mesh(cur, pmids)
        union_gen = earliest_genetic(ym)
        if union_gen is None or g2p_gen_year is None:
            continue
        both += 1
        diffs.append(union_gen - g2p_gen_year)  # + = union LATER than G2P

    print(f"\n[B] G2P/ClinGen diseases with a comparable HPO-union genetic year: {both}")
    if diffs:
        later = sum(1 for d in diffs if d > 0)
        earlier = sum(1 for d in diffs if d < 0)
        same = sum(1 for d in diffs if d == 0)
        print(f"[B] union LATER than G2P:   {later} ({100*later/both:.0f}%)")
        print(f"[B] union EARLIER than G2P: {earlier} ({100*earlier/both:.0f}%)")
        print(f"[B] union SAME:             {same} ({100*same/both:.0f}%)")
        print(f"[B] median diff (union - G2P): {median(diffs):+.0f} years")
        print(f"[B] mean diff:                 {sum(diffs)/len(diffs):+.1f} years")

    conn.close()


if __name__ == "__main__":
    main()
