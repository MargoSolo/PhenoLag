"""Apply the genetic-date union fallback as a targeted UPDATE (experiments branch).

Recomputes first_genetic_year / lag / suspect flags ONLY for rows that have a
clinical date but no genetic date, by dating the gene from HPO/ORPHA-cited
PMIDs with the same molecular-genetics MeSH filter used for G2P.

Targeted UPDATE (not a full calculate_lag rerun) so the enrichment + ML-feature
columns populated by later pipeline steps are preserved, and no network is needed.
"""
from __future__ import annotations
from pipeline.db import get_conn
from pipeline.pubmed_dates import get_earliest_genetic_year
from pipeline.lag_calculator import _flag_suspect, collect_hpo_pmids as hpo_pmids_for


def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT orpha_code, first_clinical_pmid, first_clinical_year, clin_source
        FROM phenolag
        WHERE first_clinical_year IS NOT NULL AND first_genetic_year IS NULL
    """)
    targets = cur.fetchall()
    print(f"[FALLBACK] candidate rows: {len(targets)}")

    updates = []
    no_pmids = recovered = 0
    for orpha, clin_pmid, clin_year, clin_source in targets:
        cur.execute("SELECT omim_ids FROM disorders WHERE orpha_code = ?", (orpha,))
        row = cur.fetchone()
        omim_ids_str = row[0] if row else None
        pmids = hpo_pmids_for(cur, orpha, omim_ids_str)
        if not pmids:
            no_pmids += 1
            continue
        gen_pmid, gen_year, gen_status = get_earliest_genetic_year(list(pmids))
        if gen_year is None:
            continue
        recovered += 1
        lag = gen_year - clin_year
        gen_pre = (gen_status == "pre_molecular")
        suspect, reason = _flag_suspect(
            clin_year, gen_year, clin_source, gen_pre,
            clin_pmid=clin_pmid, gen_pmid=gen_pmid)
        updates.append((gen_pmid, gen_year, lag, suspect, reason, orpha))

    cur.executemany(
        "UPDATE phenolag SET first_genetic_pmid=?, first_genetic_year=?, "
        "lag_years=?, suspect_dating=?, suspect_reason=? WHERE orpha_code=?",
        updates)
    conn.commit()

    print(f"[FALLBACK] no HPO/ORPHA PMIDs:  {no_pmids}")
    print(f"[FALLBACK] genetic year filled: {recovered}")

    # Post-update funnel
    for label, sql in [
        ("has genetic year", "first_genetic_year IS NOT NULL"),
        ("lag>=0", "lag_years>=0"),
        ("lag>=0 & not suspect", "lag_years>=0 AND suspect_dating=0"),
        ("lag>=0 & not suspect & clin>=1946",
         "lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946"),
    ]:
        cur.execute(f"SELECT COUNT(*) FROM phenolag WHERE {sql}")
        print(f"[FUNNEL] {label}: {cur.fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    main()
