"""Worklist of 'forgotten' rare diseases: clinically described long ago but still
with NO known causative gene (absent from G2P/ClinGen/HPO/OMIM gene links).
Ranked by years waiting. Includes phenotype/inheritance fields and a suggested
gene-discovery strategy. -> forgotten_diseases_top50.xlsx

Run: PYTHONPATH=. python -m pipeline.build_forgotten_list
"""
from __future__ import annotations
import sqlite3
import pandas as pd
from pipeline.config import DB_PATH, SNAPSHOT_YEAR

ROOT = DB_PATH.parent


def strategy(inh: str, n_hpo: int) -> str:
    i = inh or ""
    s = []
    if "Autosomal recessive" in i:
        s.append("Recessive: homozygosity mapping + WES/WGS in consanguineous/multiplex families; recessive gene-burden across cohorts")
    if "Autosomal dominant" in i:
        s.append("Dominant/sporadic: trio WES/WGS for de novo variants")
    if "X-linked" in i:
        s.append("X-linked: X-exome + segregation in affected males")
    if not s:
        s.append("Inheritance unknown: trio WGS as first line")
    s.append("submit to Matchmaker Exchange / GeneMatcher (UDN, IRUD) to aggregate unrelated cases")
    if n_hpo and n_hpo >= 10:
        s.append("distinctive multi-system phenotype aids matching")
    s.append("if exome-negative: RNA-seq + long-read WGS for non-coding/structural causes")
    return "; ".join(s)


def main():
    con = sqlite3.connect(str(DB_PATH))
    dis = pd.read_sql("""SELECT orpha_code, name, omim_ids FROM disorders
        WHERE disorder_type='Disease' AND orpha_code NOT IN (SELECT orpha_code FROM disorder_genes)""", con)
    omap = dict(con.execute("SELECT omim_id, earliest_year FROM omim_clinical_cache WHERE earliest_year IS NOT NULL").fetchall())
    nref = dict(con.execute("SELECT omim_id, n_omim_refs FROM omim_clinical_cache WHERE n_omim_refs IS NOT NULL").fetchall())
    epi = pd.read_sql("SELECT orpha_code, inheritance, age_of_onset FROM epidemiology", con).set_index("orpha_code")
    cls = pd.read_sql("SELECT orpha_code, parent_name FROM classifications", con).drop_duplicates("orpha_code").set_index("orpha_code")

    def clin_year(ids):
        ys = [omap[m.strip()] for m in (ids or "").split(",") if m.strip() in omap]
        return min(ys) if ys else None

    def n_hpo(orpha, ids):
        keys = [f"ORPHA:{orpha}"] + [f"OMIM:{m.strip()}" for m in (ids or "").split(",") if m.strip()]
        ph = ",".join("?" * len(keys))
        return con.execute(f"SELECT COUNT(DISTINCT hpo_id) FROM hpo_annotations WHERE database_id IN ({ph})", keys).fetchone()[0]

    def refs(ids):
        vs = [nref[m.strip()] for m in (ids or "").split(",") if m.strip() in nref]
        return max(vs) if vs else None

    rows = []
    for _, r in dis.iterrows():
        cy = clin_year(r.omim_ids)
        if cy is None or cy < 1946:
            continue
        inh = epi.loc[r.orpha_code, "inheritance"] if r.orpha_code in epi.index else None
        onset = epi.loc[r.orpha_code, "age_of_onset"] if r.orpha_code in epi.index else None
        klass = cls.loc[r.orpha_code, "parent_name"] if r.orpha_code in cls.index else None
        nh = n_hpo(r.orpha_code, r.omim_ids)
        mim = (r.omim_ids or "").split(",")[0].strip()
        rows.append({
            "orpha_code": r.orpha_code, "disease": r["name"], "class": klass,
            "first_described": int(cy), "years_waiting": SNAPSHOT_YEAR - int(cy),
            "inheritance": inh, "age_of_onset": onset,
            "n_HPO_terms": nh, "n_OMIM_refs": refs(r.omim_ids),
            "OMIM": f"https://omim.org/entry/{mim}" if mim else "",
            "Orphanet": f"https://www.orpha.net/en/disease/detail/{r.orpha_code}",
            "suggested_gene_discovery_approach": strategy(inh, nh),
        })
    con.close()

    df = pd.DataFrame(rows).sort_values("years_waiting", ascending=False).head(50).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    notes = pd.DataFrame({"PhenoLag — candidate gene-discovery / curation worklist": [
        "Definition: Orphanet 'Disease' with NO causative gene in our curated sources",
        "(G2P / ClinGen / HPO gene-disease links), clinically described in the MEDLINE era (>=1946),",
        "ranked by years waiting (snapshot year = %d)." % SNAPSHOT_YEAR,
        "",
        "*** READ BEFORE USE — this is a SCREEN, not a verified list of geneless diseases ***",
        "'No gene' here means 'no gene in OUR curated catalogue'. The list therefore mixes:",
        "  (a) genuinely unsolved diseases (true gene-discovery targets), AND",
        "  (b) diseases whose gene IS already known but is not linked in our sources — a curation gap",
        "      (e.g. dystrophic epidermolysis bullosa -> COL7A1; familial hypocalciuric hypercalcemia -> CASR;",
        "       leukoencephalopathy with spheroids -> CSF1R). These are false positives for 'no gene'.",
        "  (c) a few with very recent or non-nuclear (mtDNA) genes not yet captured.",
        "Triage every row against OMIM/GeneReviews before acting. Both outcomes are actionable:",
        "find the gene (a) OR fix the curation link (b).",
        "",
        "Clinical year = earliest OMIM-cited PMID (literature-based proxy).",
        "years_waiting = %d - first_described; n_HPO_terms = phenotype richness (matching power);" % SNAPSHOT_YEAR,
        "n_OMIM_refs = literature volume; suggested approach derived from reported inheritance.",
    ]})

    out = ROOT / "forgotten_diseases_top50.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="forgotten_top50", index=False)
        notes.to_excel(xl, sheet_name="notes", index=False)
    print(f"candidates with datable year: {len(rows)}; wrote top 50 -> {out}")
    print(df[["rank", "disease", "first_described", "years_waiting", "inheritance"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
