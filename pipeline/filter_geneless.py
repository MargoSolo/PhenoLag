"""Filter the 'forgotten' candidate list down to GENUINELY geneless diseases by
cross-checking OMIM: for each disease's OMIM phenotype id, NCBI elink (omim -> gene)
tells whether OMIM links a causative gene. If any linked gene exists, the disease has
a known molecular basis (curation gap, not gene-absence) -> drop it.

Caches results in omim_gene_cache. Rebuilds forgotten_diseases_top50.xlsx ->
forgotten_diseases_GENELESS_top50.xlsx (verified subset).

Run: PYTHONPATH=. python -m pipeline.filter_geneless
"""
from __future__ import annotations
import time
import requests
import sqlite3
import pandas as pd
from pipeline.config import DB_PATH, NCBI_API_KEY, NCBI_BASE, SNAPSHOT_YEAR
from pipeline.build_forgotten_list import strategy

ROOT = DB_PATH.parent


def elink_omim_gene(mims, retries=4):
    params = [("dbfrom", "omim"), ("db", "gene"), ("cmd", "neighbor"), ("retmode", "json")]
    for m in mims:
        params.append(("id", m))
    if NCBI_API_KEY:
        params.append(("api_key", NCBI_API_KEY))
    for a in range(retries):
        try:
            r = requests.get(f"{NCBI_BASE}/elink.fcgi", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if a == retries - 1:
                print("  [WARN] elink omim->gene failed:", e)
                return set(), {}
            time.sleep(0.5 * 2 ** a)
    queried = set()   # source ids that came back (so a 0 means true 0, not a failure)
    out = {}
    for ls in data.get("linksets", []):
        ids = ls.get("ids", [])
        if not ids:
            continue
        src = str(ids[0]); queried.add(src)
        genes = []
        for ldb in ls.get("linksetdbs", []):
            if "gene" in (ldb.get("linkname") or ""):
                genes += [str(g) for g in ldb.get("links", [])]
        out[src] = genes
    return queried, out


def main():
    con = sqlite3.connect(str(DB_PATH))
    con.execute("CREATE TABLE IF NOT EXISTS omim_gene_cache (omim_id TEXT PRIMARY KEY, n_genes INTEGER)")
    con.commit()
    dis = pd.read_sql("""SELECT orpha_code, name, omim_ids FROM disorders
        WHERE disorder_type='Disease' AND orpha_code NOT IN (SELECT orpha_code FROM disorder_genes)""", con)
    omap = dict(con.execute("SELECT omim_id, earliest_year FROM omim_clinical_cache WHERE earliest_year IS NOT NULL").fetchall())

    # candidate OMIMs (datable, geneless-in-our-sources)
    cand_mims = set()
    for _, r in dis.iterrows():
        for m in (r.omim_ids or "").split(","):
            m = m.strip()
            if m in omap and omap[m] >= 1946:
                cand_mims.add(m)
    cached = {r[0] for r in con.execute("SELECT omim_id FROM omim_gene_cache")}
    todo = sorted(cand_mims - cached)
    print(f"[GENELESS] {len(cand_mims)} candidate OMIMs, {len(todo)} to elink (omim->gene)")
    fails = 0
    for i in range(0, len(todo), 20):
        queried, res = elink_omim_gene(todo[i:i + 20])
        for m in todo[i:i + 20]:
            if m in queried:   # only cache successfully-queried ids (0 = true no-gene)
                con.execute("INSERT OR REPLACE INTO omim_gene_cache VALUES (?,?)", (m, len(res.get(m, []))))
            else:
                fails += 1     # leave uncached -> retried on next run
        if (i // 20) % 15 == 0:
            print(f"  ... {min(i+20,len(todo))}/{len(todo)}")
        con.commit()
        time.sleep(0.2)
    if fails:
        print(f"  {fails} OMIMs not returned (left uncached for retry); re-run to complete")

    has_gene = {r[0]: r[1] for r in con.execute("SELECT omim_id, n_genes FROM omim_gene_cache")}
    epi = pd.read_sql("SELECT orpha_code, inheritance, age_of_onset FROM epidemiology", con).set_index("orpha_code")
    cls = pd.read_sql("SELECT orpha_code, parent_name FROM classifications", con).drop_duplicates("orpha_code").set_index("orpha_code")
    nref = dict(con.execute("SELECT omim_id, n_omim_refs FROM omim_clinical_cache WHERE n_omim_refs IS NOT NULL").fetchall())

    def n_hpo(orpha, ids):
        keys = [f"ORPHA:{orpha}"] + [f"OMIM:{m.strip()}" for m in (ids or "").split(",") if m.strip()]
        ph = ",".join("?" * len(keys))
        return con.execute(f"SELECT COUNT(DISTINCT hpo_id) FROM hpo_annotations WHERE database_id IN ({ph})", keys).fetchone()[0]

    rows = []
    for _, r in dis.iterrows():
        mims = [m.strip() for m in (r.omim_ids or "").split(",") if m.strip()]
        datable = [m for m in mims if m in omap and omap[m] >= 1946]
        if not datable:
            continue
        # GENUINELY geneless: NONE of its OMIM phenotype ids links a gene
        if any(has_gene.get(m, 0) > 0 for m in mims):
            continue
        inh = epi.loc[r.orpha_code, "inheritance"] if r.orpha_code in epi.index else None
        # Mendelian only: must report AR / AD / X-linked (drop multifactorial / None / unknown)
        if not (isinstance(inh, str) and ("Autosomal" in inh or "X-linked" in inh)):
            continue
        cy = min(omap[m] for m in datable)
        nh = n_hpo(r.orpha_code, r.omim_ids)
        rows.append({
            "orpha_code": r.orpha_code, "disease": r["name"],
            "class": cls.loc[r.orpha_code, "parent_name"] if r.orpha_code in cls.index else None,
            "first_described": int(cy), "years_waiting": SNAPSHOT_YEAR - int(cy),
            "inheritance": inh,
            "age_of_onset": epi.loc[r.orpha_code, "age_of_onset"] if r.orpha_code in epi.index else None,
            "n_HPO_terms": nh,
            "n_OMIM_refs": max([nref[m] for m in datable if m in nref], default=None),
            "OMIM": f"https://omim.org/entry/{datable[0]}",
            "Orphanet": f"https://www.orpha.net/en/disease/detail/{r.orpha_code}",
            "suggested_gene_discovery_approach": strategy(inh, nh),
        })
    con.close()

    df = pd.DataFrame(rows).sort_values("years_waiting", ascending=False).head(50).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    notes = pd.DataFrame({"PhenoLag — Mendelian geneless rare diseases (gene-discovery worklist)": [
        "Orphanet 'Disease' meeting ALL of:",
        "  - geneless in our curated sources (G2P / ClinGen / HPO), AND",
        "  - NO causative gene linked in OMIM (verified by NCBI elink omim->gene) -> removes curation-gap",
        "    false positives (diseases whose gene is already known, e.g. SLC26A3, COL7A1, CASR), AND",
        "  - reported Mendelian inheritance (autosomal recessive / dominant / X-linked) -> removes",
        "    multifactorial / complex / unknown-inheritance entries that are not single-gene targets.",
        "Clinically described >=1946, ranked by years waiting (snapshot %d)." % SNAPSHOT_YEAR,
        "",
        "These are the cleanest single-gene discovery targets. Still verify each against current OMIM/GeneReviews:",
        "OMIM links lag the literature; a few may have very recent or non-nuclear (mtDNA) genes.",
    ]})
    out = ROOT / "forgotten_diseases_GENELESS_top50.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="geneless_top50", index=False)
        notes.to_excel(xl, sheet_name="notes", index=False)
    print(f"verified geneless candidates: {len(rows)}; wrote top 50 -> {out}")
    print(df[["rank", "disease", "first_described", "years_waiting", "inheritance"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
