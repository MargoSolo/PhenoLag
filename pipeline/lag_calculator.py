from __future__ import annotations
import re
from pipeline.db import get_conn
from pipeline.pubmed_dates import get_earliest_year, get_earliest_genetic_year, get_years


_MIM_YEAR_BREAKPOINTS = [
    (100000, 1966), (150000, 1975), (175000, 1980), (200000, 1985),
    (250000, 1990), (270000, 1994), (300000, 1966),
    (400000, 1992), (600000, 1994), (605000, 1998), (607000, 2000),
    (609000, 2003), (610000, 2005), (612000, 2008), (614000, 2011),
    (616000, 2014), (617000, 2016), (618000, 2018), (619000, 2020),
    (620000, 2022), (621000, 2024),
]

MEDLINE_CUTOFF = 1946
CORE_SOURCES = {"G2P", "ClinGen"}
_CONF_RANK = {"definitive": 4, "strong": 3, "moderate": 2, "limited": 1, "disputed": 0}


def _mim_to_approx_year(mim_str):
    try:
        mim = int(mim_str.strip())
    except (ValueError, AttributeError):
        return None
    for i in range(len(_MIM_YEAR_BREAKPOINTS) - 1, -1, -1):
        if mim >= _MIM_YEAR_BREAKPOINTS[i][0]:
            return _MIM_YEAR_BREAKPOINTS[i][1]
    return None


def _extract_pmids(pubs_str):
    if not pubs_str:
        return set()
    return {p.strip() for p in re.split(r"[;,\s]+", pubs_str) if p.strip().isdigit()}


def _collect_all_pmids(conn):
    cur = conn.cursor()
    cur.execute("SELECT publications FROM g2p WHERE publications IS NOT NULL")
    g2p_pmids = set()
    for (pubs,) in cur.fetchall():
        g2p_pmids |= _extract_pmids(pubs)

    cur.execute("SELECT DISTINCT reference FROM hpo_annotations WHERE reference LIKE 'PMID:%'")
    hpo_pmids = set()
    for (ref,) in cur.fetchall():
        pmid = ref.replace("PMID:", "").strip()
        if pmid.isdigit():
            hpo_pmids.add(pmid)
    return g2p_pmids, hpo_pmids


def prefetch_pmids():
    conn = get_conn()
    g2p_pmids, hpo_pmids = _collect_all_pmids(conn)
    all_pmids = list(g2p_pmids | hpo_pmids)
    conn.close()

    print(f"[LAG] Total unique PMIDs to fetch: {len(all_pmids)} "
          f"(G2P: {len(g2p_pmids)}, HPO: {len(hpo_pmids)})")

    batch_size = 200
    for i in range(0, len(all_pmids), batch_size):
        batch = all_pmids[i:i + batch_size]
        get_years(batch)
        if (i // batch_size) % 50 == 0:
            print(f"  ... cached {min(i + batch_size, len(all_pmids))}/{len(all_pmids)}")
    print(f"[LAG] PMID cache warmed")


def _flag_suspect(clin_year, gen_year, clin_source, gen_pre_molecular=False,
                  clin_pmid=None, gen_pmid=None):
    """Return (suspect_dating: int, reason: str or None)."""
    reasons = []
    if clin_source == "MIM_breakpoint":
        reasons.append("mim_breakpoint")
    if clin_source == "HPO_fallback":
        reasons.append("codiscovery")
    # Same-PMID codiscovery: regardless of clin_source, if the same
    # paper became both the "earliest clinical" and the "earliest
    # genetic" PMID, the lag is an artefact of one paper describing
    # both phenotype and mutation simultaneously.  These slipped past
    # the HPO_fallback check when OMIM_elink happened to surface the
    # same PMID first; flag them all as codiscovery.
    if (clin_pmid is not None and gen_pmid is not None
            and str(clin_pmid).strip() and str(clin_pmid) == str(gen_pmid)):
        if "codiscovery" not in reasons:
            reasons.append("codiscovery")
    if clin_year is not None and gen_year is not None and clin_year > gen_year:
        reasons.append("neg_lag")
    if clin_year is not None and clin_year < MEDLINE_CUTOFF:
        reasons.append("pre_medline")
    # Gene verification year fell back to a pre-1970 paper because all
    # G2P publications for this gene predate the recombinant-DNA era —
    # almost always a sign that the "earliest" PMID is biochemistry, not
    # gene–disease verification.
    if gen_pre_molecular:
        reasons.append("gen_pre_molecular")
    if reasons:
        return 1, ",".join(reasons)
    return 0, None


def collect_hpo_pmids(cur, orpha, omim_ids_str):
    """All HPO/ORPHA-cited PMIDs for a disease (ORPHA:<code> + OMIM:<mim> keys).
    Single source of truth — used by the genetic-date fallback and analysis scripts.
    """
    pmids = set()
    keys = [f"ORPHA:{orpha}"]
    if omim_ids_str:
        keys += [f"OMIM:{m.strip()}" for m in omim_ids_str.split(",") if m.strip()]
    for k in keys:
        cur.execute(
            "SELECT DISTINCT reference FROM hpo_annotations "
            "WHERE database_id = ? AND reference LIKE 'PMID:%'", (k,))
        for (ref,) in cur.fetchall():
            p = ref.replace("PMID:", "").strip()
            if p.isdigit():
                pmids.add(p)
    return pmids


def enforce_lag_integrity(conn):
    """Data-integrity guard (idempotent): keep lag_years == gen - clin and ensure
    every gen < clin row is flagged neg_lag. Prevents the stale-lag / unflagged
    negative-lag drift that creeps in after manual curation edits to year columns.
    """
    cur = conn.cursor()
    cur.execute(
        "UPDATE phenolag SET lag_years = first_genetic_year - first_clinical_year "
        "WHERE first_genetic_year IS NOT NULL AND first_clinical_year IS NOT NULL "
        "AND (lag_years IS NULL OR lag_years != first_genetic_year - first_clinical_year)")
    n_lag = cur.rowcount
    cur.execute(
        "UPDATE phenolag SET suspect_dating = 1, "
        "suspect_reason = CASE "
        "  WHEN suspect_reason IS NULL OR suspect_reason = '' THEN 'neg_lag' "
        "  WHEN suspect_reason LIKE '%neg_lag%' THEN suspect_reason "
        "  ELSE suspect_reason || ',neg_lag' END "
        "WHERE first_genetic_year < first_clinical_year "
        "AND (suspect_dating = 0 OR suspect_reason NOT LIKE '%neg_lag%')")
    n_flag = cur.rowcount
    conn.commit()
    if n_lag or n_flag:
        print(f"[LAG] integrity: recomputed {n_lag} lag_years, flagged {n_flag} neg_lag rows")
    return n_lag, n_flag


def calculate_lag():
    """For each ORPHA disorder with gene links, compute lag and record flags."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT d.orpha_code, d.name, d.omim_ids
        FROM disorders d
        JOIN disorder_genes dg ON d.orpha_code = dg.orpha_code
        WHERE d.disorder_type = 'Disease'
    """)
    disorders = cur.fetchall()
    print(f"[LAG] Processing {len(disorders)} disorders with gene links...")

    results = []
    for orpha, name, omim_ids_str in disorders:
        cur.execute(
            "SELECT DISTINCT gene_symbol, source FROM disorder_genes WHERE orpha_code = ?",
            (orpha,)
        )
        rows = cur.fetchall()
        genes = sorted({r[0] for r in rows})
        sources_for_disease = {r[1] for r in rows if r[1]}
        gene_sources_str = ",".join(sorted(sources_for_disease)) if sources_for_disease else None
        in_core = 1 if sources_for_disease & CORE_SOURCES else 0

        # --- Per-gene genetic verification ---
        gene_candidates = []
        for gene in genes:
            gene_pmids = set()
            gene_conf = None
            if omim_ids_str:
                for omim_id in omim_ids_str.split(","):
                    cur.execute(
                        "SELECT publications, confidence FROM g2p WHERE gene_symbol = ? AND disease_mim = ?",
                        (gene, omim_id.strip())
                    )
                    for pubs, conf in cur.fetchall():
                        gene_pmids |= _extract_pmids(pubs)
                        if conf and (gene_conf is None or _CONF_RANK.get(conf, 0) > _CONF_RANK.get(gene_conf, 0)):
                            gene_conf = conf

            if not gene_pmids:
                cur.execute(
                    "SELECT publications, confidence FROM g2p WHERE gene_symbol = ?",
                    (gene,)
                )
                for pubs, conf in cur.fetchall():
                    gene_pmids |= _extract_pmids(pubs)
                    if conf and (gene_conf is None or _CONF_RANK.get(conf, 0) > _CONF_RANK.get(gene_conf, 0)):
                        gene_conf = conf

            if gene_pmids:
                # MeSH-aware genetic-era filter: prefer earliest PMID
                # whose MeSH descriptors mark it as a molecular-genetic
                # paper (cloning, mutation, linkage); fall back to
                # year >= 1980 floor; last resort is absolute min.
                gen_pmid, gen_year, gen_status = get_earliest_genetic_year(
                    list(gene_pmids)
                )
                gene_candidates.append((
                    gene,
                    _CONF_RANK.get(gene_conf, -1),
                    gen_pmid, gen_year,
                    gene_pmids,
                    gene_conf,
                    gen_status,
                ))

        gen_status = ""
        if gene_candidates:
            gene_candidates.sort(key=lambda x: (-x[1], x[3] or 9999))
            best = gene_candidates[0]
            (best_gene, _, gen_pmid, gen_year, genetic_pmids,
             confidence, gen_status) = best
        else:
            best_gene = genes[0] if genes else None
            gen_pmid, gen_year = None, None
            genetic_pmids = set()
            confidence = None
        gen_pre_molecular = (gen_status == "pre_molecular")

        # --- Clinical description year ---
        clin_pmid, clin_year, clin_source = None, None, None

        # Source 1: OMIM elink (preferred)
        if omim_ids_str:
            for omim_id in omim_ids_str.split(","):
                cur.execute(
                    "SELECT earliest_pmid, earliest_year FROM omim_clinical_cache "
                    "WHERE omim_id = ? AND earliest_year IS NOT NULL",
                    (omim_id.strip(),)
                )
                row = cur.fetchone()
                if row and row[1] is not None:
                    if clin_year is None or row[1] < clin_year:
                        clin_pmid, clin_year = row[0], row[1]
                        clin_source = "OMIM_elink"

        # Source 2/3: HPO PMIDs
        if clin_year is None:
            all_hpo_pmids = set()
            if omim_ids_str:
                for omim_id in omim_ids_str.split(","):
                    cur.execute(
                        "SELECT DISTINCT reference FROM hpo_annotations "
                        "WHERE database_id = ? AND reference LIKE 'PMID:%'",
                        (f"OMIM:{omim_id.strip()}",)
                    )
                    for (ref,) in cur.fetchall():
                        pmid = ref.replace("PMID:", "").strip()
                        if pmid.isdigit():
                            all_hpo_pmids.add(pmid)
            cur.execute(
                "SELECT DISTINCT reference FROM hpo_annotations "
                "WHERE database_id = ? AND reference LIKE 'PMID:%'",
                (f"ORPHA:{orpha}",)
            )
            for (ref,) in cur.fetchall():
                pmid = ref.replace("PMID:", "").strip()
                if pmid.isdigit():
                    all_hpo_pmids.add(pmid)

            clinical_pmids = all_hpo_pmids - genetic_pmids
            if clinical_pmids:
                clin_pmid, clin_year = get_earliest_year(list(clinical_pmids))
                if clin_year is not None:
                    clin_source = "HPO_non_g2p"
            if clin_year is None and all_hpo_pmids:
                clin_pmid, clin_year = get_earliest_year(list(all_hpo_pmids))
                if clin_year is not None:
                    clin_source = "HPO_fallback"

        # Source 4: MIM breakpoint
        if clin_year is None and omim_ids_str:
            for omim_id in omim_ids_str.split(","):
                mim_year = _mim_to_approx_year(omim_id)
                if mim_year:
                    clin_year = mim_year
                    clin_pmid = f"MIM:{omim_id.strip()}"
                    clin_source = "MIM_breakpoint"
                    break

        # --- Genetic-date fallback: union of HPO/ORPHA-cited PMIDs ---
        # Most HPO-tier diseases have no G2P publication list, so the gene
        # has no datable "first molecular paper" and would be dropped.
        # Recover one from HPO/ORPHA-cited PMIDs using the SAME molecular-
        # genetics MeSH filter used for G2P. Validated against G2P dating
        # on 403 overlapping diseases: 61% identical, median diff 0y,
        # mean +0.7y — no systematic bias (see pipeline/union_analysis.py).
        if gen_year is None:
            fb_pmids = collect_hpo_pmids(cur, orpha, omim_ids_str)
            if fb_pmids:
                fb_pmid, fb_year, fb_status = get_earliest_genetic_year(list(fb_pmids))
                if fb_year is not None:
                    gen_pmid, gen_year = fb_pmid, fb_year
                    gen_status = fb_status
                    gen_pre_molecular = (fb_status == "pre_molecular")

        lag = None
        if clin_year is not None and gen_year is not None:
            lag = gen_year - clin_year

        suspect, suspect_reason = _flag_suspect(
            clin_year, gen_year, clin_source, gen_pre_molecular,
            clin_pmid=clin_pmid, gen_pmid=gen_pmid,
        )

        # --- Enrichment ---
        cur.execute("SELECT inheritance FROM epidemiology WHERE orpha_code = ?", (orpha,))
        row = cur.fetchone()
        inheritance = row[0] if row else None

        loeuf, pli = None, None
        if best_gene:
            cur.execute(
                "SELECT lof_oe_lower, lof_pli FROM constraint_metrics WHERE gene_symbol = ? LIMIT 1",
                (best_gene,)
            )
            row = cur.fetchone()
            if row:
                loeuf, pli = row

        if confidence is None:
            for gene in genes:
                cur.execute(
                    "SELECT confidence FROM g2p WHERE gene_symbol = ? AND confidence IS NOT NULL LIMIT 1",
                    (gene,)
                )
                row = cur.fetchone()
                if row:
                    confidence = row[0]
                    break

        # n_hpo_terms: phenotypic complexity of the *gene-specific* form
        # of the disease, not of the whole umbrella diagnosis.
        #
        # Resolution priority:
        #   1. OMIM matching `best_gene` via G2P `disease_mim` (most
        #      specific gene-disease pair we have on this row).
        #   2. ORPHA-level annotations.
        #   3. UNION across ORPHA + all linked OMIMs (legacy fallback).
        n_hpo = 0
        gene_specific_mims: list[str] = []
        if best_gene and omim_ids_str:
            disorder_mims = {m.strip() for m in omim_ids_str.split(",") if m.strip()}
            cur.execute(
                "SELECT DISTINCT disease_mim FROM g2p "
                "WHERE gene_symbol = ? AND disease_mim IS NOT NULL AND disease_mim != ''",
                (best_gene,),
            )
            for (mim,) in cur.fetchall():
                m = (mim or "").strip()
                if m and m in disorder_mims:
                    gene_specific_mims.append(m)

        if gene_specific_mims:
            placeholders = ",".join(["?"] * len(gene_specific_mims))
            cur.execute(
                f"SELECT COUNT(DISTINCT hpo_id) FROM hpo_annotations "
                f"WHERE database_id IN ({placeholders})",
                [f"OMIM:{m}" for m in gene_specific_mims],
            )
            n_hpo = cur.fetchone()[0] or 0

        if n_hpo == 0:
            cur.execute(
                "SELECT COUNT(DISTINCT hpo_id) FROM hpo_annotations "
                "WHERE database_id = ?",
                (f"ORPHA:{orpha}",),
            )
            n_hpo = cur.fetchone()[0] or 0

        if n_hpo == 0 and omim_ids_str:
            ids = [f"ORPHA:{orpha}"] + [
                f"OMIM:{m.strip()}" for m in omim_ids_str.split(",") if m.strip()
            ]
            placeholders = ",".join(["?"] * len(ids))
            cur.execute(
                f"SELECT COUNT(DISTINCT hpo_id) FROM hpo_annotations "
                f"WHERE database_id IN ({placeholders})",
                ids,
            )
            n_hpo = cur.fetchone()[0] or 0

        cur.execute(
            "SELECT parent_name FROM classifications WHERE orpha_code = ? LIMIT 1",
            (orpha,)
        )
        row = cur.fetchone()
        disorder_class = row[0] if row else None

        results.append((
            orpha, name, omim_ids_str, best_gene,
            clin_pmid, clin_year, gen_pmid, gen_year, lag,
            inheritance, loeuf, pli, confidence, n_hpo, disorder_class,
            in_core, clin_source, suspect, suspect_reason, gene_sources_str,
        ))

    conn.executemany(
        """INSERT OR REPLACE INTO phenolag (
            orpha_code, name, omim_id, gene_symbol,
            first_clinical_pmid, first_clinical_year,
            first_genetic_pmid, first_genetic_year, lag_years,
            inheritance, loeuf, pli, confidence, n_hpo_terms, disorder_class,
            in_core_cohort, clin_source, suspect_dating, suspect_reason, gene_sources
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        results
    )
    conn.commit()
    enforce_lag_integrity(conn)

    n_total = len(results)
    n_core = sum(1 for r in results if r[15] == 1)
    with_lag = sum(1 for r in results if r[8] is not None)
    core_with_lag = sum(1 for r in results if r[15] == 1 and r[8] is not None)
    core_clean = sum(1 for r in results if r[15] == 1 and r[8] is not None and r[17] == 0)
    print(f"[LAG] Written {n_total} records. "
          f"Core cohort (G2P/ClinGen): {n_core}. "
          f"With lag: {with_lag} ({core_with_lag} in core). "
          f"Main sample (core, non-suspect, lag present): {core_clean}.")

    conn.close()
    return results
