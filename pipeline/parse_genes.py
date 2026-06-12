import csv
import pandas as pd
from pipeline.config import SOURCES
from pipeline.db import get_conn


def parse_hgnc():
    """Parse hgnc_complete_set.txt -> genes table."""
    print("[PARSE] hgnc_complete_set.txt ...")
    df = pd.read_csv(SOURCES / "hgnc_complete_set.txt", sep="\t", low_memory=False)
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("hgnc_id", "")),
            str(r.get("symbol", "")),
            str(r.get("name", "")) if pd.notna(r.get("name")) else None,
            str(r.get("locus_group", "")) if pd.notna(r.get("locus_group")) else None,
            str(r.get("locus_type", "")) if pd.notna(r.get("locus_type")) else None,
            str(r.get("status", "")) if pd.notna(r.get("status")) else None,
            str(r.get("location", "")) if pd.notna(r.get("location")) else None,
            str(r.get("alias_symbol", "")) if pd.notna(r.get("alias_symbol")) else None,
            str(r.get("prev_symbol", "")) if pd.notna(r.get("prev_symbol")) else None,
            str(r.get("entrez_id", "")) if pd.notna(r.get("entrez_id")) else None,
            str(r.get("ensembl_gene_id", "")) if pd.notna(r.get("ensembl_gene_id")) else None,
            str(r.get("omim_id", "")) if pd.notna(r.get("omim_id")) else None,
            str(r.get("uniprot_ids", "")) if pd.notna(r.get("uniprot_ids")) else None,
        ))
    conn = get_conn()
    conn.executemany("INSERT OR REPLACE INTO genes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} genes")


def parse_phenotype_to_genes():
    """Parse phenotype_to_genes.txt -> disorder_genes for ORPHA diseases."""
    print("[PARSE] phenotype_to_genes.txt ...")
    seen = set()
    rows = []
    with open(SOURCES / "phenotype_to_genes.txt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("hpo_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            # hpo_id, hpo_name, ncbi_gene_id, gene_symbol, disease_id
            gene_symbol = parts[3]
            disease_id = parts[4]
            ncbi_gene_id = parts[2]
            if disease_id.startswith("ORPHA:"):
                orpha = disease_id.replace("ORPHA:", "")
                key = (orpha, gene_symbol)
                if key not in seen:
                    seen.add(key)
                    rows.append((orpha, gene_symbol, ncbi_gene_id, "HPO"))

    conn = get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO disorder_genes VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} disease-gene links (ORPHA)")


def parse_g2p():
    """Parse G2P CSV -> g2p table."""
    print("[PARSE] G2P_all_2026-02-07.csv ...")
    rows = []
    with open(SOURCES / "G2P_all_2026-02-07.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pubs = r.get("publications", "")
            rows.append((
                r.get("g2p id", ""),
                r.get("gene symbol", ""),
                r.get("gene mim", "") or None,
                r.get("hgnc id", "") or None,
                r.get("disease name", ""),
                r.get("disease mim", "") or None,
                r.get("disease MONDO", "") or None,
                r.get("allelic requirement", "") or None,
                r.get("confidence", "") or None,
                r.get("variant consequence", "") or None,
                r.get("molecular mechanism", "") or None,
                r.get("panel", "") or None,
                pubs.replace("; ", ";") if pubs else None,
                r.get("date of last review", "") or None,
            ))

    conn = get_conn()
    conn.executemany("INSERT OR REPLACE INTO g2p VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} G2P entries")


def promote_g2p_to_disorder_genes():
    """Create disorder_genes rows with source='G2P' by mapping G2P disease_mim
    or disease_mondo back to ORPHA via the disorders table."""
    print("[PARSE] promoting G2P → disorder_genes ...")
    conn = get_conn()
    cur = conn.cursor()

    # Build MIM -> ORPHA index (one OMIM can map to several ORPHAs; keep all)
    mim_to_orpha: dict[str, list[str]] = {}
    cur.execute("SELECT orpha_code, omim_ids FROM disorders WHERE omim_ids IS NOT NULL")
    for orpha, omim_ids in cur.fetchall():
        for mim in omim_ids.split(","):
            mim = mim.strip()
            if mim:
                mim_to_orpha.setdefault(mim, []).append(orpha)

    mondo_to_orpha: dict[str, list[str]] = {}
    cur.execute("SELECT orpha_code, mondo_id FROM disorders WHERE mondo_id IS NOT NULL")
    for orpha, mondo in cur.fetchall():
        mondo_to_orpha.setdefault(mondo.strip(), []).append(orpha)

    cur.execute("SELECT gene_symbol, disease_mim, disease_mondo FROM g2p")
    rows = []
    unmatched = 0
    for gene, mim, mondo in cur.fetchall():
        orphas: list[str] = []
        if mim and mim.strip() in mim_to_orpha:
            orphas.extend(mim_to_orpha[mim.strip()])
        if mondo:
            key = mondo.strip()
            if not key.startswith("MONDO:") and key.isdigit():
                key = f"MONDO:{key}"
            if key in mondo_to_orpha:
                orphas.extend(mondo_to_orpha[key])
        if not orphas:
            unmatched += 1
            continue
        for orpha in set(orphas):
            rows.append((orpha, gene, None, "G2P"))

    conn.executemany(
        "INSERT OR IGNORE INTO disorder_genes VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} G2P→ORPHA links ({unmatched} G2P entries without ORPHA match)")


def parse_clingen():
    """Parse ClinGen_gene_curation_list_GRCh38.tsv.

    ClinGen lists per-gene haploinsufficiency/triplosensitivity. We promote to
    disorder_genes when a MONDO disease ID is present AND the gene has evidence
    (Haploinsufficiency Score >= 1 or Triplosensitivity Score >= 1, i.e. not
    'No evidence available' (0) / 'Not yet evaluated' (empty)).
    """
    print("[PARSE] ClinGen_gene_curation_list_GRCh38.tsv ...")
    path = SOURCES / "ClinGen_gene_curation_list_GRCh38.tsv"

    # Build MONDO -> ORPHA lookup
    conn = get_conn()
    cur = conn.cursor()
    mondo_to_orpha: dict[str, list[str]] = {}
    cur.execute("SELECT orpha_code, mondo_id FROM disorders WHERE mondo_id IS NOT NULL")
    for orpha, mondo in cur.fetchall():
        mondo_to_orpha.setdefault(mondo.strip(), []).append(orpha)

    rows = []
    n_rows = 0
    n_unmatched = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 23:
                continue
            gene = parts[0].strip()
            if not gene:
                continue
            # Columns (0-indexed): 0=Gene Symbol, 4=HI Score, 12=TS Score,
            # 20=Date Last Evaluated, 21=HI Disease ID, 22=TS Disease ID
            hi_score = parts[4].strip()
            ts_score = parts[12].strip()
            hi_disease = parts[21].strip()
            ts_disease = parts[22].strip()

            # Only keep rows with at least one MONDO disease AND an evidenced score
            def _score_ok(s: str) -> bool:
                try:
                    return int(s) >= 1
                except ValueError:
                    return False

            candidates: list[str] = []
            if hi_disease and _score_ok(hi_score):
                candidates.append(hi_disease)
            if ts_disease and _score_ok(ts_score):
                candidates.append(ts_disease)
            if not candidates:
                continue
            n_rows += 1

            mapped = False
            for mondo in candidates:
                key = mondo.strip()
                if not key.startswith("MONDO:"):
                    continue
                if key in mondo_to_orpha:
                    for orpha in set(mondo_to_orpha[key]):
                        rows.append((orpha, gene, None, "ClinGen"))
                    mapped = True
            if not mapped:
                n_unmatched += 1

    conn.executemany(
        "INSERT OR IGNORE INTO disorder_genes VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} ClinGen→ORPHA links "
          f"({n_rows} evidenced ClinGen rows, {n_unmatched} without ORPHA match)")


def parse_gnomad():
    """Parse gnomAD constraint metrics (parquet) -> constraint_metrics."""
    print("[PARSE] gnomad.v4.1.constraint_metrics.parquet ...")
    df = pd.read_parquet(SOURCES / "gnomad.v4.1.constraint_metrics.parquet")

    # Keep only canonical MANE transcripts where possible
    canon = df[df["canonical"] == True].copy()
    if len(canon) == 0:
        canon = df.copy()

    rows = []
    for _, r in canon.iterrows():
        def g(col):
            v = r.get(col)
            if pd.isna(v):
                return None
            return float(v) if isinstance(v, (int, float)) else v

        rows.append((
            str(r.get("gene", "")),
            str(r.get("gene_id", "")),
            str(r.get("transcript", "")),
            str(r.get("canonical", "")),
            str(r.get("mane_select", "")),
            g("lof.obs"), g("lof.exp"), g("lof.oe"),
            g("lof.oe_ci.lower"), g("lof.oe_ci.upper"),
            g("lof.pLI"),
            g("mis.obs"), g("mis.exp"), g("mis.oe"), g("mis.z_raw"),
            g("syn.obs"), g("syn.exp"), g("syn.oe"), g("syn.z_raw"),
        ))

    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO constraint_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} constraint records")


def parse_hpoa():
    """Parse phenotype.hpoa -> hpo_annotations."""
    print("[PARSE] phenotype.hpoa ...")
    rows = []
    with open(SOURCES / "phenotype.hpoa") as f:
        for line in f:
            if line.startswith("#") or line.startswith("database_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            rows.append((
                parts[0],   # database_id
                parts[1],   # disease_name
                parts[2],   # qualifier
                parts[3],   # hpo_id
                parts[4],   # reference
                parts[5],   # evidence
                parts[6],   # onset
                parts[7],   # frequency
                parts[10],  # aspect
            ))

    conn = get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO hpo_annotations VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} HPO annotations")


def run_all():
    parse_hgnc()
    parse_g2p()
    parse_phenotype_to_genes()
    promote_g2p_to_disorder_genes()
    parse_clingen()
    parse_gnomad()
    parse_hpoa()
