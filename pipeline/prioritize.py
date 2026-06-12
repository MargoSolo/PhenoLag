"""Undiagnosed Disease Prioritization module.

Converts PhenoLag from a descriptive analysis into an actionable clinical tool.
Computes a composite priority score for each disease based on:

1. Diagnostic lag (longer = higher priority)
2. Severity — multi-source:
   - Orphanet age of onset (neonatal/infantile = highest)
   - HPO mortality annotations (HP:0011420-0011425, HP:0001522)
   - HPO progressive course (HP:0003676)
   - HPO seizures (HP:0001250), intellectual disability (HP:0001249)
3. Genetic tractability:
   - gnomAD pLI / LOEUF (haploinsufficiency = findable by WES/WGS)
   - ClinGen haploinsufficiency score
   - Number of associated genes
4. Evidence maturity (G2P confidence: limited = underserved)
5. Phenotypic richness (HPO term count — not just case reports)

Output: ranked list for Undiagnosed Diseases Networks, newborn genomic
screening panels, orphan drug prioritization, and exome reanalysis triage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline.db import get_conn
from pipeline.config import FIGURES, SOURCES


def _percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, na_option="keep")


# ── HPO severity terms ──────────────────────────────────────

# Mortality
HPO_DEATH = {
    "HP:0011420": 1.0,   # neonatal death
    "HP:0011421": 0.95,  # death in infancy
    "HP:0001522": 0.95,  # death in infancy (legacy)
    "HP:0011422": 0.85,  # death in childhood
    "HP:0011423": 0.70,  # death in adolescence
    "HP:0011424": 0.55,  # death in early adulthood
    "HP:0011425": 0.40,  # death in adulthood
}

# Clinical course modifiers
HPO_SEVERITY_FEATURES = {
    "HP:0003676": 0.3,   # progressive
    "HP:0001249": 0.2,   # intellectual disability
    "HP:0001250": 0.2,   # seizures
    "HP:0001252": 0.1,   # muscular hypotonia
    "HP:0001257": 0.1,   # spasticity
}


def _onset_score(onset_str):
    if pd.isna(onset_str) or onset_str == "No data available":
        return 0.3
    onset = onset_str.lower()
    if "antenatal" in onset or "neonatal" in onset:
        return 1.0
    elif "infancy" in onset:
        return 0.9
    elif "childhood" in onset:
        return 0.7
    elif "adolescent" in onset:
        return 0.5
    elif "adult" in onset:
        return 0.3
    elif "elderly" in onset:
        return 0.2
    elif "all ages" in onset:
        return 0.5
    return 0.3


def _confidence_need_score(conf):
    if pd.isna(conf):
        return 0.7
    return {"definitive": 0.2, "strong": 0.4, "moderate": 0.6,
            "limited": 0.9, "disputed": 0.5}.get(conf, 0.5)


def _load_clingen():
    """Load ClinGen haploinsufficiency scores."""
    path = SOURCES / "ClinGen_gene_curation_list_GRCh38.tsv"
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#Gene Symbol"):
                header = line.lstrip("#").strip().split("\t")
                continue
            if header is None or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                gene = parts[0]
                try:
                    hi_score = int(parts[4])
                except (ValueError, IndexError):
                    hi_score = None
                rows.append({"gene_symbol": gene, "clingen_hi_score": hi_score})
    return pd.DataFrame(rows)


def compute_priority_scores() -> pd.DataFrame:
    """Compute composite priority score for all diseases with lag data."""
    conn = get_conn()

    # Base data
    df = pd.read_sql("""
        SELECT p.*, e.age_of_onset
        FROM phenolag p
        LEFT JOIN epidemiology e ON p.orpha_code = e.orpha_code
    """, conn)

    df = df[df["lag_years"].notna()].copy()

    # ── HPO-based severity signals ──────────────────────────

    # Mortality: max death severity for each disease
    death_hpos = "','".join(HPO_DEATH.keys())
    mort_df = pd.read_sql(f"""
        SELECT database_id, hpo_id FROM hpo_annotations
        WHERE hpo_id IN ('{death_hpos}')
    """, conn)
    # Map to orpha
    mort_scores = {}
    for _, row in mort_df.iterrows():
        db_id = row["database_id"]
        score = HPO_DEATH.get(row["hpo_id"], 0)
        # Match to OMIM or ORPHA
        if db_id.startswith("OMIM:"):
            omim_num = db_id.replace("OMIM:", "")
            mort_scores.setdefault(("omim", omim_num), []).append(score)
        elif db_id.startswith("ORPHA:"):
            orpha_num = db_id.replace("ORPHA:", "")
            mort_scores.setdefault(("orpha", orpha_num), []).append(score)

    def _get_mortality(row):
        best = 0
        if pd.notna(row.get("omim_id")):
            for omim in str(row["omim_id"]).split(","):
                key = ("omim", omim.strip())
                if key in mort_scores:
                    best = max(best, max(mort_scores[key]))
        key = ("orpha", str(row["orpha_code"]))
        if key in mort_scores:
            best = max(best, max(mort_scores[key]))
        return best

    df["mortality_score"] = df.apply(_get_mortality, axis=1)

    # Severity features: progressive, seizures, ID
    feat_hpos = "','".join(HPO_SEVERITY_FEATURES.keys())
    feat_df = pd.read_sql(f"""
        SELECT database_id, hpo_id FROM hpo_annotations
        WHERE hpo_id IN ('{feat_hpos}')
    """, conn)
    feat_map = {}
    for _, row in feat_df.iterrows():
        db_id = row["database_id"]
        score = HPO_SEVERITY_FEATURES.get(row["hpo_id"], 0)
        if db_id.startswith("OMIM:"):
            feat_map.setdefault(("omim", db_id.replace("OMIM:", "")), set()).add(row["hpo_id"])
        elif db_id.startswith("ORPHA:"):
            feat_map.setdefault(("orpha", db_id.replace("ORPHA:", "")), set()).add(row["hpo_id"])

    def _get_feat_score(row):
        total = 0
        seen = set()
        if pd.notna(row.get("omim_id")):
            for omim in str(row["omim_id"]).split(","):
                key = ("omim", omim.strip())
                if key in feat_map:
                    seen |= feat_map[key]
        key = ("orpha", str(row["orpha_code"]))
        if key in feat_map:
            seen |= feat_map[key]
        for hpo_id in seen:
            total += HPO_SEVERITY_FEATURES.get(hpo_id, 0)
        return min(total, 1.0)  # cap at 1.0

    df["feature_severity"] = df.apply(_get_feat_score, axis=1)

    # ── Prevalence (Orphanet product9) ────────────────────────
    try:
        prev_df = pd.read_sql("""
            SELECT orpha_code,
                MAX(CASE WHEN prev_type = 'Cases/families' THEN n_cases END) as n_reported_cases,
                MAX(CASE WHEN prev_type = 'Point prevalence' THEN prev_class_numeric END) as prevalence_per_million,
                MAX(CASE WHEN prev_type = 'Point prevalence' THEN prev_class END) as prevalence_class
            FROM prevalence
            WHERE geographic = 'Worldwide' OR geographic = 'Europe'
            GROUP BY orpha_code
        """, conn)
        df = df.merge(prev_df, on="orpha_code", how="left")
    except Exception:
        df["n_reported_cases"] = None
        df["prevalence_per_million"] = None
        df["prevalence_class"] = None

    conn.close()

    # ── ClinGen haploinsufficiency ──────────────────────────
    clingen = _load_clingen()
    df = df.merge(clingen, on="gene_symbol", how="left")
    # ClinGen HI score: 3=sufficient evidence, 2=some, 1=little, 0=none, 30=AR, 40=dosage sensitivity unlikely
    df["clingen_hi_norm"] = df["clingen_hi_score"].apply(
        lambda x: {3: 1.0, 2: 0.7, 1: 0.4, 0: 0.1, 30: 0.3, 40: 0.1}.get(x, 0.2)
        if pd.notna(x) else 0.2
    )

    # ══════════════════════════════════════════════════════════
    # COMPONENT SCORES (all 0-1)
    # ══════════════════════════════════════════════════════════

    # 1. LAG SCORE
    df["lag_score"] = _percentile_rank(df["lag_years"])

    # 2. SEVERITY SCORE (multi-source composite)
    df["onset_score"] = df["age_of_onset"].apply(_onset_score)
    df["severity_score"] = (
        0.40 * df["onset_score"] +
        0.35 * df["mortality_score"] +
        0.25 * df["feature_severity"]
    ).clip(0, 1)

    # 3. GENETIC TRACTABILITY (gnomAD + ClinGen + gene count)
    df["pli_score"] = df["pli"].fillna(0).clip(0, 1)
    df["loeuf_score"] = 1 - df["loeuf"].fillna(1).clip(0, 1.5) / 1.5
    df["tractability_score"] = (
        0.40 * df["pli_score"] +
        0.30 * df["loeuf_score"] +
        0.30 * df["clingen_hi_norm"]
    ).clip(0, 1)

    # 4. UNMET NEED
    df["need_score"] = df["confidence"].apply(_confidence_need_score)

    # 5. EVIDENCE DEPTH + PATIENT VOLUME
    df["n_hpo_log"] = np.log1p(df["n_hpo_terms"])
    df["hpo_depth_score"] = _percentile_rank(df["n_hpo_log"]).fillna(0.3)

    # Prevalence score: more patients = more impact from genomic diagnostics
    # Log-scale prevalence ranking; ultra-rare (< 1/million) gets lower score
    df["prevalence_score"] = df["prevalence_per_million"].apply(
        lambda x: min(np.log1p(x) / np.log1p(5000), 1.0) if pd.notna(x) and x > 0 else 0.3
    )
    # Case count bonus: diseases with > 10 cases have enough evidence to act
    df["case_volume_score"] = df["n_reported_cases"].apply(
        lambda x: min(np.log1p(x) / np.log1p(1000), 1.0) if pd.notna(x) and x > 0 else 0.3
    )

    df["evidence_score"] = (
        0.40 * df["hpo_depth_score"] +
        0.35 * df["prevalence_score"] +
        0.25 * df["case_volume_score"]
    ).clip(0, 1)

    # ══════════════════════════════════════════════════════════
    # COMPOSITE SCORE
    # ══════════════════════════════════════════════════════════
    weights = {
        "lag_score": 0.25,
        "severity_score": 0.30,
        "tractability_score": 0.20,
        "need_score": 0.15,
        "evidence_score": 0.10,
    }

    df["priority_score"] = sum(df[k] * w for k, w in weights.items())

    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("priority_rank")

    df["priority_tier"] = pd.cut(
        df["priority_score"],
        bins=[0, 0.35, 0.45, 0.55, 1.0],
        labels=["Low", "Moderate", "High", "Critical"]
    )

    print(f"[PRIORITY] Scored {len(df)} diseases")
    print(f"  Critical: {(df['priority_tier'] == 'Critical').sum()}")
    print(f"  High:     {(df['priority_tier'] == 'High').sum()}")
    print(f"  Moderate: {(df['priority_tier'] == 'Moderate').sum()}")
    print(f"  Low:      {(df['priority_tier'] == 'Low').sum()}")

    # Print severity signal coverage
    n_mort = (df["mortality_score"] > 0).sum()
    n_feat = (df["feature_severity"] > 0).sum()
    n_clingen = (df["clingen_hi_score"].notna()).sum()
    n_prev = (df["prevalence_per_million"].notna()).sum()
    n_cases = (df["n_reported_cases"].notna() & (df["n_reported_cases"] > 0)).sum()
    print(f"  Severity signals: mortality HPO={n_mort}, "
          f"progressive/seizures/ID={n_feat}, ClinGen HI={n_clingen}")
    print(f"  Evidence signals: prevalence class={n_prev}, "
          f"case counts={n_cases}")

    return df


def save_supplement(df: pd.DataFrame):
    out_cols = [
        "priority_rank", "priority_tier", "priority_score",
        "orpha_code", "name", "gene_symbol", "omim_id",
        "lag_years", "first_clinical_year", "first_genetic_year",
        "inheritance", "age_of_onset",
        "loeuf", "pli", "confidence",
        "n_hpo_terms", "disorder_class",
        "lag_score", "severity_score", "onset_score",
        "mortality_score", "feature_severity",
        "tractability_score", "pli_score", "loeuf_score", "clingen_hi_norm",
        "need_score",
        "evidence_score", "hpo_depth_score", "prevalence_score", "case_volume_score",
        "prevalence_class", "prevalence_per_million", "n_reported_cases",
    ]
    cols_present = [c for c in out_cols if c in df.columns]
    df_out = df[cols_present].copy()

    supp_path = FIGURES / "supplement_priority_list.csv"
    df_out.to_csv(supp_path, index=False)
    print(f"  [SUPP] Full ranked list: {supp_path} ({len(df_out)} diseases)")

    top_path = FIGURES / "supplement_top50.tsv"
    df_out.head(50).to_csv(top_path, sep="\t", index=False)
    print(f"  [SUPP] Top 50 list: {top_path}")


def run_prioritization():
    df = compute_priority_scores()
    save_supplement(df)
    return df
