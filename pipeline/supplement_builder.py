"""Build detailed supplement for undiagnosed disease prioritization.

Generates a comprehensive multi-sheet Excel workbook and rich CSV with:
- Full priority rankings with all component scores
- Disease characterization (definition, synonyms, classification)
- Gene details (all associated genes, constraint metrics)
- Publication provenance (clinical & genetic paper titles, years)
- Actionable use-case assignment
- Summary statistics by disease class and tier
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline.db import get_conn
from pipeline.config import FIGURES
from pipeline.prioritize import compute_priority_scores


def _assign_use_case(row):
    """Assign actionable use-case based on quadrant position."""
    lag = row.get("lag_years", 0) or 0
    tract = row.get("tractability_score", 0) or 0
    lag_med = 2  # approximate median for positive lag
    tract_med = 0.3

    if lag >= lag_med and tract >= tract_med:
        return "Exome/genome reanalysis"
    elif lag < lag_med and tract >= tract_med:
        return "Newborn screening candidate"
    elif lag >= lag_med and tract < tract_med:
        return "Gene discovery priority"
    else:
        return "Monitoring / low priority"


def build_detailed_supplement():
    """Build the comprehensive supplement."""
    conn = get_conn()

    # --- 1. Priority scores ---
    df = compute_priority_scores()

    # --- 2. Enrich with disease details ---
    # Definition
    defs = pd.read_sql("SELECT orpha_code, definition FROM disorders", conn)
    df = df.merge(defs, on="orpha_code", how="left")

    # Synonyms (comma-separated)
    syns = pd.read_sql("""
        SELECT orpha_code, GROUP_CONCAT(synonym, '; ') as synonyms
        FROM disorder_synonyms GROUP BY orpha_code
    """, conn)
    df = df.merge(syns, on="orpha_code", how="left")

    # Full classification chain
    cls = pd.read_sql("""
        SELECT orpha_code, GROUP_CONCAT(DISTINCT parent_name) as classification_path
        FROM classifications GROUP BY orpha_code
    """, conn)
    df = df.merge(cls, on="orpha_code", how="left")

    # --- 3. Gene details ---
    # All genes for each disease
    genes_all = pd.read_sql("""
        SELECT orpha_code, GROUP_CONCAT(DISTINCT gene_symbol) as all_genes,
               COUNT(DISTINCT gene_symbol) as n_genes
        FROM disorder_genes GROUP BY orpha_code
    """, conn)
    df = df.merge(genes_all, on="orpha_code", how="left")

    # Best gene constraint details (one row per gene — canonical MANE)
    constraint = pd.read_sql("""
        SELECT gene_symbol,
               lof_oe as loeuf_value, lof_pli as pli_value,
               lof_obs, lof_exp,
               mis_oe as mis_oeuf, mis_z as mis_z_score
        FROM constraint_metrics
        WHERE canonical = 'True' AND mane_select = 'True'
    """, conn)
    constraint = constraint.drop_duplicates(subset=["gene_symbol"], keep="first")
    df = df.merge(constraint, on="gene_symbol", how="left")

    # --- 4. Publication provenance ---
    # Clinical paper
    clin_papers = pd.read_sql("""
        SELECT pmid, year as clinical_pub_year,
               title as clinical_paper_title,
               journal as clinical_journal
        FROM pubmed_cache WHERE title IS NOT NULL
    """, conn)
    df = df.merge(
        clin_papers.rename(columns={"pmid": "first_clinical_pmid"}),
        on="first_clinical_pmid", how="left"
    )

    # Genetic paper
    gen_papers = pd.read_sql("""
        SELECT pmid, year as genetic_pub_year,
               title as genetic_paper_title,
               journal as genetic_journal
        FROM pubmed_cache WHERE title IS NOT NULL
    """, conn)
    df = df.merge(
        gen_papers.rename(columns={"pmid": "first_genetic_pmid"}),
        on="first_genetic_pmid", how="left"
    )

    # --- 5. Semantic Scholar citations ---
    try:
        s2_clin = pd.read_sql("""
            SELECT pmid, citation_count as clinical_citations,
                   influential_citations as clinical_influential
            FROM s2_cache WHERE citation_count IS NOT NULL
        """, conn)
        df = df.merge(
            s2_clin.rename(columns={"pmid": "first_clinical_pmid"}),
            on="first_clinical_pmid", how="left"
        )
    except Exception:
        df["clinical_citations"] = None
        df["clinical_influential"] = None

    try:
        s2_gen = pd.read_sql("""
            SELECT pmid, citation_count as genetic_citations,
                   influential_citations as genetic_influential
            FROM s2_cache WHERE citation_count IS NOT NULL
        """, conn)
        df = df.merge(
            s2_gen.rename(columns={"pmid": "first_genetic_pmid"}),
            on="first_genetic_pmid", how="left"
        )
    except Exception:
        df["genetic_citations"] = None
        df["genetic_influential"] = None

    # --- 6. G2P details ---
    g2p_details = pd.read_sql("""
        SELECT gene_symbol,
               GROUP_CONCAT(DISTINCT allelic_req) as allelic_requirement,
               GROUP_CONCAT(DISTINCT mechanism) as molecular_mechanism,
               GROUP_CONCAT(DISTINCT panel) as g2p_panels
        FROM g2p GROUP BY gene_symbol
    """, conn)
    df = df.merge(g2p_details, on="gene_symbol", how="left")

    conn.close()

    # --- 7. Actionable use-case ---
    df["actionable_use_case"] = df.apply(_assign_use_case, axis=1)

    # --- 8. External links ---
    df["orphanet_url"] = "https://www.orpha.net/en/disease/detail/" + df["orpha_code"].astype(str)
    df["omim_url"] = df["omim_id"].apply(
        lambda x: f"https://omim.org/entry/{x.split(',')[0].strip()}" if pd.notna(x) else None
    )

    # Deduplicate (merges may create duplicates)
    df = df.drop_duplicates(subset=["orpha_code"], keep="first")

    # Sort by priority
    df = df.sort_values("priority_rank")

    # ============================================================
    # OUTPUT: Multi-sheet Excel
    # ============================================================

    out_path = FIGURES / "supplement_undiagnosed_disease_prioritization.xlsx"

    # --- Sheet 1: Priority rankings (main table) ---
    sheet1_cols = [
        "priority_rank", "priority_tier", "priority_score",
        "actionable_use_case",
        "orpha_code", "name", "synonyms",
        "gene_symbol", "all_genes", "n_genes",
        "omim_id",
        "lag_years", "first_clinical_year", "first_genetic_year",
        "inheritance", "age_of_onset", "disorder_class",
        "lag_score",
        "severity_score", "onset_score", "mortality_score", "feature_severity",
        "tractability_score", "pli_score", "loeuf_score", "clingen_hi_norm",
        "need_score", "evidence_score",
        "loeuf", "pli", "confidence", "clingen_hi_score",
        "n_hpo_terms",
        "orphanet_url", "omim_url",
    ]
    sheet1 = df[[c for c in sheet1_cols if c in df.columns]].copy()

    # --- Sheet 2: Publication provenance ---
    sheet2_cols = [
        "priority_rank", "orpha_code", "name", "gene_symbol",
        "first_clinical_pmid", "first_clinical_year",
        "clinical_paper_title", "clinical_journal",
        "first_genetic_pmid", "first_genetic_year",
        "genetic_paper_title", "genetic_journal",
        "lag_years",
    ]
    # Add citation data if available
    for c in ["clinical_citations", "clinical_influential",
              "genetic_citations", "genetic_influential"]:
        if c in df.columns:
            sheet2_cols.append(c)

    sheet2 = df[[c for c in sheet2_cols if c in df.columns]].copy()

    # --- Sheet 3: Gene characterization ---
    sheet3_cols = [
        "priority_rank", "orpha_code", "name",
        "gene_symbol", "all_genes", "n_genes",
        "loeuf", "pli",
        "loeuf_value", "pli_value",
        "lof_obs", "lof_exp",
        "mis_oeuf", "mis_z_score",
        "confidence",
        "allelic_requirement", "molecular_mechanism", "g2p_panels",
    ]
    sheet3 = df[[c for c in sheet3_cols if c in df.columns]].copy()

    # --- Sheet 4: Disease description ---
    sheet4_cols = [
        "priority_rank", "orpha_code", "name", "synonyms",
        "definition",
        "disorder_class", "classification_path",
        "inheritance", "age_of_onset",
        "orphanet_url", "omim_url",
    ]
    sheet4 = df[[c for c in sheet4_cols if c in df.columns]].copy()

    # --- Sheet 5: Summary statistics ---
    summary_rows = []

    # By tier
    for tier in ["Critical", "High", "Moderate", "Low"]:
        sub = df[df["priority_tier"] == tier]
        summary_rows.append({
            "category": "Priority tier",
            "group": tier,
            "n_diseases": len(sub),
            "median_lag": sub["lag_years"].median(),
            "mean_lag": sub["lag_years"].mean(),
            "median_priority_score": sub["priority_score"].median(),
            "median_severity": sub["severity_score"].median(),
            "median_tractability": sub["tractability_score"].median(),
            "median_n_hpo": sub["n_hpo_terms"].median(),
            "pct_neonatal_infantile": (sub["age_of_onset"].str.contains(
                "Neonatal|Infancy", na=False, case=False)).mean() * 100,
            "pct_high_pli": (sub["pli"] > 0.9).mean() * 100 if sub["pli"].notna().any() else None,
        })

    # By use case
    for uc in df["actionable_use_case"].unique():
        sub = df[df["actionable_use_case"] == uc]
        summary_rows.append({
            "category": "Use case",
            "group": uc,
            "n_diseases": len(sub),
            "median_lag": sub["lag_years"].median(),
            "mean_lag": sub["lag_years"].mean(),
            "median_priority_score": sub["priority_score"].median(),
            "median_severity": sub["severity_score"].median(),
            "median_tractability": sub["tractability_score"].median(),
            "median_n_hpo": sub["n_hpo_terms"].median(),
            "pct_neonatal_infantile": (sub["age_of_onset"].str.contains(
                "Neonatal|Infancy", na=False, case=False)).mean() * 100,
            "pct_high_pli": (sub["pli"] > 0.9).mean() * 100 if sub["pli"].notna().any() else None,
        })

    # By disease class (top 10)
    top_cls = df["disorder_class"].value_counts().head(10).index
    for cls in top_cls:
        sub = df[df["disorder_class"] == cls]
        summary_rows.append({
            "category": "Disease class",
            "group": cls,
            "n_diseases": len(sub),
            "median_lag": sub["lag_years"].median(),
            "mean_lag": sub["lag_years"].mean(),
            "median_priority_score": sub["priority_score"].median(),
            "median_severity": sub["severity_score"].median(),
            "median_tractability": sub["tractability_score"].median(),
            "median_n_hpo": sub["n_hpo_terms"].median(),
            "pct_neonatal_infantile": (sub["age_of_onset"].str.contains(
                "Neonatal|Infancy", na=False, case=False)).mean() * 100,
            "pct_high_pli": (sub["pli"] > 0.9).mean() * 100 if sub["pli"].notna().any() else None,
        })

    sheet5 = pd.DataFrame(summary_rows)

    # --- Sheet 6: Methods & scoring description ---
    methods_data = [
        {"parameter": "Composite priority score formula",
         "description": "Weighted sum: 0.25*lag + 0.30*severity + 0.20*tractability + 0.15*need + 0.10*evidence"},
        {"parameter": "Lag score (weight 0.25)",
         "description": "Percentile rank of lag_years (genetic verification year - clinical description year). Higher lag = higher score."},
        {"parameter": "Severity score (weight 0.30)",
         "description": "Multi-source composite: 0.40*onset + 0.35*mortality + 0.25*features. Onset: Orphanet age of onset (neonatal=1.0, infancy=0.9, childhood=0.7, adult=0.3). Mortality: HPO death annotations (HP:0011420-25, HP:0001522). Features: HPO progressive course (HP:0003676), seizures (HP:0001250), intellectual disability (HP:0001249), hypotonia (HP:0001252), spasticity (HP:0001257)."},
        {"parameter": "Tractability score (weight 0.20)",
         "description": "0.40*pLI + 0.30*(1-LOEUF/1.5) + 0.30*ClinGen_HI_norm. gnomAD pLI and LOEUF indicate haploinsufficiency. ClinGen haploinsufficiency score (3=sufficient evidence -> 1.0, 2=some -> 0.7, 1=little -> 0.4, 0/40=none/unlikely -> 0.1, 30=AR phenotype -> 0.3)."},
        {"parameter": "Unmet need score (weight 0.15)",
         "description": "Based on G2P evidence confidence: limited=0.9, moderate=0.6, strong=0.4, definitive=0.2, unknown=0.7. Less mature evidence = higher unmet diagnostic need."},
        {"parameter": "Evidence depth score (weight 0.10)",
         "description": "Percentile rank of log(1 + n_hpo_terms). More HPO terms = better phenotypic characterization = more actionable for clinical genomics."},
        {"parameter": "Priority tiers",
         "description": "Critical: score > 0.55; High: 0.45-0.55; Moderate: 0.35-0.45; Low: < 0.35"},
        {"parameter": "Clinical description year",
         "description": "Earliest PubMed publication from HPO annotations NOT in G2P curated list, or OMIM MIM number approximate year as fallback."},
        {"parameter": "Genetic verification year",
         "description": "Earliest PubMed publication from G2P curated publications (gene-disease discovery papers)."},
        {"parameter": "Actionable use cases",
         "description": "Quadrant assignment based on lag (above/below median) and tractability (above/below median): Exome reanalysis, Newborn screening, Gene discovery, Monitoring."},
        {"parameter": "Data sources",
         "description": "Orphanet (en_product1.xml, en_product7.xml, orphanet_epidemiology.xml), HPO (phenotype.hpoa, phenotype_to_genes.txt), gnomAD v4.1 (constraint_metrics), HGNC, G2P (2026-02-07), PubMed (NCBI E-utilities), Semantic Scholar."},
        {"parameter": "Software",
         "description": "PhenoLag Pipeline v1.0. Python 3.9, lifelines (Cox PH), statsmodels (OLS/VIF), scipy (Spearman/Kruskal-Wallis), matplotlib/seaborn."},
        {"parameter": "Limitations",
         "description": "Severity enriched with HPO mortality/progressive/seizure/ID annotations and ClinGen HI scores, but does not include prevalence data or treatment availability. Clinical description year depends on HPO reference PMIDs (may underestimate for poorly annotated diseases). Genetic tractability proxy uses gnomAD constraint + ClinGen; does not account for variant type specificity or structural variants."},
    ]
    sheet6 = pd.DataFrame(methods_data)

    # ============================================================
    # Write Excel
    # ============================================================
    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        sheet1.to_excel(writer, sheet_name="1_Priority_rankings", index=False)
        sheet2.to_excel(writer, sheet_name="2_Publication_provenance", index=False)
        sheet3.to_excel(writer, sheet_name="3_Gene_characterization", index=False)
        sheet4.to_excel(writer, sheet_name="4_Disease_description", index=False)
        sheet5.to_excel(writer, sheet_name="5_Summary_statistics", index=False)
        sheet6.to_excel(writer, sheet_name="6_Methods_scoring", index=False)

    print(f"  [SUPP] Excel workbook: {out_path}")
    print(f"         Sheet 1: Priority rankings ({len(sheet1)} diseases)")
    print(f"         Sheet 2: Publication provenance")
    print(f"         Sheet 3: Gene characterization")
    print(f"         Sheet 4: Disease descriptions")
    print(f"         Sheet 5: Summary statistics ({len(sheet5)} rows)")
    print(f"         Sheet 6: Methods & scoring")

    # Also save flat CSV for maximum compatibility
    csv_path = FIGURES / "supplement_full_prioritization.csv"
    # Merge all useful columns into one flat table
    flat_cols = list(set(
        sheet1.columns.tolist() +
        ["clinical_paper_title", "clinical_journal",
         "genetic_paper_title", "genetic_journal",
         "definition", "classification_path",
         "allelic_requirement", "molecular_mechanism", "g2p_panels",
         "clinical_citations", "genetic_citations"]
    ))
    flat_cols = [c for c in flat_cols if c in df.columns]
    df[flat_cols].to_csv(csv_path, index=False)
    print(f"  [SUPP] Flat CSV: {csv_path}")

    return df
