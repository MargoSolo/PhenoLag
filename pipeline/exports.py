"""Ad-hoc Excel exports for poster supplementary tables.

Run after run_pipeline.py + apply_curation has populated phenolag.db:

    python3 -m pipeline.exports

Produces in figures/:
  - export_genes_1950_1980.xlsx   genes with first_genetic_year in [1950, 1980]
  - export_hpo_complex.xlsx       diseases with n_hpo_terms > 1000

Both use only the main analysis sample (suspect_dating = 0).
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from pipeline.config import DB_PATH, FIGURES


def _conn():
    return sqlite3.connect(str(DB_PATH))


def export_genes_1950_1980() -> pd.DataFrame:
    """Diseases whose genetic verification year is 1950-1980 inclusive."""
    q = """
    SELECT
        orpha_code, name, gene_symbol,
        first_clinical_year, first_clinical_pmid, clin_source,
        first_genetic_year,  first_genetic_pmid,
        lag_years,
        inheritance, disorder_class, confidence,
        loeuf, pli, n_hpo_terms,
        gene_sources
    FROM phenolag
    WHERE first_genetic_year BETWEEN 1950 AND 1980
      AND suspect_dating = 0
    ORDER BY first_genetic_year, gene_symbol
    """
    with _conn() as c:
        df = pd.read_sql(q, c)

    out = FIGURES / "export_genes_1950_1980.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="genes_1950_1980", index=False)
        # decade summary on a second sheet
        summary = (
            df.assign(decade=(df["first_genetic_year"] // 10 * 10).astype("Int64"))
              .groupby("decade")
              .agg(n=("gene_symbol", "count"),
                   mean_lag=("lag_years", "mean"))
              .round(1)
              .reset_index()
        )
        summary.to_excel(w, sheet_name="by_decade", index=False)
    print(f"[export] genes 1950-1980: {len(df):>4} rows -> {out}")
    return df


def export_hpo_complex(threshold: int = 1000) -> pd.DataFrame:
    """Diseases with more than `threshold` HPO terms (highly complex phenotypes)."""
    q = f"""
    SELECT
        orpha_code, name, gene_symbol,
        n_hpo_terms,
        first_clinical_year, first_genetic_year, lag_years,
        clin_source, inheritance, disorder_class, confidence,
        gene_sources
    FROM phenolag
    WHERE n_hpo_terms > {int(threshold)}
      AND suspect_dating = 0
    ORDER BY n_hpo_terms DESC
    """
    with _conn() as c:
        df = pd.read_sql(q, c)

    out = FIGURES / "export_hpo_complex.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=f"n_hpo_gt_{threshold}", index=False)
        # class-level summary
        if not df.empty and "disorder_class" in df:
            cls = (df.groupby("disorder_class")
                     .agg(n=("orpha_code", "count"),
                          median_hpo=("n_hpo_terms", "median"),
                          mean_lag=("lag_years", "mean"))
                     .round(1)
                     .reset_index()
                     .sort_values("n", ascending=False))
            cls.to_excel(w, sheet_name="by_class", index=False)
    print(f"[export] HPO > {threshold}: {len(df):>4} rows -> {out}")
    return df


if __name__ == "__main__":
    export_genes_1950_1980()
    export_hpo_complex(threshold=1000)
