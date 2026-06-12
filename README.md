# PhenoLag

**Measuring the phenotype-to-genotype lag in rare Mendelian disease  (the years between a disease's first clinical description and the identification of its causal gene).**

PhenoLag pairs two dated events for each rare disease (year of first clinical description and year the causal gene was verified) and computes the gap across the whole rare-Mendelian-disease landscape. Built for ESHG 2026 (poster P15.151.A).

---

## Headline results

Computed on the final dataset of **N = 1,533** rare Mendelian diseases (snapshot year 2026), each with a verified clinical-description year and a verified causal-gene year.

| Metric | Value |
|---|---|
| Median phenotype-to-genotype lag | **19 years** (mean 19.8; IQR 2–34; max 71) |
| Pre-NGS diseases (described < 2010) | median lag **24 years** |
| NGS-era diseases (described ≥ 2010) | median lag **0 years**; 94% solved within 2 years |
| Co-discoveries (lag = 0) | 23% of all diseases |

**By inheritance** (time-to-genetic-verification): autosomal recessive median 14 y, autosomal dominant 21 y, X-linked 25 y (log-rank p < 0.001). After Cox adjustment for the decade of description, **AR is genuinely faster than AD** (HR ≈ 0.83, p = 0.003); the **apparent X-linked disadvantage disappears**   it was confounding by era of description.

---

## Methodology

1. **At-risk denominator, not solved-only.** All survival/Kaplan–Meier analyses include still-unsolved diseases as right-censored observations. Counting only solved diseases creates survivorship bias.
2. **Co-discoveries (lag = 0) are legitimate** for genomic-era diseases (clinical year ≥ 2005): phenotype and gene reported together. Not filtered as artifacts.
3. **Fixed 10-year window** for cross-era RMST comparisons — the maximal follow-up common to every cohort.
4. **Validated dating.** Clinical-year priority: OMIM elink → HPO non-G2P PMIDs → HPO fallback → MIM-number breakpoint. Records where dating is unreliable are flagged `suspect_dating = 1` and excluded from the main sample.
5. **Core cohort**: diseases with at least one G2P or ClinGen gene link (`in_core_cohort = 1`). Main sample = core cohort ∩ `suspect_dating = 0` ∩ `lag_years IS NOT NULL`.

`SNAPSHOT_YEAR = 2026` in `pipeline/config.py` is the single source of truth for right-censoring unsolved diseases.

---

## Data sources

All inputs are **public, open databases**: Orphanet · OMIM · HPO · Gene2Phenotype · ClinGen · HGNC · gnomAD · NCBI/PubMed.

---

## Repository layout

```
run_pipeline.py              # main entry point
requirements.txt
pipeline/
  config.py                  # paths, SNAPSHOT_YEAR, API keys
  db.py                      # SQLite schema + helpers
  parse_orphanet.py          # ingest Orphanet XML (disorders, classifications, epidemiology)
  parse_genes.py             # gene-disease links: G2P, ClinGen, HPO, HGNC, gnomAD
  pubmed_dates.py            # NCBI E-utilities — fetch & cache publication years
  omim_dates.py              # OMIM clinical-description dating via elink
  lag_calculator.py          # lag = gene_year − clinical_year; suspect-dating flags
  semantic_scholar.py        # Semantic Scholar enrichment for top papers
  visualize.py               # all poster figures (PNG + PDF)
```

---

## Running

Requires Python 3.10+ and the packages in `requirements.txt`.

```bash
# create env (Linux/macOS)
python3 -m venv .venv && source .venv/bin/activate

# create env (Windows)
python -m venv .venv && .venv\Scripts\activate

pip install -r requirements.txt
```

Place source files in `sources/` (Orphanet XMLs, G2P, ClinGen, HPO, HGNC, gnomAD) and set API keys in `.env`:

```
NCBI_API_KEY=your_key_here 
```

### Full pipeline

```bash
python run_pipeline.py
```

Runs all steps in order: parse → fetch PMIDs → fetch OMIM dates → compute lag → enrich → generate figures.

### Individual steps

```bash
python run_pipeline.py --step parse   # parse all source files → phenolag.db
python run_pipeline.py --step fetch   # pre-fetch & cache PubMed dates
python run_pipeline.py --step omim    # fetch OMIM clinical-description dates
python run_pipeline.py --step lag     # compute lag (requires parsed DB)
python run_pipeline.py --step s2      # enrich top papers via Semantic Scholar
python run_pipeline.py --step viz     # generate figures (requires lag step)
```

Figures are saved to `figures/` as both PNG (200 dpi screen) and PDF (300 dpi, editable text).

---

## Citation

Soloshenko M., Borovikov A. *PhenoLag: the phenotype-to-genotype lag across rare Mendelian disease.* ESHG 2026, poster P15.151.A.

## License

MIT  see [LICENSE](LICENSE).
