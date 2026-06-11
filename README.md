# PhenoLag

**Measuring the phenotype-to-genotype lag in rare Mendelian disease — the years between a disease's first clinical description and the identification of its causal gene.**

PhenoLag pairs two dated events for each rare disease — the year of first clinical description and the year the causal gene was verified — and computes the gap between them across the whole rare-Mendelian-disease landscape. Built for ESHG 2026 (poster P15.151.A).

---

## Headline results

Computed on the final dataset of **N = 1,533** rare Mendelian diseases (snapshot year 2026), each with a verified clinical-description year and a verified causal-gene year.

| Metric | Value |
|---|---|
| Median phenotype-to-genotype lag | **19 years** (mean 19.8; IQR 2–34; max 71) |
| Pre-NGS diseases (described < 2010) | median lag **24 years** |
| NGS-era diseases (described ≥ 2010) | median lag **0 years**; 94% solved within 2 years |
| Co-discoveries (lag = 0) | 23% of all diseases |

**By inheritance** (time-to-genetic-verification): autosomal recessive median 14 y, autosomal dominant 21 y, X-linked 25 y (log-rank p < 0.001). After Cox adjustment for the decade of description, **AR is genuinely faster than AD** (HR ≈ 0.83, p = 0.003); the **apparent X-linked disadvantage disappears** — it was confounding by era of description.

---

## Methodology (the parts that matter)

1. **At-risk denominator, not solved-only.** All survival/Kaplan–Meier analyses include still-unsolved diseases as right-censored observations. Counting only already-solved diseases creates survivorship bias and falsely flatters recent decades. Reusable loader: `pipeline/fig_acceleration.py::load_at_risk()`.
2. **Co-discoveries (lag = 0) are legitimate** for genomic-era diseases (clinical year ≥ 2005): the phenotype and the gene are reported together. They are not filtered out as artifacts.
3. **Fixed 10-year window** for cross-era comparisons of restricted mean survival time (RMST). Ten years is the maximal follow-up common to every cohort (diseases described in the 2010s have under 15 years of follow-up), so it is the only apples-to-apples horizon.
4. **Validated dating.** Automated dating is benchmarked against a 119-disease hand-curated gold set (`pipeline/validate_gold.py`). The long-lag tail was hand-checked in PubMed and artifacts (acquired cancers, modifier genes, mis-attributed early genes) were flagged out.
5. **No difference-score tautology.** We never report corr(clinical_year, lag): because lag = gene_year − clinical_year, it correlates with clinical_year by construction (≈ −0.92 even under independence). Where a correlation is reported, it is corr(clinical_year, gene_year) = +0.52.

`SNAPSHOT_YEAR = 2026` (in `pipeline/config.py`) is the single source of truth for "now".

---

## Data sources

All inputs are **public, open databases**: Orphanet · OMIM · HPO · Gene2Phenotype · ClinGen · HGNC · gnomAD · NCBI/PubMed.

---

## Repository layout

```
pipeline/
  config.py                    # DB path, SNAPSHOT_YEAR, source versions
  parse_orphanet.py            # ingest Orphanet catalogue
  parse_genes.py               # gene-disease curation (G2P/ClinGen/HPO)
  pubmed_dates.py              # NCBI E-utilities date collection
  omim_dates.py                # OMIM clinical-description dating
  lag_calculator.py            # lag = gene_year - clinical_year, integrity enforced
  validate_gold.py             # dating validation vs 119-disease gold set

  fig_acceleration.py          # load_at_risk() + KM cumulative incidence by era
  fig_pipeline12.py            # 11-step cohort-construction funnel
  fig_lag_violin_clean.py      # pre/post-NGS lag distribution (mean ± SD, 95% CI)
  fig_debt_heatmap.py          # diagnostic-debt heatmap (% per class x decade)
  fig_rmst_lines_by_class.py   # RMST(10) vs decade, one line per disorder class, 95% CI
  fig_inheritance_stats.py     # inheritance ECDF + log-rank + Cox forest (era-adjusted)
  build_forgotten_list.py      # shortlist of long-unsolved / geneless diseases
  ...
requirements.txt
run_pipeline.py
```

---

## Running

Requires Python 3.10+ and the packages in `requirements.txt` (pandas, numpy, lifelines, matplotlib, scipy, python-dotenv, requests, openpyxl).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# generate any figure module from the repo root:
PYTHONPATH=. python3 -m pipeline.fig_rmst_lines_by_class
PYTHONPATH=. python3 -m pipeline.fig_inheritance_stats
```

The processed SQLite database and raw source databases are not included in this repository (the DB is large and the sources are re-downloadable from the providers above). API keys for NCBI/PubMed dating go in a local `.env` (see `pipeline/config.py`); never commit them.

---

## Citation

Soloshenko M. *PhenoLag: the phenotype-to-genotype lag across rare Mendelian disease.* ESHG 2026, poster P15.151.A.

## License

MIT — see [LICENSE](LICENSE).
