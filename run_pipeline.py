#!/usr/bin/env python3
"""
PhenoLag Pipeline — Main orchestrator.

Calculates the time lag between first clinical description and genetic
verification of rare diseases, using data from Orphanet, HPO, gnomAD,
HGNC, G2P/ClinGen, PubMed, and Semantic Scholar.

Usage:
    python run_pipeline.py                 # full pipeline
    python run_pipeline.py --step parse    # only parse sources
    python run_pipeline.py --step fetch    # only pre-fetch PMIDs
    python run_pipeline.py --step lag      # only compute lag (requires parsed DB)
    python run_pipeline.py --step s2       # only enrich with Semantic Scholar
    python run_pipeline.py --step viz      # only generate figures (requires lag)
"""
import argparse
import sys
import time

from pipeline.db import init_db
from pipeline.parse_orphanet import run_all as parse_orphanet
from pipeline.parse_genes import run_all as parse_genes
from pipeline.lag_calculator import prefetch_pmids, calculate_lag
from pipeline.omim_dates import fetch_omim_clinical_dates
from pipeline.semantic_scholar import enrich_top_papers
from pipeline.visualize import generate_all as visualize


def step_parse():
    print("\n" + "=" * 60)
    print("STEP 1: PARSE ALL SOURCE FILES")
    print("=" * 60)
    init_db()
    parse_orphanet()
    parse_genes()


def step_fetch():
    print("\n" + "=" * 60)
    print("STEP 2: PRE-FETCH PUBMED DATES")
    print("=" * 60)
    prefetch_pmids()


def step_omim():
    print("\n" + "=" * 60)
    print("STEP 2b: FETCH OMIM CLINICAL DATES")
    print("=" * 60)
    fetch_omim_clinical_dates()


def step_lag():
    print("\n" + "=" * 60)
    print("STEP 3: CALCULATE PHENOLAG")
    print("=" * 60)
    calculate_lag()


def step_s2():
    print("\n" + "=" * 60)
    print("STEP 3b: ENRICH WITH SEMANTIC SCHOLAR")
    print("=" * 60)
    enrich_top_papers(limit=200)


def step_viz():
    print("\n" + "=" * 60)
    print("STEP 4: GENERATE VISUALIZATIONS")
    print("=" * 60)
    visualize()


def step_stats():
    print("\n" + "=" * 60)
    print("STEP 5: STATISTICAL ANALYSIS (experimental)")
    print("=" * 60)
    from pipeline.stats_figures import generate_stats_figures
    generate_stats_figures()


def step_priority():
    print("\n" + "=" * 60)
    print("STEP 6: UNDIAGNOSED DISEASE PRIORITIZATION (experimental)")
    print("=" * 60)
    from pipeline.prioritize import run_prioritization
    from pipeline.priority_figures import generate_priority_figures
    from pipeline.supplement_builder import build_detailed_supplement
    from pipeline.validation_figures import generate_validation_figures
    df = run_prioritization()
    generate_priority_figures(df)
    build_detailed_supplement()
    generate_validation_figures()


def main():
    parser = argparse.ArgumentParser(description="PhenoLag Pipeline")
    parser.add_argument("--step", choices=["parse", "fetch", "omim", "lag", "s2", "viz", "stats", "priority"],
                        help="Run only a specific step")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("PHENOLAG PIPELINE")
    print("Phenotype-to-Genotype Lag in Rare Diseases")
    print("=" * 60)

    if args.step:
        {"parse": step_parse, "fetch": step_fetch, "omim": step_omim,
         "lag": step_lag, "s2": step_s2, "viz": step_viz,
         "stats": step_stats, "priority": step_priority}[args.step]()
    else:
        # Default pipeline: data → lag → poster figures only.
        # Experimental steps (stats, priority) are available via --step.
        step_parse()
        step_fetch()
        step_omim()
        step_lag()
        step_s2()
        step_viz()

    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
