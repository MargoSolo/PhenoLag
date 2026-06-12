"""Cohort-construction funnel for the poster (current numbers), architecture-style.

Six disease-count stages (the funnel) with side annotations for the engineering
steps (OMIM/PubMed dating, AI consensus, QC) — a clean version of the 12-step
processing pipeline.

Run: PYTHONPATH=. python -m pipeline.fig_cohort_funnel
Saves figures/fig_cohort_funnel.png
"""
from __future__ import annotations
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from pipeline.config import DB_PATH, FIGURES

NAVY, TEAL, RUST, GREY = "#065A82", "#2E8B8B", "#B85042", "#D9DCE0"


def main():
    con = sqlite3.connect(str(DB_PATH))
    q = lambda s: con.execute(s).fetchone()[0]
    n_pmid = q("SELECT COUNT(*) FROM pubmed_cache")
    n_ai = q("SELECT COUNT(*) FROM ai_consensus_omim")
    stages = [
        ("All Orphanet disorders", q("SELECT COUNT(*) FROM disorders"), GREY, ""),
        ("Rare diseases (disorder type)", q("SELECT COUNT(*) FROM disorders WHERE disorder_type='Disease'"), GREY, ""),
        ("With curated causative gene", q("SELECT COUNT(*) FROM phenolag"), GREY,
         "G2P · ClinGen · HPO · OMIM"),
        ("Phenotype + genotype datable", q("SELECT COUNT(*) FROM phenolag WHERE lag_years IS NOT NULL"), TEAL,
         f"OMIM-elink + PubMed ({n_pmid:,} refs) · Gemini Flash+Pro consensus (n={n_ai:,})"),
        ("Dating-quality filtered", q("SELECT COUNT(*) FROM phenolag WHERE lag_years>=0 AND suspect_dating=0"), TEAL,
         "drop neg-lag / co-discovery / pre-MEDLINE"),
        ("Primary analysis sample (≥1946)",
         q("SELECT COUNT(*) FROM phenolag WHERE lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946"),
         NAVY, "median lag 23 y"),
    ]
    con.close()
    top = stages[0][1]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    n = len(stages)
    h = 0.66
    for i, (label, val, col, note) in enumerate(stages):
        y = n - 1 - i
        w = max(0.12, val / top)
        x0 = (1 - w) / 2
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h,
                     boxstyle="round,pad=0.004,rounding_size=0.02",
                     linewidth=0, facecolor=col, mutation_aspect=0.45))
        ax.text(0.5, y, f"{val:,}", ha="center", va="center",
                color="#1a1a1a", fontsize=19, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
        ax.text(1.04, y + 0.12, label, ha="left", va="center", fontsize=12.5, fontweight="bold")
        if note:
            ax.text(1.04, y - 0.20, note, ha="left", va="center", fontsize=9.5, color="#666")
    ax.set_xlim(-0.02, 1.95)
    ax.set_ylim(-0.55, n - 0.3)
    ax.axis("off")
    fig.text(0.06, 0.015,
             "AI verification uses OMIM-cited PMIDs only (no fabricated citations) · "
             "119 human-validated: 90% genetic / 83% clinical PMID accuracy",
             fontsize=10, color="#555")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cohort_funnel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("stages:", [(s[0][:22], s[1]) for s in stages])
    print("saved", FIGURES / "fig_cohort_funnel.png")


if __name__ == "__main__":
    main()
