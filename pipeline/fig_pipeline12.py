"""12-step data-processing pipeline (engineering funnel) for the poster — faithful to
the original PhenoLag_architecture.pptx layout, with CURRENT numbers.

Two columns of 6 numbered cards, arrows between consecutive steps. Colour groups:
parse/filter (grey) -> genetics (teal) -> dating/QC (navy) -> AI (accent).

Run: PYTHONPATH=. python -m pipeline.fig_pipeline12
Saves figures/fig_pipeline12.png
"""
from __future__ import annotations
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pipeline.config import DB_PATH, FIGURES

GREY, TEAL, NAVY, RUST = "#C9CDD4", "#2E8B8B", "#065A82", "#B85042"


def main():
    con = sqlite3.connect(str(DB_PATH))
    q = lambda s: con.execute(s).fetchone()[0]
    n_phenolag = q("SELECT COUNT(*) FROM phenolag")
    steps = [
        ("Parsing", "Orphanet catalogue", "11,456", "rows", GREY),
        ("Filtration", "Orphanet: type = Disease", "4,713", "nosologies", GREY),
        ("Monogenicity", "Orphanet gene-disease set", "2,779", "eligible", GREY),
        ("Identification", "OMIM phenotype IDs", "2,460", "with OMIM", GREY),
        ("Genetics", "G2P · ClinGen · HPO", f"{n_phenolag:,}", "with gene", TEAL),
        ("Canonization", "HGNC gene symbols", f"{n_phenolag:,}", "no loss", TEAL),
        ("PubMed collection", "NCBI E-utilities (PubMed)", f"{q('SELECT COUNT(*) FROM pubmed_cache'):,}", "PMIDs", TEAL),
        ("Lag calculation", "gene year − clinical year", f"{n_phenolag:,}", "rows", TEAL),
        ("Date cleaning", "both dates resolved", f"{q('SELECT COUNT(*) FROM phenolag WHERE lag_years IS NOT NULL'):,}", "records", NAVY),
        ("Validation", "dating-quality filters", f"{q('SELECT COUNT(*) FROM phenolag WHERE lag_years>=0 AND suspect_dating=0'):,}", "records", NAVY),
        ("Final dataset", "clinical year ≥ 1946", f"{q('SELECT COUNT(*) FROM phenolag WHERE lag_years>=0 AND suspect_dating=0 AND first_clinical_year>=1946'):,}", "diseases", NAVY),
    ]
    con.close()

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 2); ax.set_ylim(-0.35, 6.4); ax.axis("off")
    BW, BH = 0.90, 0.82
    cols = {0: 0.06, 1: 1.06}  # left/right column x-start
    pos = {}
    for i, (title, desc, num, unit, col) in enumerate(steps):
        c = 0 if i < 6 else 1
        row = i if i < 6 else i - 6
        y = 5.5 - row * 1.02
        x = cols[c]
        pos[i] = (x + BW / 2, y, x, y + BH / 2, y - BH / 2)
        ax.add_patch(FancyBboxPatch((x, y - BH / 2), BW, BH,
                     boxstyle="round,pad=0.01,rounding_size=0.04", linewidth=0, facecolor=col))
        light = col in (NAVY, TEAL, RUST)
        tc = "white" if light else "#222"
        ax.text(x + 0.06, y + 0.22, f"{i+1}. {title}", fontsize=13.5, fontweight="bold",
                color=tc, va="center")
        ax.text(x + 0.06, y - 0.01, desc, fontsize=11, color=("#eaf2f2" if light else "#555"), va="center")
        ax.text(x + 0.06, y - 0.25, f"{num} {unit}", fontsize=13, fontweight="bold",
                color=tc, va="center")
    # arrows within columns (1->...->6, 7->...->12)
    arrow = dict(arrowstyle="-|>", color="#9AA0A6", lw=1.6, mutation_scale=14)
    for a, b in [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10)]:
        ax.add_patch(FancyArrowPatch((pos[a][0], pos[a][4]), (pos[b][0], pos[b][3]), **arrow))
    fig.text(0.5, 0.015, "All inputs are public, open databases:  "
             "Orphanet · OMIM · HPO · Gene2Phenotype · ClinGen · HGNC · gnomAD · NCBI/PubMed",
             ha="center", fontsize=11, color="#444", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_pipeline12.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIGURES / "fig_pipeline12.png")


if __name__ == "__main__":
    main()
