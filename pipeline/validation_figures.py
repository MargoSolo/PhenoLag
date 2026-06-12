"""Validation figures: sensitivity analysis + G2P coverage + OMIM date improvement.

Fig 22: Weight sensitivity analysis (Tornado + rank stability)
Fig 23: G2P coverage analysis by disease class
Fig 24: OMIM elink dates vs MIM fallback comparison (after OMIM fetch)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pipeline.config import FIGURES
from pipeline.db import get_conn
from pipeline.prioritize import compute_priority_scores

PALETTE = {
    "bg": "#FAFAFA", "accent1": "#2C73D2", "accent2": "#FF6F61",
    "accent3": "#45B7A0", "accent4": "#FFC75F", "accent5": "#845EC2",
    "text": "#2D2D2D", "grid": "#E0E0E0",
    "critical": "#D32F2F",
}


def _setup():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


# ============================================================
# Fig 22: Weight sensitivity analysis
# ============================================================

def fig22_weight_sensitivity(df_base: pd.DataFrame):
    """Test how priority rankings change under different weight schemes."""
    _setup()

    score_cols = ["lag_score", "severity_score", "tractability_score",
                  "need_score", "evidence_score"]
    comp_names = ["Lag", "Severity", "Tractability", "Unmet need", "Evidence"]

    # Weight scenarios
    scenarios = {
        "Baseline\n(0.25/0.30/0.20/0.15/0.10)": [0.25, 0.30, 0.20, 0.15, 0.10],
        "Lag-focused\n(0.45/0.20/0.15/0.10/0.10)": [0.45, 0.20, 0.15, 0.10, 0.10],
        "Severity-focused\n(0.15/0.45/0.15/0.15/0.10)": [0.15, 0.45, 0.15, 0.15, 0.10],
        "Tractability-focused\n(0.15/0.20/0.40/0.15/0.10)": [0.15, 0.20, 0.40, 0.15, 0.10],
        "Equal weights\n(0.20/0.20/0.20/0.20/0.20)": [0.20, 0.20, 0.20, 0.20, 0.20],
    }

    # Compute rankings under each scenario
    rankings = {}
    for name, weights in scenarios.items():
        df_tmp = df_base.copy()
        df_tmp["score"] = sum(df_tmp[c] * w for c, w in zip(score_cols, weights))
        df_tmp["rank"] = df_tmp["score"].rank(ascending=False, method="min").astype(int)
        rankings[name] = df_tmp.set_index("orpha_code")["rank"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Panel A: Top 20 rank stability heatmap ---
    ax = axes[0]
    baseline = list(scenarios.keys())[0]
    top20_baseline = df_base.sort_values("priority_rank").head(20)["orpha_code"].values

    rank_matrix = []
    for orpha in top20_baseline:
        row = []
        for name in scenarios:
            row.append(rankings[name].get(orpha, np.nan))
        rank_matrix.append(row)

    rank_df = pd.DataFrame(rank_matrix,
                           index=[df_base[df_base["orpha_code"] == o]["name"].values[0][:35]
                                  for o in top20_baseline],
                           columns=list(scenarios.keys()))

    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="YlOrRd_r",
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Rank (lower = higher priority)"})
    ax.set_title("A. Top 20 rank stability across weight scenarios")
    ax.set_xlabel("")

    # --- Panel B: Rank correlation between scenarios ---
    ax = axes[1]
    from scipy.stats import spearmanr
    n_scenarios = len(scenarios)
    corr_mat = np.ones((n_scenarios, n_scenarios))
    scenario_names = list(scenarios.keys())
    short_names = ["Baseline", "Lag-focus", "Severity-focus", "Tract-focus", "Equal"]

    for i in range(n_scenarios):
        for j in range(n_scenarios):
            r, _ = spearmanr(
                rankings[scenario_names[i]].values,
                rankings[scenario_names[j]].values
            )
            corr_mat[i, j] = r

    sns.heatmap(corr_mat, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, linewidths=0.5, linecolor="white", vmin=0.8, vmax=1.0,
                cbar_kws={"label": "Spearman r"})
    ax.set_title("B. Rank correlation between weight scenarios")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig22_weight_sensitivity.png")
    fig.savefig(FIGURES / "fig22_weight_sensitivity.pdf")
    plt.close(fig)
    print("  [FIG22] Weight sensitivity saved")


# ============================================================
# Fig 23: G2P coverage analysis
# ============================================================

def fig23_g2p_coverage():
    """Show G2P coverage gaps by disease class and gene count."""
    _setup()
    conn = get_conn()

    # All ORPHA diseases with gene associations
    df_all = pd.read_sql("""
        SELECT d.orpha_code, d.name, d.disorder_type,
               COUNT(DISTINCT dg.gene_symbol) as n_genes
        FROM disorders d
        JOIN disorder_genes dg ON d.orpha_code = dg.orpha_code
        WHERE d.disorder_type = 'Disease'
        GROUP BY d.orpha_code
    """, conn)

    # Diseases with G2P entries
    df_g2p = pd.read_sql("""
        SELECT DISTINCT dg.orpha_code
        FROM disorder_genes dg
        JOIN g2p g ON dg.gene_symbol = g.gene_symbol
    """, conn)
    g2p_orphas = set(df_g2p["orpha_code"])

    # Classification
    cls = pd.read_sql("""
        SELECT orpha_code, parent_name as disorder_class
        FROM classifications
    """, conn)
    cls = cls.drop_duplicates(subset=["orpha_code"], keep="first")
    df_all = df_all.merge(cls, on="orpha_code", how="left")

    conn.close()

    df_all["has_g2p"] = df_all["orpha_code"].isin(g2p_orphas)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Panel A: G2P coverage by disease class ---
    ax = axes[0]
    top_cls = df_all["disorder_class"].value_counts().head(12).index
    df_cls = df_all[df_all["disorder_class"].isin(top_cls)].copy()

    def _short(s):
        if not s:
            return "Unknown"
        s = s.replace("Rare ", "").replace(" disease", "")
        return s[0].upper() + s[1:]

    ct = pd.crosstab(
        df_cls["disorder_class"].apply(_short),
        df_cls["has_g2p"],
        normalize="index"
    ) * 100
    ct.columns = ["No G2P", "Has G2P"]
    ct = ct.sort_values("Has G2P", ascending=True)

    ct.plot.barh(ax=ax, stacked=True,
                 color=[PALETTE["grid"], PALETTE["accent3"]],
                 alpha=0.8, edgecolor="white")
    ax.set_xlabel("% of diseases")
    ax.set_title("A. G2P gene-disease evidence coverage")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate total counts
    for i, cls_name in enumerate(ct.index):
        full_name = [c for c in top_cls if _short(c) == cls_name]
        if full_name:
            n = len(df_cls[df_cls["disorder_class"] == full_name[0]])
            ax.text(101, i, f"n={n}", va="center", fontsize=9, color="gray")

    # --- Panel B: G2P coverage by phenolag presence ---
    ax = axes[1]
    conn2 = get_conn()
    df_phenolag = pd.read_sql("SELECT orpha_code, lag_years FROM phenolag", conn2)
    conn2.close()

    df_all2 = df_all.copy()
    df_all2 = df_all2.merge(df_phenolag, on="orpha_code", how="left")
    df_all2["has_lag"] = df_all2["lag_years"].notna()

    categories = [
        ("Has gene link\n(all)", len(df_all2)),
        ("Has G2P entry", df_all2["has_g2p"].sum()),
        ("Has computable\nlag", df_all2["has_lag"].sum()),
        ("Has lag > 0", (df_all2["lag_years"] > 0).sum()),
    ]

    bars = ax.bar(range(len(categories)), [c[1] for c in categories],
                  color=[PALETTE["accent1"], PALETTE["accent3"],
                         PALETTE["accent4"], PALETTE["accent2"]],
                  alpha=0.7, edgecolor="white")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c[0] for c in categories], fontsize=11)
    ax.set_ylabel("Number of diseases")
    ax.set_title("B. Data completeness funnel")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, (_, val) in zip(bars, categories):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(val)), ha="center", fontsize=12, fontweight="bold")

    # Add percentage annotations
    total = categories[0][1]
    for i, (_, val) in enumerate(categories):
        if i > 0:
            pct = val / total * 100
            ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                    bars[i].get_height() / 2,
                    f"{pct:.0f}%", ha="center", fontsize=11, color="white",
                    fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig23_g2p_coverage.png")
    fig.savefig(FIGURES / "fig23_g2p_coverage.pdf")
    plt.close(fig)
    print("  [FIG23] G2P coverage saved")


# ============================================================
# Fig 24: OMIM elink dates comparison
# ============================================================

def fig24_omim_date_comparison():
    """Compare OMIM elink dates vs MIM fallback vs HPO dates."""
    _setup()
    conn = get_conn()

    # Check if omim_clinical_cache exists and has data
    try:
        df_omim = pd.read_sql("""
            SELECT omim_id, earliest_year, n_omim_refs
            FROM omim_clinical_cache
            WHERE earliest_year IS NOT NULL
        """, conn)
    except Exception:
        print("  [FIG24] No OMIM cache yet, skipping")
        conn.close()
        return

    if len(df_omim) < 10:
        print(f"  [FIG24] Only {len(df_omim)} OMIM entries cached, skipping")
        conn.close()
        return

    # Get phenolag data with clinical years
    df_lag = pd.read_sql("SELECT * FROM phenolag WHERE first_clinical_year IS NOT NULL", conn)
    conn.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Distribution of OMIM earliest years ---
    ax = axes[0]
    ax.hist(df_omim["earliest_year"], bins=40, color=PALETTE["accent1"],
            alpha=0.7, edgecolor="white")
    ax.set_xlabel("Earliest OMIM-cited publication year")
    ax.set_ylabel("Number of OMIM entries")
    ax.set_title(f"A. OMIM elink clinical dates (n={len(df_omim)})")
    ax.axvline(df_omim["earliest_year"].median(), color=PALETTE["accent2"],
               ls="--", lw=2, label=f"Median = {df_omim['earliest_year'].median():.0f}")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: Compare OMIM elink year vs current clinical year ---
    ax = axes[1]
    # Match OMIM dates to phenolag diseases
    matched = []
    for _, row in df_lag.iterrows():
        if pd.isna(row["omim_id"]):
            continue
        for mim in str(row["omim_id"]).split(","):
            mim = mim.strip()
            omim_row = df_omim[df_omim["omim_id"] == mim]
            if len(omim_row) > 0:
                matched.append({
                    "current_clinical_year": row["first_clinical_year"],
                    "omim_elink_year": omim_row.iloc[0]["earliest_year"],
                    "lag_years": row["lag_years"],
                })
                break

    if matched:
        df_match = pd.DataFrame(matched)
        ax.scatter(df_match["current_clinical_year"], df_match["omim_elink_year"],
                   s=15, alpha=0.4, color=PALETTE["accent5"])

        # Diagonal
        lims = [min(df_match["current_clinical_year"].min(), df_match["omim_elink_year"].min()),
                max(df_match["current_clinical_year"].max(), df_match["omim_elink_year"].max())]
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)

        # Cases where OMIM elink is earlier (below diagonal)
        earlier = (df_match["omim_elink_year"] < df_match["current_clinical_year"]).sum()
        total = len(df_match)
        ax.text(0.05, 0.95,
                f"OMIM elink earlier: {earlier}/{total} ({earlier/total*100:.0f}%)\n"
                f"Median shift: {(df_match['current_clinical_year'] - df_match['omim_elink_year']).median():.0f} years",
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Current clinical year (HPO/MIM fallback)")
    ax.set_ylabel("OMIM elink earliest year")
    ax.set_title(f"B. Date improvement ({len(matched)} diseases matched)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig24_omim_dates.png")
    fig.savefig(FIGURES / "fig24_omim_dates.pdf")
    plt.close(fig)
    print("  [FIG24] OMIM date comparison saved")


def generate_validation_figures():
    """Generate all validation figures."""
    FIGURES.mkdir(exist_ok=True)
    df = compute_priority_scores()
    fig22_weight_sensitivity(df)
    fig23_g2p_coverage()
    fig24_omim_date_comparison()
    print("[VALIDATION] All validation figures generated")
