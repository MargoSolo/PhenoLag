from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from pipeline.config import FIGURES
from pipeline.db import get_conn


PALETTE = {
    "bg": "#FAFAFA",
    "ink": "#2D2D2D",
    "grid": "#E0E0E0",
    "core": "#2C73D2",       # main sample
    "accent": "#FF6F61",      # highlights, medians
    "genetic": "#845EC2",
    "clinical": "#45B7A0",
    "flag": "#FFC75F",
}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink"],
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.labelpad": 6,
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "pdf.fonttype": 42,  # TrueType for editable text in PDF
        "ps.fonttype": 42,
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name: str):
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


# ---------- data loaders ----------

def load_main() -> pd.DataFrame:
    """Main analysis sample: any gene-linked Orphanet disease with both
    years resolvable and no dating flag. Core vs extended tier is kept
    as a column (`in_core_cohort`) for optional stratification."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM phenolag "
        "WHERE suspect_dating = 0 AND lag_years IS NOT NULL",
        conn,
    )
    conn.close()
    return df


def load_all() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM phenolag", conn)
    conn.close()
    return df


# Back-compat for any external callers.
load_data = load_main


# ---------- figures ----------

def fig_cohort_cascade():
    """N waterfall: Orphanet → any gene link → core (G2P/ClinGen) →
    computable lag → main sample. Shows both diseases and gene-disease pairs,
    plus number of PubMed articles that fed the main analysis."""
    _setup_style()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM disorders WHERE disorder_type='Disease'")
    n_orphanet = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*), COUNT(DISTINCT dg.orpha_code) FROM disorder_genes dg "
        "JOIN disorders d ON d.orpha_code = dg.orpha_code "
        "WHERE d.disorder_type='Disease'"
    )
    pairs_any, dis_any = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM phenolag WHERE lag_years IS NOT NULL")
    n_with_lag = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM phenolag WHERE lag_years IS NOT NULL AND suspect_dating=0"
    )
    n_non_suspect = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM phenolag "
        "WHERE lag_years IS NOT NULL AND suspect_dating=0 "
        "AND first_clinical_year >= 1946"
    )
    n_main = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(DISTINCT first_clinical_pmid), COUNT(DISTINCT first_genetic_pmid) "
        "FROM phenolag WHERE lag_years IS NOT NULL AND suspect_dating=0 "
        "AND first_clinical_year >= 1946"
    )
    n_clin_pmids, n_gen_pmids = cur.fetchone()
    conn.close()

    # Strict monotone waterfall — each row is a subset of the previous.
    # The G2P/ClinGen-curated tier is shown as a *subset* of the
    # primary sample (separate, lighter bar) instead of an earlier
    # cascade step, because forcing 'core' between 'any gene link'
    # (1,954) and 'lag computable' (1,431) would produce a confusing
    # 1,954 -> 499 -> 1,431 swing on the chart.
    steps = [
        ("1. Orphanet catalogue\n(disorder_type = Disease)",            None,      n_orphanet),
        ("2. Any gene link\n(Orphanet / HPO / G2P / ClinGen)",          pairs_any, dis_any),
        ("3. Lag computable\n(both years resolved)",                    None,      n_with_lag),
        ("4. Non-suspect dating\n(codisc / neg_lag / pre_medline / mim filtered out)",
                                                                        None,      n_non_suspect),
        ("5. Primary analysis sample\n(MEDLINE era, clin ≥ 1946)",      None,      n_main),
    ]
    labels = [s[0] for s in steps]
    pairs  = [s[1] for s in steps]
    dis    = [s[2] for s in steps]
    colors = [PALETTE["grid"], PALETTE["grid"], PALETTE["clinical"],
              PALETTE["genetic"], PALETTE["accent"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, dis, color=colors, edgecolor="white")
    for yi, d_, p_ in zip(y, dis, pairs):
        txt = f"  {d_:,} diseases" + (f"  |  {p_:,} pairs" if p_ is not None else "")
        ax.text(d_, yi, txt, va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Number of diseases")
    ax.set_title("Cohort construction")
    _despine(ax)
    ax.set_xlim(0, max(dis) * 1.35)
    caption = (f"PubMed articles analyzed for main sample: "
               f"{n_clin_pmids:,} clinical + {n_gen_pmids:,} genetic PMIDs")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic",
             color=PALETTE.get("muted", "#555"))
    fig.subplots_adjust(bottom=0.12)
    _save(fig, "fig_cohort_cascade")
    print(f"  [FIG] cohort cascade: diseases={dis} pairs={pairs}")


def fig1_lag_distribution(df: pd.DataFrame):
    """Histogram + KDE of lag on the full main-sample range.

    Earlier versions clipped the histogram at the 99th percentile so the
    long historical tail (up to ~180 y) wouldn't dwarf the bulk.  That
    made readers infer a max of ~120 y from the x-axis, which is wrong
    — the real max is the curated 19th-century cases.  We now show the
    full range and use bin width = 5 y so the historical tail stays
    visible without distorting the bulk.
    """
    _setup_style()
    lag = df["lag_years"].dropna()
    if len(lag) == 0:
        return

    # Force lag to non-negative (neg_lag is already flagged out, but guard).
    lag = lag[lag >= 0]
    lo, hi = 0, int(lag.max()) + 5
    bins = np.arange(lo, hi + 5, 5)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(lag, bins=bins, color=PALETTE["core"], alpha=0.75,
            edgecolor="white", linewidth=0.5, density=True)
    if len(lag) >= 5 and lag.std() > 0:
        kde_x = np.linspace(0, lag.max(), 300)
        kde = stats.gaussian_kde(lag)
        ax.plot(kde_x, kde(kde_x), color=PALETTE["accent"], lw=2.5)

    med = lag.median()
    ax.axvline(med, color=PALETTE["genetic"], ls="--", lw=2,
               label=f"median = {med:.0f} y")

    n_long = int((lag > 50).sum())
    stats_text = (
        f"N = {len(lag):,}\n"
        f"median = {med:.0f} y\n"
        f"IQR = [{lag.quantile(0.25):.0f}, {lag.quantile(0.75):.0f}] y\n"
        f"mean = {lag.mean():.1f} y\n"
        f"max = {int(lag.max())} y\n"
        f"lag > 50 y : {n_long:,}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.85, edgecolor=PALETTE["grid"]))

    ax.set_xlim(0, hi)
    ax.set_xlabel("Lag (years): genetic verification − clinical description")
    ax.set_ylabel("Density")
    ax.set_title("Phenotype-to-genotype lag — main sample")
    ax.legend(frameon=False, loc="upper center")
    _despine(ax)
    _save(fig, "fig1_lag_distribution")
    print(f"  [FIG1] lag distribution: N={len(lag)}, median={med:.0f}y")


def fig2_lag_by_era(df: pd.DataFrame):
    """Lag by decade of genetic verification (single-panel boxplot)."""
    _setup_style()
    d = df.dropna(subset=["first_genetic_year", "lag_years"]).copy()
    if len(d) == 0:
        return
    d["decade"] = (d["first_genetic_year"] // 10 * 10).astype(int)
    counts = d["decade"].value_counts()
    decades = sorted(counts[counts >= 1].index)
    d = d[d["decade"].isin(decades)]
    groups = [d[d["decade"] == dd]["lag_years"].values for dd in decades]
    if not decades:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(
        groups, positions=range(len(decades)), widths=0.6, patch_artist=True,
        showfliers=False,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        boxprops=dict(facecolor=PALETTE["core"], alpha=0.7, edgecolor=PALETTE["ink"]),
        whiskerprops=dict(color=PALETTE["ink"]),
        capprops=dict(color=PALETTE["ink"]),
    )
    for i, g in enumerate(groups):
        if len(g):
            jitter = np.random.uniform(-0.12, 0.12, len(g))
            ax.scatter(i + jitter, g, s=6, alpha=0.3, color=PALETTE["core"], zorder=3)

    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels([f"{dd}s" for dd in decades], rotation=30)
    ax.set_xlabel("Decade of genetic verification")
    ax.set_ylabel("Lag (years)")
    ax.set_title("Lag by era of genetic verification")
    _despine(ax)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    y_label = ax.get_ylim()[1] * 0.98
    for i, g in enumerate(groups):
        ax.text(i, y_label, f"n = {len(g)}", ha="center", va="top",
                fontsize=9, color=PALETTE["ink"], alpha=0.65)
    _save(fig, "fig2_lag_by_era")
    print(f"  [FIG2] lag by era: {len(decades)} decades")


def fig3_timeline_scatter(df: pd.DataFrame):
    """Clinical year vs genetic year, color = lag."""
    _setup_style()
    d = df.dropna(subset=["first_clinical_year", "first_genetic_year"]).copy()
    d = d[d["first_clinical_year"] >= 1946]
    if len(d) == 0:
        return

    cmap = LinearSegmentedColormap.from_list(
        "lag", [PALETTE["clinical"], PALETTE["flag"], PALETTE["accent"]]
    )

    fig, ax = plt.subplots(figsize=(8.5, 8))
    vmax = max(1, d["lag_years"].quantile(0.95))
    sc = ax.scatter(
        d["first_clinical_year"], d["first_genetic_year"],
        c=d["lag_years"], cmap=cmap, s=16, alpha=0.65,
        edgecolors="none", vmin=0, vmax=vmax,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Lag (years)")

    lo = min(d["first_clinical_year"].min(), d["first_genetic_year"].min())
    hi = max(d["first_clinical_year"].max(), d["first_genetic_year"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color=PALETTE["ink"], alpha=0.35,
            lw=1, label="lag = 0")
    ax.set_xlim(lo - 2, hi + 2)
    ax.set_ylim(lo - 2, hi + 2)

    ax.set_xlabel("Year of first clinical description")
    ax.set_ylabel("Year of genetic verification")
    ax.set_title("Clinical vs genetic first-description timeline")
    ax.legend(frameon=False, loc="upper left")
    _despine(ax)
    _save(fig, "fig3_timeline_scatter")
    print(f"  [FIG3] timeline scatter: N={len(d)}")


def fig4_cumulative(df: pd.DataFrame, x_end: int = 2026):
    """Cumulative curves: clinical descriptions vs genetic verifications.

    Both curves reach the full sample size by construction (every
    disease in `df` has both years resolved).  See fig13 for a panel
    that also draws the Orphanet-catalogue ceiling and unsolved
    backlog.
    """
    _setup_style()
    if len(df) == 0:
        return

    def cumulative(series: pd.Series) -> pd.Series:
        s = series.dropna().astype(int)
        s = s[s >= 1946].sort_values()
        if len(s) == 0:
            return pd.Series(dtype=int)
        cum = pd.Series(range(1, len(s) + 1), index=s.values)
        return cum.groupby(cum.index).max()

    clin = cumulative(df["first_clinical_year"])
    gen = cumulative(df["first_genetic_year"])
    if len(clin) == 0 or len(gen) == 0:
        return

    # Forward-fill to x_end so the plateau is visible.
    if clin.index.max() < x_end:
        clin = pd.concat([clin, pd.Series([clin.iloc[-1]], index=[x_end])])
    if gen.index.max() < x_end:
        gen = pd.concat([gen, pd.Series([gen.iloc[-1]], index=[x_end])])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(clin.index, clin.values, alpha=0.25, color=PALETTE["clinical"])
    ax.plot(clin.index, clin.values, color=PALETTE["clinical"], lw=2.5,
            label="First clinical descriptions")
    ax.fill_between(gen.index, gen.values, alpha=0.25, color=PALETTE["genetic"])
    ax.plot(gen.index, gen.values, color=PALETTE["genetic"], lw=2.5,
            label="Genetic verifications")

    # Annotate the gap at the year that maximises (clinical − genetic)
    common_years = clin.index.intersection(gen.index)
    if len(common_years) > 0:
        diff = clin.loc[common_years] - gen.loc[common_years]
        peak_year = int(diff.idxmax())
        c_val = int(clin.loc[peak_year])
        g_val = int(gen.loc[peak_year])
        if c_val - g_val >= 5:
            ax.annotate(
                "",
                xy=(peak_year, g_val), xytext=(peak_year, c_val),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.8),
            )
            ax.text(
                peak_year + 1, (c_val + g_val) / 2,
                f"gap in {peak_year}\n{c_val - g_val} diseases",
                fontsize=10, color=PALETTE["accent"],
                va="center", ha="left",
            )

    ax.set_xlim(1946, x_end)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Cumulative first clinical descriptions vs genetic verifications")
    ax.legend(frameon=False, loc="upper left")
    _despine(ax)
    _save(fig, "fig4_cumulative")
    print(f"  [FIG4] cumulative: clin={int(clin.iloc[-1])}, gen={int(gen.iloc[-1])}, "
          f"x_end={x_end}")


def fig14_phenotype_focus(df: pd.DataFrame):
    """Scatter: mean pairwise HPO graph distance vs lag."""
    _setup_style()
    d = df.dropna(subset=["mean_pairwise_dist", "lag_years"]).copy()
    if len(d) < 5:
        return
    rho, p = stats.spearmanr(d["mean_pairwise_dist"], d["lag_years"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(d["mean_pairwise_dist"], d["lag_years"],
               s=14, alpha=0.35, color=PALETTE["core"], edgecolor="none")

    # rolling-bin median trend
    bins = np.linspace(d["mean_pairwise_dist"].min(),
                       d["mean_pairwise_dist"].max(), 8)
    centers, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = d[(d["mean_pairwise_dist"] >= lo) & (d["mean_pairwise_dist"] < hi)]
        if len(sub) >= 10:
            centers.append((lo + hi) / 2)
            meds.append(sub["lag_years"].median())
    if centers:
        ax.plot(centers, meds, "-o", color=PALETTE["accent"], lw=2.2,
                ms=6, label="bin median")

    ax.set_xlabel("Mean pairwise HPO graph distance "
                  "(small = focused phenotype, large = scattered)")
    ax.set_ylabel("Lag (years)")
    txt = (f"Spearman ρ = {rho:+.3f}\n"
           f"p = {p:.2e}\n"
           f"N = {len(d):,}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.85, edgecolor=PALETTE["grid"]))
    ax.set_title("Phenotype focus (HPO graph distance) vs diagnostic lag")
    ax.legend(frameon=False, loc="lower right")
    _despine(ax)
    _save(fig, "fig14_phenotype_focus")
    print(f"  [FIG14] phenotype focus: rho={rho:+.3f} p={p:.2e} N={len(d)}")


def fig15_organ_systems(df: pd.DataFrame):
    """Boxplot of lag stratified by number of involved organ systems."""
    _setup_style()
    d = df.dropna(subset=["n_organ_systems", "lag_years"]).copy()
    if len(d) < 5:
        return
    d["bin"] = pd.cut(
        d["n_organ_systems"],
        bins=[0, 2, 5, 10, 30],
        labels=["1–2", "3–5", "6–10", "11+"],
    )
    bins_order = ["1–2", "3–5", "6–10", "11+"]
    groups = [d[d["bin"] == b]["lag_years"].values for b in bins_order]
    rho, p = stats.spearmanr(d["n_organ_systems"], d["lag_years"])

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.boxplot(
        groups, positions=range(len(bins_order)), widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        boxprops=dict(facecolor=PALETTE["core"], alpha=0.7,
                      edgecolor=PALETTE["ink"]),
        whiskerprops=dict(color=PALETTE["ink"]),
        capprops=dict(color=PALETTE["ink"]),
    )
    for i, g in enumerate(groups):
        if len(g):
            jitter = np.random.uniform(-0.12, 0.12, len(g))
            ax.scatter(i + jitter, g, s=6, alpha=0.25,
                       color=PALETTE["core"], zorder=3)

    xlabels = [f"{b}\nn = {len(g)}, med = {np.median(g):.0f} y"
               if len(g) else b
               for b, g in zip(bins_order, groups)]
    ax.set_xticks(range(len(bins_order)))
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_xlabel("Distinct HPO organ-system ancestors involved")
    ax.set_ylabel("Lag (years)")
    ax.set_title(
        f"Multisystem involvement vs lag  (Spearman ρ = {rho:+.3f}, p = {p:.2e})"
    )
    _despine(ax)
    fig.subplots_adjust(bottom=0.20)
    _save(fig, "fig15_organ_systems")
    print(f"  [FIG15] organ systems: rho={rho:+.3f} p={p:.2e}")


def fig16_term_specificity(df: pd.DataFrame):
    """Scatter: mean HPO term depth (specificity) vs lag."""
    _setup_style()
    d = df.dropna(subset=["mean_term_depth", "lag_years"]).copy()
    if len(d) < 5:
        return
    rho, p = stats.spearmanr(d["mean_term_depth"], d["lag_years"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(d["mean_term_depth"], d["lag_years"],
               s=14, alpha=0.35, color=PALETTE["core"], edgecolor="none")

    bins = np.linspace(d["mean_term_depth"].min(),
                       d["mean_term_depth"].max(), 8)
    centers, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = d[(d["mean_term_depth"] >= lo) & (d["mean_term_depth"] < hi)]
        if len(sub) >= 10:
            centers.append((lo + hi) / 2)
            meds.append(sub["lag_years"].median())
    if centers:
        ax.plot(centers, meds, "-o", color=PALETTE["accent"], lw=2.2,
                ms=6, label="bin median")

    ax.set_xlabel("Mean HPO term depth from root "
                  "(higher = more specific terms)")
    ax.set_ylabel("Lag (years)")
    sig_str = "n.s." if p >= 0.05 else f"p = {p:.2e}"
    txt = f"Spearman ρ = {rho:+.3f}\n{sig_str}\nN = {len(d):,}"
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.85, edgecolor=PALETTE["grid"]))
    ax.set_title("HPO term specificity vs diagnostic lag")
    ax.legend(frameon=False, loc="lower right")
    _despine(ax)
    _save(fig, "fig16_term_specificity")
    print(f"  [FIG16] term specificity: rho={rho:+.3f} p={p:.2e} N={len(d)}")


def fig13_unsolved_backlog(df: pd.DataFrame, x_end: int = 2026):
    """Cumulative curves with the Orphanet-catalogue ceiling and the
    current unsolved-backlog annotation.

    The same `df` as fig4, but the figure additionally shows:
      - dotted ceiling at the total Orphanet rare-disease catalogue
        size (~4,713 diseases of disorder_type=Disease)
      - the current backlog = N_orphanet − N_with_curated_validity,
        annotated with an arrow at year x_end

    This makes the limitation of the analysis sample explicit:
    figure 4 only counts the diseases that already have a curated
    G2P/ClinGen gene–disease entry; the real-world unsolved set
    (clinical description present, gene not yet curated) is much
    larger.
    """
    _setup_style()
    if len(df) == 0:
        return

    def cumulative(series: pd.Series) -> pd.Series:
        s = series.dropna().astype(int)
        s = s[s >= 1946].sort_values()
        if len(s) == 0:
            return pd.Series(dtype=int)
        cum = pd.Series(range(1, len(s) + 1), index=s.values)
        return cum.groupby(cum.index).max()

    clin = cumulative(df["first_clinical_year"])
    gen = cumulative(df["first_genetic_year"])
    if len(clin) == 0 or len(gen) == 0:
        return
    if clin.index.max() < x_end:
        clin = pd.concat([clin, pd.Series([clin.iloc[-1]], index=[x_end])])
    if gen.index.max() < x_end:
        gen = pd.concat([gen, pd.Series([gen.iloc[-1]], index=[x_end])])

    conn = get_conn()
    n_orphanet = conn.execute(
        "SELECT COUNT(*) FROM disorders WHERE disorder_type='Disease'"
    ).fetchone()[0]
    conn.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(clin.index, clin.values, alpha=0.25, color=PALETTE["clinical"])
    ax.plot(clin.index, clin.values, color=PALETTE["clinical"], lw=2.5,
            label=f"Clinical descriptions in sample (N = {int(clin.iloc[-1]):,})")
    ax.fill_between(gen.index, gen.values, alpha=0.25, color=PALETTE["genetic"])
    ax.plot(gen.index, gen.values, color=PALETTE["genetic"], lw=2.5,
            label=f"Genetic verifications (N = {int(gen.iloc[-1]):,})")

    ax.axhline(n_orphanet, color=PALETTE["grid"], ls=":", lw=1.5,
               label=f"Orphanet catalogue ({n_orphanet:,} diseases)")

    backlog = n_orphanet - int(clin.iloc[-1])
    pct_solved = 100.0 * int(clin.iloc[-1]) / n_orphanet
    ax.annotate(
        f"unsolved as of {x_end}: {backlog:,} diseases\n"
        f"({100 - pct_solved:.1f}% of Orphanet catalogue\nstill without curated gene-disease validity)",
        xy=(x_end, n_orphanet),
        xytext=(x_end - 28, n_orphanet - (n_orphanet - clin.iloc[-1]) * 0.45),
        fontsize=10, color="dimgray", va="top", ha="left",
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_xlim(1946, x_end)
    ax.set_ylim(0, n_orphanet * 1.08)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative number of diseases")
    ax.set_title("Rare-disease curation backlog: clinical vs molecular vs catalogue")
    ax.legend(frameon=False, loc="center left")
    _despine(ax)
    _save(fig, "fig13_unsolved_backlog")
    print(f"  [FIG13] backlog: solved={int(clin.iloc[-1])}, "
          f"unsolved={backlog}, total_orphanet={n_orphanet}")


def fig5_lag_by_class(df: pd.DataFrame):
    """Horizontal boxplot of lag by Orphanet top-level disease class."""
    _setup_style()
    d = df.dropna(subset=["disorder_class"]).copy()
    if len(d) == 0:
        return
    top = d["disorder_class"].value_counts().head(10).index.tolist()
    d = d[d["disorder_class"].isin(top)]
    medians = d.groupby("disorder_class")["lag_years"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    groups = [d[d["disorder_class"] == c]["lag_years"].values for c in order]
    bp = ax.boxplot(
        groups, positions=range(len(order)), vert=False, widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        boxprops=dict(facecolor=PALETTE["core"], alpha=0.7, edgecolor=PALETTE["ink"]),
        whiskerprops=dict(color=PALETTE["ink"]),
        capprops=dict(color=PALETTE["ink"]),
    )
    def _short(s):
        s = s.replace("Rare ", "").replace(" disease", "")
        return s[:1].upper() + s[1:] if s else s

    # Bake n + median into the y-tick labels so they never overlap
    # with whisker tails or scatter points (the previous version put
    # them inside the axes at xmax * 0.99, which collided with the
    # right-most boxes for skin/ophthalmic).
    ylabels = []
    for c in order:
        n = len(d[d["disorder_class"] == c])
        ylabels.append(f"{_short(c)}\nn = {n}, med = {medians[c]:.0f} y")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Lag (years)")
    ax.set_title("Lag by disease category (top 10)")
    _despine(ax)
    fig.subplots_adjust(left=0.28)
    _save(fig, "fig5_lag_by_class")
    print(f"  [FIG5] lag by class: {len(order)} classes")


def fig6_inheritance_lag(df: pd.DataFrame):
    """Lag distribution by primary inheritance pattern."""
    _setup_style()
    d = df.dropna(subset=["inheritance"]).copy()
    if len(d) == 0:
        return

    def primary(s):
        # Order matters: X-linked / mitochondrial must be checked
        # before generic 'recessive'/'dominant', otherwise
        # 'X-linked recessive' is wrongly bucketed as 'Autosomal
        # recessive' (which previously hid all 71 X-linked diseases).
        # We scan ALL semicolon-separated tags, not only the first,
        # so an entry tagged 'Autosomal dominant; X-linked recessive'
        # is recognised as X-linked.
        tags = [t.strip().lower() for t in s.split(";")]
        if any("x-linked" in t for t in tags):
            return "X-linked"
        if any("mitochondrial" in t for t in tags):
            return "Mitochondrial"
        if any("y-linked" in t for t in tags):
            return "Y-linked"
        first = tags[0]
        if "recessive" in first:
            return "Autosomal recessive"
        if "dominant" in first:
            return "Autosomal dominant"
        return None

    d["inh"] = d["inheritance"].apply(primary)
    d = d.dropna(subset=["inh"])
    if len(d) == 0:
        return
    counts = d["inh"].value_counts()
    keep = counts[counts >= 5].index
    d = d[d["inh"].isin(keep)]
    if len(d) == 0:
        return
    order = d.groupby("inh")["lag_years"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(9, 6.2))
    groups = [d[d["inh"] == k]["lag_years"].values for k in order]
    bp = ax.boxplot(
        groups, positions=range(len(order)), widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        boxprops=dict(facecolor=PALETTE["core"], alpha=0.7, edgecolor=PALETTE["ink"]),
        whiskerprops=dict(color=PALETTE["ink"]),
        capprops=dict(color=PALETTE["ink"]),
    )
    for i, g in enumerate(groups):
        jitter = np.random.uniform(-0.12, 0.12, len(g))
        ax.scatter(i + jitter, g, s=8, alpha=0.25, color=PALETTE["core"], zorder=3)

    # Put n + median directly into the x-tick labels — they used to
    # be a separate text layer near the bottom and collided with
    # boxplot whiskers / jitter / footnote.
    xlabels = []
    for k in order:
        sub = d[d["inh"] == k]["lag_years"]
        xlabels.append(f"{k}\nn = {len(sub)}, med = {sub.median():.0f} y")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Lag (years)")

    # ---- Statistical tests --------------------------------------------
    # Kruskal–Wallis: overall difference of lag medians across groups
    # Pairwise Mann–Whitney U with Bonferroni correction.  Distributions
    # are right-skewed (lag in years) and group sizes differ a lot, so
    # nonparametric tests are the appropriate choice.
    kw_stat, kw_p = stats.kruskal(*[g for g in groups if len(g) > 0])
    n_pairs = len(order) * (len(order) - 1) // 2

    pairwise = []  # list of (i, j, p_corr)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            gi, gj = groups[i], groups[j]
            if len(gi) >= 3 and len(gj) >= 3:
                _, p = stats.mannwhitneyu(gi, gj, alternative="two-sided")
                p_corr = min(1.0, p * n_pairs)  # Bonferroni
                pairwise.append((i, j, p_corr))

    # Annotate sig brackets above boxes
    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    sig_pairs = [(i, j, p) for i, j, p in pairwise if p < 0.05]
    sig_pairs.sort(key=lambda t: (t[1] - t[0], t[0]))  # short brackets first

    lo, hi = ax.get_ylim()
    span = hi - lo
    bracket_y_start = hi
    step = span * 0.06
    for k, (i, j, p) in enumerate(sig_pairs):
        y = bracket_y_start + step * k
        ax.plot([i, i, j, j], [y - step * 0.15, y, y, y - step * 0.15],
                color=PALETTE["ink"], lw=1.0)
        ax.text((i + j) / 2, y + step * 0.08, stars(p),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    if sig_pairs:
        ax.set_ylim(lo, bracket_y_start + step * (len(sig_pairs) + 1.2))
    else:
        ax.set_ylim(lo, hi + 0.08 * span)

    # Title with Kruskal–Wallis result
    kw_str = (f"Kruskal–Wallis χ² = {kw_stat:.1f}, "
              f"p = {kw_p:.1e}" if kw_p < 0.001
              else f"Kruskal–Wallis χ² = {kw_stat:.1f}, p = {kw_p:.3f}")
    ax.set_title(f"Lag by inheritance pattern\n{kw_str}")

    _despine(ax)

    # Significance-encoding footnote outside the axes (below xlabels)
    fig.text(0.99, 0.01,
             "* p<0.05   ** p<0.01   *** p<0.001   (Bonferroni-adjusted)",
             ha="right", va="bottom",
             fontsize=8, color=PALETTE["ink"], alpha=0.6)
    fig.subplots_adjust(bottom=0.18)

    _save(fig, "fig6_inheritance_lag")
    print(f"  [FIG6] inheritance: {len(order)} groups, "
          f"KW p={kw_p:.2e}, "
          f"{len(sig_pairs)}/{n_pairs} significant pairs")


def fig7_constraint_vs_lag(df: pd.DataFrame):
    """LOEUF and pLI vs lag — does gene constraint track diagnostic delay?"""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # LOEUF
    ax = axes[0]
    d = df.dropna(subset=["loeuf", "lag_years"]).copy()
    d = d[(d["loeuf"] > 0) & (d["loeuf"] < 2)]
    if len(d) >= 20:
        ax.scatter(d["loeuf"], d["lag_years"], s=14, alpha=0.35,
                   color=PALETTE["core"], edgecolors="none")
        z = np.polyfit(d["loeuf"], d["lag_years"], 1)
        xs = np.linspace(d["loeuf"].min(), d["loeuf"].max(), 80)
        ax.plot(xs, np.poly1d(z)(xs), color=PALETTE["accent"], lw=2, ls="--")
        r, p = stats.spearmanr(d["loeuf"], d["lag_years"])
        ax.text(0.04, 0.95,
                f"Spearman ρ = {r:.2f}\np = {p:.1e}\nN = {len(d)}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          alpha=0.85, edgecolor=PALETTE["grid"]))
    ax.set_xlabel("LOEUF (lower = more constrained)")
    ax.set_ylabel("Lag (years)")
    ax.set_title("LOEUF vs lag")
    _despine(ax)

    # pLI
    ax = axes[1]
    d = df.dropna(subset=["pli", "lag_years"]).copy()
    bins = [0, 0.1, 0.5, 0.9, 1.01]
    labels = ["<0.1", "0.1–0.5", "0.5–0.9", "≥0.9"]
    d["bin"] = pd.cut(d["pli"], bins=bins, labels=labels, include_lowest=True)
    d = d.dropna(subset=["bin"])
    if len(d) >= 20:
        groups = [d[d["bin"] == b]["lag_years"].values for b in labels]
        bp = ax.boxplot(
            groups, positions=range(len(labels)), widths=0.6,
            patch_artist=True, showfliers=False,
            medianprops=dict(color=PALETTE["accent"], lw=2),
            boxprops=dict(facecolor=PALETTE["genetic"], alpha=0.7,
                          edgecolor=PALETTE["ink"]),
            whiskerprops=dict(color=PALETTE["ink"]),
            capprops=dict(color=PALETTE["ink"]),
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))
        y_label = ax.get_ylim()[1] * 0.98
        for i, b in enumerate(labels):
            sub = d[d["bin"] == b]["lag_years"]
            ax.text(i, y_label, f"n = {len(sub)}",
                    ha="center", va="top", fontsize=9,
                    color=PALETTE["ink"], alpha=0.65)
    ax.set_xlabel("pLI bin")
    ax.set_ylabel("Lag (years)")
    ax.set_title("pLI vs lag")
    _despine(ax)

    fig.tight_layout()
    _save(fig, "fig7_constraint_vs_lag")
    print(f"  [FIG7] constraint vs lag")


def fig8_hpo_complexity(df: pd.DataFrame):
    """Number of HPO phenotype terms vs lag."""
    _setup_style()
    d = df.dropna(subset=["n_hpo_terms", "lag_years"]).copy()
    d = d[d["n_hpo_terms"] > 0]
    if len(d) < 20:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(d["n_hpo_terms"], d["lag_years"], s=14, alpha=0.35,
               color=PALETTE["genetic"], edgecolors="none")
    r, p = stats.spearmanr(d["n_hpo_terms"], d["lag_years"])
    ax.text(0.96, 0.95,
            f"Spearman ρ = {r:.2f}\np = {p:.1e}\nN = {len(d)}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.85, edgecolor=PALETTE["grid"]))
    ax.set_xlabel("Number of HPO phenotype terms")
    ax.set_ylabel("Lag (years)")
    ax.set_title("Phenotypic complexity vs lag")
    _despine(ax)
    _save(fig, "fig8_hpo_complexity")
    print(f"  [FIG8] HPO complexity: N={len(d)}")


def fig9_confidence_lag(df: pd.DataFrame):
    """G2P evidence confidence vs lag (core cohort only)."""
    _setup_style()
    d = df.dropna(subset=["confidence"]).copy()
    order = ["definitive", "strong", "moderate", "limited"]
    d = d[d["confidence"].isin(order)]
    if len(d) < 10:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups = [d[d["confidence"] == c]["lag_years"].values for c in order]
    bp = ax.boxplot(
        groups, positions=range(len(order)), widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        boxprops=dict(facecolor=PALETTE["clinical"], alpha=0.7,
                      edgecolor=PALETTE["ink"]),
        whiskerprops=dict(color=PALETTE["ink"]),
        capprops=dict(color=PALETTE["ink"]),
    )
    for i, g in enumerate(groups):
        if len(g):
            jitter = np.random.uniform(-0.12, 0.12, len(g))
            ax.scatter(i + jitter, g, s=7, alpha=0.3,
                       color=PALETTE["clinical"], zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Lag (years)")
    ax.set_title("G2P evidence confidence vs lag")
    _despine(ax)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    y_label = ax.get_ylim()[1] * 0.98
    for i, c in enumerate(order):
        sub = d[d["confidence"] == c]["lag_years"]
        ax.text(i, y_label,
                f"n = {len(sub)}" + (f"\nmed = {sub.median():.0f} y" if len(sub) else ""),
                ha="center", va="top", fontsize=9,
                color=PALETTE["ink"], alpha=0.65)
    _save(fig, "fig9_confidence_lag")
    print(f"  [FIG9] confidence: groups {[len(g) for g in groups]}")


def fig10_clin_source(df: pd.DataFrame):
    """Horizontal bar of clinical-year source distribution (+ median lag per
    source) — makes data provenance explicit."""
    _setup_style()
    d = df.dropna(subset=["clin_source"]).copy()
    if len(d) == 0:
        return
    order = ["OMIM_elink", "HPO_non_g2p", "HPO_fallback", "MIM_breakpoint"]
    order = [o for o in order if (d["clin_source"] == o).any()]
    counts = [int((d["clin_source"] == o).sum()) for o in order]
    meds = [d[d["clin_source"] == o]["lag_years"].median() for o in order]
    color_map = {
        "OMIM_elink": PALETTE["core"],
        "HPO_non_g2p": PALETTE["genetic"],
        "HPO_fallback": PALETTE["flag"],
        "MIM_breakpoint": PALETTE["accent"],
    }
    colors = [color_map.get(o, PALETTE["core"]) for o in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = list(range(len(order)))[::-1]
    ax.barh(y, counts, color=colors, edgecolor="white")
    for yi, n, m, lab in zip(y, counts, meds, order):
        txt = f"  n = {n}   median lag = {m:.0f} y"
        ax.text(n, yi, txt, va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Diseases")
    ax.set_title("Clinical-year source used in the main sample")
    _despine(ax)
    ax.set_xlim(0, max(counts) * 1.35)
    _save(fig, "fig10_clin_source")
    print(f"  [FIG10] clin source: {dict(zip(order, counts))}")


def fig11_top_lag(df: pd.DataFrame):
    """Top 20 diseases with the longest lag — poster sidebar."""
    _setup_style()
    top = df.nlargest(20, "lag_years").sort_values("lag_years")
    if len(top) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    y = list(range(len(top)))
    ax.barh(y, top["lag_years"].values, color=PALETTE["accent"], alpha=0.85,
            edgecolor="white")
    for yi, row in zip(y, top.itertuples()):
        ax.text(row.lag_years, yi,
                f"  {int(row.lag_years)} y   "
                f"{int(row.first_clinical_year)} → {int(row.first_genetic_year)}",
                va="center", fontsize=9)

    labels = []
    for row in top.itertuples():
        name = row.name if row.name else row.orpha_code
        if len(name) > 48:
            name = name[:45] + "…"
        gene = f" ({row.gene_symbol})" if row.gene_symbol else ""
        labels.append(name + gene)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Lag (years)")
    ax.set_title("Top 20 longest phenotype-to-genotype odysseys")
    _despine(ax)
    ax.set_xlim(0, top["lag_years"].max() * 1.25)
    _save(fig, "fig11_top_lag")
    print(f"  [FIG11] top 20: max lag = {int(top['lag_years'].max())}")


def fig12_decade_rate(df: pd.DataFrame):
    """Discovery rate per decade — number of diseases first clinically
    described vs genetically verified in each decade."""
    _setup_style()
    d = df.dropna(subset=["first_clinical_year", "first_genetic_year"]).copy()
    if len(d) == 0:
        return
    d["clin_dec"] = (d["first_clinical_year"] // 10 * 10).astype(int)
    d["gen_dec"] = (d["first_genetic_year"] // 10 * 10).astype(int)

    clin_counts = d["clin_dec"].value_counts().sort_index()
    gen_counts = d["gen_dec"].value_counts().sort_index()

    decades = sorted(set(clin_counts.index) | set(gen_counts.index))
    decades = [dd for dd in decades if dd >= 1940]
    clin_vals = [int(clin_counts.get(dd, 0)) for dd in decades]
    gen_vals = [int(gen_counts.get(dd, 0)) for dd in decades]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(decades))
    w = 0.4
    ax.bar(x - w / 2, clin_vals, width=w, color=PALETTE["clinical"],
           edgecolor="white", label="First clinical description")
    ax.bar(x + w / 2, gen_vals, width=w, color=PALETTE["genetic"],
           edgecolor="white", label="Genetic verification")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dd}s" for dd in decades])
    ax.set_xlabel("Decade")
    ax.set_ylabel("Diseases")
    ax.set_title("Discovery rate per decade")
    ax.legend(loc="upper left")
    _despine(ax)
    for xi, v in zip(x - w / 2, clin_vals):
        if v:
            ax.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=8,
                    color=PALETTE["ink"], alpha=0.7)
    for xi, v in zip(x + w / 2, gen_vals):
        if v:
            ax.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=8,
                    color=PALETTE["ink"], alpha=0.7)
    _save(fig, "fig12_decade_rate")
    print(f"  [FIG12] decade rate: {len(decades)} decades")


def fig_historical_review():
    """Bar chart of dating-flag reasons + CSV export for manual curation."""
    _setup_style()
    df = load_all()
    suspects = df[df["suspect_dating"] == 1].copy()
    if len(suspects) == 0:
        print("  [FIG] no suspect rows — historical review skipped")
        return

    reason_counts: dict[str, int] = {}
    for r in suspects["suspect_reason"].dropna():
        for token in str(r).split(","):
            t = token.strip()
            if t:
                reason_counts[t] = reason_counts.get(t, 0) + 1

    order = sorted(reason_counts, key=reason_counts.get, reverse=True)
    vals = [reason_counts[k] for k in order]
    color_map = {
        "neg_lag": PALETTE["accent"],
        "pre_medline": PALETTE["genetic"],
        "mim_breakpoint": PALETTE["flag"],
    }
    colors = [color_map.get(k, PALETTE["core"]) for k in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(order, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:,}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Diseases flagged")
    ax.set_title("Dating flags — excluded from main sample, kept for manual review")
    _despine(ax)
    _save(fig, "fig_historical_review")

    cols = ["orpha_code", "name", "omim_id", "gene_symbol",
            "first_clinical_pmid", "first_clinical_year", "clin_source",
            "first_genetic_pmid", "first_genetic_year", "lag_years",
            "suspect_reason", "gene_sources"]
    suspects[cols].sort_values(
        ["suspect_reason", "first_clinical_year"]
    ).to_csv(FIGURES / "historical_review.csv", index=False)
    print(f"  [FIG] historical review: {len(suspects):,} flagged → historical_review.csv")


# ---------- primary (MEDLINE-era) variants of the most-affected figures ----------

# Earliest year for which MEDLINE indexing is reliable.  Clinical
# descriptions before this come from OMIM history references or
# manual curation against pre-MEDLINE literature; they are valid but
# require human verification, so we exclude them from the *primary*
# poster figure and present them in a sidebar.
PRIMARY_CLIN_FLOOR = 1946


def _save_with_subdir(fig, subdir: str, name: str):
    """Save into figures/<subdir>/ instead of figures/."""
    out = FIGURES / subdir
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig1_primary(df_primary: pd.DataFrame):
    """fig1 on the MEDLINE-era subset (clin_year >= 1946)."""
    _setup_style()
    lag = df_primary["lag_years"].dropna()
    lag = lag[lag >= 0]
    if len(lag) == 0:
        return
    lo, hi = 0, int(lag.max()) + 5
    bins = np.arange(lo, hi + 5, 5)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(lag, bins=bins, color=PALETTE["core"], alpha=0.75,
            edgecolor="white", linewidth=0.5, density=True)
    if len(lag) >= 5 and lag.std() > 0:
        kde_x = np.linspace(0, lag.max(), 300)
        kde = stats.gaussian_kde(lag)
        ax.plot(kde_x, kde(kde_x), color=PALETTE["accent"], lw=2.5)
    med = lag.median()
    ax.axvline(med, color=PALETTE["genetic"], ls="--", lw=2,
               label=f"median = {med:.0f} y")
    stats_text = (
        f"N = {len(lag):,}\n"
        f"median = {med:.0f} y\n"
        f"IQR = [{lag.quantile(0.25):.0f}, {lag.quantile(0.75):.0f}] y\n"
        f"mean = {lag.mean():.1f} y\n"
        f"max = {int(lag.max())} y"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      alpha=0.85, edgecolor=PALETTE["grid"]))
    ax.set_xlim(0, hi)
    ax.set_xlabel("Lag (years): genetic verification − clinical description")
    ax.set_ylabel("Density")
    ax.set_title(f"Phenotype-to-genotype lag — MEDLINE era (clin ≥ {PRIMARY_CLIN_FLOOR})")
    ax.legend(frameon=False, loc="upper center")
    _despine(ax)
    _save_with_subdir(fig, "primary", "fig1_lag_distribution")


def fig_historical_top(df_full: pd.DataFrame, n_top: int = 12):
    """Sidebar figure: pre-1900 clinical descriptions (manual_curation).

    Lists the n_top diseases with the longest lag among entries whose
    clinical year predates 1900.  Designed as a small inset/sidebar
    next to the main figure.
    """
    _setup_style()
    pre1900 = df_full[
        (df_full["first_clinical_year"].notna())
        & (df_full["first_clinical_year"] < 1900)
        & (df_full["lag_years"].notna())
    ].copy()
    if len(pre1900) == 0:
        return
    top = pre1900.nlargest(n_top, "lag_years").iloc[::-1]  # bottom = longest

    fig, ax = plt.subplots(figsize=(8, 0.45 * n_top + 1.4))
    y = np.arange(len(top))
    ax.barh(y, top["lag_years"], color=PALETTE["accent"],
            edgecolor="white", linewidth=0.5)
    for yi, (_, row) in zip(y, top.iterrows()):
        label = (
            f"{row['name'][:42]} ({row['gene_symbol']}) "
            f"— clin {int(row['first_clinical_year'])} → "
            f"gen {int(row['first_genetic_year'])}"
        )
        ax.text(row["lag_years"] + 1, yi,
                f" {int(row['lag_years'])} y",
                va="center", fontsize=10, fontweight="bold")
        ax.text(0, yi, " " + label, va="center", fontsize=9,
                color="white" if row["lag_years"] > 30 else "black")
    ax.set_yticks([])
    ax.set_xlabel("Lag (years)")
    ax.set_title(
        f"Historical perspective: top {len(top)} pre-1900 clinical "
        f"descriptions (curated against OMIM History)"
    )
    _despine(ax)
    ax.spines["left"].set_visible(False)
    _save_with_subdir(fig, "primary", "fig_historical_top")


# ---------- orchestrator ----------

def generate_all():
    FIGURES.mkdir(exist_ok=True)
    fig_cohort_cascade()

    df_full = load_main()
    print(f"[VIZ] Main sample (full, all eras): N = {len(df_full):,}")

    # All main figures use the primary (MEDLINE-era) subset by default.
    # Pre-1946 curated cases (Wardrop 1809 etc.) are shown only on the
    # historical-perspective sidebar — they no longer skew the
    # distribution figures, era plots, top-lag bars or summary stats.
    df = df_full[df_full["first_clinical_year"] >= PRIMARY_CLIN_FLOOR].copy()
    print(f"[VIZ] Primary sample (clin >= {PRIMARY_CLIN_FLOOR}): N = {len(df):,}")

    if len(df) > 0:
        fig1_lag_distribution(df)
        fig2_lag_by_era(df)
        fig3_timeline_scatter(df)
        fig4_cumulative(df)
        fig5_lag_by_class(df)
        fig6_inheritance_lag(df)
        fig7_constraint_vs_lag(df)
        fig8_hpo_complexity(df)
        fig9_confidence_lag(df)
        fig10_clin_source(df)
        fig11_top_lag(df)
        fig12_decade_rate(df)
        fig13_unsolved_backlog(df)
        fig14_phenotype_focus(df)
        fig15_organ_systems(df)
        fig16_term_specificity(df)

        lag = df["lag_years"].dropna()
        print("\n" + "=" * 56)
        print(f"MAIN SAMPLE SUMMARY (primary tier, clin >= {PRIMARY_CLIN_FLOOR})")
        print("=" * 56)
        print(f"  N diseases      : {len(df):,}")
        print(f"  median lag      : {lag.median():.0f} y")
        print(f"  IQR             : [{lag.quantile(0.25):.0f}, "
              f"{lag.quantile(0.75):.0f}] y")
        print(f"  mean lag        : {lag.mean():.1f} y")
        print(f"  max lag         : {int(lag.max())} y")
        print(f"  lag > 20 y      : {(lag > 20).sum():,} "
              f"({(lag > 20).mean() * 100:.1f}%)")
        print(f"  lag < 5 y       : {(lag < 5).sum():,} "
              f"({(lag < 5).mean() * 100:.1f}%)")
        print("=" * 56)
    else:
        print("[VIZ] empty primary sample — only cohort cascade rendered")

    fig_historical_review()

    # ---- Historical sidebar (pre-1900 curated cases) --------------------
    # The full-tier sample (df_full) is used here only — these diseases
    # are NOT in the main figures above.
    if len(df_full) > len(df):
        fig_historical_top(df_full)
        n_historical = len(df_full) - len(df)
        lag_full = df_full["lag_years"].dropna()
        print("\n" + "=" * 56)
        print("HISTORICAL EXTENSION (pre-MEDLINE curated cases)")
        print("=" * 56)
        print(f"  N additional      : {n_historical}")
        print(f"  Full-tier N       : {len(df_full):,}")
        print(f"  Full-tier max lag : {int(lag_full.max())} y")
        print("=" * 56)
        print(f"  → figures/primary/fig_historical_top.{{pdf,png}}")
