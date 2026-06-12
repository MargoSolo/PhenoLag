"""Visualizations for Undiagnosed Disease Prioritization."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pipeline.config import FIGURES

PALETTE = {
    "bg": "#FAFAFA", "accent1": "#2C73D2", "accent2": "#FF6F61",
    "accent3": "#45B7A0", "accent4": "#FFC75F", "accent5": "#845EC2",
    "text": "#2D2D2D", "grid": "#E0E0E0",
    "critical": "#D32F2F", "high": "#FF6F61", "moderate": "#FFC75F", "low": "#B0BEC5",
}

TIER_COLORS = {
    "Critical": PALETTE["critical"],
    "High": PALETTE["high"],
    "Moderate": PALETTE["moderate"],
    "Low": PALETTE["low"],
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
# Fig 16: Bubble chart — lag vs severity, sized by tractability
# ============================================================

def fig16_bubble_chart(df: pd.DataFrame):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 8))

    df16 = df.dropna(subset=["lag_years", "severity_score", "tractability_score"]).copy()
    df16 = df16[df16["lag_years"] >= 0]

    # Size = tractability, color = priority tier
    sizes = df16["tractability_score"] * 200 + 10
    colors = df16["priority_tier"].map(TIER_COLORS).fillna(PALETTE["low"])

    ax.scatter(df16["lag_years"], df16["severity_score"],
               s=sizes, c=colors, alpha=0.6, edgecolors="white", linewidth=0.5)

    # Label top 10 critical
    top = df16[df16["priority_tier"] == "Critical"].nsmallest(10, "priority_rank")
    for _, row in top.iterrows():
        label = row["name"]
        if len(label) > 35:
            label = label[:32] + "..."
        ax.annotate(label, (row["lag_years"], row["severity_score"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Diagnostic lag (years)")
    ax.set_ylabel("Severity score (onset-based)")
    ax.set_title("Undiagnosed disease prioritization: lag vs severity")

    # Legend for tiers
    for tier, color in TIER_COLORS.items():
        ax.scatter([], [], c=color, s=80, label=tier, alpha=0.7)
    ax.legend(title="Priority tier", frameon=False, fontsize=10, loc="upper right")

    # Legend for size
    ax.text(0.02, 0.02, "Bubble size = genetic tractability\n(pLI + LOEUF)",
            transform=ax.transAxes, fontsize=9, color="gray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES / "fig16_priority_bubble.png")
    fig.savefig(FIGURES / "fig16_priority_bubble.pdf")
    plt.close(fig)
    print("  [FIG16] Priority bubble chart saved")


# ============================================================
# Fig 17: Top-30 ranked diseases (horizontal lollipop)
# ============================================================

def fig17_top_diseases(df: pd.DataFrame):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 10))

    top = df.head(30).copy()
    top = top.iloc[::-1]  # reverse for bottom-to-top

    y_pos = np.arange(len(top))
    colors = top["priority_tier"].map(TIER_COLORS).fillna(PALETTE["low"]).values

    # Lollipop
    ax.hlines(y_pos, 0, top["priority_score"], colors=colors, lw=2.5, alpha=0.7)
    ax.scatter(top["priority_score"], y_pos, c=colors, s=80, zorder=5, edgecolors="white")

    # Labels
    labels = []
    for _, row in top.iterrows():
        name = row["name"]
        if len(name) > 45:
            name = name[:42] + "..."
        gene = row["gene_symbol"] if pd.notna(row["gene_symbol"]) else ""
        lag = int(row["lag_years"]) if pd.notna(row["lag_years"]) else "?"
        labels.append(f"{name} ({gene}, lag={lag}y)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Composite priority score")
    ax.set_title("Top 30 diseases for undiagnosed disease prioritization")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tier legend
    for tier, color in TIER_COLORS.items():
        ax.scatter([], [], c=color, s=80, label=tier, alpha=0.7)
    ax.legend(title="Priority tier", frameon=False, fontsize=10, loc="lower right")

    fig.savefig(FIGURES / "fig17_top_diseases.png")
    fig.savefig(FIGURES / "fig17_top_diseases.pdf")
    plt.close(fig)
    print("  [FIG17] Top 30 diseases saved")


# ============================================================
# Fig 18: Radar/spider chart for top 5 diseases
# ============================================================

def fig18_radar_top5(df: pd.DataFrame):
    _setup()

    top5 = df.head(5).copy()
    categories = ["Diagnostic\nlag", "Severity\n(onset)", "Genetic\ntractability",
                   "Unmet\nneed", "Evidence\ndepth"]
    score_cols = ["lag_score", "severity_score", "tractability_score",
                  "need_score", "evidence_score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
              PALETTE["accent5"], PALETTE["accent4"]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[c] for c in score_cols]
        values += values[:1]
        name = row["name"]
        if len(name) > 35:
            name = name[:32] + "..."
        ax.plot(angles, values, "o-", lw=2, color=colors[i], label=name, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.set_title("Priority profile: top 5 diseases", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=9)

    fig.savefig(FIGURES / "fig18_radar_top5.png")
    fig.savefig(FIGURES / "fig18_radar_top5.pdf")
    plt.close(fig)
    print("  [FIG18] Radar chart saved")


# ============================================================
# Fig 19: Priority score distribution by disease class
# ============================================================

def fig19_priority_by_class(df: pd.DataFrame):
    _setup()
    fig, ax = plt.subplots(figsize=(12, 7))

    df19 = df.dropna(subset=["disorder_class"]).copy()
    top_cls = df19["disorder_class"].value_counts().head(10).index
    df19 = df19[df19["disorder_class"].isin(top_cls)]

    medians = df19.groupby("disorder_class")["priority_score"].median().sort_values(ascending=False)
    order = medians.index.tolist()

    def _short(s):
        s = s.replace("Rare ", "").replace(" disease", "")
        return s[0].upper() + s[1:] if s else s

    bp = ax.boxplot(
        [df19[df19["disorder_class"] == c]["priority_score"].values for c in order],
        vert=False, widths=0.6, patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
        medianprops=dict(color=PALETTE["critical"], lw=2),
    )
    palette = sns.color_palette("coolwarm_r", len(order))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([_short(c) for c in order], fontsize=11)
    ax.set_xlabel("Composite priority score")
    ax.set_title("Priority score distribution by disease category")

    # Mark critical threshold
    ax.axvline(0.6, color=PALETTE["critical"], ls="--", lw=1, alpha=0.7, label="Critical threshold")
    ax.legend(frameon=False, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, c in enumerate(order):
        n = len(df19[df19["disorder_class"] == c])
        n_crit = ((df19["disorder_class"] == c) & (df19["priority_tier"] == "Critical")).sum()
        ax.text(ax.get_xlim()[1] * 0.98, i + 1, f"n={n}, crit={n_crit}",
                va="center", fontsize=9, color="gray")

    fig.savefig(FIGURES / "fig19_priority_by_class.png")
    fig.savefig(FIGURES / "fig19_priority_by_class.pdf")
    plt.close(fig)
    print("  [FIG19] Priority by class saved")


# ============================================================
# Fig 20: Component contribution stacked bar (top 20)
# ============================================================

def fig20_component_breakdown(df: pd.DataFrame):
    _setup()
    fig, ax = plt.subplots(figsize=(14, 8))

    top20 = df.head(20).copy()
    top20 = top20.iloc[::-1]

    score_cols = ["lag_score", "severity_score", "tractability_score",
                  "need_score", "evidence_score"]
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    comp_names = ["Diagnostic lag", "Severity", "Genetic tractability",
                  "Unmet need", "Evidence depth"]
    comp_colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
                   PALETTE["accent5"], PALETTE["accent4"]]

    y_pos = np.arange(len(top20))
    left = np.zeros(len(top20))

    for col, w, name, color in zip(score_cols, weights, comp_names, comp_colors):
        vals = (top20[col] * w).values
        ax.barh(y_pos, vals, left=left, height=0.6, color=color, alpha=0.8,
                label=name, edgecolor="white", linewidth=0.5)
        left += vals

    labels = []
    for _, row in top20.iterrows():
        name = row["name"]
        if len(name) > 40:
            name = name[:37] + "..."
        labels.append(f"#{int(row['priority_rank'])}  {name}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Weighted priority score")
    ax.set_title("Priority score decomposition: top 20 diseases")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES / "fig20_component_breakdown.png")
    fig.savefig(FIGURES / "fig20_component_breakdown.pdf")
    plt.close(fig)
    print("  [FIG20] Component breakdown saved")


# ============================================================
# Fig 21: Actionable use-case mapping
# ============================================================

def fig21_use_case_quadrant(df: pd.DataFrame):
    """2x2 quadrant: tractability vs lag, annotated by use case."""
    _setup()
    fig, ax = plt.subplots(figsize=(11, 9))

    df21 = df[df["lag_years"].notna() & df["tractability_score"].notna()].copy()
    df21 = df21[df21["lag_years"] >= 0]

    lag_med = df21["lag_years"].median()
    tract_med = df21["tractability_score"].median()

    colors = df21["severity_score"].values
    cmap = LinearSegmentedColormap.from_list("sev",
        [PALETTE["accent3"], PALETTE["accent4"], PALETTE["critical"]])

    sc = ax.scatter(df21["lag_years"], df21["tractability_score"],
                    c=colors, cmap=cmap, s=40, alpha=0.6,
                    edgecolors="white", linewidth=0.3,
                    vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Severity score", fontsize=12)

    # Quadrant lines
    ax.axhline(tract_med, color="gray", ls="--", lw=1, alpha=0.4)
    ax.axvline(lag_med, color="gray", ls="--", lw=1, alpha=0.4)

    # Quadrant labels
    props = dict(fontsize=11, fontweight="bold", alpha=0.7,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()

    ax.text(lag_med + (x_hi - lag_med) * 0.5, tract_med + (y_hi - tract_med) * 0.8,
            "EXOME REANALYSIS\nHigh lag + tractable gene",
            ha="center", color=PALETTE["critical"], **props)

    ax.text(lag_med * 0.4, tract_med + (y_hi - tract_med) * 0.8,
            "NEWBORN SCREENING\nShort lag + tractable gene",
            ha="center", color=PALETTE["accent3"], **props)

    ax.text(lag_med + (x_hi - lag_med) * 0.5, tract_med * 0.4,
            "GENE DISCOVERY\nHigh lag + hard to find",
            ha="center", color=PALETTE["accent5"], **props)

    ax.text(lag_med * 0.4, tract_med * 0.4,
            "MONITORING\nShort lag + low tractability",
            ha="center", color="gray", **props)

    ax.set_xlabel("Diagnostic lag (years)")
    ax.set_ylabel("Genetic tractability score")
    ax.set_title("Actionable use-case mapping for rare diseases")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES / "fig21_use_case_quadrant.png")
    fig.savefig(FIGURES / "fig21_use_case_quadrant.pdf")
    plt.close(fig)
    print("  [FIG21] Use-case quadrant saved")


# ============================================================
# Entry point
# ============================================================

def generate_priority_figures(df: pd.DataFrame):
    """Generate all prioritization figures."""
    FIGURES.mkdir(exist_ok=True)
    print(f"[PRIORITY VIZ] Generating figures for {len(df)} scored diseases")

    fig16_bubble_chart(df)
    fig17_top_diseases(df)
    fig18_radar_top5(df)
    fig19_priority_by_class(df)
    fig20_component_breakdown(df)
    fig21_use_case_quadrant(df)

    print("[PRIORITY VIZ] All prioritization figures generated")
