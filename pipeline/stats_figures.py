"""Advanced biostatistical figures for ESHG poster.

Fig 10: Kaplan-Meier survival analysis
Fig 11: Forest plot from multivariable regression
Fig 12: Sensitivity analysis (lag >= 0)
Fig 13: Missing data analysis
Fig 14: Correlation heatmap with significance
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pipeline.config import FIGURES, SNAPSHOT_YEAR
from pipeline.db import get_conn

PALETTE = {
    "bg": "#FAFAFA", "accent1": "#2C73D2", "accent2": "#FF6F61",
    "accent3": "#45B7A0", "accent4": "#FFC75F", "accent5": "#845EC2",
    "text": "#2D2D2D", "grid": "#E0E0E0",
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


def load_all() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM phenolag", conn)
    conn.close()
    return df


def _primary_inheritance(s):
    if pd.isna(s):
        return None
    s = s.split(";")[0].strip().lower()
    if "recessive" in s:
        return "AR"
    elif "dominant" in s:
        return "AD"
    elif "x-linked" in s:
        return "XL"
    return None


def _era_bin(year):
    if pd.isna(year):
        return None
    y = int(year)
    if y < 1990:
        return "Before 1990"
    elif y < 2000:
        return "1990-1999"
    elif y < 2010:
        return "2000-2009"
    else:
        return "2010+"


# ============================================================
# Fig 10: Kaplan-Meier survival analysis
# ============================================================

def fig10_kaplan_meier(df: pd.DataFrame):
    """KM curves: time from clinical description to genetic verification,
    stratified by era and inheritance."""
    _setup()

    # For KM we need lag >= 0 (time-to-event must be non-negative)
    # Diseases with lag < 0 are right-censored at time 0
    # Diseases without genetic verification are censored
    df_km = df.copy()
    df_km["duration"] = df_km["lag_years"].clip(lower=0)
    df_km["event"] = (df_km["lag_years"].notna() & (df_km["lag_years"] >= 0)).astype(int)
    # Also include diseases without genetic year as censored
    no_gen = df_km["first_genetic_year"].isna() & df_km["first_clinical_year"].notna()
    df_km.loc[no_gen, "duration"] = SNAPSHOT_YEAR - df_km.loc[no_gen, "first_clinical_year"]
    df_km.loc[no_gen, "event"] = 0
    df_km = df_km.dropna(subset=["duration"])
    df_km = df_km[df_km["duration"] >= 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: by era ---
    ax = axes[0]
    df_km["era"] = df_km["first_clinical_year"].apply(_era_bin)
    df_km_era = df_km.dropna(subset=["era"])

    era_order = ["Before 1990", "1990-1999", "2000-2009", "2010+"]
    era_colors = [PALETTE["accent5"], PALETTE["accent1"], PALETTE["accent3"], PALETTE["accent2"]]

    kmf = KaplanMeierFitter()
    for era, color in zip(era_order, era_colors):
        mask = df_km_era["era"] == era
        if mask.sum() < 5:
            continue
        kmf.fit(df_km_era.loc[mask, "duration"],
                event_observed=df_km_era.loc[mask, "event"],
                label=f"{era} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_show=True, ci_alpha=0.15)

    # Log-rank test
    groups_era = df_km_era.dropna(subset=["era"])
    if len(groups_era["era"].unique()) > 1:
        lr = multivariate_logrank_test(groups_era["duration"], groups_era["era"], groups_era["event"])
        ax.text(0.95, 0.95, f"Log-rank p = {lr.p_value:.2e}",
                transform=ax.transAxes, fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Years from clinical description")
    ax.set_ylabel("Probability of no genetic verification")
    ax.set_title("A. Kaplan-Meier by era of clinical description")
    ax.set_xlim(0, 50)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: by inheritance ---
    ax = axes[1]
    df_km["inh"] = df_km["inheritance"].apply(_primary_inheritance)
    df_km_inh = df_km.dropna(subset=["inh"])

    inh_order = ["AD", "AR", "XL"]
    inh_colors = [PALETTE["accent1"], PALETTE["accent3"], PALETTE["accent4"]]
    inh_labels = {"AD": "Autosomal dominant", "AR": "Autosomal recessive", "XL": "X-linked"}

    kmf = KaplanMeierFitter()
    for inh, color in zip(inh_order, inh_colors):
        mask = df_km_inh["inh"] == inh
        if mask.sum() < 5:
            continue
        kmf.fit(df_km_inh.loc[mask, "duration"],
                event_observed=df_km_inh.loc[mask, "event"],
                label=f"{inh_labels[inh]} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_show=True, ci_alpha=0.15)

    # Log-rank
    groups_inh = df_km_inh.dropna(subset=["inh"])
    if len(groups_inh["inh"].unique()) > 1:
        lr = multivariate_logrank_test(groups_inh["duration"], groups_inh["inh"], groups_inh["event"])
        ax.text(0.95, 0.95, f"Log-rank p = {lr.p_value:.2e}",
                transform=ax.transAxes, fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Years from clinical description")
    ax.set_ylabel("Probability of no genetic verification")
    ax.set_title("B. Kaplan-Meier by inheritance pattern")
    ax.set_xlim(0, 50)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_kaplan_meier.png")
    fig.savefig(FIGURES / "fig10_kaplan_meier.pdf")
    plt.close(fig)
    print("  [FIG10] Kaplan-Meier saved")


# ============================================================
# Fig 11: Forest plot from multivariable regression
# ============================================================

def fig11_forest_plot(df: pd.DataFrame):
    """Cox PH model with biological predictors only (no tautological variables).

    Excluded: clinical_year, era, genetic_year — these are part of the outcome
    definition (lag = genetic_year - clinical_year) and would create circular
    reasoning.

    Kept: LOEUF, pLI, n_hpo_terms, inheritance, disease class, G2P confidence
    — genuine biological / knowledge-based predictors.

    Output: 3-panel figure:
      A. Cox PH forest plot (hazard ratios with 95% CI)
      B. VIF collinearity check
      C. Schoenfeld residuals (PH assumption test)
    """
    _setup()

    # --- Prepare data for Cox PH ---
    # Duration = lag (clipped to >= 0); event = 1 if lag >= 0
    df_cox = df.copy()
    df_cox["duration"] = df_cox["lag_years"].clip(lower=0)
    df_cox["event"] = (df_cox["lag_years"].notna() & (df_cox["lag_years"] >= 0)).astype(int)
    # Include censored diseases (no genetic year) with duration from clinical year
    no_gen = df_cox["first_genetic_year"].isna() & df_cox["first_clinical_year"].notna()
    df_cox.loc[no_gen, "duration"] = SNAPSHOT_YEAR - df_cox.loc[no_gen, "first_clinical_year"]
    df_cox.loc[no_gen, "event"] = 0
    df_cox = df_cox.dropna(subset=["duration"])
    df_cox = df_cox[df_cox["duration"] >= 0]
    # Remove zero durations (Cox doesn't handle well)
    df_cox.loc[df_cox["duration"] == 0, "duration"] = 0.5

    # Biological predictors
    df_cox["inh"] = df_cox["inheritance"].apply(_primary_inheritance)
    df_cox["loeuf_z"] = (df_cox["loeuf"] - df_cox["loeuf"].median()) / df_cox["loeuf"].std()
    df_cox["n_hpo_log"] = np.log1p(df_cox["n_hpo_terms"])
    df_cox["n_hpo_log_z"] = (df_cox["n_hpo_log"] - df_cox["n_hpo_log"].mean()) / df_cox["n_hpo_log"].std()

    # Inheritance: AD vs AR only (drop XL — too few for stable estimation)
    df_cox = df_cox[df_cox["inh"].isin(["AD", "AR"])].copy()
    # Fix zero durations AFTER filtering
    df_cox.loc[df_cox["duration"] == 0, "duration"] = 0.5
    df_cox["inh_AD"] = (df_cox["inh"] == "AD").astype(float)

    # Disease classes: only include those with enough cases in this subset
    cls_counts = df_cox["disorder_class"].value_counts()
    for cls_name in cls_counts.index:
        if cls_counts[cls_name] >= 50:
            col = "cls_" + cls_name[:15].replace(" ", "_").lower()
            df_cox[col] = (df_cox["disorder_class"] == cls_name).astype(float)

    # Find which class columns exist
    cls_cols = [c for c in df_cox.columns if c.startswith("cls_")]

    # G2P confidence: definitive vs other (binary, stable)
    df_cox["conf_not_definitive"] = (
        df_cox["confidence"].notna() & (df_cox["confidence"] != "definitive")
    ).astype(float)

    # Select columns for Cox model
    cox_cols = ["duration", "event",
                "loeuf_z", "n_hpo_log_z",
                "inh_AD",
                "conf_not_definitive"] + cls_cols
    df_fit = df_cox[cox_cols].dropna()

    # Drop any columns with near-zero variance (< 2% or > 98%)
    for col in list(cls_cols):
        if col in df_fit.columns:
            frac = df_fit[col].mean()
            if frac < 0.03 or frac > 0.97:
                df_fit = df_fit.drop(columns=[col])
                cox_cols = [c for c in cox_cols if c != col]

    print(f"  [FIG11] Cox PH fitting on N = {len(df_fit)} "
          f"(events = {df_fit['event'].sum():.0f})")

    # Fit Cox PH with L2 penalization for stability
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_fit, duration_col="duration", event_col="event")

    # --- Panel A: Forest plot (hazard ratios) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             gridspec_kw={"width_ratios": [3, 1.2, 1.5]})

    ax = axes[0]
    summary = cph.summary
    hr = np.exp(summary["coef"])
    hr_lo = np.exp(summary["coef"] - 1.96 * summary["se(coef)"])
    hr_hi = np.exp(summary["coef"] + 1.96 * summary["se(coef)"])
    pvals = summary["p"]

    rename = {
        "loeuf_z": "LOEUF (per SD)",
        "n_hpo_log_z": "log(HPO terms) (per SD)",
        "inh_AD": "AD (vs AR)",
        "conf_not_definitive": "G2P not definitive (vs definitive)",
    }
    # Add dynamic class column labels
    for col in cox_cols:
        if col.startswith("cls_") and col not in rename:
            pretty = col.replace("cls_", "").replace("_", " ").strip()
            pretty = pretty[0].upper() + pretty[1:] + " (vs other)"
            rename[col] = pretty
    labels = [rename.get(c, c) for c in summary.index]

    # Sort by HR
    order = np.argsort(hr.values)
    y_pos = np.arange(len(order))

    colors = []
    for p in pvals.values[order]:
        if p < 0.001:
            colors.append(PALETTE["accent2"])
        elif p < 0.05:
            colors.append(PALETTE["accent1"])
        else:
            colors.append(PALETTE["grid"])

    # Plot HRs as points with CI whiskers
    for i, idx in enumerate(order):
        ax.plot([hr_lo.values[idx], hr_hi.values[idx]], [i, i],
                color=colors[i], lw=2, solid_capstyle="round")
        ax.plot(hr.values[idx], i, "o", color=colors[i], markersize=10, zorder=5)

    ax.axvline(1, color=PALETTE["text"], lw=1, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=12)
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_title("A. Cox proportional hazards model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # HR > 1 = faster genetic verification (shorter lag)
    ax.text(0.98, 0.02,
            "HR > 1: faster verification\n"
            f"N = {len(df_fit)}, events = {df_fit['event'].sum():.0f}\n"
            f"Concordance = {cph.concordance_index_:.3f}\n"
            f"Log-likelihood ratio p = {cph.log_likelihood_ratio_test().p_value:.2e}",
            transform=ax.transAxes, fontsize=10, color="gray",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["accent2"], alpha=0.7, label="p < 0.001"),
        mpatches.Patch(facecolor=PALETTE["accent1"], alpha=0.7, label="p < 0.05"),
        mpatches.Patch(facecolor=PALETTE["grid"], alpha=0.7, label="p >= 0.05"),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    # --- Panel B: VIF collinearity ---
    ax = axes[1]
    predictor_cols = [c for c in cox_cols if c not in ("duration", "event")]
    X_vif = df_fit[predictor_cols].copy()
    X_vif = sm.add_constant(X_vif)
    vif_data = []
    for i, col in enumerate(X_vif.columns):
        if col == "const":
            continue
        try:
            vif_val = variance_inflation_factor(X_vif.values, i)
        except Exception:
            vif_val = np.nan
        vif_data.append((rename.get(col, col), vif_val))

    vif_labels = [v[0] for v in vif_data]
    vif_vals = [v[1] for v in vif_data]
    vif_colors = [PALETTE["accent2"] if v > 5 else
                  PALETTE["accent4"] if v > 2.5 else
                  PALETTE["accent3"] for v in vif_vals]

    ax.barh(range(len(vif_labels)), vif_vals, color=vif_colors, alpha=0.7,
            edgecolor="white", height=0.6)
    ax.axvline(5, color=PALETTE["accent2"], ls="--", lw=1, alpha=0.7, label="VIF = 5")
    ax.axvline(2.5, color=PALETTE["accent4"], ls="--", lw=1, alpha=0.7, label="VIF = 2.5")
    ax.set_yticks(range(len(vif_labels)))
    ax.set_yticklabels(vif_labels, fontsize=10)
    ax.set_xlabel("VIF")
    ax.set_title("B. Collinearity (VIF)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel C: Schoenfeld test (PH assumption) ---
    ax = axes[2]
    from lifelines.statistics import proportional_hazard_test
    ph_result = proportional_hazard_test(cph, df_fit, time_transform="rank")
    ph_summary = ph_result.summary

    ph_labels = [rename.get(c, c) for c in ph_summary.index.get_level_values(0)]
    ph_pvals = ph_summary["p"].values

    colors_ph = [PALETTE["accent2"] if p < 0.05 else PALETTE["accent3"] for p in ph_pvals]
    ax.barh(range(len(ph_labels)), -np.log10(ph_pvals), color=colors_ph, alpha=0.7,
            edgecolor="white", height=0.6)
    ax.axvline(-np.log10(0.05), color=PALETTE["accent2"], ls="--", lw=1, alpha=0.7,
               label="p = 0.05")
    ax.set_yticks(range(len(ph_labels)))
    ax.set_yticklabels(ph_labels, fontsize=10)
    ax.set_xlabel("-log10(p)")
    ax.set_title("C. PH assumption (Schoenfeld)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: if all p > 0.05, PH assumption holds
    n_violated = (ph_pvals < 0.05).sum()
    if n_violated == 0:
        ax.text(0.5, 0.02, "PH assumption satisfied\nfor all covariates",
                transform=ax.transAxes, fontsize=10, ha="center",
                color=PALETTE["accent3"], fontweight="bold")
    else:
        ax.text(0.5, 0.02, f"PH violated for {n_violated} covariate(s)",
                transform=ax.transAxes, fontsize=10, ha="center",
                color=PALETTE["accent2"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_forest_plot.png")
    fig.savefig(FIGURES / "fig11_forest_plot.pdf")
    plt.close(fig)
    print(f"  [FIG11] Cox PH forest plot saved (concordance = {cph.concordance_index_:.3f})")

    # Save model summary
    with open(FIGURES / "model_summary.txt", "w") as f:
        f.write("COX PROPORTIONAL HAZARDS MODEL\n")
        f.write("=" * 60 + "\n")
        f.write(f"N = {len(df_fit)}, events = {df_fit['event'].sum():.0f}\n")
        f.write(f"Concordance index = {cph.concordance_index_:.4f}\n")
        f.write(f"Log-likelihood ratio test p = {cph.log_likelihood_ratio_test().p_value:.2e}\n")
        f.write(f"AIC = {cph.AIC_partial_}\n\n")
        f.write("NOTE: Only biological predictors included.\n")
        f.write("clinical_year / era / genetic_year EXCLUDED to avoid\n")
        f.write("circular reasoning (lag = genetic_year - clinical_year).\n\n")
        f.write("HR > 1 means faster genetic verification (shorter lag).\n")
        f.write("HR < 1 means slower genetic verification (longer lag).\n\n")
        f.write(cph.summary.to_string())
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("VIF (Variance Inflation Factors):\n")
        for name, val in vif_data:
            flag = " *** HIGH" if val > 5 else ""
            f.write(f"  {name}: {val:.2f}{flag}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Schoenfeld test (PH assumption):\n")
        f.write(ph_summary.to_string())
    print("  [FIG11] Model summary saved to figures/model_summary.txt")


# ============================================================
# Fig 12: Sensitivity analysis
# ============================================================

def fig12_sensitivity(df: pd.DataFrame):
    """Sensitivity analysis: compare full dataset vs lag >= 0 only."""
    _setup()

    df_all = df.dropna(subset=["lag_years"]).copy()
    df_pos = df_all[df_all["lag_years"] >= 0].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- A: Distributions side by side ---
    ax = axes[0, 0]
    bins = np.arange(-15, 65, 2)
    ax.hist(df_all["lag_years"].clip(-15, 65), bins=bins, alpha=0.5,
            color=PALETTE["accent1"], label=f"All (n={len(df_all)})", density=True)
    ax.hist(df_pos["lag_years"].clip(0, 65), bins=bins[bins >= 0], alpha=0.5,
            color=PALETTE["accent2"], label=f"Lag >= 0 (n={len(df_pos)})", density=True)
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel("Density")
    ax.set_title("A. Distribution comparison")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- B: Era comparison ---
    ax = axes[0, 1]
    for label, data, color, marker in [
        ("All data", df_all, PALETTE["accent1"], "o"),
        ("Lag >= 0", df_pos, PALETTE["accent2"], "s"),
    ]:
        data_era = data.copy()
        data_era["era"] = data_era["first_clinical_year"].apply(_era_bin)
        data_era = data_era.dropna(subset=["era"])
        medians = data_era.groupby("era")["lag_years"].median()
        era_order = ["Before 1990", "1990-1999", "2000-2009", "2010+"]
        meds = [medians.get(e, np.nan) for e in era_order]
        ax.plot(range(len(era_order)), meds, marker=marker, lw=2, color=color,
                label=label, markersize=8)

    ax.set_xticks(range(len(era_order)))
    ax.set_xticklabels(era_order, rotation=30, ha="right")
    ax.set_ylabel("Median lag (years)")
    ax.set_title("B. Median lag by era")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- C: Inheritance comparison ---
    ax = axes[1, 0]
    df_all["inh"] = df_all["inheritance"].apply(_primary_inheritance)
    df_pos["inh"] = df_pos["inheritance"].apply(_primary_inheritance)
    inh_order = ["AD", "AR", "XL"]
    inh_labels = {"AD": "Autosomal\ndominant", "AR": "Autosomal\nrecessive", "XL": "X-linked"}

    width = 0.35
    x = np.arange(len(inh_order))
    for i, (label, data, color) in enumerate([
        ("All data", df_all, PALETTE["accent1"]),
        ("Lag >= 0", df_pos, PALETTE["accent2"]),
    ]):
        meds = [data[data["inh"] == inh]["lag_years"].median() for inh in inh_order]
        iqr_lo = [data[data["inh"] == inh]["lag_years"].quantile(0.25) for inh in inh_order]
        iqr_hi = [data[data["inh"] == inh]["lag_years"].quantile(0.75) for inh in inh_order]
        err_lo = [m - lo for m, lo in zip(meds, iqr_lo)]
        err_hi = [hi - m for m, hi in zip(meds, iqr_hi)]
        ax.bar(x + i * width - width / 2, meds, width, color=color, alpha=0.7, label=label,
               yerr=[err_lo, err_hi], capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([inh_labels[i] for i in inh_order])
    ax.set_ylabel("Median lag (years), IQR")
    ax.set_title("C. Lag by inheritance")
    ax.legend(frameon=False, fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- D: Regression coefficient comparison ---
    ax = axes[1, 1]
    coefs_all, coefs_pos = [], []
    labels_comp = []
    for label_name, col in [("Year (z)", "first_clinical_year"),
                            ("LOEUF (z)", "loeuf"),
                            ("log(HPO) (z)", "n_hpo_terms")]:
        for data, store in [(df_all, coefs_all), (df_pos, coefs_pos)]:
            d = data.dropna(subset=[col]).copy()
            if col == "n_hpo_terms":
                x_var = np.log1p(d[col])
            else:
                x_var = d[col]
            mu, sd = x_var.mean(), x_var.std()
            if sd > 0:
                x_z = (x_var - mu) / sd
            else:
                x_z = x_var * 0
            X_ols = sm.add_constant(x_z)
            try:
                m = sm.OLS(d["lag_years"], X_ols).fit()
                store.append((m.params.iloc[1], m.conf_int().iloc[1, 0], m.conf_int().iloc[1, 1]))
            except Exception:
                store.append((0, 0, 0))
        labels_comp.append(label_name)

    y_comp = np.arange(len(labels_comp))
    offset = 0.15
    for i, (data, color, label) in enumerate([
        (coefs_all, PALETTE["accent1"], "All data"),
        (coefs_pos, PALETTE["accent2"], "Lag >= 0"),
    ]):
        vals = [d[0] for d in data]
        lo = [d[0] - d[1] for d in data]
        hi = [d[2] - d[0] for d in data]
        ax.errorbar(vals, y_comp + (i - 0.5) * offset * 2, xerr=[lo, hi],
                    fmt="o", color=color, capsize=4, lw=2, markersize=8, label=label)

    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_yticks(y_comp)
    ax.set_yticklabels(labels_comp, fontsize=11)
    ax.set_xlabel("Univariate regression coefficient")
    ax.set_title("D. Effect size comparison")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Sensitivity analysis: all data vs positive lag only", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_sensitivity.png")
    fig.savefig(FIGURES / "fig12_sensitivity.pdf")
    plt.close(fig)
    print("  [FIG12] Sensitivity analysis saved")


# ============================================================
# Fig 13: Missing data analysis
# ============================================================

def fig13_missing_data(df_all: pd.DataFrame):
    """Characterise diseases with vs without computable lag."""
    _setup()

    df_all = df_all.copy()
    df_all["has_lag"] = df_all["lag_years"].notna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- A: Missingness pattern ---
    ax = axes[0, 0]
    cols = ["omim_id", "first_clinical_year", "first_genetic_year",
            "lag_years", "loeuf", "confidence", "inheritance", "disorder_class"]
    miss_pct = [(c, df_all[c].isna().mean() * 100) for c in cols]
    miss_pct.sort(key=lambda x: x[1])
    labels_m = [m[0].replace("_", " ") for m in miss_pct]
    vals_m = [m[1] for m in miss_pct]
    colors_m = [PALETTE["accent2"] if v > 20 else PALETTE["accent1"] for v in vals_m]
    ax.barh(range(len(labels_m)), vals_m, color=colors_m, alpha=0.7, edgecolor="white")
    ax.set_yticks(range(len(labels_m)))
    ax.set_yticklabels(labels_m, fontsize=11)
    ax.set_xlabel("% missing")
    ax.set_title("A. Missingness by variable")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- B: Disease class distribution: has_lag vs no_lag ---
    ax = axes[0, 1]
    df_cls = df_all.dropna(subset=["disorder_class"]).copy()
    top_cls = df_cls["disorder_class"].value_counts().head(8).index
    df_cls = df_cls[df_cls["disorder_class"].isin(top_cls)]

    def _short(s):
        return s.replace("Rare ", "").replace(" disease", "")[:25]

    ct = pd.crosstab(df_cls["disorder_class"].apply(_short), df_cls["has_lag"], normalize="index") * 100
    ct = ct.sort_values(True, ascending=True)
    ct.plot.barh(ax=ax, stacked=True,
                 color=[PALETTE["grid"], PALETTE["accent3"]], alpha=0.8, edgecolor="white")
    ax.set_xlabel("% of diseases")
    ax.set_title("B. Lag availability by disease class")
    ax.legend(["No lag", "Has lag"], frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- C: HPO terms has_lag vs no_lag ---
    ax = axes[1, 0]
    has = df_all[df_all["has_lag"]]["n_hpo_terms"]
    no = df_all[~df_all["has_lag"]]["n_hpo_terms"]
    ax.hist(has.clip(0, 200), bins=40, alpha=0.5, color=PALETTE["accent3"],
            label=f"Has lag (n={len(has)})", density=True)
    ax.hist(no.clip(0, 200), bins=40, alpha=0.5, color=PALETTE["accent2"],
            label=f"No lag (n={len(no)})", density=True)
    u_stat, u_p = stats.mannwhitneyu(has, no, alternative="two-sided")
    ax.text(0.95, 0.95, f"Mann-Whitney p = {u_p:.2e}",
            transform=ax.transAxes, fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Number of HPO terms")
    ax.set_ylabel("Density")
    ax.set_title("C. Phenotypic complexity by lag availability")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- D: LOEUF has_lag vs no_lag ---
    ax = axes[1, 1]
    has_l = df_all[df_all["has_lag"]]["loeuf"].dropna()
    no_l = df_all[~df_all["has_lag"]]["loeuf"].dropna()
    if len(has_l) > 0 and len(no_l) > 0:
        ax.hist(has_l.clip(0, 2), bins=40, alpha=0.5, color=PALETTE["accent3"],
                label=f"Has lag (n={len(has_l)})", density=True)
        ax.hist(no_l.clip(0, 2), bins=40, alpha=0.5, color=PALETTE["accent2"],
                label=f"No lag (n={len(no_l)})", density=True)
        u_stat2, u_p2 = stats.mannwhitneyu(has_l, no_l, alternative="two-sided")
        ax.text(0.95, 0.95, f"Mann-Whitney p = {u_p2:.2e}",
                transform=ax.transAxes, fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("LOEUF")
    ax.set_ylabel("Density")
    ax.set_title("D. Gene constraint by lag availability")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Missing data analysis", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_missing_data.png")
    fig.savefig(FIGURES / "fig13_missing_data.pdf")
    plt.close(fig)
    print("  [FIG13] Missing data analysis saved")


# ============================================================
# Fig 14: Correlation heatmap with significance
# ============================================================

def fig14_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix of all numeric variables with significance stars."""
    _setup()

    df_corr = df.dropna(subset=["lag_years"]).copy()
    df_corr["n_hpo_log"] = np.log1p(df_corr["n_hpo_terms"])

    cols = {
        "lag_years": "Lag (years)",
        "first_clinical_year": "Clinical year",
        "first_genetic_year": "Genetic year",
        "loeuf": "LOEUF",
        "pli": "pLI",
        "n_hpo_terms": "HPO terms",
        "n_hpo_log": "log(HPO terms)",
    }

    data = df_corr[list(cols.keys())].rename(columns=cols)

    # Spearman correlation + p-values
    n = len(cols)
    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))
    col_names = list(cols.values())

    for i in range(n):
        for j in range(n):
            x = data[col_names[i]].values
            y = data[col_names[j]].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 5:
                r, p = stats.spearmanr(x[mask], y[mask])
                corr_mat[i, j] = r
                pval_mat[i, j] = p
            else:
                corr_mat[i, j] = np.nan
                pval_mat[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    sns.heatmap(corr_mat, mask=mask_upper, annot=True, fmt=".2f",
                xticklabels=col_names, yticklabels=col_names,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Spearman r", "shrink": 0.8})

    # Add significance stars
    for i in range(n):
        for j in range(i):
            p = pval_mat[i, j]
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(j + 0.5, i + 0.75, star, ha="center", va="center",
                        fontsize=10, color="black", fontweight="bold")

    ax.set_title("Spearman correlation matrix (* p<0.05, ** p<0.01, *** p<0.001)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_correlation_heatmap.png")
    fig.savefig(FIGURES / "fig14_correlation_heatmap.pdf")
    plt.close(fig)
    print("  [FIG14] Correlation heatmap saved")


# ============================================================
# Fig 15: Formal group comparison tests
# ============================================================

def fig15_group_tests(df: pd.DataFrame):
    """Pairwise group comparisons with Kruskal-Wallis and Dunn post-hoc."""
    _setup()

    df_test = df.dropna(subset=["lag_years"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- A: Disease class Kruskal-Wallis ---
    ax = axes[0]
    df_cls = df_test.dropna(subset=["disorder_class"]).copy()
    top = df_cls["disorder_class"].value_counts().head(8).index
    df_cls = df_cls[df_cls["disorder_class"].isin(top)]

    groups = [df_cls[df_cls["disorder_class"] == c]["lag_years"].values for c in top]
    groups = [g for g in groups if len(g) >= 5]

    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)

        medians = df_cls.groupby("disorder_class")["lag_years"].median().loc[top].sort_values()
        order_cls = medians.index.tolist()

        def _short_cls(s):
            s = s.replace("Rare ", "").replace(" disease", "")
            return s[0].upper() + s[1:] if s else s

        bp = ax.boxplot(
            [df_cls[df_cls["disorder_class"] == c]["lag_years"].values for c in order_cls],
            vert=False, widths=0.6, patch_artist=True, showfliers=False,
            medianprops=dict(color=PALETTE["accent2"], lw=2),
        )
        colors_cls = sns.color_palette("husl", len(order_cls))
        for patch, color in zip(bp["boxes"], colors_cls):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_yticks(range(1, len(order_cls) + 1))
        ax.set_yticklabels([_short_cls(c) for c in order_cls], fontsize=10)
        ax.set_xlabel("Lag (years)")
        ax.set_title(f"A. Kruskal-Wallis H = {kw_stat:.1f}, p = {kw_p:.2e}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- B: Inheritance pairwise Mann-Whitney ---
    ax = axes[1]
    df_inh = df_test.copy()
    df_inh["inh"] = df_inh["inheritance"].apply(_primary_inheritance)
    df_inh = df_inh.dropna(subset=["inh"])
    inh_order = ["AD", "AR", "XL"]
    inh_full = {"AD": "Autosomal dominant", "AR": "Autosomal recessive", "XL": "X-linked"}

    bp2 = ax.boxplot(
        [df_inh[df_inh["inh"] == i]["lag_years"].values for i in inh_order],
        widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color=PALETTE["accent2"], lw=2),
    )
    inh_colors = [PALETTE["accent1"], PALETTE["accent3"], PALETTE["accent4"]]
    for patch, color in zip(bp2["boxes"], inh_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(inh_order) + 1))
    ax.set_xticklabels([inh_full[i] for i in inh_order])
    ax.set_ylabel("Lag (years)")

    # Pairwise tests
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = ["AD vs AR", "AD vs XL", "AR vs XL"]
    y_max = ax.get_ylim()[1]
    for k, (i, j) in enumerate(pairs):
        g1 = df_inh[df_inh["inh"] == inh_order[i]]["lag_years"].values
        g2 = df_inh[df_inh["inh"] == inh_order[j]]["lag_years"].values
        if len(g1) >= 5 and len(g2) >= 5:
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            # Bonferroni correction
            p_adj = min(p * len(pairs), 1.0)
            star = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
            h = y_max + 3 + k * 5
            ax.plot([i + 1, j + 1], [h, h], color=PALETTE["text"], lw=1)
            ax.text((i + j + 2) / 2, h + 0.5, f"{star}\n(p={p_adj:.3f})",
                    ha="center", fontsize=9, color=PALETTE["text"])

    ax.set_ylim(ax.get_ylim()[0], y_max + 20)
    ax.set_title("B. Pairwise Mann-Whitney U (Bonferroni-corrected)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig15_group_tests.png")
    fig.savefig(FIGURES / "fig15_group_tests.pdf")
    plt.close(fig)
    print("  [FIG15] Group comparison tests saved")


# ============================================================
# Entry point
# ============================================================

def generate_stats_figures():
    """Generate all advanced statistical figures."""
    FIGURES.mkdir(exist_ok=True)
    df = load_all()
    print(f"[STATS] Loaded {len(df)} total records ({df['lag_years'].notna().sum()} with lag)")

    fig10_kaplan_meier(df)
    fig11_forest_plot(df)
    fig12_sensitivity(df)
    fig13_missing_data(df)
    fig14_correlation_heatmap(df)
    fig15_group_tests(df)

    print("[STATS] All statistical figures generated")
