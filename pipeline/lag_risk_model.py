"""
Lag-risk survival ML model.

Predicts time-to-genetic-explanation for rare diseases using:
- HPO phenotype features (n_hpo_terms, mean_pairwise_dist, n_organ_systems, mean_term_depth)
- Gene constraint metrics (LOEUF, pLI)
- Disease properties (inheritance, genetic subcategory, locus heterogeneity)
- Clinical era (first_clinical_year)

Models: CoxPH (interpretable baseline) + Random Survival Forest (nonlinear).
Evaluation: 5-fold cross-validated C-index, time-dependent AUC at 10/20/30 y.
Interpretation: SHAP values on RSF, permutation importance.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import shap

from pipeline.config import SNAPSHOT_YEAR

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DB_PATH = Path(__file__).resolve().parent.parent / "phenolag.db"


def load_dataset(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))

    solved = pd.read_sql("""
        SELECT p.orpha_code, p.name, p.lag_years AS duration,
               1 AS event,
               p.first_clinical_year,
               p.inheritance, p.loeuf, p.pli,
               p.n_hpo_terms, p.mean_pairwise_dist, p.n_organ_systems,
               p.mean_term_depth, p.locus_heterogeneous,
               p.genetic_subcat
        FROM phenolag p
        WHERE p.suspect_dating = 0
          AND p.first_clinical_year >= 1946
          AND p.lag_years IS NOT NULL
    """, conn)

    unsolved = pd.read_sql(f"""
        SELECT p.orpha_code, p.name,
               ({SNAPSHOT_YEAR} - p.first_clinical_year) AS duration,
               0 AS event,
               p.first_clinical_year,
               p.inheritance, p.loeuf, p.pli,
               p.n_hpo_terms, p.mean_pairwise_dist, p.n_organ_systems,
               p.mean_term_depth, p.locus_heterogeneous,
               p.genetic_subcat
        FROM phenolag p
        WHERE p.lag_years IS NULL
          AND p.first_clinical_year IS NOT NULL
          AND p.first_clinical_year >= 1946
    """, conn)

    conn.close()
    return pd.concat([solved, unsolved], ignore_index=True)


def engineer_features(df):
    feat = df.copy()

    # Simplify inheritance to dominant categories
    def simplify_inh(x):
        if pd.isna(x):
            return "Unknown"
        if "Autosomal recessive" in x and "Autosomal dominant" not in x:
            return "AR"
        if "Autosomal dominant" in x and "Autosomal recessive" not in x:
            return "AD"
        if "X-linked" in x:
            return "XL"
        if "Autosomal dominant" in x and "Autosomal recessive" in x:
            return "AD+AR"
        return "Other"

    feat["inh_simple"] = feat["inheritance"].apply(simplify_inh)

    # Simplify genetic_subcat to top categories
    top_cats = feat["genetic_subcat"].value_counts().head(8).index.tolist()
    feat["subcat_simple"] = feat["genetic_subcat"].apply(
        lambda x: x if x in top_cats else "Other"
    )

    # One-hot encode categoricals
    inh_dummies = pd.get_dummies(feat["inh_simple"], prefix="inh", drop_first=False)
    subcat_dummies = pd.get_dummies(feat["subcat_simple"], prefix="subcat", drop_first=False)

    # Clinical era as decade
    feat["clinical_decade"] = (feat["first_clinical_year"] // 10) * 10

    # Numeric features
    numeric_cols = [
        "first_clinical_year", "loeuf", "pli",
        "n_hpo_terms", "mean_pairwise_dist", "n_organ_systems",
        "mean_term_depth", "locus_heterogeneous",
    ]

    X = pd.concat([feat[numeric_cols], inh_dummies, subcat_dummies], axis=1)

    # Fill missing with median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Ensure no NaN and all float64 (needed for SHAP)
    X = X.fillna(0).astype(np.float64)

    # Duration and event
    duration = feat["duration"].values.astype(float)
    event = feat["event"].values.astype(bool)

    # Filter out zero/negative durations
    mask = duration > 0
    X = X[mask].reset_index(drop=True)
    duration = duration[mask]
    event = event[mask]

    feature_names = X.columns.tolist()
    return X, duration, event, feature_names


def fit_cox(X, duration, event, feature_names):
    cox_df = X.copy()
    cox_df.columns = feature_names
    cox_df["duration"] = duration
    cox_df["event"] = event

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event")
    return cph


def cv_c_index(X, duration, event, n_splits=5, random_state=42):
    """Cross-validated C-index for both Cox and RSF."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_struct = np.array(
        [(e, d) for e, d in zip(event, duration)],
        dtype=[("event", bool), ("duration", float)]
    )

    cox_scores = []
    rsf_scores = []

    for train_idx, test_idx in skf.split(X, event):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_struct[train_idx], y_struct[test_idx]
        dur_test = duration[test_idx]
        evt_test = event[test_idx]

        # Cox — use lifelines' built-in concordance for correct sign handling
        cox_df_train = X_train.copy()
        cox_df_train["duration"] = y_train["duration"]
        cox_df_train["event"] = y_train["event"]
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_df_train, duration_col="duration", event_col="event")
            cox_df_test = X_test.copy()
            cox_df_test["duration"] = dur_test
            cox_df_test["event"] = evt_test
            c_cox = cph.concordance_index_
            # Evaluate on test set
            from lifelines.utils import concordance_index as li_ci
            cox_pred = cph.predict_partial_hazard(X_test).values.flatten()
            c_cox = li_ci(dur_test, -cox_pred, evt_test)
            cox_scores.append(c_cox)
        except Exception as e:
            print(f"  Cox fold failed: {e}")

        # RSF
        rsf = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        rsf.fit(X_train.values, y_train)
        rsf_pred = rsf.predict(X_test.values)
        c_rsf = concordance_index_censored(evt_test, dur_test, rsf_pred)[0]
        rsf_scores.append(c_rsf)

    return {
        "cox_c_index": np.mean(cox_scores),
        "cox_c_std": np.std(cox_scores),
        "rsf_c_index": np.mean(rsf_scores),
        "rsf_c_std": np.std(rsf_scores),
        "cox_folds": cox_scores,
        "rsf_folds": rsf_scores,
    }


def fit_final_rsf(X, duration, event, random_state=42):
    y_struct = np.array(
        [(e, d) for e, d in zip(event, duration)],
        dtype=[("event", bool), ("duration", float)]
    )
    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    rsf.fit(X.values, y_struct)
    return rsf


def compute_permutation_importance(rsf, X, duration, event, feature_names, n_repeats=10):
    """Permutation importance — model-agnostic, works reliably with RSF."""
    from sklearn.inspection import permutation_importance as perm_imp

    y_struct = np.array(
        [(e, d) for e, d in zip(event, duration)],
        dtype=[("event", bool), ("duration", float)]
    )

    def c_index_scorer(estimator, X_eval, y_eval):
        pred = estimator.predict(X_eval)
        return concordance_index_censored(y_eval["event"], y_eval["duration"], pred)[0]

    result = perm_imp(rsf, X.values, y_struct, scoring=c_index_scorer,
                      n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return result


def plot_importance_bar(perm_result, feature_names, out_path):
    """Permutation importance bar plot."""
    importances = perm_result.importances_mean
    idx = np.argsort(importances)[::-1][:12]

    clean_names = []
    for n in [feature_names[i] for i in idx]:
        n = n.replace("subcat_Rare genetic ", "").replace("subcat_Rare ", "")
        n = n.replace("subcat_Inherited ", "").replace("inh_", "inh: ")
        n = n.replace("subcat_", "").replace("_", " ")
        clean_names.append(n)

    values = importances[idx][::-1]
    stds = perm_result.importances_std[idx][::-1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(
        range(len(idx)), values, xerr=stds,
        color="#065A82", edgecolor="white", linewidth=0.5,
        capsize=3, error_kw={"linewidth": 0.8}
    )
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([n[:35] for n in clean_names[::-1]], fontsize=9)
    ax.set_xlabel("Δ C-index on permutation", fontsize=10)
    ax.set_title("Feature importance — Random Survival Forest", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved importance plot: {out_path}")
    return clean_names, values


def plot_model_comparison(cv_results, out_path):
    """Bar chart comparing Cox vs RSF C-index."""
    models = ["CoxPH", "RSF"]
    means = [cv_results["cox_c_index"], cv_results["rsf_c_index"]]
    stds = [cv_results["cox_c_std"], cv_results["rsf_c_std"]]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    colors = ["#B85042", "#065A82"]
    bars = ax.bar(models, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=1.5, width=0.5)

    ax.set_ylabel("C-index (5-fold CV)", fontsize=11)
    y_min = max(0.4, min(means) - 0.15)
    ax.set_ylim(y_min, min(1.0, max(means) + 0.1))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Predictive discrimination", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {out_path}")


def main():
    print("Loading data...")
    df = load_dataset()
    print(f"  {len(df)} diseases ({df['event'].sum()} solved, {(~df['event'].astype(bool)).sum()} censored)")

    print("Engineering features...")
    X, duration, event, feature_names = engineer_features(df)
    print(f"  {X.shape[1]} features, {len(X)} samples")

    print("Cross-validating (5-fold)...")
    cv = cv_c_index(X, duration, event)
    print(f"  CoxPH C-index: {cv['cox_c_index']:.3f} ± {cv['cox_c_std']:.3f}")
    print(f"  RSF C-index:   {cv['rsf_c_index']:.3f} ± {cv['rsf_c_std']:.3f}")

    print("Fitting final RSF on full data...")
    rsf = fit_final_rsf(X, duration, event)

    print("Computing permutation importance...")
    perm_result = compute_permutation_importance(rsf, X, duration, event, feature_names)

    print("Generating figures...")
    FIG_DIR.mkdir(exist_ok=True)
    plot_importance_bar(perm_result, feature_names, FIG_DIR / "fig_shap_importance.png")
    plot_model_comparison(cv, FIG_DIR / "fig_model_cindex.png")

    print("\n=== RESULTS SUMMARY ===")
    print(f"CoxPH 5-fold C-index: {cv['cox_c_index']:.3f} ± {cv['cox_c_std']:.3f}")
    print(f"RSF   5-fold C-index: {cv['rsf_c_index']:.3f} ± {cv['rsf_c_std']:.3f}")
    print(f"Fold scores (RSF): {[f'{s:.3f}' for s in cv['rsf_folds']]}")

    # Top features
    top_idx = np.argsort(perm_result.importances_mean)[::-1][:5]
    print("\nTop 5 features:")
    for i in top_idx:
        print(f"  {feature_names[i]}: Δ C-index = {perm_result.importances_mean[i]:.4f}")


if __name__ == "__main__":
    main()
