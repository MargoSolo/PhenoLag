"""Apply manual-curation decisions to the phenolag table.

Reads `figures/curation_applied.csv` (produced from `manual_curation.xlsx`)
and updates rows in `phenolag` for diseases curated by the analyst:

  - decision == "keep" with positive `new_lag`:
        clear `suspect_dating`, set `first_clinical_year = curated_clin_year`,
        recompute `lag_years`, append `manual_curation` to `clin_source`.
  - decision == "keep" with new_lag == 0:
        same as above; lag is genuine codiscovery.
  - decision == "keep" with new_lag < 0:
        leave as suspect; the curated year still produces a negative lag,
        flag stays for re-review (`suspect_reason = neg_lag_after_curation`).
  - decision == "drop" / "unclear":
        leave row untouched; record stays in historical_review only.

Run after `run_pipeline.py` has rebuilt the DB:

    python3 -m pipeline.apply_curation

Idempotent — re-running it is safe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from pipeline.config import DB_PATH, FIGURES

CURATION_CSV = FIGURES / "curation_applied.csv"


def apply() -> dict[str, int]:
    if not CURATION_CSV.exists():
        raise FileNotFoundError(f"Curation file not found: {CURATION_CSV}")

    cur_df = pd.read_csv(CURATION_CSV, dtype={"orpha_code": str})
    keep = cur_df[cur_df["decision"] == "keep"].copy()
    keep = keep[keep["curated_clin_year"].notna() & keep["first_genetic_year"].notna()]
    keep["new_lag"] = (
        keep["first_genetic_year"].astype(int) - keep["curated_clin_year"].astype(int)
    )

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    n_unsuspended = 0
    n_still_neg = 0
    n_skipped_no_row = 0

    for _, r in keep.iterrows():
        orpha = str(r["orpha_code"])
        new_clin = int(r["curated_clin_year"])
        new_lag = int(r["new_lag"])
        pmid = str(r.get("curated_pmid") or "")

        cur.execute("SELECT 1 FROM phenolag WHERE orpha_code = ?", (orpha,))
        if cur.fetchone() is None:
            n_skipped_no_row += 1
            continue

        if new_lag >= 0:
            cur.execute(
                """
                UPDATE phenolag
                SET first_clinical_year = ?,
                    first_clinical_pmid = COALESCE(NULLIF(?, ''), first_clinical_pmid),
                    lag_years           = ?,
                    clin_source         = 'manual_curation',
                    suspect_dating      = 0,
                    suspect_reason      = NULL
                WHERE orpha_code = ?
                """,
                (new_clin, pmid, new_lag, orpha),
            )
            n_unsuspended += 1
        else:
            cur.execute(
                """
                UPDATE phenolag
                SET first_clinical_year = ?,
                    first_clinical_pmid = COALESCE(NULLIF(?, ''), first_clinical_pmid),
                    lag_years           = ?,
                    clin_source         = 'manual_curation',
                    suspect_dating      = 1,
                    suspect_reason      = 'neg_lag_after_curation'
                WHERE orpha_code = ?
                """,
                (new_clin, pmid, new_lag, orpha),
            )
            n_still_neg += 1

    conn.commit()
    conn.close()

    summary = {
        "curated_rows":          int(len(cur_df)),
        "kept_with_new_lag":     int(len(keep)),
        "unsuspended":           n_unsuspended,
        "still_negative":        n_still_neg,
        "skipped_missing_row":   n_skipped_no_row,
    }
    print("[curation] applied:", summary)
    return summary


if __name__ == "__main__":
    apply()
