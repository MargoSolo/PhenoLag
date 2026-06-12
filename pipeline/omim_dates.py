from __future__ import annotations
import time
import requests
from pipeline.config import NCBI_API_KEY, NCBI_BASE
from pipeline.db import get_conn
from pipeline.pubmed_dates import get_years


def _elink_batch(mim_ids: list[str]) -> dict[str, list[str]]:
    """Resolve a batch of OMIM IDs to their cited PubMed PMIDs.

    NCBI elink with `id=A,B,C,...` returns ONE merged linkset for the
    whole batch — not one linkset per source ID — so the per-source
    mapping is lost.  Sending the IDs as repeated `&id=A&id=B&id=C`
    parameters with `&cmd=neighbor` makes ESearch return one linkset
    per source ID, preserving the mapping.

    With NCBI_API_KEY the rate limit is 10 req/s; we sleep 0.15s
    between calls (≈6 req/s, comfortable margin).
    """
    # Use list of tuples so requests serializes as &id=A&id=B&id=C
    params = [
        ("dbfrom", "omim"),
        ("db", "pubmed"),
        ("cmd", "neighbor"),
        ("retmode", "json"),
    ]
    for mim in mim_ids:
        params.append(("id", mim))
    if NCBI_API_KEY:
        params.append(("api_key", NCBI_API_KEY))

    # retry with exponential backoff — NCBI intermittently drops connections
    # ("Response ended prematurely"), which previously silently lost whole batches.
    data = None
    for attempt in range(4):
        try:
            resp = requests.get(f"{NCBI_BASE}/elink.fcgi", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 3:
                print(f"  [WARN] elink batch failed after 4 attempts: {e}")
                return {}
            time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s
    if data is None:
        return {}

    result = {}
    for ls in data.get("linksets", []):
        source_ids = ls.get("ids", [])
        if not source_ids:
            continue
        # When the request used repeated id= parameters, each linkset's
        # `ids` contains exactly one source OMIM ID.
        source_id = str(source_ids[0])

        pmids = []
        for ldb in ls.get("linksetdbs", []):
            if ldb.get("linkname") == "omim_pubmed_cited":
                pmids = [str(p) for p in ldb.get("links", [])]
        result[source_id] = pmids

    return result


def fetch_omim_clinical_dates():
    """Fetch earliest OMIM-referenced publication year for all diseases.

    Strategy to avoid 429:
    1. Batch elink requests (up to 20 OMIM IDs per request)
    2. Collect all unique PMIDs, then batch-fetch years
    3. 0.2s delay between elink requests (5 req/s, well under limit)
    """
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS omim_clinical_cache (
            omim_id TEXT PRIMARY KEY,
            earliest_pmid TEXT,
            earliest_year INTEGER,
            n_omim_refs INTEGER,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    cur = conn.cursor()

    # Read OMIM IDs from `disorders.omim_ids` (the source), not from the
    # `phenolag` table — at this stage in the pipeline `phenolag` is still
    # empty (calculate_lag runs in STEP 3, after this step).
    #
    # Restrict to disorders with ANY gene link (G2P, ClinGen, or HPO).
    # The earlier G2P/ClinGen-only restriction caused ~5,000 OMIM IDs to
    # never be fetched — including diseases whose G2P entries failed to
    # promote into disorder_genes.source = G2P (a known parse_genes.py
    # quirk for some omim<->orpha mappings, e.g. DMD orpha 98896).
    # Without OMIM_elink data, these diseases fell back to HPO_non_g2p,
    # which often picks a recent review and yields negative lag.
    cur.execute(
        """
        SELECT DISTINCT d.omim_ids
        FROM disorders d
        JOIN disorder_genes dg ON d.orpha_code = dg.orpha_code
        WHERE d.omim_ids IS NOT NULL AND d.omim_ids != ''
        """
    )
    all_omim_strs = [r[0] for r in cur.fetchall()]

    all_omims = set()
    for s in all_omim_strs:
        for mim in s.split(","):
            mim = mim.strip()
            if mim:
                all_omims.add(mim)

    cur.execute("SELECT omim_id FROM omim_clinical_cache")
    cached = {r[0] for r in cur.fetchall()}

    missing = [m for m in all_omims if m not in cached]
    print(f"[OMIM] {len(all_omims)} unique OMIM IDs, {len(cached)} cached, "
          f"{len(missing)} to fetch")

    if not missing:
        conn.close()
        return

    # --- Phase 1: Batch elink to collect all PMIDs ---
    print(f"[OMIM] Phase 1: elink (batches of 20)...")
    omim_pmids = {}  # mim_id -> [pmids]
    all_pmids_needed = set()
    elink_batch_size = 20

    for i in range(0, len(missing), elink_batch_size):
        batch = missing[i:i + elink_batch_size]
        result = _elink_batch(batch)
        for mim_id in batch:
            pmids = result.get(mim_id, [])
            omim_pmids[mim_id] = pmids
            all_pmids_needed.update(pmids)

        if (i // elink_batch_size) % 20 == 0:
            print(f"  ... elink {min(i + elink_batch_size, len(missing))}/{len(missing)}")

        time.sleep(0.25)  # 4 req/s, safe margin

    print(f"[OMIM] Phase 1 done. {len(all_pmids_needed)} unique PMIDs to resolve.")
    conn.close()  # Close before get_years opens its own connection

    # --- Phase 2: Batch-fetch all PMID years ---
    print(f"[OMIM] Phase 2: fetching PMID years...")
    all_years = get_years(list(all_pmids_needed))
    print(f"[OMIM] Phase 2 done. {sum(1 for v in all_years.values() if v is not None)} years resolved.")

    # --- Phase 3: Write results ---
    conn = get_conn()
    for mim_id, pmids in omim_pmids.items():
        if pmids:
            valid = [(p, all_years.get(p)) for p in pmids if all_years.get(p) is not None]
            if valid:
                earliest = min(valid, key=lambda x: x[1])
                conn.execute(
                    "INSERT OR REPLACE INTO omim_clinical_cache VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    (mim_id, earliest[0], earliest[1], len(pmids))
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO omim_clinical_cache (omim_id, n_omim_refs) VALUES (?,?)",
                    (mim_id, len(pmids))
                )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO omim_clinical_cache (omim_id, n_omim_refs) VALUES (?,0)",
                (mim_id,)
            )

    conn.commit()
    conn.close()
    print(f"[OMIM] Done. Cached {len(omim_pmids)} OMIM entries.")

