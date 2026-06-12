"""OMIM-cited PMID classifier (no scraping).

For each disease we already have:
  - omim_clinical_cache : list of PMIDs that OMIM elink returned
                          for each MIM ID
  - pubmed_cache        : real title, journal, year, MeSH per PMID

AI receives a list of REAL candidates for one disease and picks the
two best PMIDs (clinical + gene).  Hallucination is impossible:
the PMID must be one of the indices in the list we showed.
"""
from __future__ import annotations
import json, os, re, sqlite3, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv("_env.crash")
_LOCAL_KEY = Path(__file__).resolve().parent.parent / "sources" / "key.json"
if _LOCAL_KEY.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_LOCAL_KEY)

from google import genai
from google.genai import types

from pipeline.config import DB_PATH

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "obzor-science-1776948898")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")


SYSTEM_PROMPT = """You receive a list of PubMed citations that
OMIM associates with a single rare disease.  Pick TWO of them:

clinical_index = EARLIEST candidate that delineates this specific
   disease entity (foundational case report / family description
   / syndrome paper).  Reject tangential mentions, papers on a
   different condition, biochemistry-only, reviews.

gene_index = EARLIEST candidate that established gene-disease
   causality through patient mutations or cloning-with-mutations.
   Reject mapping-only, linkage-only, LOH, expression-only,
   exclusion studies.

Earliest wins; do NOT upgrade to a later "cleaner" paper.

Indices = 0-based positions in the array shown.  If no candidate
matches, return null for that field.

Return strictly:
{
  "clinical_index": <int|null>,
  "clinical_reason": "<one short sentence>",
  "gene_index": <int|null>,
  "gene_evidence_type": "<mutation|cloning|linkage_with_mutation|other|null>",
  "gene_reason": "<one short sentence>"
}
"""


def get_candidates(conn: sqlite3.Connection, mim_ids: list[str]) -> list[dict]:
    """For a list of OMIM IDs, return all PMIDs cited by OMIM
    elink, joined with pubmed_cache for real metadata.  Sort by
    year ascending so the model sees them chronologically."""
    out = []
    seen = set()
    for mim in mim_ids:
        # We don't have a per-PMID list per OMIM in our schema —
        # omim_clinical_cache has earliest_pmid only.  But we have
        # all PMIDs via the underlying elink calls.  We can
        # reconstruct from hpo_annotations + g2p + a fresh elink
        # call if needed.  For pilot, redo elink for this one MIM.
        # (Full implementation below uses omim_dates._elink_batch.)
        pass
    return out


def fetch_omim_pmids(mim: str, ncbi_api_key: str | None) -> list[str]:
    """Fresh elink call for one OMIM ID — returns ALL cited PMIDs
    across every link-name (cited, calc_misc, etc.)."""
    import requests
    params = [
        ("dbfrom", "omim"),
        ("db", "pubmed"),
        ("cmd", "neighbor"),
        ("id", mim),
        ("retmode", "json"),
    ]
    if ncbi_api_key:
        params.append(("api_key", ncbi_api_key))
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                     params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pmids: set[str] = set()
    for ls in data.get("linksets", []):
        for ldb in ls.get("linksetdbs", []):
            ln = ldb.get("linkname", "")
            if ln.startswith("omim_pubmed"):
                for p in ldb.get("links", []):
                    pmids.add(str(p))
    return sorted(pmids)


def build_candidate_list(conn: sqlite3.Connection, mim_ids: list[str],
                         ncbi_api_key: str | None) -> list[dict]:
    """Return [{idx, year, pmid, title, journal, mesh}, ...] sorted by year."""
    all_pmids = set()
    for mim in mim_ids:
        try:
            for p in fetch_omim_pmids(mim, ncbi_api_key):
                all_pmids.add(p)
        except Exception as exc:
            print(f"  [warn] elink {mim} failed: {exc}", file=sys.stderr)

    if not all_pmids:
        return []

    placeholders = ",".join("?" * len(all_pmids))
    rows = conn.execute(
        f"SELECT pmid, year, title, journal, mesh_terms "
        f"FROM pubmed_cache WHERE pmid IN ({placeholders})",
        list(all_pmids),
    ).fetchall()
    cached = {r[0]: r for r in rows}

    # PMIDs not in pubmed_cache — fetch on the fly
    missing = [p for p in all_pmids if p not in cached]
    if missing:
        from pipeline.pubmed_dates import _fetch_batch
        for i in range(0, len(missing), 200):
            batch = missing[i:i + 200]
            try:
                results = _fetch_batch(batch)
                for p, info in results.items():
                    cached[p] = (p, info.get("year"), info.get("title", ""),
                                 info.get("journal", ""), info.get("mesh_terms", ""))
            except Exception as exc:
                print(f"  [warn] efetch missing batch failed: {exc}", file=sys.stderr)

    cands = []
    for pmid, row in cached.items():
        if row is None:
            continue
        _, yr, title, journal, mesh = row
        if yr is None:
            continue
        cands.append({"pmid": pmid, "year": int(yr),
                      "title": title or "",
                      "journal": journal or "",
                      "mesh": mesh or ""})
    cands.sort(key=lambda c: (c["year"], c["pmid"]))
    for i, c in enumerate(cands):
        c["idx"] = i
    return cands


def render_for_prompt(disease: str, gene: str | None, cands: list[dict]) -> str:
    lines = [
        f"Disease: {disease}",
        f"Gene: {gene or '(unknown)'}",
        f"Candidates ({len(cands)}, sorted by year):",
    ]
    for c in cands:
        # truncate MeSH to keep prompt reasonable
        mesh = c["mesh"][:200] + ("…" if len(c["mesh"]) > 200 else "")
        lines.append(
            f"[{c['idx']}] year={c['year']} pmid={c['pmid']}\n"
            f"     title=\"{c['title']}\"\n"
            f"     journal=\"{c['journal']}\"\n"
            f"     mesh=\"{mesh}\""
        )
    return "\n".join(lines)


def _client() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"): t = t[:-3]
    s = t.find("{"); e = t.rfind("}")
    if s < 0 or e <= s: return None
    try: return json.loads(t[s:e+1])
    except: return None


def query_classifier(client, model: str, prompt: str) -> dict | None:
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return _extract_json(resp.text or "")
    except Exception as exc:
        print(f"  [{model}] {exc}", file=sys.stderr)
        return None


def resolve(parsed: dict, cands: list[dict]) -> dict:
    """Map indices to {pmid, year, title} entries (or None)."""
    out = {"clinical_pmid": None, "clinical_year": None, "clinical_title": None,
           "gene_pmid": None, "gene_year": None, "gene_title": None,
           "clinical_reason": parsed.get("clinical_reason"),
           "gene_reason": parsed.get("gene_reason"),
           "gene_evidence_type": parsed.get("gene_evidence_type")}
    if parsed.get("clinical_index") is not None:
        ci = parsed["clinical_index"]
        if 0 <= ci < len(cands):
            c = cands[ci]
            out["clinical_pmid"] = c["pmid"]; out["clinical_year"] = c["year"]
            out["clinical_title"] = c["title"]
    if parsed.get("gene_index") is not None:
        gi = parsed["gene_index"]
        if 0 <= gi < len(cands):
            c = cands[gi]
            out["gene_pmid"] = c["pmid"]; out["gene_year"] = c["year"]
            out["gene_title"] = c["title"]
    return out


def run_pilot(orpha_codes: list[str] | None = None,
              out_filename: str = "omim_only_pilot.json"):
    """Run pilot on the given orpha_codes (uses all OMIMs per disease)."""
    if orpha_codes is None:
        # Default: 5 new cases the user hasn't manually verified yet
        orpha_codes = ['228366', '91414', '90186', '2398', '2909']
    ncbi_key = os.environ.get("NCBI_API_KEY")
    conn = sqlite3.connect(str(DB_PATH))
    client = _client()

    ph = ",".join("?" * len(orpha_codes))
    rows = conn.execute(
        f"SELECT p.orpha_code, p.name, p.gene_symbol, d.omim_ids "
        f"FROM phenolag p LEFT JOIN disorders d ON p.orpha_code = d.orpha_code "
        f"WHERE p.orpha_code IN ({ph})",
        orpha_codes,
    ).fetchall()

    out = {}
    for orpha, name, gene, omim_ids_str in rows:
        omim_list = [m.strip() for m in (omim_ids_str or "").split(",") if m.strip()]
        print(f"\n=== {name} ({gene}, OMIMs {omim_list}) ===")
        if not omim_list:
            print("  no OMIM IDs — skipping")
            continue
        cands = build_candidate_list(conn, omim_list, ncbi_key)
        print(f"  {len(cands)} OMIM-cited PMIDs with year+title "
              f"across {len(omim_list)} OMIM(s)")
        if not cands:
            continue
        prompt = render_for_prompt(name, gene, cands)
        for model in ("gemini-2.5-flash", "gemini-2.5-pro"):
            parsed = query_classifier(client, model, prompt)
            if parsed is None:
                print(f"  [{model}] FAIL")
                continue
            resolved = resolve(parsed, cands)
            print(f"  [{model}] clin={resolved['clinical_year']}/"
                  f"{resolved['clinical_pmid']}  "
                  f"gen={resolved['gene_year']}/{resolved['gene_pmid']}")
            out.setdefault(orpha, {})[model] = {
                "name": name, "gene": gene,
                "n_candidates": len(cands),
                **parsed, **resolved,
            }
        time.sleep(1)
    conn.close()
    Path("figures").mkdir(exist_ok=True)
    Path(f"figures/{out_filename}").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved -> figures/{out_filename}")
    return out


def _ensure_consensus_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_consensus_omim (
            orpha_code TEXT PRIMARY KEY,
            n_candidates INTEGER,
            flash_clin_pmid TEXT, flash_clin_year INTEGER, flash_clin_title TEXT,
            flash_gen_pmid  TEXT, flash_gen_year  INTEGER, flash_gen_title  TEXT,
            flash_clin_reason TEXT, flash_gen_reason TEXT,
            pro_clin_pmid   TEXT, pro_clin_year   INTEGER, pro_clin_title   TEXT,
            pro_gen_pmid    TEXT, pro_gen_year    INTEGER, pro_gen_title    TEXT,
            pro_clin_reason TEXT, pro_gen_reason TEXT,
            consensus_clin_pmid TEXT, consensus_clin_year INTEGER,
            consensus_gen_pmid  TEXT, consensus_gen_year  INTEGER,
            agreement_clin TEXT,   -- both / pro_only / flash_only / disagree / null
            agreement_gen  TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _process_one(client, conn_path: str, ncbi_key: str | None,
                 orpha: str, name: str, gene: str | None,
                 omim_list: list[str]) -> dict | None:
    conn = sqlite3.connect(conn_path)
    try:
        cands = build_candidate_list(conn, omim_list, ncbi_key)
    finally:
        conn.close()
    if not cands:
        return {"orpha_code": orpha, "n_candidates": 0}

    prompt = render_for_prompt(name, gene, cands)
    out = {"orpha_code": orpha, "n_candidates": len(cands)}
    by_model = {}
    for model in ("gemini-2.5-flash", "gemini-2.5-pro"):
        parsed = query_classifier(client, model, prompt)
        if parsed is None:
            by_model[model] = None
            continue
        resolved = resolve(parsed, cands)
        by_model[model] = {**parsed, **resolved}

    flash = by_model.get("gemini-2.5-flash") or {}
    pro = by_model.get("gemini-2.5-pro") or {}
    out["flash_clin_pmid"] = flash.get("clinical_pmid")
    out["flash_clin_year"] = flash.get("clinical_year")
    out["flash_clin_title"] = flash.get("clinical_title")
    out["flash_clin_reason"] = flash.get("clinical_reason")
    out["flash_gen_pmid"] = flash.get("gene_pmid")
    out["flash_gen_year"] = flash.get("gene_year")
    out["flash_gen_title"] = flash.get("gene_title")
    out["flash_gen_reason"] = flash.get("gene_reason")
    out["pro_clin_pmid"] = pro.get("clinical_pmid")
    out["pro_clin_year"] = pro.get("clinical_year")
    out["pro_clin_title"] = pro.get("clinical_title")
    out["pro_clin_reason"] = pro.get("clinical_reason")
    out["pro_gen_pmid"] = pro.get("gene_pmid")
    out["pro_gen_year"] = pro.get("gene_year")
    out["pro_gen_title"] = pro.get("gene_title")
    out["pro_gen_reason"] = pro.get("gene_reason")

    # Consensus: pro wins on disagreement, fall back to flash if pro null
    def consensus(field):
        f = flash.get(field); p = pro.get(field)
        if p == f and p is not None:
            return p, "both"
        if p is not None and f is None:
            return p, "pro_only"
        if f is not None and p is None:
            return f, "flash_only"
        if p is not None and f is not None and p != f:
            return p, "disagree"
        return None, "null"

    cp_pmid, ag_clin = consensus("clinical_pmid")
    cp_year, _ = consensus("clinical_year")
    gp_pmid, ag_gen = consensus("gene_pmid")
    gp_year, _ = consensus("gene_year")
    out["consensus_clin_pmid"] = cp_pmid
    out["consensus_clin_year"] = cp_year
    out["consensus_gen_pmid"] = gp_pmid
    out["consensus_gen_year"] = gp_year
    out["agreement_clin"] = ag_clin
    out["agreement_gen"] = ag_gen
    return out


def _save_one(conn: sqlite3.Connection, row: dict):
    cols = [
        "orpha_code","n_candidates",
        "flash_clin_pmid","flash_clin_year","flash_clin_title",
        "flash_gen_pmid","flash_gen_year","flash_gen_title",
        "flash_clin_reason","flash_gen_reason",
        "pro_clin_pmid","pro_clin_year","pro_clin_title",
        "pro_gen_pmid","pro_gen_year","pro_gen_title",
        "pro_clin_reason","pro_gen_reason",
        "consensus_clin_pmid","consensus_clin_year",
        "consensus_gen_pmid","consensus_gen_year",
        "agreement_clin","agreement_gen",
    ]
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO ai_consensus_omim ({','.join(cols)}) "
        f"VALUES ({placeholders})",
        [row.get(c) for c in cols],
    )
    conn.commit()


def run_full(scope: str = "main", workers: int = 6):
    """scope: 'main' = clin>=1946 + suspect=0 + lag computable;
              'all'  = all phenolag rows with omim_ids."""
    ncbi_key = os.environ.get("NCBI_API_KEY")
    conn = sqlite3.connect(str(DB_PATH))
    _ensure_consensus_table(conn)

    if scope == "main":
        sql = """
            SELECT p.orpha_code, p.name, p.gene_symbol, d.omim_ids
            FROM phenolag p
            LEFT JOIN disorders d ON p.orpha_code = d.orpha_code
            WHERE p.suspect_dating = 0
              AND p.lag_years IS NOT NULL
              AND p.first_clinical_year >= 1946
              AND d.omim_ids IS NOT NULL AND d.omim_ids != ''
        """
    else:
        sql = """
            SELECT p.orpha_code, p.name, p.gene_symbol, d.omim_ids
            FROM phenolag p
            LEFT JOIN disorders d ON p.orpha_code = d.orpha_code
            WHERE d.omim_ids IS NOT NULL AND d.omim_ids != ''
        """

    rows = conn.execute(sql).fetchall()
    done = {r[0] for r in conn.execute(
        "SELECT orpha_code FROM ai_consensus_omim").fetchall()}
    todo = [r for r in rows if r[0] not in done]
    print(f"[OMIM-AI] scope={scope}, total={len(rows)}, "
          f"already done={len(done)}, todo={len(todo)}")
    if not todo:
        conn.close()
        return

    client = _client()
    conn_path = str(DB_PATH)
    n_done = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for orpha, name, gene, omim_str in todo:
            omim_list = [m.strip() for m in (omim_str or "").split(",") if m.strip()]
            if not omim_list:
                continue
            f = ex.submit(_process_one, client, conn_path, ncbi_key,
                          orpha, name, gene, omim_list)
            futs[f] = orpha
        for fut in as_completed(futs):
            orpha = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"  [{orpha}] FAIL: {exc}", file=sys.stderr)
                n_fail += 1
                continue
            if row is None:
                n_fail += 1
                continue
            _save_one(conn, row)
            n_done += 1
            if n_done % 25 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / rate if rate else 0
                print(f"  ... {n_done}/{len(todo)} "
                      f"({rate:.2f}/s ETA {eta/60:.0f}m)")
    conn.close()
    print(f"[OMIM-AI] done: {n_done} ok, {n_fail} fail, "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "full", "all"], default="pilot")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if args.mode == "pilot":
        run_pilot()
    elif args.mode == "full":
        run_full(scope="main", workers=args.workers)
    else:
        run_full(scope="all", workers=args.workers)
