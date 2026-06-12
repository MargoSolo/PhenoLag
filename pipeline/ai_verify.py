from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load env first so GOOGLE_APPLICATION_CREDENTIALS is set
from dotenv import load_dotenv
load_dotenv("_env.crash")
# Override credentials path: _env.crash uses Windows path; on this
# machine the key actually lives at sources/key.json.
_LOCAL_KEY = Path(__file__).resolve().parent.parent / "sources" / "key.json"
if _LOCAL_KEY.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_LOCAL_KEY)

from google import genai
from google.genai import types

from pipeline.db import get_conn

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "obzor-science-1776948898")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]


SYSTEM_PROMPT = """You are verifying two publication years for a
phenotype-to-genotype lag analysis on rare diseases.  The lag is the
gap between when a disease was first recognised as a distinct
clinical entity and when a causal gene was first established for it.
We need defensible PubMed-indexed dates for an academic poster.

CORE PRINCIPLE — DO NOT CONFLATE THESE:
  - "OMIM PubMed-first clinical lineage" = the earliest PubMed paper
    OMIM cites for THIS phenotype lineage (used for main figures).
  - "Strict modern molecular phenotype" = the paper that first
    delineated the exact molecular form (used as a sensitivity
    field; can differ from above when a gene's earliest reported
    phenotype differs from the specific OMIM entry).
  Return both whenever they diverge.

================================================================
WHAT clinical_year MEANS FOR MAIN ANALYSIS
================================================================
The earliest PubMed-indexed clinical paper that OMIM cites for the
same phenotype lineage as the disease in question, anywhere in the
selected OMIM entry's text (Clinical Features preferred, then
Description, then Inheritance, History, Cytogenetics, etc.).

INCLUDE
  - Original case reports describing the disease lineage.
  - Family/pedigree descriptions establishing inheritance.
  - Eponymous foundational papers.

EXCLUDE
  - Tangential mentions ("X in pregnancy", "screening for X").
  - Papers about a related/parent/sibling condition (cutaneous
    melanoma when disease is uveal melanoma; generic NCL when
    disease is the CLN7 subtype).
  - Reviews of an already-established entity.
  - Pure biochemistry/enzymology with no clinical case description.
  - LOH / cytogenetic / expression artifacts.

If the disease has classical pre-MEDLINE description (Wardrop 1809
retinoblastoma, Hart 1750 Harlequin ichthyosis), record the year
in `historical_year` and the secondary-source PMID that cites it
in `historical_source_pmid`.  These are NOT used as clinical_year.

If the OMIM PubMed-first clinical paper is older than the modern
molecular delineation of the syndrome, ALSO populate the
`strict_modern_clinical_year` / `strict_modern_clinical_pmid`
sensitivity fields with the modern syndrome paper.

================================================================
WHAT gene_year MEANS FOR MAIN ANALYSIS
================================================================
The earliest PubMed-indexed paper, cited by OMIM or verified in
PubMed, that established causal involvement of the specified gene
in the **selected OMIM phenotype lineage** — NOT the first time the
gene was linked to any human disorder.

INCLUDE
  - Patient-mutation papers showing pathogenic variants in
    affected individuals of the selected lineage.
  - Cloning papers that identified the gene and disease-causing
    mutations for the lineage in the same study.
  - Linkage papers ONLY if the same paper also identified the
    causal gene and patient mutations.
  - For cancer phenotypes: somatic driver / tumour-suppressor
    mutation papers are eligible if OMIM uses them as gene-disease
    evidence for the cancer phenotype entry.  If the OMIM entry is
    explicitly an inherited cancer predisposition syndrome, prefer
    germline-susceptibility evidence.

EXCLUDE
  - Gene papers for a different phenotype caused by the same gene.
  - Linkage-only or mapping-only studies (locus localised, gene
    not yet identified).
  - LOH, cytogenetic, expression, biomarker, prognosis studies
    without causal mutation evidence.
  - Candidate-gene exclusion studies.
  - Reviews or replication-only cohorts.

================================================================
SEARCH PROTOCOL — DO THIS, do not answer from memory
================================================================
S1. Open OMIM for the MIM ID(s) provided.  If multiple, pick the
    OMIM whose phenotype lineage best matches the Orphanet name AND
    is the entry where the gene is curated as causal.
S2. For clinical_pmid: walk citations across the entry in
    chronological order across ALL sections (Clinical Features,
    Description, Inheritance, History, Cytogenetics).  For each
    candidate, open https://pubmed.ncbi.nlm.nih.gov/<PMID>/, read
    title + abstract, and apply INCLUDE/EXCLUDE.  Use the FIRST
    paper that survives.  Record which OMIM section it appeared in.
S3. For gene_pmid: walk Molecular Genetics / Mapping in
    chronological order.  Apply INCLUDE/EXCLUDE.
S4. Always quote the EXACT PubMed title and journal that the PMID
    page shows — never paraphrase.  PMIDs are sequential integers
    >= 1; do NOT claim a PMID is invalid without verification.
S5. Maintain an audit trail: every candidate you considered and
    rejected goes into `rejected_clinical_candidates` /
    `rejected_gene_candidates` with a one-line reason.

================================================================
HARD CHRONOLOGICAL SELECTION RULE  (CRITICAL — do not violate)
================================================================
Among all papers that pass INCLUDE/EXCLUDE for the same
gene–phenotype lineage, `clinical_pmid` and `gene_pmid` MUST be
the chronologically EARLIEST.

A later, more polished, more "syndrome-named", or more "cleanly
delineated" paper does NOT displace an earlier valid causal paper
for the same gene–phenotype pair.  Once an earlier survivor is
found, stop upgrading.

================================================================
PMID VALIDATION RULE  (CRITICAL — every PMID you return)
================================================================
Every accepted PMID must exactly match the PubMed record:
  - title returned by you = exact title shown on
    https://pubmed.ncbi.nlm.nih.gov/<PMID>/
  - journal returned = exact journal abbreviation
  - year returned = PubMed publication year (use ePub year if
    earlier than print year and the project rule prefers earliest)

Do NOT paraphrase the title.  Do NOT swap to a "more descriptive"
title.  If you find the candidate PMID's title does not match the
content you expected, the PMID is wrong — go back to S2/S3 and
pick a different candidate.

================================================================
WORKED EXAMPLES (use these to calibrate, do not copy verbatim)
================================================================

Example A — Harlequin ichthyosis (ABCA12, Orphanet 457):
  WRONG  clinical_pmid=null because "the only first description is
         Hart 1750 diary".  Hart 1750 is historical_year, NOT
         clinical_year.
  RIGHT  clinical_year=1951, clinical_pmid=14847077,
         title="Congenital ichthyosis", Lattuada & Parker 1951.
         This is the earliest PubMed-indexed clinical paper cited
         by OMIM #242500 for the same phenotype lineage; in OMIM
         it appears under Inheritance, not Clinical Features.
  RIGHT  gene_year=2005, gene_pmid=15756637 (Kelsell et al.,
         identified ABCA12 mutations in Harlequin patients).

Example B — Uveal melanoma (BAP1, Orphanet 39044):
  WRONG  clinical_pmid=14821926 (Pack & Scharnagel 1951) — that
         paper is about cutaneous melanoma in pregnancy, NOT
         uveal melanoma; OMIM cites it under endocrine influence
         in a tangential context.
  WRONG  clinical_year=1809 — that is Burns/Wardrop historical,
         report it in historical_year only.
  RIGHT  clinical_year=1988, clinical_pmid=3288276
         (Canning & Hungerford "Familial uveal melanoma",
         OMIM #155720 Clinical Features, first PubMed-indexed
         familial entity description for the lineage).
  RIGHT  gene_year=2010, gene_pmid=21051595 (Harbour et al. 2010,
         frequent BAP1 mutations in metastasising uveal melanomas;
         OMIM uses this as gene-disease evidence for the uveal
         melanoma entry).

Example C — DEPDC5 epileptic encephalopathy lineage:
  Two candidate gene papers exist for the same lineage:
    a) Dibbens et al. 2013, PMID 23542697 — first to identify
       loss-of-function DEPDC5 mutations in familial focal
       epilepsy patients with segregation in pedigrees.
    b) Ververi et al. 2022, PMID ~35723916 — dedicated
       developmental-and-epileptic-encephalopathy phenotype paper.
  RIGHT  gene_pmid=23542697, gene_year=2013.  The earlier valid
         causal paper wins.  A later "cleaner" syndrome-named
         paper does NOT displace it.

Example D — KIAA1109 / BLTP1 (Alkuraya–Kucinskas) syndrome:
  Two candidate papers for the same gene–phenotype lineage:
    a) Alazami et al. 2014/2015, PMID 25558065 — exome screen of
       prescreened multiplex consanguineous families that
       co-reported the gene and the affected fetus phenotype
       (homozygous KIAA1109 nonsense, hydrocephalus, Dandy-Walker,
       arthrogryposis).  ePub 2014-12-31, print 2015-01-13.  The
       PubMed record's exact title is "Accelerating novel
       candidate gene discovery in neurogenetic disorders via
       whole-exome sequencing of prescreened multiplex
       consanguineous families".
    b) Gueneau et al. 2018, PMID 29290337 — dedicated syndrome
       delineation paper with cohort.
  RIGHT  clinical_pmid=25558065 and gene_pmid=25558065,
         clinical_year=2014, gene_year=2014, lag=0 (codiscovery in
         the same paper).  Use ePub year 2014.  Do NOT paraphrase
         the title.

Example E — ACAN short stature/advanced bone age/early osteoarthritis:
  RIGHT  clinical_pmid=24762113 and gene_pmid=24762113
         (Nilsson et al. 2014, "Short stature, accelerated bone
         maturation, and early growth cessation due to
         heterozygous aggrecan mutations").  PMID 24297857 is a
         DIFFERENT, unrelated paper — quoting that PMID would be
         a hallucinated mismatch and is rejected.

================================================================
OUTPUT
================================================================
Return strictly the following JSON, no commentary:

{
  "clinical_year":               <int|null>,
  "clinical_pmid":               <string|null>,
  "clinical_title":              "<EXACT PubMed title or null>",
  "clinical_journal":            "<journal or null>",
  "clinical_omim_section":       "<Clinical Features|Description|Inheritance|History|Cytogenetics|other|null>",
  "clinical_rule":               "OMIM_PUBMED_FIRST_CLINICAL_LINEAGE",

  "historical_year":             <int|null>,
  "historical_source_pmid":      <string|null>,
  "historical_source_note":      "<one sentence or null>",

  "strict_modern_clinical_year": <int|null>,
  "strict_modern_clinical_pmid": <string|null>,
  "strict_modern_note":          "<one sentence or null>",

  "gene_year":                   <int|null>,
  "gene_pmid":                   <string|null>,
  "gene_title":                  "<EXACT PubMed title or null>",
  "gene_journal":                "<journal or null>",
  "gene_evidence_type":          "<mutation|cloning|linkage_with_mutation|somatic_cancer|other|null>",
  "gene_rule":                   "FIRST_CAUSAL_GENE_FOR_SELECTED_OMIM_PHENOTYPE_LINEAGE",

  "rejected_clinical_candidates": [
    {"pmid": "...", "year": ..., "reason": "..."}
  ],
  "rejected_gene_candidates": [
    {"pmid": "...", "year": ..., "reason": "..."}
  ],

  "clinical_justification":      "<one sentence quoting OMIM section + author>",
  "gene_justification":          "<one sentence with author/year + what they showed>",
  "confidence":                  "<high|medium|low>",
  "rule_violations":             []
}
"""


def _client() -> genai.Client:
    """Vertex AI client (uses GOOGLE_APPLICATION_CREDENTIALS)."""
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def _user_prompt(row: dict) -> str:
    omim = row.get("omim_ids") or ""
    return (
        f"Disease (Orphanet): {row['name']}\n"
        f"Orphanet ID: ORPHA:{row['orpha_code']}\n"
        f"OMIM phenotype ID(s): {omim}\n"
        f"Gene symbol: {row.get('gene_symbol') or '(unknown)'}\n"
        f"Known gene aliases if visible in OMIM: use them.\n\n"
        f"Automated extractor candidates below may be wrong, unrelated,\n"
        f"or section artifacts.  Do NOT trust them; verify each\n"
        f"independently:\n\n"
        f"Candidate clinical year: {row.get('first_clinical_year')}\n"
        f"Candidate clinical PMID: {row.get('first_clinical_pmid')}\n"
        f"Candidate genetic year:  {row.get('first_genetic_year')}\n"
        f"Candidate genetic PMID:  {row.get('first_genetic_pmid')}\n\n"
        f"Task:\n"
        f"  1. Select the OMIM phenotype entry that best matches the\n"
        f"     Orphanet disease and gene.\n"
        f"  2. Apply OMIM PubMed-first clinical-lineage rule for\n"
        f"     clinical_year (and populate strict_modern_* fields if\n"
        f"     they diverge).\n"
        f"  3. Apply selected-phenotype-lineage gene-causality rule\n"
        f"     for gene_year.\n"
        f"  4. Verify every accepted PMID on PubMed; quote exact\n"
        f"     titles.\n"
        f"  5. List every rejected candidate with a reason.\n"
        f"  6. Return strict JSON only.\n"
    )


def _ensure_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_verification (
            orpha_code TEXT,
            model TEXT,
            verified_clin_year INTEGER,
            verified_clin_pmid TEXT,
            verified_clin_source TEXT,
            verified_gen_year INTEGER,
            verified_gen_pmid TEXT,
            confidence TEXT,
            justification TEXT,
            raw_response TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (orpha_code, model)
        )
        """
    )
    # Per-disease consensus row (one per orpha_code).  agreement is
    # 'both', 'arbitrated', or 'unresolved'.  When the two models
    # already agree on both years, agreement='both' and the values
    # come straight from them.  Otherwise the pro model is re-queried
    # with both answers visible and asked to pick or override.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_consensus (
            orpha_code TEXT PRIMARY KEY,
            clin_year INTEGER,
            clin_pmid TEXT,
            gen_year INTEGER,
            gen_pmid TEXT,
            confidence TEXT,
            agreement TEXT,
            arbitration_note TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a possibly-wrapped response.

    Handles:
      - Plain JSON object
      - JSON wrapped in ```json ... ``` fences
      - JSON with leading/trailing prose
      - JSON with trailing extra data (e.g. citations after closing })
    """
    if not text:
        return None
    # Strip markdown fences first
    t = text.strip()
    if t.startswith("```"):
        # remove leading fence (```json or ```)
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    s = t.find("{")
    e = t.rfind("}")
    if s < 0 or e <= s:
        return None
    blob = t[s:e + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # try truncating progressively from the end
        for cut in range(e, s, -1):
            if t[cut] == "}":
                try:
                    return json.loads(t[s:cut + 1])
                except json.JSONDecodeError:
                    continue
        return None


_FAIL_LOG = Path(__file__).resolve().parent.parent / "figures" / "ai_verify_failures.jsonl"


def _query_one(client: genai.Client, model: str, row: dict,
               max_retries: int = 2) -> dict | None:
    """Single Gemini call with grounded Google Search.  Retries up to
    `max_retries` times on JSON parse failures or empty responses.
    On final failure, dumps raw response text to ai_verify_failures.jsonl
    for later inspection.

    Known issue: gemini-2.5-flash sometimes returns an empty response
    (text_len=0, finish_reason=STOP) when Google Search grounding is
    enabled.  We detect this and retry without the search tool — for
    well-curated rare diseases the model has enough prior knowledge
    to answer.
    """
    cfg_grounded = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.1,
        max_output_tokens=8192,
    )
    cfg_no_search = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",  # ok without search
    )
    last_err = None
    last_text = ""
    last_finish = ""
    for attempt in range(max_retries + 1):
        # First attempts use grounded search; final attempt drops it
        # if previous calls returned empty text (flash bug).
        empty_so_far = last_text == "" and last_err == "no parseable JSON"
        cfg = cfg_no_search if (attempt > 0 and empty_so_far) else cfg_grounded
        try:
            resp = client.models.generate_content(
                model=model,
                contents=_user_prompt(row),
                config=cfg,
            )
            last_text = resp.text or ""
            try:
                last_finish = str(resp.candidates[0].finish_reason)
            except Exception:
                last_finish = ""
            parsed = _extract_json(last_text)
            if parsed is not None:
                return parsed
            last_err = "no parseable JSON"
        except Exception as exc:
            last_err = str(exc)
        if attempt < max_retries:
            time.sleep(2 + attempt * 2)  # backoff

    # Final failure — log full context for diagnosis
    try:
        _FAIL_LOG.parent.mkdir(exist_ok=True)
        with _FAIL_LOG.open("a") as fh:
            fh.write(json.dumps({
                "orpha_code": row["orpha_code"],
                "name": row["name"],
                "model": model,
                "error": last_err,
                "finish_reason": last_finish,
                "text_len": len(last_text),
                "text_head": last_text[:600],
                "text_tail": last_text[-300:] if len(last_text) > 600 else "",
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    print(f"  [{model}] {row['orpha_code']} {row['name'][:40]!r}: "
          f"failed after {max_retries + 1} tries — {last_err} "
          f"(finish={last_finish}, len={len(last_text)})",
          file=sys.stderr)
    return None


def _validate_pmid_titles(conn: sqlite3.Connection, parsed: dict) -> dict:
    """Cross-check AI-returned PMIDs against pubmed_cache.

    For each PMID in the parsed answer, fetch the cached title from
    pubmed_cache.  If the cached title exists and differs from
    AI-returned title (case- and punctuation-insensitive substring
    check), flag it.  We DO NOT silently overwrite — we add
    `pmid_validation` to the parsed dict so the downstream review
    xlsx can show the truth alongside the AI claim.

    Note: only flags if BOTH a cached title and an AI title exist;
    cache misses (PMID we never fetched) leave the field empty.
    """
    issues = []
    for prefix in ("clinical", "gene"):
        pmid = parsed.get(f"{prefix}_pmid")
        ai_title = parsed.get(f"{prefix}_title")
        ai_year = parsed.get(f"{prefix}_year")
        if not pmid:
            continue
        row = conn.execute(
            "SELECT year, title FROM pubmed_cache WHERE pmid = ?",
            (str(pmid).strip(),),
        ).fetchone()
        if row is None:
            continue
        cached_year, cached_title = row
        if cached_title and ai_title:
            ct = "".join(c.lower() for c in cached_title if c.isalnum())
            at = "".join(c.lower() for c in ai_title if c.isalnum())
            # Substring tolerance: AI sometimes truncates or rephrases;
            # if neither title contains a 25-char prefix of the other,
            # flag a mismatch.
            if ct and at:
                short = min(ct[:25], at[:25])
                if short and short not in ct and short not in at:
                    issues.append(
                        f"{prefix}_pmid={pmid}: cached title "
                        f"{cached_title!r} ≠ AI title {ai_title!r}"
                    )
        if cached_year and ai_year and abs(int(cached_year) - int(ai_year)) > 1:
            issues.append(
                f"{prefix}_pmid={pmid}: cached year={cached_year} "
                f"vs AI year={ai_year}"
            )
    if issues:
        parsed = dict(parsed)
        parsed["pmid_validation"] = issues
    return parsed


def _store(conn: sqlite3.Connection, orpha: str, model: str, parsed: dict | None):
    if parsed is None:
        return
    parsed = _validate_pmid_titles(conn, parsed)
    conn.execute(
        """
        INSERT OR REPLACE INTO ai_verification (
            orpha_code, model,
            verified_clin_year, verified_clin_pmid, verified_clin_source,
            verified_gen_year, verified_gen_pmid,
            confidence, justification, raw_response
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            orpha, model,
            parsed.get("clinical_year"),
            parsed.get("clinical_pmid"),
            parsed.get("clinical_source"),
            parsed.get("gene_year"),
            parsed.get("gene_pmid"),
            parsed.get("confidence"),
            (parsed.get("clinical_justification") or "")
                + " | "
                + (parsed.get("gene_justification") or ""),
            json.dumps(parsed, ensure_ascii=False),
        ),
    )
    conn.commit()


def _select_rows(conn: sqlite3.Connection, mode: str) -> list[dict]:
    """Fetch the rows to verify based on mode (pilot / suspect / all)."""
    base = """
        SELECT p.orpha_code, p.name, p.gene_symbol,
               p.first_clinical_year, p.first_clinical_pmid,
               p.first_genetic_year,  p.first_genetic_pmid,
               p.lag_years,
               d.omim_ids
        FROM phenolag p
        LEFT JOIN disorders d ON p.orpha_code = d.orpha_code
        WHERE p.suspect_dating = 0
          AND p.lag_years IS NOT NULL
          AND p.first_clinical_year >= 1946
    """
    if mode == "pilot":
        # 10 longest lag + 10 lag=0 cases
        rows1 = conn.execute(base + " ORDER BY p.lag_years DESC LIMIT 10").fetchall()
        rows2 = conn.execute(base + " AND p.lag_years = 0 LIMIT 10").fetchall()
        rows = rows1 + rows2
    elif mode == "suspect":
        # extreme lag, lag=0, clin=1946, very recent
        rows = conn.execute(
            base + " AND (p.lag_years > 50 OR p.lag_years = 0 "
            "OR p.first_clinical_year = 1946 "
            "OR p.first_genetic_year >= 2020) "
            "ORDER BY p.lag_years DESC"
        ).fetchall()
    elif mode == "all":
        rows = conn.execute(base + " ORDER BY p.orpha_code").fetchall()
    else:
        raise ValueError(f"unknown mode: {mode}")
    cols = [
        "orpha_code", "name", "gene_symbol",
        "first_clinical_year", "first_clinical_pmid",
        "first_genetic_year", "first_genetic_pmid",
        "lag_years", "omim_ids",
    ]
    return [dict(zip(cols, r)) for r in rows]


ARBITER_SYSTEM = """You are arbitrating between two AI assistants
that independently answered the same clinical-genetics question.
Both answers are below.  Decide which is correct, OR provide a
third corrected answer if both are wrong.  Use OMIM, GeneReviews,
and PubMed primary sources via search.  Return strictly JSON in the
same schema:

{
  "clinical_year": <int|null>,
  "clinical_pmid": <string|null>,
  "clinical_source": <"OMIM"|"GeneReviews"|"PubMed"|"history">,
  "clinical_justification": "<one sentence>",
  "gene_year": <int|null>,
  "gene_pmid": <string|null>,
  "gene_justification": "<one sentence>",
  "confidence": <"high"|"medium"|"low">,
  "arbitration_note": "<why this answer over the others, 1-2 sentences>"
}
"""


def _arbitrate(client: genai.Client, row: dict,
               flash_ans: dict, pro_ans: dict) -> dict | None:
    cfg = types.GenerateContentConfig(
        system_instruction=ARBITER_SYSTEM,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.1,
    )
    user = (
        _user_prompt(row)
        + "\n--- gemini-2.5-flash answered ---\n"
        + json.dumps(flash_ans, ensure_ascii=False, indent=2)
        + "\n\n--- gemini-2.5-pro answered ---\n"
        + json.dumps(pro_ans, ensure_ascii=False, indent=2)
        + "\n\nDecide. If you confirm one, copy its values verbatim. "
        "If both are wrong, provide a corrected answer."
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=user,
            config=cfg,
        )
        text = resp.text or ""
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s:e + 1])
        return None
    except Exception as exc:
        print(f"  [arbiter] {row['orpha_code']}: {exc}", file=sys.stderr)
        return None


def _agree_on(a: dict, b: dict, key: str) -> bool:
    """Two answers agree on a key if both are not None and equal."""
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return va is None and vb is None
    return str(va).strip() == str(vb).strip()


def _build_consensus(conn: sqlite3.Connection, row: dict,
                     flash_ans: dict | None, pro_ans: dict | None,
                     client: genai.Client):
    """Compare flash vs pro; arbitrate if needed; write ai_consensus."""
    if flash_ans is None and pro_ans is None:
        return
    if flash_ans is None:
        winner = pro_ans
        agreement, note = "pro_only", "flash failed"
    elif pro_ans is None:
        winner = flash_ans
        agreement, note = "flash_only", "pro failed"
    else:
        clin_ok = _agree_on(flash_ans, pro_ans, "clinical_year")
        gen_ok = _agree_on(flash_ans, pro_ans, "gene_year")
        if clin_ok and gen_ok:
            winner = pro_ans  # values identical, take pro for justification
            agreement, note = "both", "models agreed on both years"
        else:
            arb = _arbitrate(client, row, flash_ans, pro_ans)
            if arb is None:
                winner = pro_ans  # default to pro on arbitration failure
                agreement = "arbitration_failed"
                note = "arbiter call errored, defaulting to pro"
            else:
                winner = arb
                agreement = "arbitrated"
                note = arb.get("arbitration_note", "")

    conn.execute(
        """
        INSERT OR REPLACE INTO ai_consensus (
            orpha_code, clin_year, clin_pmid, gen_year, gen_pmid,
            confidence, agreement, arbitration_note
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            row["orpha_code"],
            winner.get("clinical_year"),
            winner.get("clinical_pmid"),
            winner.get("gene_year"),
            winner.get("gene_pmid"),
            winner.get("confidence"),
            agreement,
            note,
        ),
    )
    conn.commit()


def run(mode: str, models: list[str] = None, max_workers: int = 4) -> dict:
    models = models or MODELS
    conn = get_conn()
    _ensure_table(conn)
    rows = _select_rows(conn, mode)

    # Skip rows already cached for both models.
    cached = {(r[0], r[1]) for r in conn.execute(
        "SELECT orpha_code, model FROM ai_verification"
    ).fetchall()}

    todo = []
    for r in rows:
        for m in models:
            if (r["orpha_code"], m) not in cached:
                todo.append((m, r))

    print(f"[AI] mode={mode}, rows={len(rows)}, "
          f"models={models}, todo={len(todo)} (skip {len(rows)*len(models)-len(todo)} cached)")
    if not todo:
        conn.close()
        return {"rows": len(rows), "todo": 0}

    client = _client()
    n_done = 0
    n_failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_query_one, client, m, r): (m, r)
            for m, r in todo
        }
        for fut in as_completed(futures):
            m, r = futures[fut]
            parsed = fut.result()
            if parsed is None:
                n_failed += 1
            else:
                _store(conn, r["orpha_code"], m, parsed)
            n_done += 1
            if n_done % 5 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                print(f"  ... {n_done}/{len(todo)}  "
                      f"({rate:.1f}/s  ETA {eta:.0f}s, failed={n_failed})")

    elapsed_phase1 = time.time() - t0
    print(f"[AI] phase 1 done in {elapsed_phase1:.1f}s "
          f"({n_done} requests, {n_failed} failed). Building consensus...")

    # ---- Phase 2: per-disease consensus + arbitration -----------------
    rows_by_orpha = {r["orpha_code"]: r for r in rows}
    cur = conn.execute(
        "SELECT orpha_code, model, raw_response FROM ai_verification "
        "WHERE orpha_code IN (" + ",".join(["?"] * len(rows_by_orpha)) + ")",
        list(rows_by_orpha.keys()),
    )
    answers: dict[str, dict[str, dict]] = {}
    for orpha, m, raw in cur.fetchall():
        try:
            answers.setdefault(orpha, {})[m] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            answers.setdefault(orpha, {})[m] = {}

    n_both = n_arbitrated = n_partial = 0
    t1 = time.time()
    for orpha, row in rows_by_orpha.items():
        per_model = answers.get(orpha, {})
        flash_ans = per_model.get("gemini-2.5-flash")
        pro_ans = per_model.get("gemini-2.5-pro")
        if flash_ans and pro_ans:
            if (_agree_on(flash_ans, pro_ans, "clinical_year")
                    and _agree_on(flash_ans, pro_ans, "gene_year")):
                n_both += 1
            else:
                n_arbitrated += 1
        else:
            n_partial += 1
        _build_consensus(conn, row, flash_ans, pro_ans, client)
    elapsed_phase2 = time.time() - t1

    conn.close()
    print(f"[AI] phase 2 done in {elapsed_phase2:.1f}s. "
          f"agree={n_both}, arbitrated={n_arbitrated}, "
          f"partial={n_partial}")
    return {"rows": len(rows), "todo": len(todo),
            "done": n_done, "failed": n_failed,
            "agree": n_both, "arbitrated": n_arbitrated,
            "partial": n_partial}


# ---------- Reporting --------------------------------------------------

def diff_report(out_path: str = "figures/ai_verification_diff.csv"):
    """Build a CSV diffing AI-verified vs current values per disease.

    Joins both models, flags consensus (both agree) vs disagreement.
    """
    import pandas as pd
    conn = get_conn()
    pheno = pd.read_sql(
        "SELECT orpha_code, name, gene_symbol, "
        "  first_clinical_year, first_genetic_year, lag_years "
        "FROM phenolag WHERE suspect_dating=0 AND lag_years IS NOT NULL "
        "AND first_clinical_year>=1946",
        conn,
    )
    ai = pd.read_sql("SELECT * FROM ai_verification", conn)
    conn.close()
    if ai.empty:
        print("[AI] no verifications yet")
        return
    flash = ai[ai["model"] == "gemini-2.5-flash"]
    pro = ai[ai["model"] == "gemini-2.5-pro"]
    merged = pheno.merge(
        flash[["orpha_code", "verified_clin_year", "verified_clin_pmid",
               "verified_gen_year", "verified_gen_pmid",
               "confidence", "justification"]]
        .rename(columns={"verified_clin_year": "flash_clin_year",
                         "verified_clin_pmid": "flash_clin_pmid",
                         "verified_gen_year": "flash_gen_year",
                         "verified_gen_pmid": "flash_gen_pmid",
                         "confidence": "flash_conf",
                         "justification": "flash_just"}),
        on="orpha_code", how="inner",
    ).merge(
        pro[["orpha_code", "verified_clin_year", "verified_clin_pmid",
             "verified_gen_year", "verified_gen_pmid",
             "confidence", "justification"]]
        .rename(columns={"verified_clin_year": "pro_clin_year",
                         "verified_clin_pmid": "pro_clin_pmid",
                         "verified_gen_year": "pro_gen_year",
                         "verified_gen_pmid": "pro_gen_pmid",
                         "confidence": "pro_conf",
                         "justification": "pro_just"}),
        on="orpha_code", how="inner",
    )

    # Consensus columns
    merged["consensus_clin_year"] = merged.apply(
        lambda r: r["flash_clin_year"] if r["flash_clin_year"] == r["pro_clin_year"] else None,
        axis=1)
    merged["consensus_gen_year"] = merged.apply(
        lambda r: r["flash_gen_year"] if r["flash_gen_year"] == r["pro_gen_year"] else None,
        axis=1)
    merged["clin_year_diff_vs_db"] = (
        merged["consensus_clin_year"] - merged["first_clinical_year"]
    )
    merged["gen_year_diff_vs_db"] = (
        merged["consensus_gen_year"] - merged["first_genetic_year"]
    )

    Path(out_path).parent.mkdir(exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[AI] diff report: {len(merged)} rows -> {out_path}")
    print(f"     clin-year consensus reached: "
          f"{merged['consensus_clin_year'].notna().sum()}/{len(merged)}")
    print(f"     clin-year differs from DB:   "
          f"{(merged['clin_year_diff_vs_db'].abs() > 0).sum()}")
    print(f"     gen-year consensus reached:  "
          f"{merged['consensus_gen_year'].notna().sum()}/{len(merged)}")
    print(f"     gen-year differs from DB:    "
          f"{(merged['gen_year_diff_vs_db'].abs() > 0).sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "suspect", "all"],
                   default="pilot")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report", action="store_true",
                   help="only build the diff report from cache")
    args = p.parse_args()
    if args.report:
        diff_report()
    else:
        run(args.mode, max_workers=args.workers)
        diff_report()
