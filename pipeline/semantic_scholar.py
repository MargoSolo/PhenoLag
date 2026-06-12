"""Semantic Scholar API integration for enriching publication metadata."""
from __future__ import annotations
import time
import requests
from pipeline.config import S2_API_KEY
from pipeline.db import get_conn

S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_headers():
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return headers


def fetch_paper_by_pmid(pmid: str) -> dict | None:
    """Fetch paper info from Semantic Scholar by PMID."""
    url = f"{S2_BASE}/paper/PMID:{pmid}"
    params = {"fields": "title,year,citationCount,influentialCitationCount,venue,publicationDate"}
    try:
        resp = requests.get(url, params=params, headers=_s2_headers(), timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def enrich_top_papers(limit: int = 100):
    """Enrich top lag papers with Semantic Scholar citation data."""
    conn = get_conn()
    cur = conn.cursor()

    # Ensure s2_cache table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS s2_cache (
            pmid TEXT PRIMARY KEY,
            s2_id TEXT,
            citation_count INTEGER,
            influential_citations INTEGER,
            venue TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Get PMIDs from phenolag that are not yet cached
    cur.execute("""
        SELECT DISTINCT p.first_clinical_pmid FROM phenolag p
        WHERE p.first_clinical_pmid IS NOT NULL
        AND p.first_clinical_pmid NOT IN (SELECT pmid FROM s2_cache)
        LIMIT ?
    """, (limit,))
    pmids_clinical = [r[0] for r in cur.fetchall()]

    cur.execute("""
        SELECT DISTINCT p.first_genetic_pmid FROM phenolag p
        WHERE p.first_genetic_pmid IS NOT NULL
        AND p.first_genetic_pmid NOT IN (SELECT pmid FROM s2_cache)
        LIMIT ?
    """, (limit,))
    pmids_genetic = [r[0] for r in cur.fetchall()]

    all_pmids = list(set(pmids_clinical + pmids_genetic))
    print(f"[S2] Enriching {len(all_pmids)} papers from Semantic Scholar...")

    rate_limit = 10 if S2_API_KEY else 1  # requests per second
    delay = 1.0 / rate_limit

    enriched = 0
    for i, pmid in enumerate(all_pmids):
        data = fetch_paper_by_pmid(pmid)
        if data:
            conn.execute(
                "INSERT OR REPLACE INTO s2_cache VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                (pmid,
                 data.get("paperId"),
                 data.get("citationCount"),
                 data.get("influentialCitationCount"),
                 data.get("venue"))
            )
            enriched += 1
        else:
            conn.execute(
                "INSERT OR IGNORE INTO s2_cache (pmid) VALUES (?)", (pmid,)
            )

        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  ... processed {i + 1}/{len(all_pmids)}")
        time.sleep(delay)

    conn.commit()
    conn.close()
    print(f"[S2] Enriched {enriched}/{len(all_pmids)} papers")
