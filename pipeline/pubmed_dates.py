"""Fetch publication years from PubMed/NCBI E-utilities with caching."""
from __future__ import annotations
import re
import time
import sqlite3
import requests
from pipeline.config import NCBI_API_KEY, NCBI_BASE, NCBI_RATE
from pipeline.db import get_conn


# MeSH descriptors that flag a paper as molecular-genetic evidence
# (gene cloning, mutation analysis, linkage mapping, sequence work).
# A paper passes the filter iff at least one of its MeSH descriptors
# matches one of these (case-insensitive substring against the
# descriptor name; we keep the list explicit rather than a regex so
# the rule is readable and auditable).
GENETIC_EVIDENCE_MESH = {
    # Cloning / sequencing
    "Cloning, Molecular",
    "Sequence Analysis, DNA",
    "Sequence Analysis, RNA",
    "Sequence Analysis, Protein",
    "DNA, Complementary",
    "Molecular Sequence Data",
    "Base Sequence",
    # Mutations
    "Mutation",
    "Mutation, Missense",
    "DNA Mutational Analysis",
    "Polymorphism, Single Nucleotide",
    "Polymorphism, Genetic",
    "Heterozygote",
    "Homozygote",
    "Loss of Function Mutation",
    "Gain of Function Mutation",
    "Frameshift Mutation",
    "Codon, Nonsense",
    # Mapping / linkage
    "Genetic Linkage",
    "Linkage (Genetics)",
    "Chromosome Mapping",
    "Restriction Mapping",
    "Polymorphism, Restriction Fragment Length",
    "Linkage Disequilibrium",
    # Pedigree-driven studies
    "Pedigree",
    "Genes, Recessive",
    "Genes, Dominant",
    "Genetic Predisposition to Disease",
    # NGS / variant calling
    "Whole Exome Sequencing",
    "Exome Sequencing",
    "Whole Genome Sequencing",
    "High-Throughput Nucleotide Sequencing",
}
GENETIC_EVIDENCE_MESH_LOWER = {m.lower() for m in GENETIC_EVIDENCE_MESH}


_YEAR_RE = re.compile(r"(1[6-9]\d{2}|20\d{2}|21\d{2})")


def _parse_medline_date(text: str | None) -> int | None:
    """Extract the EARLIEST 4-digit year from a free-form <MedlineDate>.

    Examples: '1825-1830' -> 1825, 'Fall 1998' -> 1998, '1975 Jun-Jul' -> 1975,
    '1999 Dec-2000 Jan' -> 1999. Returns None if no plausible year found.
    """
    if not text:
        return None
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if not years:
        return None
    return min(years)


def _fetch_batch(pmids: list[str]) -> dict[str, dict]:
    """Fetch metadata for a batch of PMIDs (max 200)."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
    resp.raise_for_status()

    from lxml import etree
    root = etree.fromstring(resp.content)
    results = {}
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            continue
        pmid = pmid_el.text

        # Get publication year
        year = None
        pub_date_str = ""
        # Try ArticleDate first (electronic publication)
        for ad in article.iter("ArticleDate"):
            y = ad.findtext("Year")
            if y:
                year = int(y)
                m = ad.findtext("Month", "01")
                d = ad.findtext("Day", "01")
                pub_date_str = f"{y}-{m}-{d}"
                break

        # Fallback to PubDate (<Year> or <MedlineDate>1825-1830 / 'Fall 1998')
        if year is None:
            for pd_el in article.iter("PubDate"):
                y = pd_el.findtext("Year")
                if y:
                    try:
                        year = int(y)
                    except ValueError:
                        year = _parse_medline_date(y)
                    if year is None:
                        continue
                    m = pd_el.findtext("Month", "01")
                    d = pd_el.findtext("Day", "01")
                    month_map = {
                        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
                    }
                    if m in month_map:
                        m = month_map[m]
                    pub_date_str = f"{year}-{m}-{d}"
                    break
                # No explicit Year — try MedlineDate
                md = pd_el.findtext("MedlineDate")
                if md:
                    y = _parse_medline_date(md)
                    if y is not None:
                        year = y
                        pub_date_str = md
                        break

        title = article.findtext(".//ArticleTitle", "")
        journal = article.findtext(".//Journal/Title", "")

        # MeSH descriptor names (semicolon-separated) — used by the
        # genetic-evidence filter.  We keep all DescriptorName texts,
        # even those flagged as MajorTopic="N", because a paper can
        # carry "Mutation" as a minor MeSH and still be a real mutation
        # study.  Empty string (not NULL) means "PubMed record was
        # fetched and has no MeSH" — which itself is informative
        # (typically very recent papers not yet indexed).
        mesh_names = []
        for mh in article.iter("MeshHeading"):
            d = mh.find("DescriptorName")
            if d is not None and d.text:
                mesh_names.append(d.text)

        results[pmid] = {
            "year": year,
            "title": title,
            "journal": journal,
            "pub_date": pub_date_str,
            "mesh_terms": ";".join(mesh_names),
        }

    return results


def get_years(pmids: list[str]) -> dict[str, int | None]:
    """Get publication years for PMIDs, using cache. Returns {pmid: year}."""
    if not pmids:
        return {}

    conn = get_conn()
    cur = conn.cursor()

    # Check cache
    placeholders = ",".join("?" * len(pmids))
    cur.execute(f"SELECT pmid, year FROM pubmed_cache WHERE pmid IN ({placeholders})", pmids)
    cached = {row[0]: row[1] for row in cur.fetchall()}

    missing = [p for p in pmids if p not in cached]
    if not missing:
        conn.close()
        return cached

    # Fetch in batches of 200
    batch_size = 200
    delay = 1.0 / NCBI_RATE
    fetched = {}
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        try:
            results = _fetch_batch(batch)
            fetched.update(results)
        except Exception as e:
            print(f"  [WARN] PubMed fetch error for batch {i}: {e}")
        time.sleep(delay)

    # Make sure mesh_terms column exists (idempotent migration).
    try:
        conn.execute("ALTER TABLE pubmed_cache ADD COLUMN mesh_terms TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Cache results
    cache_rows = []
    for pmid, info in fetched.items():
        cache_rows.append((
            pmid, info["year"], info["title"], info["journal"],
            info["pub_date"], info.get("mesh_terms", ""),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO pubmed_cache "
        "(pmid, year, title, journal, pub_date, mesh_terms) VALUES (?,?,?,?,?,?)",
        cache_rows
    )
    conn.commit()

    result = {**cached}
    for pmid, info in fetched.items():
        result[pmid] = info["year"]

    # Mark missing ones as None in cache to avoid refetching
    for pmid in missing:
        if pmid not in fetched:
            conn.execute(
                "INSERT OR IGNORE INTO pubmed_cache (pmid, year) VALUES (?, NULL)", (pmid,)
            )
            result[pmid] = None
    conn.commit()
    conn.close()
    return result


def get_earliest_year(pmids: list[str]) -> tuple[str | None, int | None]:
    """Return (earliest_pmid, earliest_year) from a list of PMIDs."""
    if not pmids:
        return None, None
    years = get_years(pmids)
    valid = [(p, y) for p, y in years.items() if y is not None]
    if not valid:
        return None, None
    return min(valid, key=lambda x: x[1])


# Era of recombinant DNA / positional cloning for human disease genes.
# Before 1970, virtually all gene-disease publications are biochemical
# enzymology / clinical case-series, NOT genetic verification.  G2P's
# `publications` field bundles all relevant papers in a flat list,
# so taking absolute min(year) systematically misdates the genetic
# verification for genes whose biochemical phenotype was characterised
# decades before the gene was cloned (e.g. GLA Fabry 1967 biochem vs
# 1989 cloning; MATN3 1959 cartilage biochem vs 2001 mutation paper).
GENE_VERIFICATION_FLOOR_YEAR = 1980


def _has_genetic_evidence_mesh(mesh_terms: str | None) -> bool:
    """True if any descriptor in `mesh_terms` (semicolon-separated)
    matches GENETIC_EVIDENCE_MESH (case-insensitive)."""
    if not mesh_terms:
        return False
    for t in mesh_terms.split(";"):
        if t.strip().lower() in GENETIC_EVIDENCE_MESH_LOWER:
            return True
    return False


def fetch_mesh_for_cached(pmids: list[str] | None = None,
                          batch_size: int = 200) -> int:
    """Backfill MeSH terms for PMIDs already in the cache that don't
    have them yet.  Re-runs `_fetch_batch` (which now collects MeSH)
    and writes the mesh_terms column.

    If `pmids` is None, scans the entire cache for rows where
    `mesh_terms IS NULL` (i.e. cached before the MeSH column existed)
    and backfills all of them.

    Returns number of rows updated.
    """
    conn = get_conn()
    cur = conn.cursor()
    # Idempotent migration
    try:
        cur.execute("ALTER TABLE pubmed_cache ADD COLUMN mesh_terms TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    if pmids is None:
        cur.execute("SELECT pmid FROM pubmed_cache WHERE mesh_terms IS NULL")
        targets = [r[0] for r in cur.fetchall()]
    else:
        placeholders = ",".join("?" * len(pmids))
        cur.execute(
            f"SELECT pmid FROM pubmed_cache "
            f"WHERE pmid IN ({placeholders}) AND mesh_terms IS NULL",
            pmids,
        )
        targets = [r[0] for r in cur.fetchall()]

    if not targets:
        conn.close()
        return 0

    print(f"[MESH] backfilling MeSH for {len(targets):,} PMIDs "
          f"({batch_size}/batch, {1.0 / NCBI_RATE:.2f}s delay)...")
    delay = 1.0 / NCBI_RATE
    n_updated = 0
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        try:
            results = _fetch_batch(batch)
        except Exception as e:
            print(f"  [WARN] mesh backfill batch {i}: {e}")
            time.sleep(delay)
            continue

        rows = [(info.get("mesh_terms", ""), pmid) for pmid, info in results.items()]
        # PMIDs in `batch` not present in `results` had no record
        # returned — mark with empty string so we don't loop forever.
        returned = set(results.keys())
        for pmid in batch:
            if pmid not in returned:
                rows.append(("", pmid))
        cur.executemany(
            "UPDATE pubmed_cache SET mesh_terms = ? WHERE pmid = ?",
            rows,
        )
        n_updated += len(rows)
        conn.commit()

        if (i // batch_size) % 5 == 0:
            print(f"  ... {min(i + batch_size, len(targets)):,}/{len(targets):,}")
        time.sleep(delay)

    conn.close()
    print(f"[MESH] done. Updated {n_updated:,} rows.")
    return n_updated


def get_earliest_genetic_year(
    pmids: list[str],
    floor_year: int = GENE_VERIFICATION_FLOOR_YEAR,
    use_mesh_filter: bool = True,
) -> tuple[str | None, int | None, str]:
    """Earliest plausibly-genetic verification year.

    Returns (pmid, year, status) where `status` is one of:
      ""                    — clean MeSH-confirmed pick (or filter off)
      "no_mesh_match"       — no PMID has a genetic-evidence MeSH;
                              fell back to floor-year filter
      "pre_molecular"       — only pre-1980 papers exist; absolute min
                              used as last-resort

    Strategy:
      1. (If `use_mesh_filter`) Among cached PMIDs whose `mesh_terms`
         contains a genetic-evidence descriptor and whose year >=
         floor_year, take the earliest.  This is the preferred path:
         it picks the earliest paper that PubMed indexed as actually
         being about gene cloning / mutation / linkage.
      2. Else (no MeSH match), fall back to floor-year-only filter.
      3. Else (all pre-floor), fall back to absolute earliest and
         flag pre_molecular.
    """
    if not pmids:
        return None, None, ""
    years = get_years(pmids)
    valid = [(p, y) for p, y in years.items() if y is not None]
    if not valid:
        return None, None, ""

    if use_mesh_filter:
        # Pull mesh_terms in one query
        conn = get_conn()
        cur = conn.cursor()
        placeholders = ",".join("?" * len(pmids))
        cur.execute(
            f"SELECT pmid, mesh_terms FROM pubmed_cache "
            f"WHERE pmid IN ({placeholders})",
            pmids,
        )
        mesh_map = {r[0]: r[1] for r in cur.fetchall()}
        conn.close()

        genetic = [
            (p, y) for p, y in valid
            if y >= floor_year and _has_genetic_evidence_mesh(mesh_map.get(p))
        ]
        if genetic:
            p, y = min(genetic, key=lambda x: x[1])
            return p, y, ""

    modern = [(p, y) for p, y in valid if y >= floor_year]
    if modern:
        p, y = min(modern, key=lambda x: x[1])
        return p, y, "no_mesh_match"
    p, y = min(valid, key=lambda x: x[1])
    return p, y, "pre_molecular"
