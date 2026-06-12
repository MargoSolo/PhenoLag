"""HPO ontology graph metrics for phenotypic-description quality.

For each disease in the main sample we compute three metrics from
its gene-specific HPO term list:

  - mean_pairwise_dist : average shortest-path distance in the HPO
                         DAG between every pair of the disease's
                         terms.  Small = tightly-clustered phenotype
                         (focused description), large = scattered.
  - n_organ_systems    : number of distinct top-level "Phenotypic
                         abnormality" children (HP:0000118 direct
                         descendants) under which the disease's
                         terms fall.  1 = monosystemic, ≥ 4 = broad
                         multisystemic syndrome.
  - mean_term_depth    : average shortest-path depth of the disease's
                         terms from the HPO root HP:0000001.  Higher
                         = more specific / well-curated terms.

All three are computed on the **undirected** HPO ancestor graph
(is_a / part_of edges, ignoring direction) so distances reflect
semantic proximity in the ontology rather than direction of
specificity.

After this module runs, three new columns are populated in
`phenolag`:  mean_pairwise_dist, n_organ_systems, mean_term_depth.

Run:
    python3 -m pipeline.hpo_metrics
"""

from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path

import networkx as nx

from pipeline.config import ROOT, DB_PATH
from pipeline.db import get_conn

HP_OBO = ROOT / "sources" / "hp.obo"
HPO_ROOT = "HP:0000001"
PHENOTYPIC_ABNORMALITY = "HP:0000118"


# ---------- OBO parser ---------------------------------------------------

def parse_obo(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return (id_to_name, edges).  edges are (child, parent) tuples
    derived from `is_a:` lines inside [Term] stanzas.  Obsolete
    terms are skipped."""
    id_to_name: dict[str, str] = {}
    edges: list[tuple[str, str]] = []

    with path.open() as f:
        cur_id = None
        cur_name = None
        cur_parents: list[str] = []
        cur_obsolete = False
        in_term = False

        def flush():
            nonlocal cur_id, cur_name, cur_parents, cur_obsolete
            if cur_id and not cur_obsolete:
                if cur_name:
                    id_to_name[cur_id] = cur_name
                for p in cur_parents:
                    edges.append((cur_id, p))
            cur_id = None
            cur_name = None
            cur_parents = []
            cur_obsolete = False

        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[Term]"):
                flush()
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]"):
                flush()
                in_term = False
                continue
            if not in_term or not line.strip():
                continue
            if line.startswith("id: "):
                cur_id = line[4:].strip()
            elif line.startswith("name: "):
                cur_name = line[6:].strip()
            elif line.startswith("is_a: "):
                # is_a: HP:NNNNNNN ! some name
                rhs = line[6:].split("!")[0].strip()
                if rhs.startswith("HP:"):
                    cur_parents.append(rhs)
            elif line.startswith("is_obsolete: true"):
                cur_obsolete = True
        flush()

    return id_to_name, edges


def build_graph(edges: list[tuple[str, str]]) -> nx.Graph:
    """Undirected HPO ancestor graph (is_a edges only)."""
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


# ---------- Metric helpers ----------------------------------------------

def organ_system_ancestors(g: nx.Graph) -> dict[str, str]:
    """For each HP term, return the top-level 'Phenotypic abnormality'
    child it descends from (HP:0000118 direct children).  A single
    term may belong to several organ systems via multi-parentage; we
    keep them all in the returned dict by storing a frozenset of
    organ-system ancestors per term as a comma-joined string."""
    if PHENOTYPIC_ABNORMALITY not in g:
        return {}

    # Direct children of HP:0000118 in the *directed* is_a sense are
    # the "organ systems".  In our undirected graph they are simply
    # the neighbors of HP:0000118 that have HP:0000118 as a parent.
    # We get them from the edges file directly: any node with
    # parent==HP:0000118.
    organ_systems = set()
    for n in g.neighbors(PHENOTYPIC_ABNORMALITY):
        organ_systems.add(n)

    # For each term, find its organ-system ancestors via shortest path
    # to HP:0000118 (the path will pass through one of the organ
    # systems).  Cache results.
    out: dict[str, str] = {}
    for node in g.nodes():
        if node == PHENOTYPIC_ABNORMALITY or node == HPO_ROOT:
            continue
        try:
            # All organ-systems reachable from `node` upward.
            # Approximate: pick any organ system that has a path of
            # length less than the path from node to HP:0000118.
            path_to_pa = nx.shortest_path(g, node, PHENOTYPIC_ABNORMALITY)
            # On the undirected graph the path from node to
            # PHENOTYPIC_ABNORMALITY passes through at least one
            # organ system (the second-to-last node is a child of
            # PHENOTYPIC_ABNORMALITY).
            if len(path_to_pa) >= 2:
                second_to_last = path_to_pa[-2]
                if second_to_last in organ_systems:
                    out[node] = second_to_last
        except nx.NetworkXNoPath:
            pass
    return out


def compute_disease_metrics(
    hpo_ids: list[str],
    g: nx.Graph,
    term_to_organ: dict[str, str],
    depth_cache: dict[str, int],
) -> dict[str, float | None]:
    """Return {'mean_pairwise_dist', 'n_organ_systems',
    'mean_term_depth'} for one disease's HPO term list."""
    valid = [t for t in hpo_ids if t in g]
    if not valid:
        return {"mean_pairwise_dist": None, "n_organ_systems": None,
                "mean_term_depth": None}

    # Mean pairwise graph distance — only feasible for moderate term
    # counts.  For large lists we sample.
    if len(valid) >= 2:
        if len(valid) > 30:
            import random
            random.seed(0)
            sample = random.sample(valid, 30)
        else:
            sample = valid
        dists = []
        for a, b in combinations(sample, 2):
            try:
                d = nx.shortest_path_length(g, a, b)
                dists.append(d)
            except nx.NetworkXNoPath:
                pass
        mean_pairwise = sum(dists) / len(dists) if dists else None
    else:
        mean_pairwise = None

    # Number of distinct organ systems
    organs = {term_to_organ.get(t) for t in valid}
    organs.discard(None)
    n_organs = len(organs) if organs else None

    # Mean depth from HP:0000001 root
    depths = []
    for t in valid:
        if t in depth_cache:
            depths.append(depth_cache[t])
            continue
        try:
            d = nx.shortest_path_length(g, t, HPO_ROOT)
            depth_cache[t] = d
            depths.append(d)
        except nx.NetworkXNoPath:
            pass
    mean_depth = sum(depths) / len(depths) if depths else None

    return {
        "mean_pairwise_dist": mean_pairwise,
        "n_organ_systems": n_organs,
        "mean_term_depth": mean_depth,
    }


# ---------- DB integration ----------------------------------------------

def _ensure_columns(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(phenolag)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("mean_pairwise_dist", "n_organ_systems", "mean_term_depth"):
        if col not in existing:
            decl = "REAL" if col != "n_organ_systems" else "INTEGER"
            cur.execute(f"ALTER TABLE phenolag ADD COLUMN {col} {decl}")
    conn.commit()


def _gene_specific_omims_for_row(cur, orpha: str, gene: str | None,
                                  omim_ids_str: str | None) -> list[str]:
    """Same logic as lag_calculator.calculate_lag uses for n_hpo_terms."""
    if not gene or not omim_ids_str:
        return []
    disorder_mims = {m.strip() for m in omim_ids_str.split(",") if m.strip()}
    cur.execute(
        "SELECT DISTINCT disease_mim FROM g2p "
        "WHERE gene_symbol = ? AND disease_mim IS NOT NULL AND disease_mim != ''",
        (gene,),
    )
    return [
        m for (mim,) in cur.fetchall()
        for m in [(mim or "").strip()]
        if m and m in disorder_mims
    ]


def collect_disease_hpo_ids(conn: sqlite3.Connection,
                             orpha: str, gene: str | None,
                             omim_ids_str: str | None) -> list[str]:
    """Mirror the n_hpo_terms resolution priority from lag_calculator."""
    cur = conn.cursor()
    gene_omims = _gene_specific_omims_for_row(cur, orpha, gene, omim_ids_str)

    def query_ids(database_ids: list[str]) -> list[str]:
        if not database_ids:
            return []
        ph = ",".join("?" * len(database_ids))
        cur.execute(
            f"SELECT DISTINCT hpo_id FROM hpo_annotations "
            f"WHERE database_id IN ({ph})",
            database_ids,
        )
        return [r[0] for r in cur.fetchall() if r[0]]

    if gene_omims:
        ids = query_ids([f"OMIM:{m}" for m in gene_omims])
        if ids:
            return ids
    ids = query_ids([f"ORPHA:{orpha}"])
    if ids:
        return ids
    if omim_ids_str:
        all_sources = [f"ORPHA:{orpha}"] + [
            f"OMIM:{m.strip()}" for m in omim_ids_str.split(",") if m.strip()
        ]
        return query_ids(all_sources)
    return []


def run() -> dict[str, int]:
    print(f"[HPO] Parsing {HP_OBO} ...")
    id_to_name, edges = parse_obo(HP_OBO)
    print(f"[HPO]   {len(id_to_name):,} terms, {len(edges):,} is_a edges")

    g = build_graph(edges)
    print(f"[HPO]   graph: {g.number_of_nodes():,} nodes, "
          f"{g.number_of_edges():,} edges")

    print("[HPO] Mapping each term to its organ system "
          "(top-level child of HP:0000118)...")
    term_to_organ = organ_system_ancestors(g)
    print(f"[HPO]   {len(term_to_organ):,} terms mapped to organ systems")

    conn = get_conn()
    _ensure_columns(conn)

    # Read all main-sample diseases (we compute metrics on every row,
    # not only main, so they are available for any future filter).
    rows = conn.execute(
        "SELECT p.orpha_code, p.gene_symbol, d.omim_ids "
        "FROM phenolag p "
        "LEFT JOIN disorders d ON p.orpha_code = d.orpha_code"
    ).fetchall()
    print(f"[HPO] Computing metrics for {len(rows):,} phenolag rows...")

    depth_cache: dict[str, int] = {}
    n_done = 0
    n_with_metrics = 0
    cur = conn.cursor()
    for orpha, gene, omim_ids_str in rows:
        hpo_ids = collect_disease_hpo_ids(conn, orpha, gene, omim_ids_str)
        m = compute_disease_metrics(hpo_ids, g, term_to_organ, depth_cache)
        cur.execute(
            "UPDATE phenolag SET mean_pairwise_dist = ?, "
            "  n_organ_systems = ?, mean_term_depth = ? "
            "WHERE orpha_code = ?",
            (m["mean_pairwise_dist"], m["n_organ_systems"],
             m["mean_term_depth"], orpha),
        )
        n_done += 1
        if any(m[k] is not None for k in m):
            n_with_metrics += 1
        if n_done % 200 == 0:
            conn.commit()
            print(f"  ... {n_done:,}/{len(rows):,}")
    conn.commit()
    conn.close()
    print(f"[HPO] Done. {n_done:,} processed, {n_with_metrics:,} got metrics.")
    return {
        "rows_processed": n_done,
        "rows_with_metrics": n_with_metrics,
        "depth_cache_size": len(depth_cache),
    }


if __name__ == "__main__":
    run()
