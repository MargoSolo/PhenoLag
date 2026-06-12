"""SQLite database schema and helpers."""
import sqlite3
from pipeline.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS disorders (
    orpha_code   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    disorder_type TEXT,
    disorder_group TEXT,
    omim_ids     TEXT,  -- comma-separated
    mondo_id     TEXT,
    icd10        TEXT,
    icd11        TEXT,
    definition   TEXT
);

CREATE TABLE IF NOT EXISTS disorder_synonyms (
    orpha_code TEXT NOT NULL,
    synonym    TEXT NOT NULL,
    FOREIGN KEY (orpha_code) REFERENCES disorders(orpha_code)
);

CREATE TABLE IF NOT EXISTS epidemiology (
    orpha_code      TEXT PRIMARY KEY,
    inheritance     TEXT,  -- comma-separated
    age_of_onset    TEXT,  -- comma-separated
    FOREIGN KEY (orpha_code) REFERENCES disorders(orpha_code)
);

CREATE TABLE IF NOT EXISTS classifications (
    orpha_code      TEXT NOT NULL,
    parent_orpha    TEXT,
    parent_name     TEXT,
    FOREIGN KEY (orpha_code) REFERENCES disorders(orpha_code)
);

CREATE TABLE IF NOT EXISTS disorder_genes (
    orpha_code   TEXT NOT NULL,
    gene_symbol  TEXT NOT NULL,
    ncbi_gene_id TEXT,
    source       TEXT,  -- phenotype_to_genes / G2P / ClinGen
    UNIQUE(orpha_code, gene_symbol),
    FOREIGN KEY (orpha_code) REFERENCES disorders(orpha_code)
);

CREATE TABLE IF NOT EXISTS g2p (
    g2p_id          TEXT PRIMARY KEY,
    gene_symbol     TEXT NOT NULL,
    gene_mim        TEXT,
    hgnc_id         TEXT,
    disease_name    TEXT,
    disease_mim     TEXT,
    disease_mondo   TEXT,
    allelic_req     TEXT,
    confidence      TEXT,
    variant_consequence TEXT,
    mechanism       TEXT,
    panel           TEXT,
    publications    TEXT,  -- semicolon-separated PMIDs
    date_last_review TEXT
);

CREATE TABLE IF NOT EXISTS genes (
    hgnc_id       TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    name          TEXT,
    locus_group   TEXT,
    locus_type    TEXT,
    status        TEXT,
    location      TEXT,
    alias_symbols TEXT,
    prev_symbols  TEXT,
    entrez_id     TEXT,
    ensembl_id    TEXT,
    omim_id       TEXT,
    uniprot_id    TEXT
);

CREATE TABLE IF NOT EXISTS constraint_metrics (
    gene_symbol   TEXT,
    gene_id       TEXT,
    transcript    TEXT,
    canonical     TEXT,
    mane_select   TEXT,
    lof_obs       REAL,
    lof_exp       REAL,
    lof_oe        REAL,
    lof_oe_lower  REAL,  -- LOEUF
    lof_oe_upper  REAL,
    lof_pli       REAL,
    mis_obs       REAL,
    mis_exp       REAL,
    mis_oe        REAL,
    mis_z         REAL,
    syn_obs       REAL,
    syn_exp       REAL,
    syn_oe        REAL,
    syn_z         REAL,
    PRIMARY KEY (gene_symbol, transcript)
);

CREATE TABLE IF NOT EXISTS hpo_annotations (
    database_id   TEXT NOT NULL,
    disease_name  TEXT,
    qualifier     TEXT,
    hpo_id        TEXT NOT NULL,
    reference     TEXT,
    evidence      TEXT,
    onset         TEXT,
    frequency     TEXT,
    aspect        TEXT
);

CREATE TABLE IF NOT EXISTS pubmed_cache (
    pmid       TEXT PRIMARY KEY,
    year       INTEGER,
    title      TEXT,
    journal    TEXT,
    pub_date   TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS phenolag (
    orpha_code          TEXT PRIMARY KEY,
    name                TEXT,
    omim_id             TEXT,
    gene_symbol         TEXT,
    first_clinical_pmid TEXT,
    first_clinical_year INTEGER,
    first_genetic_pmid  TEXT,
    first_genetic_year  INTEGER,
    lag_years           INTEGER,
    inheritance         TEXT,
    loeuf               REAL,
    pli                 REAL,
    confidence          TEXT,  -- G2P confidence
    n_hpo_terms         INTEGER,
    disorder_class      TEXT,
    in_core_cohort      INTEGER DEFAULT 0,  -- 1 iff disease has G2P or ClinGen link
    clin_source         TEXT,                -- OMIM_elink / HPO_non_g2p / HPO_fallback / MIM_breakpoint
    suspect_dating      INTEGER DEFAULT 0,   -- 1 if dating should be manually reviewed
    suspect_reason      TEXT,                -- why flagged
    gene_sources        TEXT,                -- comma-separated: G2P,ClinGen,HPO
    FOREIGN KEY (orpha_code) REFERENCES disorders(orpha_code)
);

CREATE INDEX IF NOT EXISTS idx_dg_orpha ON disorder_genes(orpha_code);
CREATE INDEX IF NOT EXISTS idx_dg_gene ON disorder_genes(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_hpo_db ON hpo_annotations(database_id);
CREATE INDEX IF NOT EXISTS idx_g2p_gene ON g2p(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_g2p_mim ON g2p(disease_mim);
CREATE INDEX IF NOT EXISTS idx_constraint_gene ON constraint_metrics(gene_symbol);
"""


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_PHENOLAG_MIGRATIONS = [
    ("in_core_cohort", "INTEGER DEFAULT 0"),
    ("clin_source", "TEXT"),
    ("suspect_dating", "INTEGER DEFAULT 0"),
    ("suspect_reason", "TEXT"),
    ("gene_sources", "TEXT"),
]


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(phenolag)")
    existing = {row[1] for row in cur.fetchall()}
    for col, decl in _PHENOLAG_MIGRATIONS:
        if col not in existing:
            cur.execute(f"ALTER TABLE phenolag ADD COLUMN {col} {decl}")
    conn.commit()
    conn.close()
    print(f"[DB] Initialized {DB_PATH}")
