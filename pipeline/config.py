import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SOURCES = ROOT / "sources"
FIGURES = ROOT / "figures"
DB_PATH = ROOT / "phenolag.db"

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
S2_API_KEY = os.getenv("S2_API_KEY", "")

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_RATE = 10 if NCBI_API_KEY else 3  # requests per second

# Data-snapshot year: used as "now" for right-censoring unsolved diseases
# (duration = SNAPSHOT_YEAR - first_clinical_year). Bump when refreshing sources.
SNAPSHOT_YEAR = 2026

# Source release versions (provenance — keep in sync when sources are refreshed).
SOURCE_VERSIONS = {
    "G2P": "2026-02-07",
    "gnomAD": "v4.1",
    "snapshot_year": SNAPSHOT_YEAR,
}
