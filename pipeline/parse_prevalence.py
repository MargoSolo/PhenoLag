"""Parse Orphanet prevalence data (en_product9_prev.xml)."""
import xml.etree.ElementTree as ET
from pipeline.config import SOURCES
from pipeline.db import get_conn


# Prevalence class -> numeric midpoint (per million)
PREV_CLASS_MAP = {
    ">1 / 1000": 5000.0,
    "1-5 / 10 000": 300.0,
    "6-9 / 10 000": 75.0,
    "1-9 / 10 000": 50.0,
    "1-9 / 100 000": 5.0,
    "1-9 / 1 000 000": 0.5,
    "<1 / 1 000 000": 0.1,
    "Unknown": None,
    "Not yet documented": None,
}


def parse_prevalence():
    """Parse en_product9_prev.xml -> prevalence table."""
    path = SOURCES / "en_product9_prev.xml"
    if not path.exists():
        print("[PARSE] en_product9_prev.xml not found, skipping")
        return

    print("[PARSE] en_product9_prev.xml ...")
    tree = ET.parse(str(path))
    root = tree.getroot()

    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prevalence (
            orpha_code TEXT NOT NULL,
            prev_type TEXT,
            prev_class TEXT,
            prev_class_numeric REAL,
            n_cases REAL,
            geographic TEXT,
            source TEXT,
            UNIQUE(orpha_code, prev_type, geographic)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prev_orpha ON prevalence(orpha_code)")
    conn.commit()

    rows = []
    for d in root.iter("Disorder"):
        orpha = d.findtext("OrphaCode")
        for prev in d.iter("Prevalence"):
            prev_type = prev.findtext("PrevalenceType/Name", "")
            prev_class = prev.findtext("PrevalenceClass/Name", "")
            val_moy = prev.findtext("ValMoy", "0")
            geo = prev.findtext("PrevalenceGeographic/Name", "")
            source = prev.findtext("Source", "")

            try:
                n_cases = float(val_moy) if val_moy else None
            except ValueError:
                n_cases = None

            prev_numeric = PREV_CLASS_MAP.get(prev_class)

            rows.append((
                orpha, prev_type, prev_class, prev_numeric,
                n_cases, geo, source
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO prevalence VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    # Stats
    n_disorders = len(set(r[0] for r in rows))
    n_with_cases = len([r for r in rows if r[1] == "Cases/families" and r[4] and r[4] > 0])
    n_with_prev = len([r for r in rows if r[1] == "Point prevalence" and r[3] is not None])
    print(f"  -> {len(rows)} prevalence records for {n_disorders} disorders")
    print(f"     {n_with_cases} with case counts, {n_with_prev} with prevalence class")
