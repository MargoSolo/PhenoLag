import xml.etree.ElementTree as ET
from pipeline.config import SOURCES
from pipeline.db import get_conn


def parse_product1():
    """Parse en_product1.xml -> disorders, disorder_synonyms."""
    print("[PARSE] en_product1.xml ...")
    tree = ET.parse(str(SOURCES / "en_product1.xml"))
    root = tree.getroot()

    disorders = []
    synonyms = []

    for d in root.iter("Disorder"):
        orpha = d.findtext("OrphaCode")
        name = d.findtext("Name")
        dtype = d.findtext("DisorderType/Name", "")
        dgroup = d.findtext("DisorderGroup/Name", "")

        omim_ids, mondo_id, icd10, icd11 = [], "", "", ""
        for ref in d.iter("ExternalReference"):
            src = ref.findtext("Source", "")
            val = ref.findtext("Reference", "")
            if src == "OMIM":
                omim_ids.append(val)
            elif src == "MONDO":
                mondo_id = f"MONDO:{val}"
            elif src == "ICD-10":
                icd10 = val
            elif src == "ICD-11":
                icd11 = val

        definition = ""
        for ts in d.iter("TextSection"):
            if ts.findtext("TextSectionType/Name") == "Definition":
                definition = ts.findtext("Contents", "")

        disorders.append((
            orpha, name, dtype, dgroup,
            ",".join(omim_ids) if omim_ids else None,
            mondo_id or None, icd10 or None, icd11 or None,
            definition or None
        ))

        for syn in d.iter("Synonym"):
            if syn.text:
                synonyms.append((orpha, syn.text))

    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO disorders VALUES (?,?,?,?,?,?,?,?,?)",
        disorders
    )
    conn.executemany(
        "INSERT OR IGNORE INTO disorder_synonyms VALUES (?,?)",
        synonyms
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(disorders)} disorders, {len(synonyms)} synonyms")


def parse_epidemiology():
    """Parse orphanet_epidemiology.xml -> epidemiology."""
    print("[PARSE] orphanet_epidemiology.xml ...")
    tree = ET.parse(str(SOURCES / "orphanet_epidemiology.xml"))
    root = tree.getroot()

    rows = []
    for d in root.iter("Disorder"):
        orpha = d.findtext("OrphaCode")
        onsets = [a.findtext("Name", "") for a in d.iter("AverageAgeOfOnset")]
        inher = [t.findtext("Name", "") for t in d.iter("TypeOfInheritance")]
        rows.append((
            orpha,
            ";".join(inher) if inher else None,
            ";".join(onsets) if onsets else None,
        ))

    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO epidemiology VALUES (?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} epidemiology records")


def parse_product7():
    """Parse en_product7.xml -> classifications (parent disease categories)."""
    print("[PARSE] en_product7.xml ...")
    tree = ET.parse(str(SOURCES / "en_product7.xml"))
    root = tree.getroot()

    rows = []
    for d in root.iter("Disorder"):
        orpha = d.findtext("OrphaCode")
        for assoc in d.iter("DisorderDisorderAssociation"):
            atype = assoc.findtext("DisorderDisorderAssociationType/Name", "")
            if "parent" in atype.lower():
                target = assoc.find("TargetDisorder")
                if target is not None:
                    parent_orpha = target.findtext("OrphaCode", "")
                    parent_name = target.findtext("Name", "")
                    rows.append((orpha, parent_orpha, parent_name))

    conn = get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO classifications VALUES (?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"  -> {len(rows)} classification links")


def run_all():
    parse_product1()
    parse_epidemiology()
    parse_product7()
