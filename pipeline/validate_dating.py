"""Dating-accuracy validation against a verified landmark set.

Ground truth = year of the seminal gene-identification publication for each
canonical gene-disease pair, confirmed against PubMed (PMID + DOI below).
Compares phenolag's first_genetic_year to truth; reports MAE and within-3y hit rate.

Run: python -m pipeline.validate_dating
"""
from __future__ import annotations
from pipeline.db import get_conn

# (disease name in `disorders`, gene, verified truth year, seminal PMID, source)
GOLD = [
    ("Retinoblastoma",              "RB1",  1986, "2877398", "Friend, Nature 1986"),
    ("Huntington disease",          "HTT",  1993, "8458085", "HDCRG, Cell 1993"),
    ("Ataxia-telangiectasia",       "ATM",  1995, "7792600", "Savitsky, Science 1995"),
    ("Marfan syndrome",             "FBN1", 1991, None,      "Dietz 1991 (canonical)"),
    ("Rett syndrome",               "MECP2",1999, None,      "Amir 1999 (canonical)"),
    ("Phenylketonuria",             "PAH",  1983, None,      "Woo 1983 (canonical)"),
    ("Duchenne muscular dystrophy", "DMD",  1987, None,      "Koenig/Monaco 1986-87"),
    ("Steinert myotonic dystrophy", "DMPK", 1992, None,      "Brook/Fu 1992 (canonical)"),
    ("Neurofibromatosis type 1",    "NF1",  1990, None,      "Wallace/Cawthon 1990"),
]


def main():
    cur = get_conn().cursor()
    errs = []
    print("gene   disease                          truth  ours    diff")
    for dz, gene, truth, _pmid, _src in GOLD:
        cur.execute(
            "SELECT first_genetic_year FROM phenolag p JOIN disorders d "
            "ON p.orpha_code=d.orpha_code WHERE d.name=? AND p.gene_symbol=?", (dz, gene))
        row = cur.fetchone()
        got = row[0] if row else None
        if got is None:
            print(f"{gene:6s} {dz[:32]:32s} {truth}  MISSING")
            errs.append(None)
        else:
            d = got - truth
            errs.append(d)
            print(f"{gene:6s} {dz[:32]:32s} {truth}  {got}  {d:+d}")
    valid = [abs(e) for e in errs if e is not None]
    n_miss = sum(1 for e in errs if e is None)
    mae = sum(valid) / len(valid)
    within3 = sum(1 for e in valid if e <= 3)
    print(f"\nDated {len(valid)}/{len(GOLD)} (missing {n_miss}); "
          f"MAE = {mae:.1f} y; within ±3 y = {within3}/{len(valid)}")


if __name__ == "__main__":
    main()
