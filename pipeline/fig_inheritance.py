"""Standalone time-to-genetic-verification ECDF by inheritance pattern (AR / AD /
X-linked), on the solved at-risk sample. Legend without n (median kept).

Run: PYTHONPATH=. python -m pipeline.fig_inheritance
-> export_figures/inheritance_ttv.png
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.fig_acceleration import load_at_risk
from pipeline.config import DB_PATH


def inh_group(s):
    if not isinstance(s, str):
        return "Other"
    if "X-linked" in s:
        return "X-linked"
    if "Autosomal recessive" in s and "Autosomal dominant" not in s:
        return "AR"
    if "Autosomal dominant" in s and "Autosomal recessive" not in s:
        return "AD"
    return "Other"


def main():
    d = load_at_risk()
    sv = d[d.solved].copy()
    sv["inh"] = sv["inheritance"].map(inh_group)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for grp, col in [("AR", "#2C6FB5"), ("AD", "#C0392B"), ("X-linked", "#8E44AD")]:
        v = np.sort(sv[sv.inh == grp]["lag_years"].values.astype(float))
        if len(v) == 0:
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
                color=col, lw=2.6, label=f"{grp} (med {int(np.median(v))} y)")
    ax.axvline(20, color="grey", ls=":", lw=1)
    ax.set_xlim(0, 72); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time-to-genetic-verification (years)", fontsize=12)
    ax.set_ylabel("Cumulative fraction verified", fontsize=12)
    ax.set_title("Time-to-verification by inheritance pattern", fontsize=13.5, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right", title="Inheritance")
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = DB_PATH.parent / "export_figures" / "inheritance_ttv.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
