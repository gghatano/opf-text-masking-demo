"""Qualitative B0 smoke coverage figure (Stage 0).

Encodes the hand-checked detection status of the stock OPF model on a few
Japanese smoke sentences (see docs/findings-opf-cli.md). This is QUALITATIVE
(no labeled dataset yet); the formal B0 numbers come from 02_evaluate.py once
the synthetic eval set (#5) exists. Figure labels are English on purpose.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# (category, status): 2=detected, 1=partial, 0=missed
ROWS = [
    ("PERSON", 2),
    ("ADDRESS", 2),
    ("PHONE", 2),
    ("EMAIL", 2),
    ("DATE (western, 1990-01-02)", 2),
    ("STAFF (person part only)", 1),
    ("DATE (Japanese, 2025-nen-1-gatsu)", 0),
    ("AGE (72-sai)", 0),
    ("FACILITY (hospital name)", 0),
    ("BUSINESS_ID (uketsuke-bango)", 0),
]
COLORS = {2: "#2e7d32", 1: "#f9a825", 0: "#c62828"}
LABELS = {2: "detected", 1: "partial", 0: "missed"}


def main() -> None:
    cats = [r[0] for r in ROWS][::-1]
    vals = [r[1] for r in ROWS][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(cats, [1] * len(cats), color=[COLORS[v] for v in vals])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Stock OPF on Japanese — qualitative smoke coverage (B0 preview)")
    ax.legend(handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in (2, 1, 0)],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "qualitative_smoke.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
