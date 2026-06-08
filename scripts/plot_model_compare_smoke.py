"""OPF vs GiNZA qualitative coverage comparison (Stage 4 / #9, qualitative).

Side-by-side per-category detection status from the shared Japanese smoke
sentences. QUALITATIVE (no labeled dataset yet). Hand-coded from:
  - OPF: docs/findings-opf-cli.md smoke
  - GiNZA: outputs/ginza_smoke.txt
Figure labels are English on purpose.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# category: (OPF_status, GiNZA_status)  2=detected 1=partial 0=missed
ROWS = [
    ("PERSON (name)",            (2, 2)),
    ("ADDRESS / REGION",         (2, 1)),
    ("PHONE",                    (2, 0)),
    ("EMAIL",                    (2, 1)),
    ("DATE (western)",           (2, 0)),
    ("DATE (Japanese)",          (0, 2)),
    ("AGE",                      (0, 2)),
    ("OCCUPATION (doctor etc.)", (0, 2)),
    ("ORGANIZATION (hospital)",  (0, 0)),
    ("ID (uketsuke-bango)",      (0, 0)),
]
COLORS = {2: "#2e7d32", 1: "#f9a825", 0: "#c62828"}
LABELS = {2: "detected", 1: "partial", 0: "missed"}


def main() -> None:
    cats = [r[0] for r in ROWS][::-1]
    opf = [r[1][0] for r in ROWS][::-1]
    gin = [r[1][1] for r in ROWS][::-1]
    y = range(len(cats))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    h = 0.38
    ax.barh([i + h / 2 for i in y], [1] * len(cats), height=h,
            color=[COLORS[v] for v in opf])
    ax.barh([i - h / 2 for i in y], [1] * len(cats), height=h,
            color=[COLORS[v] for v in gin])
    for i in y:
        ax.text(1.02, i + h / 2, "OPF", va="center", fontsize=7, color="#444")
        ax.text(1.02, i - h / 2, "GiNZA", va="center", fontsize=7, color="#444")
    ax.set_yticks(list(y)); ax.set_yticklabels(cats)
    ax.set_xlim(0, 1.12); ax.set_xticks([])
    ax.set_title("OPF (stock) vs GiNZA — qualitative coverage on Japanese smoke")
    ax.legend(handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in (2, 1, 0)],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "model_compare_smoke.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
