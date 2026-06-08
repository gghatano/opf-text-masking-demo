"""Generate the score-progression / leaderboard / per-label figures (Issue #2).

Reads outputs/metrics_ledger.csv and writes PNGs to figures/. Success-criteria
lines (spec §9) are overlaid so distance-to-goal is always visible.
Figure labels are in English on purpose (matplotlib JP font issues).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "outputs" / "metrics_ledger.csv"
FIGDIR = ROOT / "figures"

STAGE_ORDER = ["B0", "B1", "B2"]
GOAL_RECALL, GOAL_PREC, GOAL_F1 = 0.90, 0.85, 0.85  # spec §9


def load_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_progression(rows: list[dict]) -> None:
    pts = {}  # stage -> (f1, recall, fpr)
    for r in rows:
        if r.get("model") == "A-OPF" and r.get("label") == "ALL" and r.get("domain") == "ALL":
            pts[r["stage"]] = (_f(r["f1"]), _f(r["recall"]), _f(r["false_pos_rate"]))
    stages = [s for s in STAGE_ORDER if s in pts]
    if not stages:
        return
    x = range(len(stages))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [pts[s][0] for s in stages], "o-", label="F1 (span)")
    ax.plot(x, [pts[s][1] for s in stages], "s-", label="Recall")
    ax.plot(x, [pts[s][2] for s in stages], "^--", label="False-positive rate")
    ax.axhline(GOAL_F1, ls=":", color="gray")
    ax.axhline(GOAL_RECALL, ls=":", color="green")
    ax.text(0, GOAL_RECALL + 0.01, "goal recall .90", color="green", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(stages)
    ax.set_ylim(0, 1); ax.set_xlabel("Stage"); ax.set_ylabel("Score")
    ax.set_title("OPF score progression (B0 -> B1 -> B2)"); ax.legend()
    fig.tight_layout(); FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "score_progression.png", dpi=130); plt.close(fig)


def plot_leaderboard(rows: list[dict]) -> None:
    best = {}  # model -> f1 (latest ALL/ALL)
    for r in rows:
        if r.get("label") == "ALL" and r.get("domain") == "ALL" and _f(r["f1"]) is not None:
            best[r["model"]] = _f(r["f1"])
    if not best:
        return
    models = list(best)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, [best[m] for m in models])
    ax.axhline(GOAL_F1, ls=":", color="gray"); ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (span)"); ax.set_title("Model leaderboard (same data/metric)")
    fig.tight_layout(); FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "model_leaderboard.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    plot_progression(rows)
    plot_leaderboard(rows)
    print(f"figures written to {FIGDIR} ({len(rows)} ledger rows)")
