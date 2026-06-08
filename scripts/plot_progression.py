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
GOAL_RECALL, GOAL_PREC = 0.90, 0.85  # spec §9 (F1 is intentionally NOT used, #24)


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
    # F1 is intentionally NOT plotted (#24): Recall and Precision are reported
    # separately because miss vs over-detection costs are asymmetric.
    pts = {}  # stage -> (recall, precision)
    for r in rows:
        if r.get("model") == "A-OPF" and r.get("label") == "ALL" and r.get("domain") == "ALL":
            pts[r["stage"]] = (_f(r["recall"]), _f(r["precision"]))
    stages = [s for s in STAGE_ORDER if s in pts]
    if not stages:
        return
    x = range(len(stages))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [pts[s][0] for s in stages], "o-", color="#1565c0", label="Recall")
    ax.plot(x, [pts[s][1] for s in stages], "s-", color="#c0392b", label="Precision")
    ax.axhline(GOAL_RECALL, ls=":", color="#1565c0")
    ax.axhline(GOAL_PREC, ls=":", color="#c0392b")
    ax.text(0, GOAL_RECALL + 0.01, "goal recall .90", color="#1565c0", fontsize=8)
    ax.text(0, GOAL_PREC + 0.01, "goal precision .85", color="#c0392b", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(stages)
    ax.set_ylim(0, 1); ax.set_xlabel("Stage"); ax.set_ylabel("Score")
    ax.set_title("OPF Recall/Precision progression (B0 -> B1 -> B2)"); ax.legend()
    fig.tight_layout(); FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "score_progression.png", dpi=130); plt.close(fig)


def plot_leaderboard(rows: list[dict]) -> None:
    # latest ALL/ALL Recall & Precision per model (sorted by date+run_id).
    best = {}  # model -> (recall, precision)
    for r in sorted(rows, key=lambda r: (r.get("date", ""), r.get("run_id", ""))):
        if r.get("label") == "ALL" and r.get("domain") == "ALL":
            rec, prec = _f(r["recall"]), _f(r["precision"])
            if rec is not None or prec is not None:
                best[r["model"]] = (rec, prec)
    if not best:
        return
    models = list(best)
    xi = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar([i - w / 2 for i in xi], [(best[m][0] or 0) for m in models], w,
           color="#1565c0", label="Recall")
    ax.bar([i + w / 2 for i in xi], [(best[m][1] or 0) for m in models], w,
           color="#c0392b", label="Precision")
    ax.axhline(GOAL_RECALL, ls=":", color="#1565c0")
    ax.axhline(GOAL_PREC, ls=":", color="#c0392b")
    ax.set_xticks(list(xi)); ax.set_xticklabels(models)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Model leaderboard (same data/matcher, Recall & Precision)"); ax.legend()
    fig.tight_layout(); FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "model_leaderboard.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    plot_progression(rows)
    plot_leaderboard(rows)
    print(f"figures written to {FIGDIR} ({len(rows)} ledger rows)")
