"""Metrics ledger helper (Issue #2).

Single source of truth for the verification's numeric history. Every eval run
APPENDS rows here (never overwrites), so B0 -> B1 -> B2 improvement stays visible.

Schema (see docs/verification-plan.md §0):
  run_id, date, stage, model, variant, domain, label,
  precision, recall, f1, miss_rate, false_pos_rate, latency_ms, seed, note
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent.parent / "outputs" / "metrics_ledger.csv"

FIELDS = [
    "run_id", "date", "stage", "model", "variant", "domain", "label",
    "precision", "recall", "f1", "miss_rate", "false_pos_rate",
    "latency_ms", "seed", "note",
]


def ensure_header(path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def append_row(row: dict, path: Path = LEDGER_PATH) -> None:
    """Append one ledger row. Missing keys default to ''; date auto-filled."""
    ensure_header(path)
    row = {**{k: "" for k in FIELDS}, **row}
    if not row.get("date"):
        row["date"] = _dt.date.today().isoformat()
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({k: row[k] for k in FIELDS})


if __name__ == "__main__":
    # Self-test: write a dummy row so the ledger + downstream plot can be exercised.
    append_row({
        "run_id": "selftest", "stage": "B0", "model": "A-OPF",
        "variant": "stock", "domain": "ALL", "label": "ALL",
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "miss_rate": 1.0, "false_pos_rate": 1.0, "note": "ledger self-test",
    })
    print(f"wrote self-test row to {LEDGER_PATH}")
