"""Run `opf eval` and record results to the metrics ledger (Issue #2).

Wraps `opf eval --metrics-out`, parses the JSON keys confirmed from upstream
source (docs/findings-opf-cli.md §4), and appends overall + per-label rows.

Usage:
  python scripts/02_evaluate.py DATASET.jsonl --stage B0 --model A-OPF \
      --variant stock --domain ALL [--checkpoint DIR] [--eval-mode typed]

Span metrics use containment matching in char space (OPF default), so other
models in Stage 4 must be scored with the same definition for fair comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from ledger import append_row  # same dir

# Labels we surface as separate per-label rows (spec §6 主要ラベル, 2026-06-08改訂). Extend freely.
KEY_LABELS = ["PERSON", "ADDRESS", "ID", "ORGANIZATION", "AGE"]


def run_opf_eval(dataset: str, metrics_out: Path, *, checkpoint: str | None,
                 eval_mode: str, device: str) -> float:
    cmd = ["opf", "eval", dataset, "--metrics-out", str(metrics_out),
           "--eval-mode", eval_mode, "--span-metrics-space", "char",
           "--per-class", "--device", device]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    # NOTE: never pass --skip-non-ascii-examples (would drop all Japanese).
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return (time.perf_counter() - t0) * 1000.0


def to_rows(metrics: dict, base: dict, latency_ms: float) -> list[dict]:
    rows: list[dict] = []
    p = metrics.get("detection.span.precision")
    r = metrics.get("detection.span.recall")
    f1 = metrics.get("detection.span.f1")
    rows.append({**base, "label": "ALL", "precision": p, "recall": r, "f1": f1,
                 "miss_rate": (1 - r) if r is not None else "",
                 "false_pos_rate": (1 - p) if p is not None else "",
                 "latency_ms": round(latency_ms, 1),
                 "note": f"token_accuracy={metrics.get('token_accuracy')}"})
    for lbl in KEY_LABELS:
        lp = metrics.get(f"by_class.{lbl}.span.precision")
        lr = metrics.get(f"by_class.{lbl}.span.recall")
        lf = metrics.get(f"by_class.{lbl}.span.f1")
        if lp is None and lr is None and lf is None:
            continue
        rows.append({**base, "label": lbl, "precision": lp, "recall": lr, "f1": lf,
                     "miss_rate": (1 - lr) if lr is not None else "",
                     "false_pos_rate": (1 - lp) if lp is not None else ""})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="opf eval -> metrics ledger")
    ap.add_argument("dataset")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--model", default="A-OPF")
    ap.add_argument("--variant", default="stock")
    ap.add_argument("--domain", default="ALL")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-mode", default="typed", choices=("typed", "untyped"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", default="")
    args = ap.parse_args(argv)

    run_id = uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory() as td:
        mpath = Path(td) / "metrics.json"
        latency = run_opf_eval(args.dataset, mpath, checkpoint=args.checkpoint,
                               eval_mode=args.eval_mode, device=args.device)
        metrics = json.loads(mpath.read_text(encoding="utf-8"))

    base = {"run_id": run_id, "stage": args.stage, "model": args.model,
            "variant": args.variant, "domain": args.domain, "seed": args.seed}
    rows = to_rows(metrics, base, latency)
    for row in rows:
        append_row(row)
    print(f"[{run_id}] appended {len(rows)} rows "
          f"(ALL f1={metrics.get('detection.span.f1')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
