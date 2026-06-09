"""B0 baseline: stock OpenAI Privacy Filter on the synthetic TEST split (#7).

Fairness (same protocol as the #23 pilot):
- Gold = data/eval/eval_300.jsonl, **test split only** (dev is reserved for B1
  threshold tuning, #20). B0 has no tuning, so measuring it on test once is fair.
- One matcher: char IoU >= 0.5, greedy. Metrics = Recall & Precision (no F1, #24).
- OPF natively emits only 6 of the 10 spec labels (the direct identifiers);
  quasi-identifiers (AGE/REGION/OCCUPATION/ORGANIZATION) are out of scope for OPF
  by design. We therefore report untyped detection over BOTH:
    * ALL gold spans            (honest ceiling; penalised for quasi it can't emit)
    * DIRECT-identifier gold     (the fair view for OPF's design, spec §9 main target)
  plus typed per-label and per-domain untyped (direct-only).
- All rows appended to outputs/metrics_ledger.csv with stage="B0".

Run: uv run python scripts/04_b0_baseline.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import append_row  # noqa: E402
from labelmap import OPF_TO_SPEC, SPEC_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval" / "eval_300.jsonl"
IOU_THRESH = 0.5
SPLIT = "test"
DIRECT = {"PERSON", "ADDRESS", "PHONE", "EMAIL", "ID"}  # direct identifiers (spec §9; DATE は準識別子 #39)

Span = tuple[str, int, int]  # (spec_label, start, end)


def iou(a: Span, b: Span) -> float:
    s = max(a[1], b[1]); e = min(a[2], b[2])
    inter = max(0, e - s)
    union = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return inter / union if union else 0.0


def match(preds: list[Span], golds: list[Span], *, typed: bool) -> int:
    """Greedy IoU>=0.5 matching; returns #true positives."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            if typed and p[0] != g[0]:
                continue
            v = iou(p, g)
            if v >= IOU_THRESH:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for _v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp += 1
    return tp


def pr(tp: int, n_pred: int, n_gold: int):
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    return p, r


def _inv(x):
    return (1 - x) if x is not None else None


def _round(x):
    return round(x, 4) if x is not None else ""


def _fmt(x):
    return f"{x:.3f}" if x is not None else "n/a"


def load_gold():
    """Return {text: (domain, [spans])} for the chosen split."""
    out = {}
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["info"]["split"] != SPLIT:
            continue
        spans = [(s["category"], s["start"], s["end"]) for s in rec["label"]]
        out[rec["text"]] = (rec["info"]["domain"], spans)
    return out


def opf_predictions(texts: list[str]):
    """Run stock OPF and return {text: [spec spans]}; second value = elapsed sec.

    Reuses the per-split cache written by scripts/05 (`data/eval/opf_raw_<split>.jsonl`)
    when present, so re-scoring (e.g. after a label-set change) needs no re-inference.
    """
    cache = ROOT / "data" / "eval" / f"opf_raw_{SPLIT}.jsonl"
    if cache.exists():
        out: dict[str, list[Span]] = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["text"]] = [tuple(s) for s in r["spans"]]
        print(f"(using cached OPF preds: {cache.name})")
        return out, 0.0
    with tempfile.TemporaryDirectory() as td:
        gold_in = Path(td) / "test.jsonl"
        with gold_in.open("w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t, "label": []}, ensure_ascii=False) + "\n")
        preds_path = Path(td) / "opf_preds.jsonl"
        t0 = time.time()
        subprocess.run(["opf", "eval", str(gold_in), "--eval-mode", "untyped",
                        "--device", "cpu", "--span-metrics-space", "char",
                        "--predictions-out", str(preds_path),
                        "--metrics-out", str(Path(td) / "m.json")], check=True)
        elapsed = time.time() - t0
        out = {}
        for line in preds_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            spans: list[Span] = []
            for key, coords in rec["predicted_spans"].items():
                native = key.split(":", 1)[0].strip()
                spec = OPF_TO_SPEC.get(native)
                if spec is None:
                    continue
                for s, e in coords:
                    spans.append((spec, s, e))
            out[rec["text"]] = spans
        with cache.open("w", encoding="utf-8") as f:  # cache for re-scoring
            for t, spans in out.items():
                f.write(json.dumps({"text": t, "spans": spans}, ensure_ascii=False) + "\n")
        return out, elapsed


def emit(run_id, domain, label, tp, n_pred, n_gold, latency, note):
    p, r = pr(tp, n_pred, n_gold)
    append_row({"run_id": run_id, "stage": "B0", "model": "A-OPF",
                "variant": "stock", "domain": domain, "label": label,
                "precision": _round(p), "recall": _round(r),
                "miss_rate": _round(_inv(r)), "false_pos_rate": _round(_inv(p)),
                "latency_ms": latency, "seed": 20260609,
                "note": f"{note}, IoU>={IOU_THRESH}, split={SPLIT}, n_gold={n_gold}"})
    return p, r


def untyped_view(run_id, gold, preds, domain_label, gset, *, latency=""):
    """untyped detection (label ignored), restricting gold to label set `gset`."""
    tp = n_pred = n_gold = 0
    for text, (_dom, gspans) in gold.items():
        g = [s for s in gspans if s[0] in gset]
        p = [s for s in preds.get(text, []) if s[0] in gset]
        tp += match(p, g, typed=False)
        n_pred += len(p); n_gold += len(g)
    label = "ALL" if gset == set(SPEC_LABELS) else "DIRECT"
    return emit(run_id, domain_label, label, tp, n_pred, n_gold, latency,
                f"untyped {label.lower()}")


def main() -> int:
    if not GOLD.exists():
        sys.exit(f"missing {GOLD}; run scripts/00_prepare_data.py first")
    gold = load_gold()
    if not gold:
        sys.exit(f"no '{SPLIT}' records in {GOLD}")
    texts = list(gold)
    preds, elapsed = opf_predictions(texts)
    latency_ms = round(elapsed * 1000 / len(texts), 1)
    run_id = uuid.uuid4().hex[:8]
    all_labels = set(SPEC_LABELS)

    print(f"B0 stock OPF on {len(texts)} '{SPLIT}' docs ({elapsed:.1f}s, "
          f"{latency_ms} ms/doc)\n")

    # (1) overall untyped: ALL vs DIRECT-only
    p_all, r_all = untyped_view(run_id, gold, preds, "ALL", all_labels, latency=latency_ms)
    p_dir, r_dir = untyped_view(run_id, gold, preds, "ALL_direct", DIRECT, latency=latency_ms)
    print(f"untyped ALL    : P={_fmt(p_all)} R={_fmt(r_all)}")
    print(f"untyped DIRECT : P={_fmt(p_dir)} R={_fmt(r_dir)}  <- fair view for OPF (spec §9)\n")

    # (2) per-domain untyped (direct-only, the fair view)
    by_dom_text = defaultdict(list)
    for text, (dom, _g) in gold.items():
        by_dom_text[dom].append(text)
    for dom, dom_texts in sorted(by_dom_text.items()):
        sub = {t: gold[t] for t in dom_texts}
        p, r = untyped_view(run_id, sub, preds, dom, DIRECT)
        print(f"  [{dom}] direct untyped: P={_fmt(p)} R={_fmt(r)}")
    print()

    # (3) typed per-label (all 10; quasi will be ~0 for OPF by design)
    for lbl in SPEC_LABELS:
        gl = [s for _t, (_d, gs) in gold.items() for s in gs if s[0] == lbl]
        pl = [s for t in gold for s in preds.get(t, []) if s[0] == lbl]
        if not gl and not pl:
            continue
        tp = sum(match([s for s in preds.get(t, []) if s[0] == lbl],
                       [s for s in gold[t][1] if s[0] == lbl], typed=True) for t in gold)
        p, r = emit(run_id, "ALL", lbl, tp, len(pl), len(gl), "", "typed")
        print(f"  {lbl:<12} P={_fmt(p)} R={_fmt(r)}  (gold={len(gl)})")

    print(f"\nappended B0 rows to outputs/metrics_ledger.csv (run_id={run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
