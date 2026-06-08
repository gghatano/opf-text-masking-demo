"""Fair model comparison: OPF vs GiNZA on one gold set, one matcher (#23).

Fairness protocol (documented in REPORT):
- Same gold (data/eval/pilot.jsonl), same span matcher (char IoU >= 0.5).
- Predictions are restricted to each model's spec-mappable PII outputs
  (labelmap.py), so we compare PII-relevant detections, not all NER spans.
- Metrics: Recall & Precision (NO F1, per #24). Two views:
    * untyped = span detection (label ignored)  -> primary fair view
    * typed   = label must match (via mapping)   -> per-label, secondary
- Results are appended to the metrics ledger and plotted.

OPF predictions come from `opf eval --predictions-out` (char spans). GiNZA from
ja_ginza (split_mode workaround). This is a PILOT on a tiny synthetic set (#5
will scale it up); treat numbers as plumbing validation, not final.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import append_row  # noqa: E402
from labelmap import OPF_TO_SPEC, GINZA_TO_SPEC, SPEC_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval" / "pilot.jsonl"
FIGDIR = ROOT / "figures"
IOU_THRESH = 0.5

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
    used_p, used_g = set(), set()
    tp = 0
    for _v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp += 1
    return tp


def pr(tp: int, n_pred: int, n_gold: int) -> tuple[float | None, float | None]:
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    return p, r


def load_gold() -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["text"]] = [(s["category"], s["start"], s["end"]) for s in rec["label"]]
    return out


def opf_predictions() -> dict[str, list[Span]]:
    with tempfile.TemporaryDirectory() as td:
        preds_path = Path(td) / "opf_preds.jsonl"
        subprocess.run(["opf", "eval", str(GOLD), "--eval-mode", "untyped",
                        "--device", "cpu", "--span-metrics-space", "char",
                        "--predictions-out", str(preds_path),
                        "--metrics-out", str(Path(td) / "m.json")], check=True)
        out: dict[str, list[Span]] = {}
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
        return out


def ginza_predictions(texts: list[str]) -> dict[str, list[Span]]:
    import spacy
    nlp = spacy.load("ja_ginza",
                     config={"components.compound_splitter.split_mode": "C"})
    out: dict[str, list[Span]] = {}
    for text in texts:
        doc = nlp(text)
        spans: list[Span] = []
        for ent in doc.ents:
            spec = GINZA_TO_SPEC.get(ent.label_)
            if spec is None:
                continue
            spans.append((spec, ent.start_char, ent.end_char))
        out[text] = spans
    return out


def score(model: str, preds: dict[str, list[Span]], gold: dict[str, list[Span]]) -> None:
    run_id = uuid.uuid4().hex[:8]
    all_p = [sp for t in gold for sp in preds.get(t, [])]
    all_g = [sp for t in gold for sp in gold[t]]

    # untyped detection (label ignored) -> ALL row
    tp_u = sum(match(preds.get(t, []), gold[t], typed=False) for t in gold)
    p_u, r_u = pr(tp_u, len(all_p), len(all_g))
    append_row({"run_id": run_id, "stage": "compare", "model": model,
                "variant": "stock", "domain": "ALL", "label": "ALL",
                "precision": _round(p_u), "recall": _round(r_u),
                "miss_rate": _round(_inv(r_u)), "false_pos_rate": _round(_inv(p_u)),
                "note": f"untyped detection, IoU>={IOU_THRESH}, n_gold={len(all_g)}"})
    print(f"[{model}] untyped detection: P={_fmt(p_u)} R={_fmt(r_u)} "
          f"(tp={tp_u}/pred={len(all_p)}/gold={len(all_g)})")

    # typed per-label
    for lbl in SPEC_LABELS:
        gl = [sp for t in gold for sp in gold[t] if sp[0] == lbl]
        pl = [sp for t in gold for sp in preds.get(t, []) if sp[0] == lbl]
        if not gl and not pl:
            continue
        tp = sum(match([s for s in preds.get(t, []) if s[0] == lbl],
                       [s for s in gold[t] if s[0] == lbl], typed=True) for t in gold)
        p, r = pr(tp, len(pl), len(gl))
        append_row({"run_id": run_id, "stage": "compare", "model": model,
                    "variant": "stock", "domain": "ALL", "label": lbl,
                    "precision": _round(p), "recall": _round(r),
                    "miss_rate": _round(_inv(r)), "false_pos_rate": _round(_inv(p)),
                    "note": f"typed, n_gold={len(gl)}"})
    return p_u, r_u


def _inv(x):
    return (1 - x) if x is not None else None


def _round(x):
    return round(x, 4) if x is not None else ""


def _fmt(x):
    return f"{x:.3f}" if x is not None else "n/a"


def plot(results: dict[str, tuple]) -> None:
    models = list(results)
    xi = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w / 2 for i in xi], [(results[m][1] or 0) for m in models], w,
           color="#1565c0", label="Recall")
    ax.bar([i + w / 2 for i in xi], [(results[m][0] or 0) for m in models], w,
           color="#c0392b", label="Precision")
    ax.axhline(0.90, ls=":", color="#1565c0"); ax.axhline(0.85, ls=":", color="#c0392b")
    ax.set_xticks(list(xi)); ax.set_xticklabels(models)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title(f"OPF vs GiNZA — untyped PII detection (pilot, IoU>={IOU_THRESH})")
    ax.legend(); fig.tight_layout(); FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "model_compare_pr.png", dpi=130); plt.close(fig)
    print(f"wrote {FIGDIR/'model_compare_pr.png'}")


def main() -> int:
    gold = load_gold()
    texts = list(gold)
    results = {}
    results["A-OPF"] = score("A-OPF", opf_predictions(), gold)
    results["C-GiNZA"] = score("C-GiNZA", ginza_predictions(texts), gold)
    plot(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
