"""T2: category-clue ablation, Japanese (#46) — reproduce Model Card Table 5.

The OpenAI Model Card reports OPF Recall 0.584 (no clue) vs 0.863 (with clue):
a nearby *category word* ("電話番号は…") strongly helps detection. Real records
(診療録/相談記録) carry weaker clues, so we quantify the clue effect in Japanese.

Design: small synthetic set focused on ONE direct-identifier value per sentence,
under three conditions (same values, same seed, only the framing differs):
  - clue_prefix (前置): "{clue}：{value}"          e.g. 電話番号：090-...
  - clue_suffix (後置): "{value}（{clue}）"          e.g. 090-...（電話番号）
  - clue_none   (無し): neutral carrier, NO category word   e.g. 本日の記録は{value}。
Only the value is gold PII (the clue word is never part of the span). We run stock
OPF (B0) per condition and report untyped Recall/Precision (IoU>=0.5), overall and
per category. ~N sentences/category across 6 emittable categories.

Fairness: same matcher as B0 (char IoU>=0.5, P/R only #24). Re-uses 00_prepare_data
value generators (fixed seed). Rows -> ledger stage="ABL". Data under data/ablation/
(git-ignored, regenerable from this script + seed). Idempotent (caches preds).

Run: uv run python scripts/08_clue_ablation.py            # 102 sents/cond, 3 conds
     uv run python scripts/08_clue_ablation.py --n-per 8  # smaller, faster
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
import uuid
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import append_row  # noqa: E402
from labelmap import OPF_TO_SPEC  # noqa: E402

_gen = import_module("00_prepare_data")  # reuse value generators (same surfaces/seed)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "ablation"
IOU_THRESH = 0.5
SEED = 20260611  # distinct from eval(…609)/train(…610) so values don't collide
SPLIT = "ablation"

Span = tuple[str, int, int]

# Direct identifiers OPF can emit (+ how to produce a value & its category clue).
# value_fn(rng) -> (value:str, clue:str). Clue = the category word a human would write.
def _person(rng):
    return _gen.person(rng, allow_foreign=False), "氏名"
def _address(rng):
    return _gen.address(rng), "住所"
def _phone(rng):
    return _gen.jp_phone(rng), "電話番号"
def _email(rng):
    return _gen.jp_email(rng), "メールアドレス"
def _date(rng):
    return _gen.jp_date(rng), "受診日"
def _id(rng):
    label, val = _gen.jp_id(rng)
    return val, label  # the ID kind label (患者番号 等) is itself the clue

CATEGORIES = [
    ("PERSON", _person), ("ADDRESS", _address), ("PHONE", _phone),
    ("EMAIL", _email), ("DATE", _date), ("ID", _id),
]

# neutral carriers for the no-clue condition (NO category word)
_NEUTRAL = ["本日の記録は次のとおり。{v}。", "メモ：{v}", "記載内容：{v}", "確認：{v}", "{v}"]


def make_sentence(cat, value, clue, condition, rng) -> tuple[str, list[Span]]:
    if condition == "clue_prefix":
        prefix = f"{clue}："
        text = prefix + value
        s = len(prefix); e = s + len(value)
    elif condition == "clue_suffix":
        text = f"{value}（{clue}）"
        s = 0; e = len(value)
    elif condition == "clue_none":
        frame = rng.choice(_NEUTRAL)
        text = frame.format(v=value)
        s = text.index(value); e = s + len(value)
    else:
        raise ValueError(condition)
    assert text[s:e] == value, (text, value)
    return text, [(cat, s, e)]


def generate(n_per, condition):
    """Deterministic per (condition): same value sequence across conditions."""
    rng = random.Random(SEED)  # reset per condition -> identical values, only frame differs
    recs = []
    for cat, fn in CATEGORIES:
        for _ in range(n_per):
            value, clue = fn(rng)
            text, spans = make_sentence(cat, value, clue, condition, rng)
            recs.append({"text": text, "spans": spans})
    return recs


# ---- matcher (same greedy IoU as B0) ---------------------------------------
def iou(a: Span, b: Span) -> float:
    s = max(a[1], b[1]); e = min(a[2], b[2])
    inter = max(0, e - s)
    union = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return inter / union if union else 0.0


def match(preds, golds, *, typed) -> int:
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            if typed and p[0] != g[0]:
                continue
            v = iou(p, g)
            if v >= IOU_THRESH:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    up, ug, tp = set(), set(), 0
    for _v, pi, gi in pairs:
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); tp += 1
    return tp


def opf_predict(recs, condition):
    """Run stock OPF on the condition's texts; cache under data/ablation/.

    Returns {text: [spec spans]}. Cached file is regenerable, git-ignored.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = OUTDIR / f"opf_raw_{condition}.jsonl"
    if cache.exists():
        out = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["text"]] = [tuple(s) for s in r["spans"]]
        print(f"  (cached: {cache.name})")
        return out, 0.0
    with tempfile.TemporaryDirectory() as td:
        gin = Path(td) / "in.jsonl"
        with gin.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({"text": r["text"], "label": []}, ensure_ascii=False) + "\n")
        pout = Path(td) / "preds.jsonl"
        t0 = time.time()
        subprocess.run(["opf", "eval", str(gin), "--eval-mode", "untyped",
                        "--device", "cpu", "--span-metrics-space", "char",
                        "--predictions-out", str(pout),
                        "--metrics-out", str(Path(td) / "m.json")], check=True)
        elapsed = time.time() - t0
        out = {}
        for line in pout.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            spans = []
            for key, coords in rec["predicted_spans"].items():
                spec = OPF_TO_SPEC.get(key.split(":", 1)[0].strip())
                if spec is None:
                    continue
                for s, e in coords:
                    spans.append((spec, s, e))
            out[rec["text"]] = spans
        with cache.open("w", encoding="utf-8") as f:
            for t, spans in out.items():
                f.write(json.dumps({"text": t, "spans": spans}, ensure_ascii=False) + "\n")
        return out, elapsed


def _r(x):
    return round(x, 4) if x is not None else ""


def _f(x):
    return f"{x:.3f}" if x is not None else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=17, help="sentences per category per condition")
    args = ap.parse_args()
    run_id = uuid.uuid4().hex[:8]
    conditions = ["clue_prefix", "clue_suffix", "clue_none"]
    n_total = args.n_per * len(CATEGORIES)
    print(f"T2 clue ablation: {len(conditions)} conditions x {n_total} sentences "
          f"({args.n_per}/category x {len(CATEGORIES)} cats), run_id={run_id}\n")

    summary = {}
    for cond in conditions:
        recs = generate(args.n_per, cond)
        preds, elapsed = opf_predict(recs, cond)
        latency = round(elapsed * 1000 / len(recs), 1) if elapsed else ""
        # overall untyped (detection, label-agnostic)
        tp = n_pred = n_gold = 0
        per_cat = {}
        for r in recs:
            g = r["spans"]; p = preds.get(r["text"], [])
            tp += match(p, g, typed=False)
            n_pred += len(p); n_gold += len(g)
        for cat, _fn in CATEGORIES:
            ctp = cgold = cpred = 0
            for r in recs:
                g = [s for s in r["spans"] if s[0] == cat]
                if not g:
                    continue
                p = preds.get(r["text"], [])
                ctp += match(p, g, typed=False)
                cgold += len(g); cpred += len(p)
            cr = ctp / cgold if cgold else None
            per_cat[cat] = cr
            append_row({"run_id": run_id, "stage": "ABL", "model": "A-OPF",
                        "variant": cond, "domain": "ablation", "label": cat,
                        "recall": _r(cr), "miss_rate": _r((1 - cr) if cr is not None else None),
                        "seed": SEED,
                        "note": f"clue ablation untyped recall, cat={cat}, n_gold={cgold}, "
                                f"IoU>={IOU_THRESH}, split={SPLIT}"})
        rec = tp / n_gold if n_gold else None
        prec = tp / n_pred if n_pred else None
        summary[cond] = (rec, prec, per_cat)
        append_row({"run_id": run_id, "stage": "ABL", "model": "A-OPF",
                    "variant": cond, "domain": "ablation", "label": "ALL",
                    "precision": _r(prec), "recall": _r(rec),
                    "miss_rate": _r((1 - rec) if rec is not None else None),
                    "false_pos_rate": _r((1 - prec) if prec is not None else None),
                    "latency_ms": latency, "seed": SEED,
                    "note": f"clue ablation untyped recall (direct), n_gold={n_gold}, "
                            f"IoU>={IOU_THRESH}, split={SPLIT}"})
        print(f"[{cond:11}] untyped R={_f(rec)} P={_f(prec)} (n_gold={n_gold})  "
              + " ".join(f"{c}={_f(per_cat[c])}" for c, _ in CATEGORIES))

    print("\n=== clue effect (前置/後置 vs 無し) ===")
    base = summary["clue_none"][0]
    for cond in ("clue_prefix", "clue_suffix"):
        r = summary[cond][0]
        if r is not None and base is not None:
            print(f"  {cond}: R={_f(r)}  (無し={_f(base)}, Δ={r-base:+.3f})")
    print(f"\nappended ABL rows to outputs/metrics_ledger.csv (run_id={run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
