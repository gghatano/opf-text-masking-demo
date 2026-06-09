"""Re-score stock OPF (B0) under the OpenAI Model Card's evaluation conventions,
so our numbers are comparable to Model Card Table 7 (Japanese) [4].

Why: our default report uses span match (char IoU>=0.5) over the full 10-label
taxonomy. The Model Card instead (i) maps gold to OPF's own categories and drops
labels with no OPF counterpart, and (ii) reports token-level recall/precision
(Table 1 is explicitly "tokens"; partial overlaps are credited per token). Those
two choices, not a real performance gap, explain most of the headline difference.

This script restricts to OPF's 6 mappable categories and reports BOTH:
  - token-level (character proxy)   <- comparable to the Model Card
  - span-level (char IoU>=0.5)       <- our default, for reference
on the same stock-OPF (B0) predictions (reusing the cached predictions).

Caveat: the Model Card uses o200k_base subword tokens; we use a *character*
proxy (no subword alignment), and a different Japanese dataset (domain text vs.
PII-Masking-300k-style synthetic). So this aligns the *metric*, not the dataset.

Run: uv run python scripts/06_openai_alignment.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import append_row  # noqa: E402
from labelmap import OPF_TO_SPEC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval" / "eval_300.jsonl"
CACHE = ROOT / "data" / "eval" / "opf_raw_test.jsonl"
SPLIT = "test"
IOU = 0.5
OPF6 = {"PERSON", "ADDRESS", "PHONE", "EMAIL", "DATE", "ID"}  # OPF-mappable categories
# Model Card Table 7, Japanese (n=968): token-level R/P/F1
MC_JA = (0.866, 0.897, 0.881)

Span = tuple[str, int, int]


def load_gold() -> dict[str, list[Span]]:
    out = {}
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["info"]["split"] != SPLIT:
            continue
        out[rec["text"]] = [(s["category"], s["start"], s["end"]) for s in rec["label"]]
    return out


def stock_predictions(texts: list[str]) -> dict[str, list[Span]]:
    if CACHE.exists():
        out = {}
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["text"]] = [tuple(s) for s in r["spans"]]
        print(f"(using cached stock OPF preds: {CACHE.name})")
        return out
    with tempfile.TemporaryDirectory() as td:
        gin = Path(td) / "in.jsonl"
        gin.write_text("".join(json.dumps({"text": t, "label": []}, ensure_ascii=False)
                               + "\n" for t in texts), encoding="utf-8")
        pout = Path(td) / "p.jsonl"
        subprocess.run(["opf", "eval", str(gin), "--eval-mode", "untyped", "--device",
                        "cpu", "--span-metrics-space", "char", "--predictions-out",
                        str(pout), "--metrics-out", str(Path(td) / "m.json")], check=True)
        out = {}
        for line in pout.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            spans = []
            for key, coords in rec["predicted_spans"].items():
                spec = OPF_TO_SPEC.get(key.split(":", 1)[0].strip())
                if spec is None:
                    continue
                spans.extend((spec, s, e) for s, e in coords)
            out[rec["text"]] = spans
        CACHE.write_text("".join(json.dumps({"text": t, "spans": s}, ensure_ascii=False)
                                 + "\n" for t, s in out.items()), encoding="utf-8")
        return out


# ---- token (character) level over OPF6 -------------------------------------
def char_idx(spans, cats):
    untyped: set[int] = set()
    typed: dict[str, set[int]] = {}
    for c, s, e in spans:
        if c in cats:
            idx = set(range(s, e))
            untyped |= idx
            typed.setdefault(c, set()).update(idx)
    return untyped, typed


def token_level(gold, preds):
    tp = npred = ngold = 0          # untyped
    ttp = tnp = tng = 0             # typed
    percat = {c: [0, 0, 0] for c in OPF6}  # cat -> [tp, npred, ngold]
    for t, gs in gold.items():
        gu, gt = char_idx(gs, OPF6)
        pu, pt = char_idx(preds.get(t, []), OPF6)
        tp += len(gu & pu); npred += len(pu); ngold += len(gu)
        for c in OPF6:
            gc, pc = gt.get(c, set()), pt.get(c, set())
            inter = len(gc & pc)
            ttp += inter; tnp += len(pc); tng += len(gc)
            percat[c][0] += inter; percat[c][1] += len(pc); percat[c][2] += len(gc)
    return (tp, npred, ngold), (ttp, tnp, tng), percat


# ---- span level (IoU>=0.5) over OPF6, for reference ------------------------
def iou(a, b):
    s = max(a[1], b[1]); e = min(a[2], b[2]); inter = max(0, e - s)
    union = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return inter / union if union else 0.0


def span_untyped(gold, preds):
    tp = npred = ngold = 0
    for t, gs in gold.items():
        G = [s for s in gs if s[0] in OPF6]
        P = [s for s in preds.get(t, []) if s[0] in OPF6]
        pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(P)
                        for gi, g in enumerate(G) if iou(p, g) >= IOU), reverse=True)
        up, ug = set(), set()
        for _v, pi, gi in pairs:
            if pi in up or gi in ug:
                continue
            up.add(pi); ug.add(gi); tp += 1
        npred += len(P); ngold += len(G)
    return tp, npred, ngold


def rp(tp, np_, ng):
    return (tp / ng if ng else 0.0), (tp / np_ if np_ else 0.0)


def f1(r, p):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> int:
    gold = load_gold()
    preds = stock_predictions(list(gold))
    run_id = uuid.uuid4().hex[:8]

    (utp, unp, ung), (ttp, tnp, tng), percat = token_level(gold, preds)
    r_tok, p_tok = rp(utp, unp, ung)
    r_tokT, p_tokT = rp(ttp, tnp, tng)
    s_tp, s_np, s_ng = span_untyped(gold, preds)
    r_sp, p_sp = rp(s_tp, s_np, s_ng)

    print(f"B0 stock OPF on {len(gold)} '{SPLIT}' docs, restricted to OPF's 6 categories\n")
    print("                              Recall  Precision  F1")
    print(f"token-level untyped (~MC)   : {r_tok:.3f}   {p_tok:.3f}     {f1(r_tok,p_tok):.3f}")
    print(f"token-level typed           : {r_tokT:.3f}   {p_tokT:.3f}     {f1(r_tokT,p_tokT):.3f}")
    print(f"span IoU>=0.5 untyped (ours): {r_sp:.3f}   {p_sp:.3f}     {f1(r_sp,p_sp):.3f}")
    print(f"\nOpenAI Model Card Table 7, Japanese (n=968): "
          f"R={MC_JA[0]} P={MC_JA[1]} F1={MC_JA[2]}")
    print("\nper-category token-level (R / P):")
    for c in sorted(OPF6):
        tp, np_, ng = percat[c]
        r, p = rp(tp, np_, ng)
        print(f"  {c:<8} R={r:.3f} P={p:.3f} (gold_chars={ng})")

    for label, (r, p) in [("OPF6-untyped(token)", (r_tok, p_tok)),
                          ("OPF6-typed(token)", (r_tokT, p_tokT)),
                          ("OPF6-untyped(span)", (r_sp, p_sp))]:
        append_row({"run_id": run_id, "stage": "B0", "model": "A-OPF",
                    "variant": "stock", "domain": "OPF6", "label": label,
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1(r, p), 4), "seed": 20260609,
                    "note": f"OpenAI Model Card alignment (Table7=R{MC_JA[0]}/P{MC_JA[1]}), "
                            f"split={SPLIT}, char-token proxy"})
    print(f"\nappended alignment rows to ledger (run_id={run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
