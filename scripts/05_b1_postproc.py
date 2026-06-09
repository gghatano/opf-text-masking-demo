"""B1: rule-based post-processing on top of stock OPF predictions (#8).

B0 (#7) showed two clear, fixable weaknesses on the synthetic set:
  - PERSON Recall 0.38 — OPF over-extends spans ("佐藤花子（72歳）は" as one span),
    so char IoU<0.5 against the gold name. Fix = trim trailing honorifics /
    particles / parenthetical / age / punctuation from PERSON spans.
  - DATE Recall 0.29 — OPF misses 和暦 (令和X年…) and 略記 (RX.X.X). Fix = add
    high-precision regex DATE spans, de-duped against existing predictions.
PHONE/EMAIL/ID are already saturated and left untouched.

Fairness (#20): tune/inspect on the **dev** split only, then run **test** once.
Same matcher as B0 (char IoU>=0.5, P/R, no F1 #24). Rows -> ledger stage="B1".

Usage:
  uv run python scripts/05_b1_postproc.py --split dev  --diagnose   # tuning
  uv run python scripts/05_b1_postproc.py --split test             # final, once
"""
from __future__ import annotations

import argparse
import json
import re
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
DIRECT = {"PERSON", "ADDRESS", "PHONE", "EMAIL", "DATE", "ID"}

Span = tuple[str, int, int]

# ---- PERSON boundary trim (generic JP rules; no template-specific vocab) ----
# Trailing honorifics / titles to drop from a PERSON span.
_HONORIFICS = ["先生", "さん", "ちゃん", "くん", "様", "氏", "君", "殿"]
# A PERSON span ends before any of these (paren / age digits / punctuation).
_TRAIL_BOUNDARY = re.compile("[（(［[【「』」、。，．！？・0-9０-９]")
_TRAIL_PARTICLES = "はがをにのへともやでだ"
_LEAD_COLONS = "：:｜|＞>"          # "退院サマリ：<name>" style record prefixes
_WS = " 　\t\n"


def trim_person(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Clean an over-extended PERSON span. Generic rules only:
    1. drop a leading "label：/|/>" record prefix (keep text after last such char),
    2. if whitespace remains, keep the last whitespace-delimited segment (OPF often
       merges "<title> <name>"; the name is last),
    3. cut at the first trailing boundary (paren/age/punct),
    4. strip trailing honorifics/particles and surrounding ws.
    """
    s, e = start, end
    surf = text[s:e]
    # 1. leading record prefix up to last colon-like char
    lead = max(surf.rfind(c) for c in _LEAD_COLONS)
    if lead >= 0:
        s += lead + 1; surf = text[s:e]
    # 2. last whitespace-delimited segment
    ws = max(surf.rfind(c) for c in _WS)
    if ws >= 0:
        s += ws + 1; surf = text[s:e]
    # strip leading ws/particles left over
    while surf and (surf[0] in _WS or surf[0] in "・の"):
        s += 1; surf = text[s:e]
    # 3. cut at first trailing boundary char
    m = _TRAIL_BOUNDARY.search(surf)
    if m:
        e = s + m.start(); surf = text[s:e]
    # 4. trailing honorifics (repeatable) + one particle + ws
    changed = True
    while changed:
        changed = False
        for h in _HONORIFICS:
            if surf.endswith(h):
                e -= len(h); surf = text[s:e]; changed = True
    if surf and surf[-1] in _TRAIL_PARTICLES:
        e -= 1; surf = text[s:e]
    while e > s and text[e - 1] in _WS:
        e -= 1
    if e <= s:
        return None
    return s, e


# ---- DATE regex (和暦 / 略記 / 西暦 / slash) --------------------------------
_DATE_PATTERNS = [
    re.compile(r"(令和|平成|昭和)\d{1,2}年\d{1,2}月\d{1,2}日"),
    re.compile(r"[RHS]\d{1,2}\.\d{1,2}\.\d{1,2}"),          # R7.1.18
    re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
    re.compile(r"\d{4}/\d{1,2}/\d{1,2}"),
]


def regex_dates(text: str) -> list[Span]:
    out: list[Span] = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            out.append(("DATE", m.start(), m.end()))
    return out


def overlaps(a: Span, b: Span) -> bool:
    return not (a[2] <= b[1] or b[2] <= a[1])


def postprocess(text: str, spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for cat, s, e in spans:
        if cat == "PERSON":
            t = trim_person(text, s, e)
            if t:
                out.append(("PERSON", t[0], t[1]))
        else:
            out.append((cat, s, e))
    # add regex DATEs that don't overlap an existing DATE prediction
    existing_dates = [sp for sp in out if sp[0] == "DATE"]
    for d in regex_dates(text):
        if not any(overlaps(d, e) for e in existing_dates):
            out.append(d)
    return out


# ---- matcher / metrics (same as B0) ---------------------------------------
def iou(a: Span, b: Span) -> float:
    s = max(a[1], b[1]); e = min(a[2], b[2])
    inter = max(0, e - s)
    union = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return inter / union if union else 0.0


def match(preds, golds, *, typed):
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


def pr(tp, n_pred, n_gold):
    return (tp / n_pred if n_pred else None), (tp / n_gold if n_gold else None)


def _inv(x):
    return (1 - x) if x is not None else None


def _round(x):
    return round(x, 4) if x is not None else ""


def _fmt(x):
    return f"{x:.3f}" if x is not None else "n/a"


def load_gold(split):
    out = {}
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["info"]["split"] != split:
            continue
        out[rec["text"]] = (rec["info"]["domain"],
                            [(s["category"], s["start"], s["end"]) for s in rec["label"]])
    return out


def opf_raw(texts, split):
    """Run stock OPF, return ({text: spec spans}, elapsed). Cache per split."""
    cache = ROOT / "data" / "eval" / f"opf_raw_{split}.jsonl"
    if cache.exists():
        out = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["text"]] = [tuple(s) for s in r["spans"]]
        print(f"(using cached OPF preds: {cache.name})")
        return out, 0.0
    with tempfile.TemporaryDirectory() as td:
        gold_in = Path(td) / "in.jsonl"
        with gold_in.open("w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t, "label": []}, ensure_ascii=False) + "\n")
        preds_path = Path(td) / "preds.jsonl"
        t0 = time.time()
        subprocess.run(["opf", "eval", str(gold_in), "--eval-mode", "untyped",
                        "--device", "cpu", "--span-metrics-space", "char",
                        "--predictions-out", str(preds_path),
                        "--metrics-out", str(Path(td) / "m.json")], check=True)
        elapsed = time.time() - t0
        out = {}
        for line in preds_path.read_text(encoding="utf-8").splitlines():
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


def diagnose(gold, raw):
    """Print PERSON over-extension and missed DATE patterns (dev tuning aid)."""
    print("\n=== PERSON: raw vs post-proc vs gold (first 12 mismatches) ===")
    shown = 0
    for text, (_dom, gspans) in gold.items():
        graw = [s for s in raw.get(text, []) if s[0] == "PERSON"]
        gpost = [s for s in postprocess(text, raw.get(text, [])) if s[0] == "PERSON"]
        ggold = [s for s in gspans if s[0] == "PERSON"]
        # show cases where a raw PERSON span doesn't IoU-match gold but post does, or neither
        for rs in graw:
            best_raw = max((iou(rs, g) for g in ggold), default=0)
            if best_raw < IOU_THRESH and shown < 12:
                psurf = [text[s[1]:s[2]] for s in gpost if not (s[2] <= rs[1] or rs[2] <= s[1])]
                print(f"  raw={text[rs[1]:rs[2]]!r:30} -> post={psurf} | gold={[text[g[1]:g[2]] for g in ggold]}")
                shown += 1
    print("\n=== DATE: gold missed by raw OPF, recovered by regex? ===")
    shown = 0
    for text, (_dom, gspans) in gold.items():
        graw = [s for s in raw.get(text, []) if s[0] == "DATE"]
        for g in [s for s in gspans if s[0] == "DATE"]:
            if max((iou(g, r) for r in graw), default=0) < IOU_THRESH and shown < 15:
                rgx = [text[d[1]:d[2]] for d in regex_dates(text)
                       if not (d[2] <= g[1] or g[2] <= d[1])]
                print(f"  gold={text[g[1]:g[2]]!r:18} raw=MISS regex={rgx}")
                shown += 1


def emit(run_id, variant, domain, label, tp, n_pred, n_gold, latency, note):
    p, r = pr(tp, n_pred, n_gold)
    append_row({"run_id": run_id, "stage": "B1", "model": "A-OPF",
                "variant": variant, "domain": domain, "label": label,
                "precision": _round(p), "recall": _round(r),
                "miss_rate": _round(_inv(r)), "false_pos_rate": _round(_inv(p)),
                "latency_ms": latency, "seed": 20260609, "note": note})
    return p, r


def untyped(run_id, variant, gold, preds, domain_label, gset, split, latency=""):
    tp = n_pred = n_gold = 0
    for text, (_dom, gspans) in gold.items():
        g = [s for s in gspans if s[0] in gset]
        p = [s for s in preds.get(text, []) if s[0] in gset]
        tp += match(p, g, typed=False)
        n_pred += len(p); n_gold += len(g)
    label = "ALL" if gset == set(SPEC_LABELS) else "DIRECT"
    return emit(run_id, variant, domain_label, label, tp, n_pred, n_gold, latency,
                f"untyped {label.lower()}, IoU>={IOU_THRESH}, split={split}, n_gold={n_gold}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()

    gold = load_gold(args.split)
    if not gold:
        sys.exit(f"no '{args.split}' records in {GOLD}")
    texts = list(gold)
    raw, elapsed = opf_raw(texts, args.split)
    latency_ms = round(elapsed * 1000 / len(texts), 1) if elapsed else ""

    if args.diagnose:
        diagnose(gold, raw)

    post = {t: postprocess(t, raw.get(t, [])) for t in texts}
    run_id = uuid.uuid4().hex[:8]
    allL = set(SPEC_LABELS)

    print(f"\nB1 post-proc on {len(texts)} '{args.split}' docs")
    for variant, preds in [("stock", raw), ("b1_postproc", post)]:
        p_d, r_d = untyped(run_id, variant, gold, preds, "ALL_direct", DIRECT, args.split, latency_ms)
        untyped(run_id, variant, gold, preds, "ALL", allL, args.split, latency_ms)
        print(f"  [{variant:12}] direct untyped: P={_fmt(p_d)} R={_fmt(r_d)}")

    # typed per-label for the post-proc variant (focus labels)
    print("  --- b1_postproc typed Recall ---")
    for lbl in SPEC_LABELS:
        gl = [s for _t, (_d, gs) in gold.items() for s in gs if s[0] == lbl]
        pl = [s for t in gold for s in post.get(t, []) if s[0] == lbl]
        if not gl and not pl:
            continue
        tp = sum(match([s for s in post.get(t, []) if s[0] == lbl],
                       [s for s in gold[t][1] if s[0] == lbl], typed=True) for t in gold)
        p, r = emit(run_id, "b1_postproc", "ALL", lbl, tp, len(pl), len(gl), "",
                    f"typed, IoU>={IOU_THRESH}, split={args.split}, n_gold={len(gl)}")
        print(f"    {lbl:<12} P={_fmt(p)} R={_fmt(r)} (gold={len(gl)})")

    print(f"\nappended B1 rows (run_id={run_id}, split={args.split})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
