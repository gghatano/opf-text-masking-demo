"""T1: practical (anonymisation-business) metrics on the stock-OPF cache (#46).

Re-inference is NOT needed: we score the cached stock-OPF predictions
(data/eval/opf_raw_test.jsonl) and their B1 post-processed version against the
gold test split. Four sub-experiments quantify the gap between OpenAI-style
headline numbers and real masking utility:

  T1a document-level leakage  : % of docs that miss >=1 DIRECT identifier, and
                                the distribution of #misses per doc (B0 & B1).
  T1b IoU threshold sweep     : untyped Recall vs IoU in [0.1, 1.0] for the SAME
                                predictions, plus the char-token recall as a
                                reference line -> figures/iou_sweep.png.
  T1c quasi-identifier coverage: untyped Recall with quasi-identifiers kept in the
                                denominator (what OPF lets through by design).
  T1d over-redaction          : characters masked beyond gold (utility cost),
                                broken down by the predicting label.

Fairness (same protocol as B0/B1): gold = test split, char IoU>=0.5 for the
headline matcher, P/R only (#24). Direct identifiers = spec §9 (DATE は準識別子 #39).
Rows -> ledger stage="T1". Figures -> figures/. Re-inference-free, idempotent.

Run: uv run python scripts/07_practical_metrics.py
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import append_row  # noqa: E402
from labelmap import SPEC_LABELS  # noqa: E402
from importlib import import_module  # noqa: E402

# reuse B1's post-processing so T1 scores the exact same b1_postproc variant
_b1 = import_module("05_b1_postproc")
postprocess = _b1.postprocess

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "eval" / "eval_300.jsonl"
CACHE = ROOT / "data" / "eval" / "opf_raw_test.jsonl"
FIGDIR = ROOT / "figures"
SPLIT = "test"
IOU_THRESH = 0.5
DIRECT = {"PERSON", "ADDRESS", "PHONE", "EMAIL", "ID"}   # DATE は準識別子 (#39)
QUASI = {"AGE", "REGION", "OCCUPATION", "ORGANIZATION"}  # OPF が出さない設計
SEED = 20260609

Span = tuple[str, int, int]


# ---- matcher (same greedy IoU as B0/B1, threshold parametrised) -------------
def iou(a: Span, b: Span) -> float:
    s = max(a[1], b[1]); e = min(a[2], b[2])
    inter = max(0, e - s)
    union = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return inter / union if union else 0.0


def matched_gold(preds: list[Span], golds: list[Span], *, typed: bool, thr: float) -> set[int]:
    """Greedy IoU>=thr matching; return the set of matched gold indices."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            if typed and p[0] != g[0]:
                continue
            v = iou(p, g)
            if v >= thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for _v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    return used_g


def char_cover(spans: list[Span]) -> set[int]:
    """Set of character offsets covered by any span (untyped union)."""
    cov: set[int] = set()
    for _c, s, e in spans:
        cov.update(range(s, e))
    return cov


def _round(x):
    return round(x, 4) if x is not None else ""


def _fmt(x):
    return f"{x:.3f}" if x is not None else "n/a"


def load_gold():
    out = {}
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["info"]["split"] != SPLIT:
            continue
        out[rec["text"]] = (rec["info"]["domain"],
                            [(s["category"], s["start"], s["end"]) for s in rec["label"]])
    return out


def load_raw():
    if not CACHE.exists():
        sys.exit(f"missing {CACHE}; run scripts/04 or 05 (--split test) first to build the cache")
    out = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["text"]] = [tuple(s) for s in r["spans"]]
    return out


def emit(run_id, variant, domain, label, p, r, note):
    inv = lambda x: (1 - x) if x is not None else None
    append_row({"run_id": run_id, "stage": "T1", "model": "A-OPF",
                "variant": variant, "domain": domain, "label": label,
                "precision": _round(p), "recall": _round(r),
                "miss_rate": _round(inv(r)), "false_pos_rate": _round(inv(p)),
                "seed": SEED, "note": note})


# ============================ T1a doc-level leakage ==========================
def t1a(run_id, gold, variants):
    print("=== T1a 文書レベル漏えい率 (DIRECT identifiers) ===")
    for variant, preds in variants.items():
        leaked_docs = 0
        docs_with_direct = 0
        miss_hist: Counter[int] = Counter()
        total_miss = 0
        for text, (_dom, gspans) in gold.items():
            g = [s for s in gspans if s[0] in DIRECT]
            if not g:
                continue
            docs_with_direct += 1
            p = [s for s in preds[text] if s[0] in DIRECT]
            mg = matched_gold(p, g, typed=False, thr=IOU_THRESH)
            n_miss = len(g) - len(mg)
            miss_hist[n_miss] += 1
            total_miss += n_miss
            if n_miss > 0:
                leaked_docs += 1
        leak_rate = leaked_docs / docs_with_direct
        # store leak rate as recall=1-leak_rate semantics in miss_rate column
        emit(run_id, variant, "ALL", "DOC_LEAK_DIRECT", None, None,
             f"doc-level leakage: {leaked_docs}/{docs_with_direct} docs miss >=1 DIRECT "
             f"(leak_rate={leak_rate:.4f}), mean_miss/doc={total_miss/docs_with_direct:.3f}, "
             f"IoU>={IOU_THRESH}, split={SPLIT}")
        # explicitly record leak_rate in miss_rate for plotting/auditing
        append_row({"run_id": run_id, "date": "", "stage": "T1", "model": "A-OPF",
                    "variant": variant, "domain": "ALL", "label": "DOC_LEAK_RATE_DIRECT",
                    "miss_rate": _round(leak_rate), "seed": SEED,
                    "note": f"frac docs with >=1 missed DIRECT, n_docs={docs_with_direct}, "
                            f"total_miss={total_miss}, IoU>={IOU_THRESH}, split={SPLIT}"})
        hist = {k: miss_hist[k] for k in sorted(miss_hist)}
        print(f"  [{variant:12}] leak_rate={leak_rate:.3f} "
              f"({leaked_docs}/{docs_with_direct} docs), "
              f"mean miss/doc={total_miss/docs_with_direct:.3f}, dist(miss->#docs)={hist}")
    print()


# ============================ T1b IoU sweep ==================================
def t1b(run_id, gold, variants):
    print("=== T1b IoU 閾値スイープ (untyped Recall) ===")
    thrs = [round(0.1 * i, 1) for i in range(1, 11)]  # 0.1 .. 1.0
    curves: dict[str, dict[str, list[float]]] = {}
    # char-token recall (IoU-independent reference): gold chars covered / total gold chars
    token_ref: dict[str, dict[str, float]] = {}
    for variant, preds in variants.items():
        curves[variant] = {}
        token_ref[variant] = {}
        for view, gset in [("DIRECT", DIRECT), ("ALL", set(SPEC_LABELS))]:
            rec_curve = []
            for thr in thrs:
                tp = n_gold = 0
                for text, (_dom, gspans) in gold.items():
                    g = [s for s in gspans if s[0] in gset]
                    p = [s for s in preds[text] if s[0] in gset]
                    tp += len(matched_gold(p, g, typed=False, thr=thr))
                    n_gold += len(g)
                rec_curve.append(tp / n_gold if n_gold else 0.0)
            curves[variant][view] = rec_curve
            # char-token recall
            cov_gold = tot_gold = 0
            for text, (_dom, gspans) in gold.items():
                g = [s for s in gspans if s[0] in gset]
                p = [s for s in preds[text] if s[0] in gset]
                gchars = char_cover(g)
                pchars = char_cover(p)
                cov_gold += len(gchars & pchars)
                tot_gold += len(gchars)
            token_ref[variant][view] = cov_gold / tot_gold if tot_gold else 0.0
        # ledger: record span recall at the headline IoU=0.5 and the token recall
        i05 = thrs.index(0.5)
        for view in ("DIRECT", "ALL"):
            emit(run_id, variant, "ALL", f"SWEEP_{view}", None, curves[variant][view][i05],
                 f"untyped span recall @IoU0.5 (sweep anchor), view={view}, split={SPLIT}")
            emit(run_id, variant, "ALL", f"TOKEN_{view}", None, token_ref[variant][view],
                 f"char-token untyped recall (IoU-independent), view={view}, split={SPLIT}")
        print(f"  [{variant:12}] DIRECT span@0.5={curves[variant]['DIRECT'][i05]:.3f} "
              f"token={token_ref[variant]['DIRECT']:.3f} | "
              f"ALL span@0.5={curves[variant]['ALL'][i05]:.3f} token={token_ref[variant]['ALL']:.3f}")

    _plot_sweep(thrs, curves, token_ref)
    print(f"  figure -> {FIGDIR / 'iou_sweep.png'}\n")
    return thrs, curves, token_ref


def _plot_sweep(thrs, curves, token_ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:  # pragma: no cover
        print(f"  (matplotlib unavailable: {ex}; skipping figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"stock": "#1f77b4", "b1_postproc": "#d62728"}
    for ax, view in zip(axes, ("DIRECT", "ALL")):
        for variant in curves:
            c = colors.get(variant, None)
            ax.plot(thrs, curves[variant][view], "-o", color=c, label=f"{variant} (span)")
            ax.axhline(token_ref[variant][view], ls="--", color=c, alpha=0.7,
                       label=f"{variant} (char-token)")
        ax.set_title(f"untyped Recall — {view}")
        ax.set_xlabel("IoU threshold")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("T1b: span-IoU recall vs char-token recall (same predictions)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "iou_sweep.png", dpi=130)
    plt.close(fig)


# ============================ T1c quasi coverage =============================
def t1c(run_id, gold, variants):
    print("=== T1c 準識別子を分母に戻した untyped Recall ===")
    for variant, preds in variants.items():
        for view, gset in [("DIRECT", DIRECT), ("QUASI", QUASI),
                           ("ALL", set(SPEC_LABELS))]:
            tp = n_gold = n_pred = 0
            for text, (_dom, gspans) in gold.items():
                g = [s for s in gspans if s[0] in gset]
                p = [s for s in preds[text] if s[0] in gset]
                tp += len(matched_gold(p, g, typed=False, thr=IOU_THRESH))
                n_gold += len(g); n_pred += len(p)
            r = tp / n_gold if n_gold else None
            p_ = tp / n_pred if n_pred else None
            emit(run_id, variant, "ALL", f"COVER_{view}", p_, r,
                 f"untyped recall, quasi in denom, view={view}, IoU>={IOU_THRESH}, "
                 f"split={SPLIT}, n_gold={n_gold}")
            print(f"  [{variant:12}] {view:7} R={_fmt(r)} P={_fmt(p_)} (n_gold={n_gold})")
    print()


# ============================ T1d over-redaction =============================
def t1d(run_id, gold, variants):
    print("=== T1d 過剰マスキング量 (over-redaction, chars beyond gold) ===")
    for variant, preds in variants.items():
        gold_chars_all = pred_chars_all = over_all = 0
        by_label_over: Counter[str] = Counter()
        by_label_pred: Counter[str] = Counter()
        for text, (_dom, gspans) in gold.items():
            gcov = char_cover(gspans)  # gold masks any labelled span (untyped redaction)
            for cat, s, e in preds[text]:
                rng = set(range(s, e))
                over = rng - gcov
                by_label_pred[cat] += len(rng)
                by_label_over[cat] += len(over)
                over_all += len(over)
                pred_chars_all += len(rng)
            gold_chars_all += len(gcov)
        over_ratio = over_all / pred_chars_all if pred_chars_all else 0.0
        emit(run_id, variant, "ALL", "OVER_REDACTION", None, None,
             f"over-redacted chars={over_all}/{pred_chars_all} pred chars "
             f"(ratio={over_ratio:.4f}); gold chars={gold_chars_all}; split={SPLIT}")
        append_row({"run_id": run_id, "date": "", "stage": "T1", "model": "A-OPF",
                    "variant": variant, "domain": "ALL", "label": "OVER_REDACTION_RATIO",
                    "false_pos_rate": _round(over_ratio), "seed": SEED,
                    "note": f"over_chars={over_all}, pred_chars={pred_chars_all}, "
                            f"gold_chars={gold_chars_all}, split={SPLIT}"})
        detail = {k: f"{by_label_over[k]}/{by_label_pred[k]}"
                  for k in sorted(by_label_pred, key=lambda x: -by_label_over[x])}
        print(f"  [{variant:12}] over/pred chars={over_all}/{pred_chars_all} "
              f"(ratio={over_ratio:.3f}) by-label(over/pred)={detail}")
        for cat in sorted(by_label_pred):
            if by_label_pred[cat] == 0:
                continue
            emit(run_id, variant, "ALL", f"OVER_{cat}", None, None,
                 f"over-redacted chars (label {cat})={by_label_over[cat]}/{by_label_pred[cat]}, "
                 f"split={SPLIT}")
    print()


def main() -> int:
    gold = load_gold()
    raw = load_raw()
    missing = [t for t in gold if t not in raw]
    if missing:
        sys.exit(f"{len(missing)} gold texts missing from cache; rebuild opf_raw_test.jsonl")
    post = {t: postprocess(t, raw[t]) for t in gold}
    variants = {"stock": raw, "b1_postproc": post}
    run_id = uuid.uuid4().hex[:8]
    print(f"T1 practical metrics on {len(gold)} '{SPLIT}' docs (run_id={run_id})\n")
    t1a(run_id, gold, variants)
    t1b(run_id, gold, variants)
    t1c(run_id, gold, variants)
    t1d(run_id, gold, variants)
    print(f"appended T1 rows to outputs/metrics_ledger.csv (run_id={run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
