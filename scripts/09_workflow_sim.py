"""業務適用シミュレーション: 検出性能 → 工数削減率の感度分析 (#14, モデル #13).

検出性能（P/R）を仮想業務フローに流し込み、匿名加工業務の **作業削減率** と
**見逃し率**（spec §9）を感度分析で試算する。検出性能と工数削減は業務フローを介して
のみ結びつく（循環論法の回避, #13）ため、単一値で断定せず、最大の仮定である通読
削減係数 ρ と最終チェック捕捉率 q をスイープして KPI のレンジを提示する。

入力は再推論なし: 検出評価の集計値（TP/FP/FN, 文書数, gold 件数）を
`outputs/metrics_ledger.csv` の T1 行（`COVER_DIRECT` と `DOC_LEAK_RATE_DIRECT`,
直接識別子・IoU≥0.5・test split）から復元する（#13「TP/FP/FN は台帳から」）。
データ再生成も OPF 再推論も不要・冪等。

仮想業務フロー (#13):
  原文書 → AI が検出・加工 → {加工不要: 人が通読し見逃し確認 / 要加工: AI 候補を人が確認・修正}
         → 最終チェック。人手単独（AI 不使用）を対照に削減率を測る。

工数モデル (#13, 1 文書あたり・秒):
  人手単独   ≒ c_read + c_mark·G + c_final
  支援併用(ρ) ≒ c_conf·(TP+FP) + ρ·c_read + c_fix·FN + c_final
  作業削減率 = 1 − Σ支援併用 / Σ人手単独
  ρ = AI 支援による通読削減係数（0=通読ゼロ … 1=全通読＝「AI 加工後に人が全部見る」）。**最大の仮定**。

見逃し率（2 定義, #13）:
  (1) 検出 FN/G = 1 − Recall（検出器そのものの取りこぼし）
  (2) 最終チェック後残存 = (1 − Recall)·(1 − q)  ← **業務 KPI**（spec §9: ≤5%）
      q = 最終チェックでの見逃し捕捉率（人手レビューが取りこぼしを拾う割合・仮定）。

係数 c_* は社内実感に基づく **仮の初期値**であり、校正は #13 の論点として残す
（実測パイロット未実施）。本シミュレーションは「削減率を感度レンジで提示」する位置づけ。

Run: uv run python scripts/09_workflow_sim.py
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import append_row, LEDGER_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
SEED = 20260609
SPLIT = "test"

# --- 工数係数（仮の初期値・秒/単位。校正は #13 の論点として残す） ----------------
# 単位は「秒」。比のみが削減率に効くため絶対値より相対比が重要。
COEF = {
    "c_read": 90,    # 1 文書を人手で通読し判断する時間
    "c_mark": 5,     # 識別子 1 件を人手でマスクする時間（特定＋加工）
    "c_conf": 7,     # AI 候補 1 件を人が確認/修正する時間（判定＋採否）
    "c_fix": 10,     # 通読中に見逃し 1 件を発見し処理する時間（無支援探索ゆえ割高）
    "c_final": 20,   # 文書あたり固定の最終チェック（両フロー共通）
}

DIRECT_VIEW = "COVER_DIRECT"          # 直接識別子（spec §9）の集計行
VARIANTS = {"stock": "B0", "b1_postproc": "B1"}
RHO_ANCHORS = [0.2, 0.3, 0.5, 1.0]     # 削減率を台帳に残す ρ アンカー
Q_ANCHORS = [0.5, 0.8, 0.95]           # 残存見逃し率を台帳に残す q アンカー


def _f(x):
    return None if x in ("", None) else float(x)


def load_counts() -> dict[str, dict]:
    """台帳 T1 行から直接識別子の集計値（R, P, n_gold, n_docs, FN）を復元する。"""
    import csv
    rows = list(csv.DictReader(LEDGER_PATH.open(encoding="utf-8")))
    out: dict[str, dict] = {}
    for variant in VARIANTS:
        cover = next((r for r in rows if r["stage"] == "T1" and r["variant"] == variant
                      and r["label"] == DIRECT_VIEW), None)
        leak = next((r for r in rows if r["stage"] == "T1" and r["variant"] == variant
                     and r["label"] == "DOC_LEAK_RATE_DIRECT"), None)
        if not cover or not leak:
            sys.exit(f"ledger に T1/{variant} の COVER_DIRECT/DOC_LEAK_RATE_DIRECT 行が無い。"
                     " 先に scripts/07_practical_metrics.py を実行してください。")
        R, P = _f(cover["recall"]), _f(cover["precision"])
        n_gold = int(re.search(r"n_gold=(\d+)", cover["note"]).group(1))
        n_docs = int(re.search(r"n_docs=(\d+)", leak["note"]).group(1))
        total_miss = int(re.search(r"total_miss=(\d+)", leak["note"]).group(1))
        tp = round(R * n_gold)
        fn = n_gold - tp
        pred = round(tp / P) if P else tp
        fp = max(pred - tp, 0)
        out[variant] = {"R": R, "P": P, "G": n_gold, "n_docs": n_docs,
                        "TP": tp, "FP": fp, "FN": fn, "leak_total_miss": total_miss}
    return out


def manual_effort(c: dict) -> float:
    return c["n_docs"] * (COEF["c_read"] + COEF["c_final"]) + COEF["c_mark"] * c["G"]


def assisted_effort(c: dict, rho: float) -> float:
    return (COEF["c_conf"] * (c["TP"] + c["FP"])
            + rho * COEF["c_read"] * c["n_docs"]
            + COEF["c_fix"] * c["FN"]
            + COEF["c_final"] * c["n_docs"])


def reduction(c: dict, rho: float) -> float:
    m = manual_effort(c)
    return 1 - assisted_effort(c, rho) / m if m else 0.0


def residual_leak(c: dict, q: float) -> float:
    """最終チェック後の残存見逃し率（業務 KPI）= (1−R)·(1−q)。"""
    return (1 - c["R"]) * (1 - q)


def emit(run_id, variant, label, *, recall=None, miss_rate=None, note=""):
    append_row({"run_id": run_id, "stage": "WF", "model": "A-OPF",
                "variant": variant, "domain": "ALL", "label": label,
                "recall": round(recall, 4) if recall is not None else "",
                "miss_rate": round(miss_rate, 4) if miss_rate is not None else "",
                "seed": SEED, "note": note})


def _plot(counts: dict, run_id: str):
    # Figure labels are in English on purpose (matplotlib JP font issues; same as
    # scripts/plot_progression.py & 07_practical_metrics.py).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:  # pragma: no cover
        print(f"  (matplotlib unavailable: {ex}; skipping figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    rhos = [i / 100 for i in range(0, 101)]
    qs = [i / 100 for i in range(0, 101)]
    colors = {"stock": "#1f77b4", "b1_postproc": "#d62728"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # left: workload reduction vs rho
    for v, c in counts.items():
        ys = [reduction(c, r) for r in rhos]
        ax1.plot(rhos, ys, "-", color=colors.get(v), label=f"{VARIANTS[v]} ({v})")
    ax1.axhline(0.50, ls="--", color="#444", alpha=0.8)
    ax1.text(0.02, 0.515, "target reduction >= 0.50 (spec 9)", fontsize=8, color="#444")
    ax1.axvline(1.0, ls=":", color="#888", alpha=0.7)
    ax1.text(0.99, -0.06, "rho=1: human re-reads\nall after AI", fontsize=7.5, color="#888",
             ha="right", va="bottom")
    ax1.set_title("Workload reduction vs skim factor rho")
    ax1.set_xlabel("rho (AI-assisted skim factor: 0=no read ... 1=full read)")
    ax1.set_ylabel("reduction = 1 - sum(assisted)/sum(manual)")
    ax1.set_xlim(0, 1); ax1.set_ylim(-0.1, 1)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="upper right")

    # right: residual miss rate after final check vs q
    for v, c in counts.items():
        ys = [residual_leak(c, q) for q in qs]
        ax2.plot(qs, ys, "-", color=colors.get(v),
                 label=f"{VARIANTS[v]} (detection miss {1-c['R']:.2f})")
    ax2.axhline(0.05, ls="--", color="#444", alpha=0.8)
    ax2.text(0.02, 0.057, "target miss rate <= 0.05 (spec 9)", fontsize=8, color="#444")
    ax2.set_title("Residual miss rate vs final-check catch rate q")
    ax2.set_xlabel("q (fraction of misses caught at final check, assumed)")
    ax2.set_ylabel("residual miss rate = (1 - R)*(1 - q)")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 0.45)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("Workflow simulation: detection -> workload reduction / residual miss "
                 "sensitivity (#14, model #13)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "workflow_reduction.png", dpi=130)
    plt.close(fig)
    print(f"  figure -> {FIGDIR / 'workflow_reduction.png'}")


def main() -> int:
    counts = load_counts()
    run_id = uuid.uuid4().hex[:8]
    print(f"業務適用シミュレーション (run_id={run_id}, split={SPLIT}, 直接識別子・IoU>=0.5)\n")
    print("係数（仮・秒）: " + ", ".join(f"{k}={v}" for k, v in COEF.items()) + "\n")

    for v, c in counts.items():
        # FN の裏取り（COVER 由来 FN と DOC_LEAK total_miss の整合）
        warn = "" if c["FN"] == c["leak_total_miss"] else f"  ⚠FN={c['FN']}≠leak{c['leak_total_miss']}"
        print(f"[{VARIANTS[v]:2} {v:11}] R={c['R']:.3f} P={c['P']:.3f} "
              f"TP={c['TP']} FP={c['FP']} FN={c['FN']} G={c['G']} n_docs={c['n_docs']}{warn}")
    print()

    print("=== 作業削減率 vs ρ ===")
    for v, c in counts.items():
        for rho in RHO_ANCHORS:
            red = reduction(c, rho)
            emit(run_id, v, "REDUCTION", recall=red,
                 note=f"作業削減率={red:.4f} @rho={rho}, TP={c['TP']},FP={c['FP']},FN={c['FN']},"
                      f"G={c['G']},n_docs={c['n_docs']}, 直接識別子, split={SPLIT}")
        cells = "  ".join(f"ρ{r}:{reduction(c, r):+.3f}" for r in RHO_ANCHORS)
        # 削減率が 50% を満たす ρ の上限（ρ をこれ以下にできれば目標達成）
        rho_break = next((r / 100 for r in range(100, -1, -1)
                          if reduction(c, r / 100) >= 0.50), None)
        bt = f"break-even ρ≤{rho_break:.2f}" if rho_break is not None else "50%未達(全 ρ)"
        print(f"  [{VARIANTS[v]}] {cells}   ({bt})")
    print()

    print("=== 見逃し率（2 定義） ===")
    for v, c in counts.items():
        det = 1 - c["R"]
        emit(run_id, v, "MISS_DETECTION", miss_rate=det,
             note=f"検出見逃し率 FN/G = 1−R = {det:.4f}, 直接識別子, split={SPLIT}")
        for q in Q_ANCHORS:
            res = residual_leak(c, q)
            emit(run_id, v, "MISS_RESIDUAL", miss_rate=res,
                 note=f"最終チェック後残存見逃し率=(1−R)(1−q)={res:.4f} @q={q}（業務KPI spec§9≤0.05）, "
                      f"split={SPLIT}")
        cells = "  ".join(f"q{q}:{residual_leak(c, q):.3f}" for q in Q_ANCHORS)
        # 残存 5% を満たす最小 q
        q_need = next((qq / 100 for qq in range(0, 101) if residual_leak(c, qq / 100) <= 0.05), None)
        qt = f"5%達成に q≥{q_need:.2f}" if q_need is not None else "q=1 でも未達"
        print(f"  [{VARIANTS[v]}] 検出見逃し={det:.3f} | 残存 {cells}   ({qt})")
    print()

    _plot(counts, run_id)
    print(f"\nWF 行を追記しました -> {LEDGER_PATH} (run_id={run_id})")
    print("※ 係数は仮値。校正（社内実測パイロット）は #13 の論点として未確定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
