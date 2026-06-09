# Overnight run log

## 2026-06-09T17:30 cycle 0
- now: 開始手順(§6)実行。main クリーン・up-to-date を確認。issue list 取得。
- done: リポジトリ状態把握（scripts/04,05, labelmap, ledger を読み、データ形式確認）。gold=eval_300.jsonl(test=225/dev=75), 予測キャッシュ=opf_raw_test.jsonl。
- decided: T1(#46 解析実験) から着手。再推論不要・最も安全。独立ブランチ feat/t1-practical-metrics で作業。
- PR/Issue: #46

## 2026-06-09T17:55 cycle 1
- now: T1 (#46 解析実験) 完了。scripts/07_practical_metrics.py を実装し再推論なしでキャッシュを再集計。
- done: T1a 文書レベル漏えい率(B0 0.782/B1 0.718), T1b IoU スイープ+figures/iou_sweep.png(token vs span 乖離), T1c 準識別子被覆(QUASI R=0.000), T1d 過剰マスキング(B0 11.9%/B1 6.9%)。台帳に T1 行追記・REPORT §3.7 追加・htmls 再ビルド。
- 裏取り: COVER_* 行が既存 B0/B1 台帳値と完全一致(DIRECT 0.6016/0.6598, ALL 0.3833/0.4617)。SWEEP@0.5 も一致。
- decided: PR 作成 → 検証済みのため squash マージ候補。次は T2 (clue ablation) へ。
- PR/Issue: #46 / PR #49 (merged 50eaf78)

## 2026-06-09T18:25 cycle 2
- now: T2 clue ablation (#46) 完了。scripts/08_clue_ablation.py 実装。
- done: OPF スループット計測(~11s load + 0.26s/文)→3条件×102文を ~83s で完走(timebox 内)。前置 R0.824 / 後置 R0.735 / 無し R0.716。台帳に ABL 行・REPORT §3.8・htmls 再ビルド。data/ablation は git-ignored。
- 裏取り: キャッシュ予測を目視確認、pred span が gold 値に一致。DATE が全条件で低い(B0 DATE R0.29 と整合)。3条件は同一値・同一シードで枠のみ変化。
- decided: 効果量(前置+0.11)は Model Card(+0.28)より小=短文・少数・天井効果のため、と正直に記載。PR 作成→マージ。次は T3 (B2 学習スモーク)。
- PR/Issue: #46
