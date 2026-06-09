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
- PR/Issue: #46
