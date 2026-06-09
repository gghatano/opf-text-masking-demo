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
- PR/Issue: #46 / PR #50 (merged e5e294c)

## 2026-06-09T19:05 cycle 3
- now: T3 B2 学習スモーク。
- blocked: **BLOCKED**。2 つの blocker。①labels_ja10.json(span_class_names のみ)が効かず head が 41 ラベルで再構築(exact=1/fallback=40)=#12 の宿題の核心。②CPU 学習が非現実的: 既定スレッドで segfault(exit139)、OMP_NUM_THREADS=1+batch2 でも ~19GB 占有・4 micro-batch 中 1 を 13 分でも未完(848s で打ち切り・プロセス kill)。本学習は budget(4h)を桁違いに超過。
- decided: 部分学習物で誤った B2 を出さない(§0)。GPU 使用可否・ラベル空間設計は人間判断(§5)→ #12 にコメントで記録。スモーク一時物は削除。T3 はコード成果物なし=PR なし。次は T4 (#45 PII-Masking-300k, ベストエフォート)。
- PR/Issue: #12 (escalation comment)

## MORNING REPORT (2026-06-09 夜間バッチ)

### 進んだこと（マージ済・main に反映）
- **T1 実用指標 (#46, PR #49 merged 50eaf78)**: 再推論なしで stock-OPF キャッシュを業務寄り 4 視点で再集計する `scripts/07_practical_metrics.py` を追加。
  - (a) 文書レベル漏えい率(直接識別子): B0 **0.782** / B1 **0.718**(≥1件見逃した文書割合)。平均見逃し 1.25→1.07 件/文書。
  - (b) IoU スイープ: char-token R 0.855 に対し span(IoU0.5) R 0.602(直接) → token は実務有効性を ~0.25pt 過大評価。図 `figures/iou_sweep.png`。
  - (c) 準識別子を分母に戻すと untyped R=**0.000**(設計上の素通し)。
  - (d) 過剰マスキング: B0 11.9% / B1 6.9%(住所・人名が主因)。
  - 台帳に T1 行・REPORT §3.7・htmls 再ビルド。COVER 行が既存 B0/B1 値と完全一致(裏取り済)。
- **T2 手がかりアブレーション (#46, PR #50 merged e5e294c)**: `scripts/08_clue_ablation.py`。同一値・同一シードで前置/後置/無しの3条件。前置 R0.824 / 後置 R0.735 / 無し R0.716。Model Card Table5 を日本語で定性再現。台帳 ABL 行・REPORT §3.8。

### BLOCKED（人間判断が必要・番号付き）
1. **B2 学習に GPU を使うか (#12)**: CPU では本学習が非現実的。既定スレッドで segfault、`OMP_NUM_THREADS=1`+batch2 でも ~19GB 占有・4 micro-batch 中 1 を 13 分でも未完。本学習は budget(4h)を桁違いに超過。→ GPU 環境の用意可否を要判断。
2. **B2 のラベル空間設計 (#12)**: `configs/labels_ja10.json`(span_class_names のみ)が効かず、head が **41 ラベル**で再構築(exact=1/fallback=40)。10/11 ラベル直接学習の意図が反映されていない。`ner_class_names` 追加 or OPF の head 構成確認が必要。
3. **PII-Masking-300k 日本語直接評価の方針 (#45)**: ai4privacy の 300k/200k/400k/500k いずれも **日本語 split が無い**(en/fr/de/it/es/nl ±hi/te)。ライセンスは other。→ #45 の前提が崩れた。非日本語 split で apples-to-apples 比較するか/別の日本語 PII データを探すか/§3.6 を最終とするか、を要判断。
4. **主要 KPI の確定 (#46)**: T1 で材料は揃った。文書レベル漏えい率と span 一致を主・token を参考とする提案の採否は議論対象のまま(既存数値は消さず併記済)。

### 推奨する次アクション
- #12: GPU 環境が用意できるなら、まずラベル空間(ner_class_names)の修正→極小スモークで 41→意図ラベルになるか確認→本学習。CPU 継続は非推奨。
- #45: 短期的には english split で OPF を評価し Model Card 英語値と並置するのが低コストで有意義(日本語ギャップは §3.6 proxy で説明)。
- #46: T1/T2 の結果をもとに主要 KPI を確定すれば #14(業務適用シミュレーション)に進める。

### 安全性
- main は常にクリーン、全変更はブランチ+PR 経由で検証後に squash マージ。破壊的操作なし。スモークの一時生成物(models/ft_smoke, 一時ログ)は削除済み。外部データ本体の取得はしていない(HF メタデータのみ)。
