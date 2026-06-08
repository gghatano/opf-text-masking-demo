# OPF CLI / 学習・評価 仕様メモ（ソース確認）

upstream `openai/privacy-filter` の**ソース直読**で確定した事実。Issue #1（環境構築）・#2（評価基盤）・#11（LoRA方式）の根拠。
確認元: `opf/_train/args.py`, `opf/_eval/args.py`, `opf/_eval/metrics.py`, `pyproject.toml`, `README.md`（いずれも main, 2026-06-08 取得）。

## 1. 動作要件・依存（実機検証済み 2026-06-08）
- `requires-python = ">=3.10"`。依存は `numpy / torch / tiktoken`（`pyproject.toml`）。
- ⚠️ **落とし穴**: 本環境の既定は Python 3.14。torch 3.14 向けホイールが無く失敗する。→ **`uv` で 3.12 venv を作る**（`setup_env.sh` が `uv venv --python 3.12`。uv が 3.12 を自動取得）。導入結果は torch==2.12.0 / tiktoken==0.13.0（`requirements.txt` に固定）。
- ⚠️ **CUDA 落とし穴**: CPU 版 torch ホイールは CUDA 無効。OPF は既定で CUDA を使おうとし `AssertionError: Torch not compiled with CUDA enabled` になる。→ **`--device cpu` 必須**（CPU 機の場合）。`02_evaluate.py` も既定 cpu。
- スモーク（CPU）確認済み: `My name is <PRIVATE_PERSON>, email <PRIVATE_EMAIL>, born <PRIVATE_DATE>.`

### 日本語 素モデル定性スモーク（参考・正式B0ではない）
- ✓ 検出: 氏名(佐藤花子/山田太郎)・住所(東京都千代田区)・電話・email・"鈴木医師"の氏名部
- ✗ 見逃し: **年齢(72歳)・業務ID(受付番号A-0012)・施設名(○○病院)・日本語日付(2025年1月3日)**
- → 仮説「基本PIIは拾うが業務ID/組織/準識別子/和暦日付は素では落ちる」と整合。B1/B2 の伸びしろ。
- ⚠️ Windows コンソールは日本語出力が文字化けするが処理自体は正常（評価はファイル/JSON経由で行うため影響なし）。

## 2. `opf train`（フルFT・本体既定値）
- **LoRA / PEFT /量子化学習のフラグは存在しない**。最適化は AdamW による**フルパラメータ FT のみ**。→ Issue #11 は「LoRA 非対応 → full FT」で確定。
- 既定ハイパラ（`opf/_train/args.py`、デモ値ではなく本体既定）:
  | 引数 | 既定 |
  |---|---|
  | `--epochs` | 1 |
  | `--batch-size` | 4 |
  | `--grad-accum-steps` | 1 |
  | `--learning-rate` | 1e-5 (AdamW) |
  | `--weight-decay` | 0.01 |
  | `--max-grad-norm` | 1.0 |
  | `--validation-split` | 0.1（`--validation-dataset` 省略時）|
  | `--shuffle-seed` | 0 |
  | `--output-param-dtype` | inherit / bf16 / fp32 |
  | `--dataset-variant` | full / message |
- 独自ラベル: `--label-space-json`（`span_class_names` 推奨、先頭に `O`）。既定8カテゴリを**置換**。
- 出力: `--output-dir` に `config.json` / `model.safetensors` / `finetune_summary.json` / `USAGE.txt`。

## 3. `opf eval`（指標）
- `--eval-mode {typed,untyped}`（既定 typed）。`--span-metrics-space {char,token}`（既定 char）。
- `--metrics-out <json>` で機械可読メトリクス、`--predictions-out` で予測スパン、`--per-class` / `--label-counts` / `--timings-out`。
- ⚠️ `--skip-non-ascii-examples` が存在 → **日本語評価では絶対に付けない**（日本語例が全除外される）。
- **スパン一致は包含(containment)判定**（`opf/_eval/metrics.py: _match_spans_containment`）。
  - precision 用: 予測スパンが gold に内包されるか（pred_in_gold）
  - recall 用: gold が予測に内包されるか（gold_in_pred）
  - → 厳密一致ではない。#2 の一致基準は他モデル比較でもこの定義に揃える。

## 4. メトリクス JSON のキー（`metrics.py` で確定）— 台帳パーサの仕様
- 全体: `n_examples`, `n_tokens`, `loss`, `token_accuracy`
- 検出(トークン級): `detection.{precision,recall,f1,f2}`
- 検出(スパン級): `detection.span.{precision,recall,f1,f2}`  ← **主指標はここ**
- ラベル別(トークン級): `by_class.<label>.{precision,recall,f1}`
- ラベル別(スパン級): `by_class.<label>.span.{precision,recall,f1,f2}`  ← **per_label_f1 図はここ**
- untyped 用: `ground_truth_label_recall.recall.<label>`（文字被覆ベース。体系が違う比較で使用）

## 5. 台帳への写像（#2 実装規約）
- `precision = detection.span.precision`, `recall = detection.span.recall`, `f1 = detection.span.f1`
- `miss_rate = 1 - recall`（漏れ率, spec §6）, `false_pos_rate = 1 - precision`（誤検出率）
- ラベル別は `by_class.<label>.span.*` を `label` 列付きで別行に追記
- `token_accuracy` は note へ
