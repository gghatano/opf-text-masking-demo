# OPF CLI / 学習・評価 仕様メモ（ソース確認）

upstream `openai/privacy-filter` の**ソース直読**で確定した事実。Issue #1（環境構築）・#2（評価基盤）・#11（LoRA方式）の根拠。
確認元: `opf/_train/args.py`, `opf/_eval/args.py`, `opf/_eval/metrics.py`, `pyproject.toml`, `README.md`（いずれも main, 2026-06-08 取得）。

## 1. 動作要件・依存
- `requires-python = ">=3.10"`。依存は `numpy / torch / tiktoken`（`pyproject.toml`）。
- ⚠️ **落とし穴**: 本環境は Python 3.14。torch は 3.14 向けホイール未提供の可能性が高く、`pip install -e .` が失敗しうる。→ **3.11/3.12 の venv を作って導入**する（`setup_env.sh` がそれを行う）。
- GPU 既定 / `--device cpu` で CPU 実行可（README）。

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
