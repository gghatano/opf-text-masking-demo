# 手法詳細: OpenAI Privacy Filter (OPF)

> 📑 凡例: `[n]`=一次情報の事実（出典）／🔎=実測・解釈／`📘`=ドキュメント由来。サマリは [評価レポート](REPORT.md) を参照。

## 概要
📘 双方向トークン分類＋Viterbi スパン復元の小型モデル（1.5B総 / 50M アクティブ MoE・128k ctx）[\[1\]](#ref1)。テキスト中の PII を検出・マスクする。生成モデルではなく**分類器**。検出は固定 **8 カテゴリ**: `private_person / private_address / private_email / private_phone / private_url / private_date / account_number / secret` [\[1\]](#ref1)。Apache-2.0、オンプレ実行可。

## CLI / 学習・評価仕様（ソース確認）
詳細は [`docs/findings-opf-cli.md`](docs/findings-opf-cli.md)。要点:

- **推論**: `opf [--device cpu] "テキスト"` / ファイル `opf -f file` / パイプ `... | opf`
- **学習**: `opf train train.jsonl --label-space-json labels.json --checkpoint <base> --output-dir <ft>`
  - **LoRA 非対応＝フルFT のみ**（AdamW）[\[2\]](#ref2)。既定: `--epochs 1 --batch-size 4 --learning-rate 1e-5 --weight-decay 0.01 --validation-split 0.1`
  - `--label-space-json` の `span_class_names`（先頭 `O`）で既定8カテゴリを**置換** → 仕様18ラベルを直接学習可
- **評価**: `opf eval data.jsonl --metrics-out m.json --eval-mode {typed,untyped}`
  - メトリクスJSON: `detection.span.{precision,recall,f1}`, `by_class.<label>.span.*`, `token_accuracy`, untyped 用 `ground_truth_label_recall.recall.<label>`
  - スパン一致は**包含(containment)判定**（厳密一致でない）→ 比較は同一定義に統一
  - ⚠️ `--skip-non-ascii-examples` は**日本語で使用禁止**（日本語例が全除外）[\[2\]](#ref2)
- **学習データ形式**: `{"text":..., "label":[{"category","start","end"}]}`（**文字オフセット**）

## 環境・落とし穴
- Python 3.14 では torch ホイール無し → `uv venv --python 3.12`（`scripts/setup_env.sh`）🔎
- CPU 機は **`--device cpu` 必須**（CPU版torchはCUDA無効で AssertionError）🔎

## 定性スモーク（参考・正式B0ではない）
CPU 実行（`opf --device cpu`）の手動確認:

| 入力（要約） | 検出 | 見逃し |
|---|---|---|
| 佐藤花子(72歳)・東京都千代田区・電話・受付番号A-0012 | 氏名・住所・電話 | 72歳・受付番号 |
| 山田太郎・2025年1月3日・○○病院・鈴木医師 | 山田太郎・鈴木(氏名部) | 和暦日付・病院名 |
| 田中一郎, tanaka@example.co.jp, 03-1234-5678 | 氏名・メール・電話 | — |

![OPF qualitative smoke coverage](figures/qualitative_smoke.png)

🔎 基本PII（氏名/住所/電話/メール/西暦日付）は日本語でも有効。**業務ID・施設名・年齢・和暦日付は素では見逃し** → B1（後処理）/B2（追加学習）の改善対象。

## 出典
- <a id="ref1"></a>[1] OpenAI Privacy Filter: https://github.com/openai/privacy-filter
- <a id="ref2"></a>[2] CLI/学習・評価 ソース確認メモ: [`docs/findings-opf-cli.md`](docs/findings-opf-cli.md)
