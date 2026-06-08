# opf-text-masking-demo

OpenAI Privacy Filter (`opf`) の**日本語自由記述データへの適用性**を、段階的に数値で検証する実証実験。
医療・自治体・その他ドメインの PII 検出性能と、匿名加工業務の工数削減効果を評価する。

- 仕様: [`docs/spec.txt`](docs/spec.txt)（実証実験計画書）
- 実行計画: [`docs/verification-plan.md`](docs/verification-plan.md)（B0→B1→B2 と数値が順次伸びる段階検証）
- OPF CLI/学習仕様メモ: [`docs/findings-opf-cli.md`](docs/findings-opf-cli.md)
- 進捗: GitHub Issues（epic: #16）

## セットアップ

```bash
bash scripts/setup_env.sh      # 3.11/3.12 venv 作成 + OPF を editable 導入 + スモーク
source .venv/Scripts/activate  # Windows（macOS/Linux は .venv/bin/activate）
```

> ⚠️ **Python 3.14 では torch ホイールが未提供**のことが多い。`setup_env.sh` は 3.11/3.12 を自動選択する。見つからない場合は 3.12 を導入して再実行（`PYTHON=...` で明示指定も可）。詳細は `docs/findings-opf-cli.md`。

## 評価の流れ（数値が順次わかる）

```bash
# 評価を実行 → outputs/metrics_ledger.csv に追記（履歴は上書きしない）
python scripts/02_evaluate.py data/eval/medical.jsonl --stage B0 --model A-OPF --domain 医療
# 改善曲線・リーダーボード図を再生成
python scripts/plot_progression.py
```

全実行は `outputs/metrics_ledger.csv` に 1 行ずつ追記され、`figures/score_progression.png`
（B0→B1→B2）に成功基準ライン（Recall .90 / Precision .85 / F1 .85, spec §9）を重ねて表示する。

## 構成

```
docs/        仕様・計画・CLIメモ・BACKLOG
scripts/     setup_env.sh / 02_evaluate.py / ledger.py / plot_progression.py ...
outputs/     metrics_ledger.csv（数値の履歴）
figures/     score_progression / model_leaderboard / per_label_f1
data/        合成評価・学習データ（.gitignore, 再生成可能）
third_party/ OPF upstream（setup が clone, .gitignore）
```

## ライセンス / 注意
評価データは**合成**を既定とし、実個人情報は扱わない。OPF は匿名化の保証ではなくデータ最小化の補助であり、人手レビュー前提で運用する（spec §10,§11）。
