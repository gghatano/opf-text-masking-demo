# OPF 日本語適用性 実証実験 — 検証タスク計画（実行計画）

正式仕様: [`docs/spec.txt`](./spec.txt)（実証実験計画書）。本書はその**実行手順・段階分割・数値ゲート**への落とし込み。
対象: OpenAI Privacy Filter (`opf`) [\[1\]](#ref1) / 比較: Presidio [\[4\]](#ref4)・GLiNER・日本語NER
作成日: 2026-06-08 / リポジトリ: gghatano/opf-text-masking-demo

> ⚠️ **スコープ宣言**
> - 対象: **日本語**自由記述（医療 / 自治体 / その他、計 300 件 [spec §3]）。
> - 主目的: PII 検出性能（漏れ最小化）＋ **匿名加工業務の工数削減効果**の定量評価 [spec §2,§7-3,§12]。単なる NER 評価にしない。
> - 評価データは**合成（synthetic）を既定**とする（後述・重要）。実データ（実診療録等）は個人情報そのものであり、評価用ラベル付けも秘匿化作業も伴うため、本実証では使わない／使う場合は別途承認・管理下で。
> - 実行範囲: ローカル推論（CPU/GPU 単体）。学習は GPU。本番運用は対象外。
>
> 📑 **凡例**: `[n]`=一次情報の事実（付録 出典）。`[spec §x]`=仕様書の記載。🔎=筆者の実測・解釈・推定。

---

## 0. ゴールと「数値が順次わかる」計装

成果物の核は、**段階（Stage）が進むほど主要スコアが伸び、その軌跡が 1 枚の図と 1 つの台帳で追える**こと。

- 全評価実行を `outputs/metrics_ledger.csv` に **1 行追記**（上書き禁止＝履歴保持）。
  列: `run_id, date, stage, model(A/B/C/D), variant, domain(医療/自治体/その他/全体), label, precision, recall, f1, miss_rate, false_pos_rate, latency_ms, seed, note`
- 図を 3 枚、各 Stage 完了時に再生成：
  1. `figures/score_progression.png` — Stage 軸（B0→B1→B2）の全体 Recall/Precision の改善曲線。**これが主役。**（F1は不使用 #24）
  2. `figures/model_leaderboard.png` — モデル A/B/C/D の横比較バー（同一データ・同一指標）。
  3. `figures/per_label_pr.png` — 主要ラベル別 Recall/Precision（PERSON/ADDRESS/ID/ORGANIZATION/AGE）[spec §6]。
- 成功基準ライン（Recall90%, Precision85%, 見逃し5%, 削減50% [spec §9]）を図に水平線で描き、**達成までの距離**を常に可視化。

---

## 1. ラベル体系と OPF 素モデルの対応（伸びしろの所在）

仕様の **10 ラベル**（2026-06-08 改訂 [spec §5]）と OPF 素の 8 カテゴリ [\[1\]](#ref1) の対応。**素モデルで構造的に出せないラベルが「改善の伸びしろ」**＝ Stage が進むと recall が跳ねる箇所。

| 群 | 仕様ラベル | OPF 素カテゴリ対応 | 素モデルでの見込み 🔎 |
|---|---|---|---|
| 直接識別子 | PERSON / ADDRESS / PHONE / EMAIL / DATE | private_person / private_address / private_phone / private_email / private_date | 比較的拾える（英語中心ゆえ日本語表記で取りこぼし懸念） |
| 直接識別子 | ID（患者/被保険者/受付/会員番号を統合） | account_number に弱く写像 / 型不一致が多い | 低い。後処理(正規表現)・追加学習で回収 |
| 準識別子 | AGE / REGION / OCCUPATION / ORGANIZATION（会社/学校/施設を統合） | 該当カテゴリ無し | ほぼ 0。本質的に難。別扱い・honest に限界を記載 |

> 🔎 OPF 素モデルは固定 8 カテゴリのトークン分類器 [\[1\]](#ref1)。ID・準識別子は素では出せないため、**B0 の全体 recall は低く出る前提**。これを正規表現フォールバック(B1)→ 10 ラベルでの追加学習(B2) [spec §7-2] で押し上げる設計が、まさに「数値が順次伸びる」物語になる。
> 🔎 準識別子（AGE/OCCUPATION/REGION/ORGANIZATION）は誤検出と表裏で難度が高い。成功基準 Recall90% は主に直接識別子（基本PII＋ID）で狙い、準識別子は限界を正直に記載する方針を推奨。
> 📌 **重要（追加学習でラベル体系を置換できる）**: `opf train --label-space-json` に `{"category_version": "...", "span_class_names": ["O", "PERSON", "ID", "AGE", ...]}` を渡すと、既定8カテゴリを**置換**して仕様の10ラベルそのもので学習できる [\[5\]](#ref5)。よって B2 ではマッピングに頼らず10ラベル直接学習が可能。一方 B0/B1（素モデル）は8カテゴリしか出ないため、B0 比較は **untyped 評価（スパン検出のみ）＋ ラベル写像での typed 部分評価** を併用する。

---

## 2. 評価指標（仕様 §6・成功基準 §9 と接続）

| 指標 | 定義 | 対応 |
|---|---|---|
| Precision / Recall（エンティティ単位, **F1不使用** #24） | スパン一致(IoU≥0.5) | spec §6 主指標。成功基準 R≥90/P≥85 |
| ラベル別 Recall / Precision | PERSON/ADDRESS/ID/ORGANIZATION/AGE 等 | spec §6。主要ラベル R≥90/P≥85 |
| 漏れ率(miss_rate) | 未検出 PII 件数 / 全 PII | spec §6。見逃し率≤5% |
| 誤検出率(false_pos_rate) | 非PIIをPII判定 / 判定総数 | spec §6。閾値調整で管理 |
| 作業削減率 | (人手単独工数 − 支援併用工数)/人手単独工数 | spec §7-3,§9。≥50% |
| latency_ms | 1 文書あたり推論時間 | spec §8 |

> OPF の `opf eval` は typed/untyped・token_accuracy・ground_truth_label_recall を出す [\[2\]](#ref2)。仕様ラベルと OPF カテゴリのズレは untyped / ground_truth_label_recall で補完評価する。

---

## 3. データセット計画（成果物1）

| 区分 | 件数 | 用途 | 方針 |
|---|---|---|---|
| 評価セット | 300（医療/自治体/その他 各100）[spec §3.2] | S1/S3 評価 | **合成生成**（テンプレート＋固有名詞置換／Claude Code 生成）。PII位置は生成時に自動ラベル。実データは使わない |
| 追加学習セット | 500〜1,000 [spec §7-2] | S2 full FT | 同上の合成＋アノテーション |
| 回帰セット | 評価から凍結 | 劣化検知 | Stage 5 で固定 |

> 🔎 **重要（privacy）**: 匿名化ツールの評価には「PII がどこにあるか」の正解ラベルが要る＝実記録を使うと秘匿対象データを直接扱うことになる。**合成データなら正解ラベルが生成時に得られ、個人情報リスクも無い**ため既定とする。合成の現実性が結果の外的妥当性を左右する点は限界として明記する。
> 📌 **生成方法（#26 決定）**: フォーマットを定めた文章の固有名詞をルールベース置換、または Claude Code（本エージェント）に合成させる。実データが無いため**疑似値の実在エンティティ衝突回避は今回不要**（実データ利用時は実在個人との衝突を避ける）。アノテーション規約は #29（敬称除外／完全住所=ADDRESS・地名単独=REGION／一致 IoU≥0.5）。

---

## 4. 段階計画（マイルストーン）

各 Stage は「作業 → **数値ゲート** → 期待する数値変化」。前段の数値を台帳に残してから次へ。

### Stage 0 — 環境構築・スモーク
- `git clone https://github.com/openai/privacy-filter && pip install -e .` でインストール [\[6\]](#ref6)、ウェイト自動取得、`opf "..."` 実行、同梱 `examples/data/sample_eval_*.jsonl` で `opf eval` 稼働。`setup_env.sh` 冪等化。CPU/最低構成での動作も確認 [spec §8]。
- **`opf train --help` / `opf eval --help` を確認**し、LoRA/省メモリ学習対応の有無・本体既定ハイパラを把握（FINETUNING.md には LoRA 記載なし [\[5\]](#ref5)）。
- ゲート: CLI が JSON スパンを返し、`opf eval --metrics-out` が `detection.span.f1` 等を含む JSON を出す。

### Stage 1 — 評価基盤・データ整備（成果物1）
- 合成 300 件生成＋自動ラベル、アノテーション一貫性(κ)確認、10ラベル↔OPF 8カテゴリ mapping 確定、`metrics_ledger.csv` と評価スクリプト整備。
- ゲート: 300 件に正解ラベルが付き、評価パイプラインが 1 行を台帳に書ける。

### Stage 2 — ベースライン **B0**（シナリオ1 [spec §7-1]）
- OPF 素モデル・既定設定で 300 件評価。ドメイン別・ラベル別に台帳記録。
- ゲート: 全体／ドメイン別／主要ラベル別の P/R・漏れ・誤検出が揃う。
- 期待: 🔎 全体 recall は低め（ID・準識別子(AGE/ORGANIZATION等)が素では出ない）。**これが基準点。**

### Stage 3 — 後処理・閾値チューニング **B1**
- 構造化PII（PHONE/EMAIL/各種ID）への正規表現フォールバック、日本語表記ゆれ対応、閾値調整で誤検出抑制 [spec §10]。
- ゲート: **全体 Recall・Precision が B0 を上回り**、誤検出率が許容内。
- 期待: 🔎 ID（番号類）の recall が正規表現で回収され全体が一段改善。

### Stage 4 — 多モデル比較（GiNZA から）
- **まず GiNZA（spaCy `ja_ginza`）** を最初の比較対象にする（`pip install` だけで日本語 NER がすぐ動く＝最短で1本の比較線が引ける）。次に Presidio、GLiNER、他の日本語NER(LUKE/DeBERTa) を順次追加 [spec §4]。
- GiNZA の NER ラベル体系（PERSON/LOC/ORG/DATE 等）↔ 仕様10ラベルの対応表を作成（GiNZA 公式で確認）。電話/メール/口座番号/secret は GiNZA 標準では拾えない＝守備範囲の差が比較軸。
- ゲート: 同一データ・同一スパン一致判定で A/B/C/D が並ぶ（スパン一致基準＝完全一致/部分重複を評価側で統一）。
- 期待: 🔎 「日本語固有表現は GiNZA 等が強い／構造化PIIは正規表現＋OPF が強い」役割分担が見える。

### Stage 5 — 日本語追加学習 **B2**（シナリオ2 [spec §7-2]・成果物3）
- 学習データを `opf` の JSONL スキーマ（`{"text":..., "label":[{"category","start","end"}]}`、**文字オフセット** [\[5\]](#ref5)）で 500〜1,000 件用意。`--label-space-json` に10ラベルの `span_class_names`（先頭 `O`）を定義。
- `opf train train.jsonl --validation-dataset val.jsonl --label-space-json labels.json --checkpoint <base> --output-dir <ft>` で学習 [\[5\]](#ref5)。LoRA 対応が確認できればそれを使い、無ければ full FT（モデルは小型なので現実的）。複数 seed で mean±std。回帰セット凍結。
- before/after は同一 test.jsonl に対し `opf eval --checkpoint <base|ft> --eval-mode typed/untyped --metrics-out *.json` を回し、`detection.span.f1`・`by_class.<label>.span.*` を台帳へ [\[5\]](#ref5)。
- ゲート: **対象ドメインで全体 Recall・Precision が B1 を上回る**。成功基準（R≥90/P≥85/主要ラベル R≥90・P≥85 [spec §9]）への到達度を測定。
- 期待: 🔎 ID・ORGANIZATION で大幅改善。準識別子(AGE/REGION/OCCUPATION)の限界も定量化。

### Stage 6 — 業務適用評価（シナリオ3 [spec §7-3]・成果物4）
- 「自由記述→Privacy Filter→候補抽出→作業者レビュー→確定」フローの試作で、**処理時間・修正件数・見逃し件数・作業工数**を測定し **作業削減率**を算出。人手単独を対照に。
- ゲート: 作業削減率≥50% かつ 見逃し率≤5% [spec §9] の達成度。
- 期待: 🔎 検出 Recall/Precision と工数削減の関係を定量化（本実証の主眼 [spec §12]）。

### Stage 7 — 総括・ガイドライン・公開（成果物2,5）
- `REPORT.md` 執筆（課題→仕組み→評価→落とし穴→まとめ）、適用ガイドライン（適用可/困難/要人手/推奨フロー [spec §11-5]）、`build_html.py`→`htmls/`→GitHub Pages 公開。スコア推移図を主要図に。
- ゲート: `https://gghatano.github.io/opf-text-masking-demo/` で Stage B0→B2 の改善と業務KPIが図で追える。

---

## 5. 成功判定ゲート（spec §9 を各 Stage に割付）

| 指標 | 目標 | 主に判定する Stage |
|---|---:|---|
| PII Recall | ≥90% | Stage 5 (B2) |
| PII Precision | ≥85% | Stage 3〜5 |
| 主要ラベル Recall / Precision | 90% / 85% | Stage 5 |
| 作業削減率 | ≥50% | Stage 6 |
| 見逃し率 | ≤5% | Stage 6 |

> 達成できない指標は「達成できなかった」と数値付きで記載する（物語化しない）。特に準識別子・日本語固有表現の限界は正直に。

---

## 6. リポジトリ構成（demo-report-builder 型）

```
opf-text-masking-demo/
├── README.md / REPORT.md / requirements.txt
├── scripts/ … setup_env.sh 00_prepare_data.py(合成生成) 01_generate.py(推論+後処理)
│             02_evaluate.py(指標+台帳追記) 03_compare_models.py 04_finetune_lora.py
│             05_workflow_sim.py(業務KPI) build_html.py
├── outputs/metrics_ledger.csv      … 全実行の追記履歴（数値推移の源泉）
├── figures/ … score_progression.png / model_leaderboard.png / per_label_f1.png
├── data/    … 合成評価・学習データ（git管理外）
├── htmls/ / docs/(spec.txt, 本書, BACKLOG.md)
└── .github/workflows/deploy-pages.yml
```

## 7. 確定事項・要確認
**確定（2026-06-08）**:
- ✅ **データ生成方針**: 合成データ（テンプレート＋LLM）で進める。実データは使わない。
- ✅ **作業削減率の測定**: 簡易シミュレーション（検出結果の修正件数・見逃し件数から工数を推定して算出）。作業者実測は行わない。

- ✅ **比較対象の優先順**: **GiNZA を最初**に着手（最短で動く）。以降 Presidio / GLiNER / LUKE / DeBERTa を順次追加。

**要確認（非ブロッキング・後続 Stage までに確定）**:
1. **LoRA 対応**: OPF 公式に LoRA 記載なし [\[5\]](#ref5)。`opf train --help` で確認し、無ければ full FT に切替（小型モデルゆえ現実的）。→ Stage 0/5。
2. **学習環境**: 追加学習用 GPU（RTX4090/A100/H100 [spec §8]）の確保状況。→ Stage 5 までに。

---

## 付録 出典
- <a id="ref1"></a>[1] OpenAI Privacy Filter（カテゴリ・制限・CLI）: https://github.com/openai/privacy-filter
- <a id="ref2"></a>[2] OPF 評価/出力モード `EVAL_AND_OUTPUT_MODES.md`: https://raw.githubusercontent.com/openai/privacy-filter/main/EVAL_AND_OUTPUT_MODES.md
- <a id="ref3"></a>[3] ai4privacy/pii-masking データセット（参考）: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
- <a id="ref4"></a>[4] microsoft/presidio / presidio-research: https://github.com/microsoft/presidio
- <a id="ref5"></a>[5] OPF 公式 FINETUNING.md（学習データ形式・`--label-space-json`・before/after 評価）: https://github.com/openai/privacy-filter/blob/main/FINETUNING.md
- <a id="ref6"></a>[6] 参考記事「OpenAI Privacy Filter を試す」(@softbase, Qiita): https://qiita.com/softbase/items/69587314c7d8be40441e
