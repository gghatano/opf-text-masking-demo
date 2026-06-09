# OPF 日本語適用性 実証実験 — 評価レポート（サマリ）

メタ: 対象 = OpenAI Privacy Filter (`opf`) ／ リポジトリ = gghatano/opf-text-masking-demo ／ 最終更新 = 2026-06-09

> 🚧 **作成途中（WIP）**: 測定済みの段階のみ数値を記入し、未実験箇所は `—` で空欄。現在 **B0・B1 完了**／**B2・300件本比較・業務適用シミュレーションは未実施**。

> 📑 **凡例** — `[n]`=一次情報の事実（付録C）／🔎=実測・解釈・推定／`📘`=ドキュメント由来。
> ⚠️ **本ページはサマリ**です。各手法の詳細・生データ・落とし穴は **手法詳細ページ**（上部タブ／付録B）に分離。実験が進むたびに §2 数値表・§3 比較と所見を更新し HTML を再ビルドします。

> ⚠️ **スコープ** — 対象は**日本語**自由記述（医療/自治体/その他, 計300件・合成）。主目的は PII 検出性能（漏れ最小化）＋匿名加工**業務の工数削減効果**の定量評価。実データは扱わない。詳細計画: [`docs/verification-plan.md`](docs/verification-plan.md)。

---

## 0. エグゼクティブサマリ

- **現在地: B0（素OPF）測定完了**。合成評価300件（#5、医療/自治体/その他 各100、dev75/test225 に層化 #20）を用い、**test 225件**で素 OPF を実測（§2）。
- 🔎 **B0→B1（直接識別子・untyped, IoU≥0.5, test225件）**: **R 0.57→0.70 / P 0.77→0.86**。B1 後処理は OPF 出力の整形のみで、**DATE 0.29→0.99**（和暦/略記の正規表現）・PERSON 境界整形（P 0.65→0.85）が効いた。**強い**: PHONE 1.0・ID 0.89・EMAIL 0.86。**残る課題**: PERSON Recall 0.49（OPF が2人目以降を非検出＝素モデルの上限）・ADDRESS P 0.48。準識別子(AGE/REGION/OCC/ORG)は**OPF設計上対象外で R=0**→ B2 で追加学習。詳細 → [OPF ページ](methods/opf.md)。
- 🔎 **モデル比較（パイロット実測 §3）**: 同一gold・同一マッチャ(IoU≥0.5)で OPF vs GiNZA を計測。**相補性が実数で確認**——OPFは PHONE/EMAIL(R=1.0)・ID等の**構造化PII**に強く、GiNZAは PERSON(0.94)・AGE/REGION/DATE(1.0)等の**人名・準識別子**に強い。OPFの境界過延長(PERSON R=0.41)は B1 の改善対象。用途で使い分け（アンサンブルは対象外 #18）。
- 📘 **確定事実**: OPF は **LoRA 非対応＝フルFT のみ**。10ラベルは `--label-space-json` で直接学習可。指標は `detection.span.*`/`by_class.<label>.span.*` を台帳へ写像 [\[2\]](#ref2)。

---

## 1. 検証パイプライン

```mermaid
flowchart LR
  D[合成データ + 正解ラベル] --> E["各モデルで推論・評価"]
  E --> P[02_evaluate.py:<br/>指標→台帳へ追記]
  P --> L[(metrics_ledger.csv)]
  L --> F[plot_progression.py:<br/>改善曲線/leaderboard]
  F --> R[本サマリ §2/§3]
```

段階: **B0**(素OPF) → **B1**(後処理・正規表現) → **B2**(日本語フルFT)。並行で多モデル比較（GiNZA→Presidio→GLiNER→日本語NER）。

---

## 2. 結果サマリ（数値）

> 成功基準 [spec §9]: Recall≥90% / Precision≥85% / 主要ラベル R≥90%・P≥85% / 作業削減率≥50% / 見逃し率≤5%。**F1（総合指標）は用いない**（見逃しと過剰検出はコスト非対称 #24）。下表は**直接識別子(PERSON/ADDRESS/PHONE/EMAIL/DATE/ID)の untyped 検出**（spec §9 の主対象, IoU≥0.5）。**test 225件**で計測（dev は B1 調整用に保持 #20）。

| Stage | 説明 | Recall | Precision | 漏れ率 | 誤検出率 |
|---|---|---:|---:|---:|---:|
| B0 | 素モデル | 0.57 | 0.77 | 0.43 | 0.23 |
| B1 | 後処理・正規表現 | **0.70** | **0.86** | 0.30 | 0.14 |
| B2 | 日本語追加学習 | — | — | — | — |

ラベル別 typed Recall（B0→B1）: **DATE 0.29→0.99**（和暦/略記の正規表現で回収, P=0.97）・**PERSON 0.38→0.49**（境界整形で P=0.65→0.85 も改善）。PHONE 1.00 / ID 0.89 / EMAIL 0.86 / ADDRESS 0.81(P=0.48) は据え置き。AGE・REGION・OCC・ORG は **0**（OPF対象外 → B2 で追加学習）。

🔎 **B1 の設計と限界**: 後処理は **OPF 自身の出力 category の整形に限定**（PERSON 境界の汎用整形＋DATE の JP 書式補完）。AGE/REGION 等の正規表現は加えていない（OPF が扱わない category であり、追加は実質アンサンブル #18／準識別子拡張は B2 の役割）。PERSON 残差は OPF が2人目以降を**そもそも非検出**な分で、後処理では作れない＝素モデルの JP 人名再現率の上限。**合成データ上の上限性能**である点に留意（#20）。

![score progression](figures/score_progression.png)
**図0**: B0→B1 の Recall/Precision 進捗（untyped 全ラベル, test 225件）。

---

## 3. モデル比較（パイロット実測）

**公平性プロトコル**（#23）: 同一 gold（合成 **15文書 / 57スパン**）・**同一マッチャ char IoU≥0.5**・各モデルの PII 該当出力に限定・**F1不使用で Recall/Precision**（#24）。untyped（検出）を主、typed（ラベル一致）を per-label。実装 `scripts/03_compare_models.py`、データ `scripts/01_make_eval.py`。

> ⚠️ **パイロット**（n=15, 準識別子多めの構成）。配管検証＋傾向把握が目的で、本評価は300件（#5）で再測定する。

### 3.1 全体（untyped 検出, IoU≥0.5）

| モデル | Recall | Precision |
|---|---:|---:|
| OPF（素） | 0.42 | 0.83 |
| GiNZA（`ja_ginza`） | 0.72 | 0.95 |

![untyped detection PR](figures/model_compare_pr.png)
**図1**: 全体の untyped 検出 P/R（このパイロットは準識別子が多く GiNZA 有利な構成）。

### 3.2 ラベル別 Recall（typed）

![per-label recall](figures/per_label_compare.png)
**図2**: ラベル別 Recall。相補性が実数で表れる。

🔎 **結論（実数で相補性を確認）**:
- **OPF が強い**: PHONE・EMAIL（R=1.0）、ID（0.67）＝**構造化PII**。
- **GiNZA が強い**: PERSON（0.94）、AGE・REGION・DATE（1.0）、OCCUPATION・ORGANIZATION（0.8）＝**人名・準識別子・和暦日付**。
- **共通の課題**: ADDRESS（OPFは**境界過延長**で R=0.67、GiNZAは住所を地名に分割し 0）。OPFの PERSON が R=0.41 と低いのも「佐藤花子（72歳）は」を1スパン化する**境界過延長**が IoU で落ちるため → **B1の境界後処理**の主対象。
→ 構造化PIIは **OPF＋正規表現**、人名・準識別子は **GiNZA等**、で使い分け。OPFを**日本語追加学習(B2)**で準識別子へ広げるのが伸びしろ。各手法の詳細は手法ページへ。

---

## 4. 制約（要点）

- ⚠️ Python 3.14 で torch 無し → `uv` で 3.12。CPU 機は `--device cpu` 必須。
- ⚠️ GiNZA 5.2 は新 spaCy と非互換 → `split_mode` 明示で回避（[GiNZA ページ](methods/ginza.md)）。
- 🔎 準識別子（年齢/地域/職業）は誤検出と表裏で難度高。成功基準 Recall90% の対象範囲は #3 で確定。
- 詳細な落とし穴は各手法ページ。

## 5. 次のマイルストーン
✅ 合成評価300件（#5）→ ✅ **B0**（#7）→ ✅ **B1**（#8, §2反映済）→ ⏭ **B2**（#12 日本語フルFT, 学習データ #6 が前提）／ ⏭ **GiNZA等の300件本比較**（#9/#10, 現状 §3 は n=15 パイロット）／ ⏭ **業務適用シミュレーション**（#14）→ 📄 適用ガイドライン・Pages公開（#15）。

---

## 付録A 再現手順
```bash
git clone https://github.com/gghatano/opf-text-masking-demo.git && cd opf-text-masking-demo
bash scripts/setup_env.sh && source .venv/Scripts/activate
python scripts/02_evaluate.py data/eval/medical.jsonl --stage B0 --model A-OPF --domain 医療
python scripts/plot_progression.py && python scripts/build_html.py
```

## 付録B レポート運用ルール（型）
- **サマリ＝本 `REPORT.md`（index）**。エグゼクティブサマリ・数値表・比較図・要点のみ。
- **詳細＝手法ごとに `methods/<name>.md`**（別ページ・タブ）。モデル仕様・生データ・型対応・落とし穴・定性所見はここに書く。
- 実験フロー: `02_evaluate.py`→台帳追記 → `plot_progression.py`→図再生成 → §2/§3 と該当手法ページを更新 → `build_html.py` で再ビルド。
- 事実に `[n]`、実測・解釈に 🔎 を必ず付け区別する。

## 付録C 手法詳細ページ・出典
- 手法ページ: [OPF](methods/opf.md) ／ [GiNZA](methods/ginza.md)（以降 Presidio/GLiNER/日本語NER を追加）
- <a id="ref1"></a>[1] OpenAI Privacy Filter: https://github.com/openai/privacy-filter
- <a id="ref2"></a>[2] OPF CLI/学習・評価 ソース確認: [`docs/findings-opf-cli.md`](docs/findings-opf-cli.md)
- <a id="ref3"></a>[3] 計画/仕様: [`docs/verification-plan.md`](docs/verification-plan.md) / [`docs/spec.txt`](docs/spec.txt)
