# 手法詳細: GiNZA (`ja_ginza`)

> 📑 凡例: `[n]`=事実／🔎=実測・解釈。サマリは [評価レポート](REPORT.md) を参照。比較対象 Issue #9。

## 概要
📘 GiNZA は spaCy ベースの日本語 NLP ライブラリ。NER は**関根の拡張固有表現体系**で型が細かい（`Person / Age / Date / City / Province / Position_Vocation / Organization` 等）。PII 専用ではなく汎用 NER のため、PII 検出には**ラベル対応（マッピング）と後処理**が要る。`pip install ginza ja_ginza` だけで日本語NERがすぐ動く＝**最短で比較線を引ける**ため最初の比較対象に採用。

## 定性スモーク結果
OPF と同一文に `ja_ginza` を適用（`outputs/ginza_smoke.txt`、`scripts/ginza_smoke.py`）。検出エンティティ（text → GiNZA label）:

| 文 | 検出エンティティ |
|---|---|
| 佐藤花子さん(72歳)、東京都千代田区…電話090-1234-5678、受付番号A-0012 | 佐藤花子=Person / さん=Title_Other / **72歳=Age** / 東京都千代田区=City / 電話=City(誤) / 090-1234=N_Organization(誤) |
| 患者の山田太郎は2025年1月3日に○○病院を受診。担当は鈴木医師 | 患者=Position_Vocation / 山田太郎=Person / **2025年1月3日=Date** / 鈴木=Person / 医師=Position_Vocation |
| Contact: 田中一郎, tanaka@example.co.jp, 03-1234-5678 | Contact=Show_Organization(誤) / 田中一郎=Person / @example.co.jp=Email(部分) / 03-1234=Time(誤) |

🔎 **所見**:
- **強み**: 年齢(Age)・和暦日付(Date)・職業(Position_Vocation)など**準識別子を拾う**（OPFの弱点を補完）。固有表現の型が豊富。
- **弱み**: 電話・メールなど**構造化PIIを誤分割/誤ラベル**（"電話"→City, 電話番号→N_Organization/Time, email部分一致）。過分割・型ノイズが出る。NER ゆえパターン系は不得手。
- → OPF/正規表現と**相補的**。構造化PIIはOPF+正規表現、準識別子はGiNZA等、と**用途で使い分け**る（アンサンブル＝統合モデルは今回スコープ外 #18。サマリ図2）。

## ラベル対応の論点（#3）
GiNZA の拡張固有表現 → 仕様10ラベル（2026-06-08改訂）の対応表が必要（例: Person→PERSON, City/Province→ADDRESS/REGION, Age→AGE, Date→DATE, Position_Vocation→OCCUPATION, Company/School/Facility系→ORGANIZATION）。電話/メール/各種IDは GiNZA 標準では不安定なため、正規表現併用または untyped 評価で扱う。確定は #3。

## 環境メモ（落とし穴）
⚠️ GiNZA 5.2 は新しい spaCy 3.8 / confection と非互換（`compound_splitter.split_mode: None` の検証エラー）。回避:
```python
nlp = spacy.load("ja_ginza", config={"components.compound_splitter.split_mode": "C"})
```
（`scripts/ginza_smoke.py` に反映済。導入: `uv pip install -U ginza ja_ginza`、spaCy 3.8.14 / ginza 5.2.0）

## 出典
- GiNZA: https://github.com/megagonlabs/ginza ／ 関根の拡張固有表現階層
