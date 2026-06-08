# BACKLOG

Issue 化済みのものは GitHub Issues (#1〜#16) を正とする。ここは Issue 未満のメモ・派生課題。

## 検討メモ
- [ ] 評価データ生成後、`requirements.txt` を `pip freeze` で固定（乱数に効く torch/numpy はコミット粒度で固定）。
- [ ] スパン一致は OPF 既定の containment 判定。他モデル比較(#10)でも同一定義に揃える共通評価器を `03_compare_models.py` に実装。
- [ ] 準識別子(AGE/REGION/OCCUPATION)は誤検出と表裏。成功基準 Recall90% の対象範囲を #3 で明確化。
- [ ] LoRA 非対応（full FT のみ）確定 → #12 の VRAM 見積もりを full FT 前提で。
- [ ] 日本語トークナイズと char-span 整合（tiktoken ベース）の確認を B0 実行時に行う。

## 将来展開（spec §12）
- 自治体向け個人情報ファイル簿作成支援 / 医療情報匿名加工支援 / 仮名加工支援 / マスキングサービス / AIレビュー支援。
