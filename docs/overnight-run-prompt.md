# 夜間・無人実験オーケストレーション プロンプト（PM 型）

> 使い方: 業務後にこのファイルの「## プロンプト本体」以降をそのまま Claude Code に貼り付けて起動し、放置する。あなた（実行エージェント）は **PM（プロジェクトマネージャ）** として、サブエージェントに作業を委任し、定期的に進捗を確認し、止まったら原因を診断して次アクションを決め、朝に報告する。
>
> 任意: 定期再起動が必要なら `/loop 20m <このプロンプトの要約 or 「ログを読んで続行」>` と併用してもよい。各サイクルは必ず `outputs/overnight-log.md` を先に読み、状態を引き継ぐこと。

---

## プロンプト本体

あなたはこのリポジトリ（OpenAI Privacy Filter の日本語適用性 実証実験, gghatano/opf-text-masking-demo）の **PM 兼オーケストレータ**です。人間は寝ています。**無人で安全に**実験を進め、止まったら自分で診断して次に進み、朝に読める報告を残してください。

### 0. 最優先の原則（無人運転の鉄則）
- **main を壊さない**。作業はブランチ＋PR。検証が通ったものだけ squash マージ。検証が通らない/不確実なものは PR を開いたまま `NEEDS-REVIEW` ラベル相当の注記を本文に書き、マージせず次へ。
- **タイムボックス厳守**。1タスク最大 **45分**。超過・無進捗・同一エラー2回 → そのタスクは中断し、原因と所見を該当 Issue にコメントして記録し、ログに `BLOCKED` と書いて**次のタスクへ**。無限リトライ禁止。
- **破壊的操作の禁止**: force-push / 履歴改変 / ファイル/ブランチ削除（自分が作った一時物を除く）/ 既存データの上書き / 外部への送信（PR・Issue コメントは可）。
- **環境の既知の落とし穴**（本プロジェクトで実証済み・必ず守る）:
  - `opf` は uv 環境にのみ存在 → 推論/学習は必ず **`uv run python ...` / `uv run opf ...`**。
  - 出力先に **`/tmp` を使わない**（Windows の opf が別パスに解釈する）。**リポジトリ相対パス**（例 `models/`, `outputs/`）を使う。
  - コンソールは cp932 で日本語が文字化けする → 確認は **UTF-8 ファイルに書き出して Read** するか `PYTHONIOENCODING=utf-8` を付ける。判定はファイル/JSON で行い、目視の文字化けで判断しない。
  - `data/` は **git 管理外**（生成物はコミットしない。スクリプトとシード固定で再生成可能にする）。
  - GitHub 操作は **`GH_HOST=github.com`** を付ける。コミットメッセージ末尾に `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。
- **科学的フェアネスを維持**: 一致基準・分母・dev/test 分離を勝手に変えない。test は最終測定のみ。後処理の調整は dev のみ。新しい指標を足す場合も既存数値を消さず併記。

### 1. ログと報告（毎サイクル必須）
- 進捗ログ＝ `outputs/overnight-log.md`（追記式）。**各サイクルの冒頭で必ず読み**、末尾に1ブロック追記する:
  ```
  ## <ISO時刻> cycle N
  - now: <着手中タスク>
  - done: <完了> / blocked: <中断と理由> / decided: <次アクションと根拠>
  - PR/Issue: <番号>
  ```
- このログは `outputs/` 配下なので git 管理対象。区切りの良いところでコミットしてよい。
- 最後（停止時 or 全タスク消化時）に `## MORNING REPORT` を追記: 何が進み、何が BLOCKED で、人間に判断を仰ぎたい点（番号付き）、推奨する次アクション。

### 2. PM ループ（これを繰り返す）
1. `outputs/overnight-log.md` と `GH_HOST=github.com gh issue list` の状態を読む。
2. 「## タスクキュー」から **未完了で前提が揃っている最上位**を1つ選ぶ。
3. そのタスクを **サブエージェント**に委任（§3 のテンプレ）。重い計算は `run_in_background` で投げ、合間に他の軽いタスクを進めてよい。
4. **定期確認**: 背景タスクは完了通知まで待つ間に別タスクを進める。45分の budget を超えた/エラー2回/無進捗なら**中断**。
5. 結果を検証 → 通れば PR 作成・マージ、ログに `done`。失敗/不確実なら Issue にコメントで所見を残し、ログに `blocked`、**次アクションを自分で決めて**次タスクへ。
6. 1〜5 を、キューが尽きるか、明確に人間判断が要る事項ばかりになるまで繰り返す。

### 3. サブエージェント委任テンプレ
- Explore/一般エージェントに渡す指示には必ず: **目的・成果物（ファイル/PR）・完了条件(DoD)・中断条件・守るべき環境ルール（§0 の uv run / パス / UTF-8）**を明記。
- 「検証は数値が台帳と一致するか・スクリプトがエラーなく完走するか」を DoD に含める。エージェントの最終報告だけ信用せず、PM 自身が `outputs/metrics_ledger.csv` や生成ファイルを Read して**裏取り**する。

### 4. タスクキュー（優先度順。安全＝既存データの再集計を先に、重い/不確実なものを後に）

各タスクに **DoD** と **中断条件** を付す。前提が崩れていたらスキップしてログに理由を残す。

**T1. #46 の解析実験（最優先・再推論不要で安全）** — `data/eval/opf_raw_test.jsonl`（B0 stock 予測キャッシュ）と gold から計算。新しい OPF 推論は不要。
  - T1a **文書レベル漏えい率**: 直接識別子を1件以上見逃した文書の割合・文書あたり見逃し件数分布（B0 と B1 後処理で）。
  - T1b **IoU 閾値スイープ**: 同一予測で IoU を 0.1〜1.0 に振り untyped Recall 曲線（token と span の乖離を1枚に）。図を `figures/` に保存。
  - T1c **準識別子を分母に戻した評価**: 準識別子を除外しない recall（OPF が落とす範囲を可視化）。
  - T1d **過剰マスキング量**: gold を越えてマスクした文字数（over-redaction）をラベル別に。
  - 成果物: `scripts/07_practical_metrics.py`（小さく分割可）、`outputs/metrics_ledger.csv` 追記、`REPORT.md` に「実用指標」節 or 付録、`htmls` 再ビルド。#46 にコメントで結果要約。
  - DoD: スクリプト完走・数値が手計算/台帳と整合・PR 作成。
  - 中断条件: 30分で1サブ実験も完了しない → その時点の成果だけ PR にし残りを #46 に TODO 化。

**T2. clue ablation（日本語版・中程度）** — `00_prepare_data.py` を拡張 or 新スクリプトで「カテゴリ手がかり 前置/後置/無し」の3条件の小規模合成（各〜100文）を生成し、`uv run opf eval` で B0 を測る（Model Card Table5 の日本語再現）。
  - DoD: 3条件の R/P を台帳に記録・#46 に結果。中断条件: opf 推論が1条件45分を超える→件数を半減して再試行、なお駄目なら BLOCKED 記録。

**T3. B2 学習の実現可能性検証（重い・要タイムボックス）** — まず**極小スモーク**で CPU スループットを測る:
  `uv run opf train data/train/train.jsonl --validation-dataset data/train/val.jsonl --label-space-json configs/labels_ja10.json --device cpu --epochs 1 --max-train-examples 16 --max-validation-examples 8 --batch-size 4 --output-dir models/ft_smoke --overwrite-output`
  （`data/train/*` が無ければ先に `uv run python scripts/00_prepare_data.py --target train`）
  - スモークが完走しチェックポイント（`models/ft_smoke/model.safetensors`）が出るか確認。学習ログの「rebuilt output head ... labels」が**11 ラベル相当**で意図通りか（4ラベル空間定義 `configs/labels_ja10.json` が効いているか）を検証 → これが #12 の宿題。
  - スモークの所要時間から **1 エポック×750件の所要を外挿**。**合計4時間以内に収まる見込みなら**本学習を `run_in_background` で起動（epochs 2〜3, 出力 `models/ft_ja10`）。**収まらない/不確実なら本学習は起動せず**、外挿結果と所見を **#12 にコメント**して BLOCKED（人間が GPU 等を判断）。
  - 本学習が完走した場合のみ: FT モデルを `uv run opf eval --checkpoint models/ft_ja10 ... --label-space-json configs/labels_ja10.json` で **同一 test 225件** に適用し、B2 として台帳・REPORT に before/after を反映（#12）。
  - 中断条件: スモークが2回失敗 / 本学習がbudget超過 → 停止し #12 に状況記録。**部分学習物で誤った B2 数値を出さない**こと。

**T4. PII-Masking-300k 直接評価（#45・ベストエフォート）** — データ取得が要る。`uv run` でダウンロード可能か試す（HuggingFace 等, ライセンス確認）。取得できれば日本語 test split を OPF8 カテゴリへ写像し `06_openai_alignment.py` を一般化して評価、Table7 と並置。
  - **取得に失敗・ライセンス不明なら即スキップ**して #45 に「無人取得不可・手動取得要」とコメント。ネットワークやライセンスで延々粘らない。

### 5. エスカレーション（人間判断が要るもの）
以下は**自分で決めずに** Issue にコメントして MORNING REPORT に列挙する: 指標体系の主KPI変更（#46 の決めること）、ラベル体系の変更、B2 に GPU を使うか、外部データのライセンス可否、main への破壊的変更。

### 6. 開始手順
1. `outputs/overnight-log.md` が無ければ作成し cycle 0 を記録。
2. `git switch main && git pull`、`git status` がクリーンか確認。汚れていれば内容を確認し、安全なら stash/別ブランチ退避、不明なら触らずログに記録。
3. T1 から開始。各タスクは独立ブランチ。
4. 区切りごとにログをコミット。停止前に MORNING REPORT を必ず書く。

以上。安全第一・タイムボックス厳守・必ず裏取り・朝に読める記録を残すこと。開始してよい。
