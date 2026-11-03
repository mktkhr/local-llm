# ローカル画像・動画生成: RTX 4080 SUPER 16 GB で最高品質を狙う(2026-11-image-video-gen)

16 GB + RAM 32 GB で動く画像生成(Qwen-Image 2.1 / FLUX.2 / Z-Image / Krea 2 / HiDream-O1)と動画生成(Wan 2.2 / HunyuanVideo 1.5 / LTX-2.5 / Kandinsky 6.0 / CogVideoX)を、品質最優先で比較し、常用構成を決める。

**進行中(Phase 1 画像: 4 構成まで計測、残りはダウンロード待ち)。調査は [notes/00-research.md](notes/00-research.md)、計画は [notes/01-plan.md](notes/01-plan.md)、取得済みモデルは [notes/02-models.md](notes/02-models.md)。**

| 項目       | 値                                                                        |
| ---------- | ------------------------------------------------------------------------- |
| 期間       | YYYY-MM-DD 〜 YYYY-MM-DD                                                  |
| 機材       | GPU / CPU / RAM / OS / Driver                                             |
| ランタイム | llama.cpp bNNNN(commit)/ Ollama vX.Y / mlx-lm X.Y                         |
| 方法論     | [docs/evaluation.md](../../docs/evaluation.md) との差分があればここに書く |
| 関連実験   | [experiments/...](../)                                                    |

## 結論

<結論だけ先に。数字は 1〜2 個、表は 1 つまで。詳細は下に>

## なぜそうなるか

<メカニズムの説明。所見が直感に反するならここで理由を示す>

## 計測条件

### Phase 0: 環境(2026-11-03)

- ComfyUI を Docker(`yanwk/comfyui-boot:cu130-slim-v2`)で起動。モデル・ノード・出力はホスト `~/comfy-data/` に bind mount。詳細と落とし穴は [notes/02-models.md](notes/02-models.md)
- 生成と計測は `scripts/comfy_api.py`(API 形式のグラフを `/prompt` に投げ、`/history` で完了を待ち、`nvidia-smi` を 1 秒間隔で採取)。同じプロンプト・同じシードは ComfyUI がキャッシュして 1 秒で返すので、時間を測るときはシードを変える
- GPU の電力制限は計測時に `nvidia-smi --query-gpu=power.limit` で記録する(Phase 0 のスモークは 320 W)

### スモーク: Z-Image Turbo(bf16、8 ステップ、res_multistep / simple、shift 3)

| 条件                              | 所要      | VRAM ピーク    | 平均 / 最大電力  |
| --------------------------------- | --------- | -------------- | ---------------- |
| 1024×1024、初回(モデルロード込み) | 53.8 s    | 15,589 MiB     | 63 / 290 W       |
| 1024×1024、2 回目以降             | **6.0 s** | 15,435 MiB     | 222〜247 / 288 W |
| 2048×2048                         | 30.1 s    | **15,915 MiB** | 271 / 298 W      |

- bf16 の DiT 12.3 GB + エンコーダ 8 GB は同時には載らないが、ComfyUI が順にロード / 退避して 16 GB で回る。ただし 2K は残り 460 MiB で限界。エンコーダを fp8 にするか DiT を fp8 / GGUF にすると余裕ができる
- 文字: チョークボードの「焼きたて パン」が「焼きたて **バ**ン」(半濁点が濁点に)、「OPEN 7:00-19:00」は正確。日本語の濁点・半濁点は文字生成の見どころになる
- 構図・写実は 8 ステップとして十分(作例は `results/raw/img/`、commit 対象外)

### Phase 1: 画像(2026-11-03〜)

- 実行は ComfyUI 公式テンプレート(`scripts/data/templates/`、MIT)を `scripts/tpl2api.py` で API 形式に変換し、`scripts/run_matrix.py` で構成 × 8 プロンプト × 3 seed(1, 2, 3)を回す。変換の正しさは、手書きグラフとテンプレート経由で同一 seed の出力が **ピクセル単位で一致** することで確認した(PNG の IDAT ハッシュ一致)
- 公式テンプレートの推奨設定をそのまま使う。変えるのはモデルファイル名・プロンプト・seed・解像度(1024×1024 相当)だけ。テンプレート同梱のプロンプト拡張(LLM による書き換え)は全構成で切る
- 電力制限は 320 W のまま(品質は seed 固定で決定的なので電力に依らない。時間の比較表は後で 220 W で測り直す)
- 構成の一覧は `scripts/data/phase1_models.json`、生データは `results/raw/phase1/runs.jsonl`(commit 対象外)、集計は `results/phase1-summary.md`、seed 1 の比較シートは `results/phase1-sheets/`

## 結果

### Phase 1(途中経過: 4 構成 × 24 枚)

| 構成                | 設定(テンプレート既定)                            | 1 枚あたり(2 回目以降 中央値) | VRAM ピーク | 平均電力 |
| ------------------- | ------------------------------------------------- | ----------------------------- | ----------- | -------- |
| Z-Image Turbo bf16  | 8 steps、cfg 1、res_multistep / simple            | **6.0 s**                     | 15.9 GB     | 232 W    |
| Z-Image Base bf16   | 25 steps、cfg 4、res_multistep / simple           | 31.1 s                        | 15.9 GB     | 289 W    |
| Qwen-Image 2.1 int8 | 25 steps、cfg 1、euler / simple、QwenImage21Cache | 9.0 s                         | 15.7 GB     | 277 W    |
| Qwen-Image 2.1 bf16 | 同上(DiT 14.2 GB をオフロードしながら)            | 19.0 s                        | 15.8 GB     | 280 W    |

初回(モデル読込込み)は 56〜186 s だが、この計測中は裏でモデルのダウンロードが走っていて読み込みが遅かった(後述の罠)。

seed 1 の比較シート([results/phase1-sheets/](results/phase1-sheets/))を見た暫定所見。人手 5 段階の本評価は全構成が揃ってから行う。

- **文字(P1 ポスター)**: Qwen-Image 2.1 は日本語見出し・英語副題・日付の 3 行すべて正確。Z-Image Turbo は「フェア」が「フェ尽」、Z-Image Base は見出しは読めるが副題が崩れ、レイアウトも乱れる。Phase 0 の「バン」もそうだが、Z-Image 系は日本語の 1〜2 文字が化ける
- **複雑構図(P2 ラーメン屋 3 人)**: 4 構成とも 3 人と役割(料理人・新聞を読む男性・食べる女性)を再現。Z-Image Base がフィルム調の自然な画で最も良く、Qwen は暗めでシネマ寄り
- **風景(P4 棚田)**: Qwen が良い。Z-Image は田んぼの形が整いすぎていて現実離れしている(ユーザー所見)。「きれいに整った幾何」はこの用途では減点
- **写実ポートレート(P3)**: いずれも破綻なし。Z-Image Base の肌の質感が最も細かく、Qwen はクリーンで整った写り
- **商品(P5 マグカップ)**: Qwen が最も商品写真らしい。Z-Image Base は指示していない文字(商品説明風の崩れた文字列)を上部に描き込んだ
- **空間指示(P8 りんご 3 個とタグ「No.2」)**: Z-Image Turbo だけ **りんごを 4 個** 描いた。他は 3 個でタグも正確
- **量子化(Qwen-Image 2.1 bf16 vs int8、同一 seed)**: 8 プロンプトすべてで構図・文字・色がほぼ同一。差は細部のテクスチャ程度で、シートの縮小では見分けがつかない。int8 は bf16 の **半分の時間**(9 s vs 19 s)なので、Qwen-Image 2.1 は int8 で使ってよい

## 落とし穴

| 罠                                     | 症状                                                                                                                                             | 対処                                                                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RAM 32 GB にモデルが溜まる             | ComfyUI は退避したモデルを CPU RAM に残す。構成を切り替えると前のモデルが残ったまま次を読み、スワップして数分固まる                              | 構成の切り替え時に `POST /free {"unload_models": true, "free_memory": true}` を呼ぶ(`run_matrix.py` が自動で呼ぶ)。RAM 25 GB → 4 GB に戻る                  |
| ダウンロードと同時に計測しない         | `hf download` の書き込み(〜380 MB/s)でモデル読み込みが 10 MB/s まで落ち、初回が 3 分以上かかる                                                   | 計測中はダウンロードを `kill -STOP`、終わったら `kill -CONT`。`pkill -f` のパターンが自分の ssh シェルにも当たるので `bash -s` のヒアドキュメントで実行する |
| ComfyUI の出力ファイル名               | `filename_prefix` に `_00001_` の連番が付く。同じ prefix で再実行すると番号がずれる                                                              | 取得時に `prompt_seed.png` へ改名して保存する(`run_matrix.py`)                                                                                              |
| サブグラフ入りテンプレートの widget 値 | サブグラフノードの `widgets_values` はノード側の inputs 順ではなく **サブグラフ定義の inputs 順**。ずれると steps に 4096 が入る等の無言の誤動作 | `tpl2api.py` は定義順で割り当て、テストで公式テンプレートの既定値(steps 25 / cfg 1 など)を固定している                                                      |

## 次

- 残りの Phase 1 構成(FLUX.3 Klein distilled / base、FLUX.2 klein 9B / 4B、Krea 2 Turbo fp8 / int8 / GGUF Q4、HiDream-O1 dev fp8 / bf16)をダウンロード完了後に同じ行列で回す
- 全構成が揃ったら人手 5 段階評価(追従・構図・文字・写実/美的・破綻)を seed 3 本で行い、2K も上位モデルで試す
- 220 W で各構成 1 枚ずつ時間を測り直して速度表を確定する
- Phase 2(動画)の T2V / I2V、Phase 3(画像 → 動画)

## ファイル

| パス       | 内容                                                              |
| ---------- | ----------------------------------------------------------------- |
| `notes/`   | 調査メモ・手順書・途中経過(複数可、番号 prefix で順序づけ)        |
| `scripts/` | この実験専用の計測スクリプト。共通 harness は `tools/` を使う     |
| `config/`  | この実験専用の llama-swap entry / compose / systemd override など |
| `results/` | 集計済みの csv / jsonl。生ログは `results/raw/`(gitignore)        |
