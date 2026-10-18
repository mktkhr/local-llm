# local-llm

オンプレ環境でローカルLLMを **長コンテキスト・GPUフルロード(CPUオフロード回避)** で動かすためのナレッジ集。

特定モデルの選定記録ではなく、以下を残すことを目的とします。

- どのモデルがどのGPUで速度を出せるのか
- 量子化(モデル重み・KVキャッシュ)をどう調整するか
- 計測結果のどこを見て調整するか
- 個別ハードウェアでの実測リファレンス

推論ランタイムは環境に応じて使い分けています:

- **Linux + NVIDIA GPU**: **Ollama**(汎用ベンチ)と **llama.cpp / llama-swap**(モデルごとに `--n-cpu-moe` / KV 量子化 / batch を個別指定したい運用)
- **macOS + Apple Silicon**: **mlx-lm**(Apple 純正 MLX)。Metal を直接叩く llama.cpp 経路より Apple Silicon 上で速い

評価方法論([docs/evaluation.md](docs/evaluation.md))と評価データ・採点ロジックは両ランタイム共通で、ランタイム実装だけが分岐します。

## リポジトリ構成

```
local-llm/
├── docs/          # ハードウェア非依存の汎用ナレッジ(セットアップ / 計測の読み方 / チューニング / 評価方法論)
├── experiments/   # 実験記録。1 ディレクトリ = 1 実験。README が結論、notes/ scripts/ config/ results/ が根拠
├── deploy/        # いま稼働している構成だけ(llama-swap + ccshim / 認証ゲート)。実験記録は持たない
└── tools/         # 実験横断で使う計測 harness(Ollama 用 / mlx-lm 用)
```

### 実験一覧

新しい順。各 README の冒頭が結論。

| 実験                                                                             | 問い                                                                            | 結論(1 行)                                                                                                      |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [2026-08-pd-split](experiments/2026-08-pd-split/README.md)                       | NVIDIA 機で Prefill した KV を Apple Silicon 機に転送して Decode を継続できるか | 成立。Wi-Fi でも 16k で 9〜23 倍速。律速は転送で、KV q4_0 と並列 Range が効く。Decode は変わらない              |
| [2026-08-claude-code-backend](experiments/2026-08-claude-code-backend/README.md) | Qwen3.6-35B-A3B を 16 GB に載せて Claude Code のバックエンドにできるか          | できる。UD-IQ4_XS + ncmoe=10 が実タスク最短。量子化を落とすと速いが手数が増えて正味遅い。256k は KV q4_0 で成立 |
| [2026-05-mlx-bench](experiments/2026-05-mlx-bench/README.md)                     | 48 GB Mac は 16 GB GPU に載らない大型モデルをどれだけの速度で動かせるか         | 載るが実用域を割る。decode は帯域律速、長文 prefill は ctx² で爆発。「動く」と「実用」が一致しない              |
| [2026-05-ollama-bench](experiments/2026-05-ollama-bench/README.md)               | GTX 1080 / RTX 4080 SUPER で 100% GPU・長 ctx を満たす構成はどれか              | 4080S は `qwen3.5:4b-q4_K_M` が 262K ctx / 135 tok/s で最良バランス。KV q4_0 はほぼフリーランチ、q8 重みは無益  |

採用推奨だけを横断で見るなら [docs/hardware-reference.md](docs/hardware-reference.md)。

### 新しい実験を始めるとき

1. `experiments/_template/` を `experiments/YYYY-MM-<slug>/` に複製する
2. 計測スクリプトは `tools/bench` / `tools/bench-mlx` の harness を使い、実験固有の差分だけ `scripts/` に置く
3. その実験専用の llama-swap entry や compose は `config/` に置く。`deploy/` には本採用が決まってから入れる
4. `results/` には集計済みの csv / jsonl だけを commit する。生ログは `results/raw/`(gitignore)
5. 結論が出たら README 冒頭に書き、本ファイルの実験一覧に 1 行追加する。採用推奨が確定したら [docs/hardware-reference.md](docs/hardware-reference.md) にも反映する

## 評価の前提(なぜこの既定値なのか)

GPU・モデルが変わっても比較できるよう、以下は **常に固定** します。「なぜ固定するか」を理由つきで残しておくことで、別 GPU での再現や、ここから記事に起こすときの根拠資料になります。

| 設定                | 値                                  | なぜ固定するか                                                                                                                    |
| ------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **CPU オフロード**  | 禁止(`100% GPU` を要件)             | 一部レイヤーが CPU 側に落ちると生成速度が一桁下がるため、実運用に耐えない。本リポジトリは「実用に耐える構成」だけを評価対象にする |
| **Flash Attention** | 常時 ON(`OLLAMA_FLASH_ATTENTION=1`) | KV キャッシュ量子化を効かせるための前提。古い GPU 世代(Pascal 等)では速度メリットが限定的だが、それでも有効化する                 |
| **並列度**          | 1(`OLLAMA_NUM_PARALLEL=1`)          | 並行リクエストが増えると KV キャッシュが並列数倍に膨らみ、同じ ctx でも VRAM 消費が変わるため比較が壊れる                         |
| **常駐モデル数**    | 1(`OLLAMA_MAX_LOADED_MODELS=1`)     | 前モデルが VRAM に残ったままだと値が壊れる                                                                                        |
| **自動アンロード**  | 無効(`OLLAMA_KEEP_ALIVE=-1`)        | 計測中にアンロードされると値が変わる                                                                                              |
| **推論ランタイム**  | Linux: Ollama / macOS: mlx-lm       | Linux は Ollama (llama.cpp ベース)、macOS は mlx-lm (Apple 純正 MLX)。Apple Silicon では MLX のほうが Metal-llama.cpp より速い    |

### 計測時に変動させるのは以下の 4 軸のみ

これらの組合せを動かしながら「100% GPU で乗る最大」「実用に耐える速度」「品質」を測ります。

- **モデル**(系列・サイズ)
- **重み量子化**(Q4_K_M, Q8_0, ...)
- **KV キャッシュ量子化**(f16, q8_0, q4_0)
- **コンテキスト長**(`num_ctx`)

詳細は [docs/evaluation.md](docs/evaluation.md) §5。

### 用語

| 用語                             | 意味                                                                                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CPU オフロード**               | モデルの一部レイヤーを VRAM ではなく CPU / メインメモリ側に置く動作。速度が大幅に落ちるため本評価では禁止                                                                                                         |
| **量子化(weight quantization)**  | モデル重みを低ビット表現に変換して VRAM を節約する技術。`Q8_0` / `Q4_K_M` などのレベルがあり、低いほど省メモリで品質も落ちる                                                                                      |
| **KV キャッシュ**                | アテンション機構で過去トークンの Key / Value を保持するメモリ領域。コンテキスト長に比例して増える                                                                                                                 |
| **KV キャッシュ量子化**          | 上記 KV キャッシュも量子化して VRAM を節約する。長コンテキスト時に効く                                                                                                                                            |
| **Flash Attention**              | アテンション計算を VRAM 効率良く行う実装。KV キャッシュ量子化を効かせるための前提条件                                                                                                                             |
| **thinking モード**              | Qwen3 系・DeepSeek-R1 等が持つ「内部思考」生成モード。応答前に大量のトークンを生成するため遅い。本評価では `think:false` を主測定値とする                                                                         |
| **TTFT**                         | Time To First Token。最初のトークンが返ってくるまでの実時間。対話型 UX を測る指標                                                                                                                                 |
| **Needle In A Haystack(needle)** | 長文の中に「秘密の数字は 603904 です」のような短い情報を埋め込み、文末で「秘密の数字は?」と問うて抽出できるかを測る手法。**長文検索能力**を見るベンチマーク。詳細は [docs/evaluation.md §7.3](docs/evaluation.md) |

理論的な背景は [docs/tuning.md](docs/tuning.md) を参照。

## ドキュメント

### 汎用ナレッジ(`docs/`)

| ファイル                                                 | 内容                                                                                   |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [docs/setup-linux.md](docs/setup-linux.md)               | Linux + NVIDIA GPU: 環境確認、Ollama 導入、systemd 経由の環境変数設定                  |
| [docs/setup-mac.md](docs/setup-mac.md)                   | macOS + Apple Silicon: 環境確認、uv / mlx-lm 導入、ストアアプリ停止                    |
| [docs/benchmark-linux.md](docs/benchmark-linux.md)       | Ollama 計測の原則と手順、`ollama ps` / `nvidia-smi` / API レスポンスのどこを見るか     |
| [docs/benchmark-mac.md](docs/benchmark-mac.md)           | mlx-lm 計測の原則、`mx.get_peak_memory()` の読み方、JSON 出力解説                      |
| [docs/tuning.md](docs/tuning.md)                         | VRAM予算の見積もり、量子化レベル選定、Flash Attention / KVキャッシュ量子化の判定フロー |
| [docs/evaluation.md](docs/evaluation.md)                 | コーディング/要約/長コンテキストの3用途で品質まで含めて評価するための方法論(汎用)      |
| [docs/hardware-reference.md](docs/hardware-reference.md) | GPU / Mac 別の採用推奨早見表。根拠は各実験へリンク                                     |

### 稼働構成(`deploy/`)

| ディレクトリ                                     | 内容                                                                                      |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| [deploy/llama-swap](deploy/llama-swap/README.md) | llama-swap + ccshim。別 PC の Claude Code から Qwen3.6-35B-A3B を使うための構成と動かし方 |
| [deploy/auth-gate](deploy/auth-gate/README.md)   | Caddy + 静的 Bearer による外部公開用の認証ゲート(Linux 専用)                              |

### 計測 harness(`tools/`)

| ディレクトリ                                 | 内容                                                                                   |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| [tools/bench](tools/bench/README.md)         | Ollama 用。速度 / ctx 二分探索 / KV 切替 / Needle / Coding / Summary(Python 3.12+、uv) |
| [tools/bench-mlx](tools/bench-mlx/README.md) | mlx-lm 用。同じ方法論の MLX 実装(Python 3.11+、uv、mlx 0.31+)                          |

## クイックスタート

### Linux + NVIDIA GPU

1. [docs/setup-linux.md](docs/setup-linux.md) で Ollama と計測用の環境変数を整える
2. [docs/benchmark-linux.md](docs/benchmark-linux.md) と [tools/bench](tools/bench/README.md) で目当てのモデルを評価
3. CPU オフロードが発生していたら [docs/tuning.md](docs/tuning.md) の判定フローで設定を絞る
4. 安定した組み合わせを実験ディレクトリに記録し、[docs/hardware-reference.md](docs/hardware-reference.md) に採用推奨を追記して資産化

### macOS + Apple Silicon

1. [docs/setup-mac.md](docs/setup-mac.md) で uv / mlx-lm を入れ、ストアアプリを停止
2. [docs/benchmark-mac.md](docs/benchmark-mac.md) と [tools/bench-mlx](tools/bench-mlx/README.md) の手順で計測
3. swap への逃避や `peak_memory > working_set` を観測したら、KV 量子化(`--kv-bits 8` / `4`)や `max_kv_size` を絞る
4. 安定した組み合わせを実験ディレクトリに記録し、[docs/hardware-reference.md](docs/hardware-reference.md) に追記

## 設計方針

- **汎用ナレッジ(方法論)と固有実績(数値)を分離**: 数値は `experiments/` に閉じ込め、`docs/` はハードウェア非依存に書く
- **実験は 1 ディレクトリで完結**: 結論・手順・スクリプト・設定・結果を同じ場所に置き、あとから 1 塊で読める / 再現できる
- **`deploy/` は稼働中の物だけ**: 実験中の構成は `experiments/*/config/` に置き、本採用で `deploy/` に移す
- **計測の再現性を最優先**: 1モデルロード固定、並行リクエスト固定、`keep_alive` 明示制御
- **判定フローは「観測 → 調整 → 再計測」のループ**: 量子化やFAは試行錯誤前提
