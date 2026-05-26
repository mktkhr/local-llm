# local-llm

オンプレ環境でローカルLLMを **長コンテキスト・GPUフルロード(CPUオフロード回避)** で動かすためのナレッジ集。

特定モデルの選定記録ではなく、以下を残すことを目的とします。

- どのモデルがどのGPUで速度を出せるのか
- 量子化(モデル重み・KVキャッシュ)をどう調整するか
- 計測結果のどこを見て調整するか
- 個別ハードウェアでの実測リファレンス

推論ランタイムは環境に応じて 2 系統を使い分けています:

- **Linux + NVIDIA GPU**: **Ollama**(llama.cpp ベース)。プリビルドバイナリが CPU 命令セット差・CUDA ランタイム差を吸収するため、古いハードウェアでも動かしやすい
- **macOS + Apple Silicon**: **mlx-lm**(Apple 純正 MLX フレームワーク)。Metal を直接叩く llama.cpp 経路より Apple Silicon 上で速い

評価方法論 ([docs/06-evaluation.md](docs/06-evaluation.md)) と評価データ・採点ロジックは両ランタイム共通で、ランタイム実装だけが分岐します。

## 評価の前提(なぜこの既定値なのか)

GPU・モデルが変わっても比較できるよう、以下は **常に固定** します。「なぜ固定するか」を理由つきで残しておくことで、別 GPU での再現や、ここから記事に起こすときの根拠資料になります。

| 設定                | 値                                  | なぜ固定するか                                                                                                                    |
| ------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **CPU オフロード**  | 禁止(`100% GPU` を要件)             | 一部レイヤーが CPU 側に落ちると生成速度が一桁下がるため、実運用に耐えない。本リポジトリは「実用に耐える構成」だけを評価対象にする |
| **Flash Attention** | 常時 ON(`OLLAMA_FLASH_ATTENTION=1`) | KV キャッシュ量子化を効かせるための前提。古い GPU 世代(Pascal 等)では速度メリットが限定的だが、それでも有効化する                 |
| **並列度**          | 1(`OLLAMA_NUM_PARALLEL=1`)          | 並行リクエストが増えると KV キャッシュが並列数倍に膨らみ、同じ ctx でも VRAM 消費が変わるため比較が壊れる                         |
| **常駐モデル数**    | 1(`OLLAMA_MAX_LOADED_MODELS=1`)     | 前モデルが VRAM に残ったままだと値が壊れる                                                                                        |
| **自動アンロード**  | 無効(`OLLAMA_KEEP_ALIVE=-1`)        | 計測中にアンロードされると値が変わる                                                                                              |
| **推論ランタイム**  | Linux: Ollama / macOS: mlx-lm        | Linux は Ollama (llama.cpp ベース)、macOS は mlx-lm (Apple 純正 MLX)。Apple Silicon では MLX のほうが Metal-llama.cpp より速い   |

### 計測時に変動させるのは以下の 4 軸のみ

これらの組合せを動かしながら「100% GPU で乗る最大」「実用に耐える速度」「品質」を測ります。

- **モデル**(系列・サイズ)
- **重み量子化**(Q4_K_M, Q8_0, ...)
- **KV キャッシュ量子化**(f16, q8_0, q4_0)
- **コンテキスト長**(`num_ctx`)

詳細は [docs/06-evaluation.md](docs/06-evaluation.md) §5。

### 用語

| 用語                            | 意味                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **CPU オフロード**              | モデルの一部レイヤーを VRAM ではなく CPU / メインメモリ側に置く動作。速度が大幅に落ちるため本評価では禁止                                 |
| **量子化(weight quantization)** | モデル重みを低ビット表現に変換して VRAM を節約する技術。`Q8_0` / `Q4_K_M` などのレベルがあり、低いほど省メモリで品質も落ちる              |
| **KV キャッシュ**               | アテンション機構で過去トークンの Key / Value を保持するメモリ領域。コンテキスト長に比例して増える                                         |
| **KV キャッシュ量子化**         | 上記 KV キャッシュも量子化して VRAM を節約する。長コンテキスト時に効く                                                                    |
| **Flash Attention**             | アテンション計算を VRAM 効率良く行う実装。KV キャッシュ量子化を効かせるための前提条件                                                     |
| **thinking モード**             | Qwen3 系・DeepSeek-R1 等が持つ「内部思考」生成モード。応答前に大量のトークンを生成するため遅い。本評価では `think:false` を主測定値とする |
| **TTFT**                        | Time To First Token。最初のトークンが返ってくるまでの実時間。対話型 UX を測る指標                                                         |

理論的な背景は [docs/03-tuning.md](docs/03-tuning.md) を参照。

## ドキュメント

### 共通(汎用方法論・実績)

| ファイル                                       | 内容                                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| [docs/03-tuning.md](docs/03-tuning.md)         | VRAM予算の見積もり、量子化レベル選定、Flash Attention / KVキャッシュ量子化の判定フロー |
| [docs/04-results.md](docs/04-results.md)       | GPU/Mac 別・モデル別の動作実績テーブル                                                 |
| [docs/05-auth-gate.md](docs/05-auth-gate.md)   | Caddy + 静的Bearer による外部公開用の認証ゲート(Linux 専用)                            |
| [docs/06-evaluation.md](docs/06-evaluation.md) | コーディング/要約/長コンテキストの3用途で品質まで含めて評価するための方法論(汎用)      |

### Linux + NVIDIA GPU + Ollama

| ファイル                                           | 内容                                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| [docs/01-setup.md](docs/01-setup.md)               | 環境確認の手順、Ollama導入、systemd経由の環境変数設定                       |
| [docs/02-benchmark.md](docs/02-benchmark.md)       | 計測の原則と手順、`ollama ps` / `nvidia-smi` / API レスポンスのどこを見るか |
| [scripts/bench/README.md](scripts/bench/README.md) | 上記方法論を実装した計測スクリプト群(Python 3.12+、uv、Ollama 必須)         |

### macOS + Apple Silicon + mlx-lm

| ファイル                                                   | 内容                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| [docs/mac/01-setup.md](docs/mac/01-setup.md)               | Apple Silicon 環境確認、uv / mlx-lm 導入、ストアアプリ停止        |
| [docs/mac/02-benchmark.md](docs/mac/02-benchmark.md)       | mlx-lm 計測の原則、`mx.get_peak_memory()` の読み方、JSON 出力解説 |
| [scripts/bench-mlx/README.md](scripts/bench-mlx/README.md) | mlx-lm 用計測スクリプト群(Python 3.11+、uv、mlx 0.31+ 必須)       |

## クイックスタート

### Linux + NVIDIA GPU

1. [docs/01-setup.md](docs/01-setup.md) で Ollama と計測用の環境変数を整える
2. [docs/02-benchmark.md](docs/02-benchmark.md) の計測スクリプトで目当てのモデルを評価
3. CPU オフロードが発生していたら [docs/03-tuning.md](docs/03-tuning.md) の判定フローで設定を絞る
4. 安定した組み合わせを [docs/04-results.md](docs/04-results.md) に追記して資産化

### macOS + Apple Silicon

1. [docs/mac/01-setup.md](docs/mac/01-setup.md) で uv / mlx-lm を入れ、ストアアプリを停止
2. [docs/mac/02-benchmark.md](docs/mac/02-benchmark.md) と [scripts/bench-mlx/README.md](scripts/bench-mlx/README.md) の手順で計測
3. swap への逃避や `peak_memory > working_set` を観測したら、KV 量子化 (`--kv-bits 8` / `4`) や `max_kv_size` を絞る
4. 安定した組み合わせを [docs/04-results.md](docs/04-results.md) の Apple Silicon セクションに追記

## 設計方針

- **汎用ナレッジ(方法論)と固有実績(数値)を分離**: GPU依存の数値は results に集約し、setup / benchmark / tuning はハードウェア非依存に書く
- **計測の再現性を最優先**: 1モデルロード固定、並行リクエスト固定、`keep_alive` 明示制御
- **判定フローは「観測 → 調整 → 再計測」のループ**: 量子化やFAは試行錯誤前提
