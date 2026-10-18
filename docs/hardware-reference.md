# ハードウェア別 採用推奨リファレンス

GPU / Mac ごとに「この構成なら 100% GPU で実用速度が出る」と確定した組み合わせだけを抜粋した早見表。
数値の根拠・計測条件・全データは各実験ディレクトリを参照する。

| ハードウェア           | 実験記録                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| GTX 1080 (8 GB)        | [experiments/2026-05-ollama-bench](../experiments/2026-05-ollama-bench/README.md)                                        |
| RTX 4080 SUPER (16 GB) | 同上。Claude Code 用途は [experiments/2026-08-claude-code-backend](../experiments/2026-08-claude-code-backend/README.md) |
| Apple M4 Pro (48 GB)   | [experiments/2026-05-mlx-bench](../experiments/2026-05-mlx-bench/README.md)                                              |

---

## NVIDIA GeForce GTX 1080 (8 GB, Pascal)

GTX 1080 (8 GB) では用途に応じて以下の2構成を使い分ける(共通設定: `OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`)。

| 用途                   | モデル       | num_ctx | 速度       | 残VRAM    | コメント                      |
| ---------------------- | ------------ | ------- | ---------- | --------- | ----------------------------- |
| **長コンテキスト優先** | `qwen3.5:2b` | 131,072 | 53.9 tok/s | 1,977 MiB | 128k フル動作、速度も最速     |
| **品質優先**           | `qwen3.5:4b` | 65,536  | 35.3 tok/s | 907 MiB   | 64k で 100% GPU、4Bの推論品質 |

備考:

- 2b は KV を `q4_0` に下げれば残VRAM 2,361 MiB まで広がる(長文参照精度との交換)
- 4b の 64k 構成は残VRAM が薄い(907 MiB)。`OLLAMA_NUM_PARALLEL` を上げる場合は再計測必須
- 重み量子化は Ollama 公式タグのデフォルト(2b は `Q8_0`、4b は `Q4_K_M`)を使用。より小さい量子化に興味がある場合は Unsloth GGUF(UD-Q3_K_XL 等)の取り込みを検討

---

## NVIDIA GeForce RTX 4080 SUPER (16 GB, Ada Lovelace)

Ollama / KV q8_0 / NUM_PARALLEL=1 での結果。llama.cpp + Qwen3.6-35B-A3B を Claude Code バックエンドにする構成は [experiments/2026-08-claude-code-backend](../experiments/2026-08-claude-code-backend/README.md) を参照。

| 用途                     | モデル                                     | Max ctx | Decode      | 採用理由                                                                             |
| ------------------------ | ------------------------------------------ | ------- | ----------- | ------------------------------------------------------------------------------------ |
| **汎用バランス**         | `qwen3.5:4b-q4_K_M`                        | 262144  | 134.6 tok/s | Coding 1.00 / Summary 0.86 / Needle ✓ / 256K フル ctx。短文も長文もこれ 1 本でカバー |
| **コーディング(短中文)** | `deepseek-coder-v2:16b-lite-instruct-q4_0` | 19456   | 268.2 tok/s | Coding 1.00、Decode 業界最速。19K ctx でも大半の実コーディングタスクは収まる         |
| **会議要約**             | `gemma4:e4b-it-q4_K_M`                     | 131072  | 130.9 tok/s | Coding 1.00 / Summary 0.86 / マルチモーダル対応。文字起こし + スクショ混在も視野     |
| **長コンテキスト要約**   | `qwen3.5:4b-q4_K_M`                        | 262144  | 134.6 tok/s | 256K フル ctx で文字起こし数時間分を一括処理可能                                     |
| **品質重視(中型)**       | `deepseek-r1:8b-0528-qwen3-q4_K_M`         | 116736  | 106.7 tok/s | Qwen3 ベースの 0528 蒸留。Coding 1.00 / Summary 0.75 / Needle ✓ / 116K ctx           |

### 推奨しない / 注意の組合せ

| モデル                                         | 理由                                                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `qwen3.5:9b-q8_0` / `qwen3.5:4b-q8_0`          | q4_K_M に対して **速度・最大 ctx・Summary すべて劣化**。VRAM を使って何も得ていない                    |
| `deepseek-r1:7b-qwen-distill-*`(q4_K_M / q8_0) | Needle 失敗 + Summary 0.53。情報抽出系の用途では使い物にならない。Qwen3 ベースの 8B-0528 か 14B を選ぶ |
| `deepseek-coder-v2:*`(長文用途)                | コード以外、特に日本語長文の検索/要約は苦手。コード補完専用に限定                                      |

---

**投機デコード(2026-10)**: Qwen3.6-35B-A3B の MTP は全層 GPU(IQ3_XXS)で decode +26〜46%、チャット・要約・思考用途で +33〜57%。Claude Code 用途では wall time が縮まず非採用。詳細は [experiments/2026-10-speculative-decoding](../experiments/2026-10-speculative-decoding/README.md)。

---

## Apple M4 Pro (48 GB Unified Memory, MLX)

| 用途                   | モデル                                       | 根拠                                                                                                |
| ---------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **汎用バランス**       | `Qwen3.5-9B-MLX-4bit` (KV q4)                | Coding 0.982 / Summary 0.70 / Needle ✓ / 256K ctx。decode 49 tok/s は対話実用域。KV q4 で長文も軽い |
| **軽量・高速**         | `Qwen3.5-4B-MLX-4bit`                        | decode 83 tok/s と速く、Coding 0.945 と十分。短〜中文の主力                                         |
| **コーディング高速**   | `DeepSeek-Coder-V2-Lite-Instruct-4bit` (MoE) | decode 114 tok/s で最速、Coding 0.963。ただし max ctx は ~49K 止まり(MoE は KV が重い)              |
| **品質最優先(短中文)** | `Qwen3.5-27B-4bit`                           | Coding 1.00 で最高品質。ただし decode 15 tok/s と遅く、長文は不可。品質が要る短中文バッチ向け       |

**KV キャッシュは q4 を既定に**(長 ctx decode +39% / メモリ減 / 品質維持)。

### 推奨しない構成

| 構成                                | 理由                                                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `DeepSeek-R1-Distill-Qwen-32B-4bit` | always-think で抑止不可 + decode 12 tok/s + 長 ctx で OS クラッシュ。48GB Mac では実用構成として成立しない |
| 全 `*-8bit` 重み量子化              | 4bit より帯域を食って decode が落ちるだけ(RTX 評価でも同結論)                                              |
| Mac で 100K+ の長文一括投入         | TTFT が分〜十数分。RAG 等で事前バッチ処理し、推論時の prompt は短くする設計が必須                          |

---

## 別ハードウェアを追加するとき

1. `experiments/_template/` を複製して計測記録を残す
2. 「採用推奨」が確定したら本ファイルに節を追加する(表 + 実験へのリンクのみ。根拠は実験側に書く)
