# Qwen3.8-Flash-Next を 16 GB + 32 GB RAM で回す(2026-11-flash-next-moe-cache)

125B MoE(活性 6B、n-gram 表 51B)を DRAM を増やさずに動かせるか。llama.cpp 本流の expert キャッシュ(`--cpu-moe --moe-cache-mib`)と n-gram 表の遅延読み込み(`--lazy-mode`)で、SATA SSD しか無い機でどこまで出るかを測る。

**進行中。計画は [notes/00-plan.md](notes/00-plan.md)。**

| 項目       | 値                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------ |
| 期間       | 2026-11-07 〜                                                                              |
| 機材       | RTX 4080 SUPER 16 GB / Ryzen 5 3600 / DDR4 32 GB / SATA SSD(BX500)/ Ubuntu 26.04 / 250 W   |
| ランタイム | llama-swap 同梱 llama.cpp build 12182(commit fd27b2034)                                    |
| 方法論     | `/metrics` 差分で tok/s(speculative-decoding 実験の `spec_bench.py` を流用)                |
| 関連実験   | [2026-10-speculative-decoding](../2026-10-speculative-decoding/)(Qwen3.6-35B-A3B の基準値) |

## 結論

(計測後に書く)

## 計測条件

- モデル: unsloth UD-IQ3_XXS(83.6 GB、n-gram 表 28.8 GB 込み)。llama-cache ボリュームの `llama.cpp/flash-next/` に `hf download --local-dir` で配置
- 構成 3 本は `deploy/llama-swap/config.yaml` の `qwen38-flash-next-*`。計測は `scripts/bench.sh`

## 結果

(計測後に書く)

## 落とし穴

| 罠                             | 症状                                                                                       | 対処                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------ |
| `hf download` の既定キャッシュ | 新しい hf CLI は hub キャッシュを別レイアウト(xet)で書き、llama-server の `-hf` が拾えない | `--local-dir` で直置きし `-m` で指す |

## ファイル

| パス               | 内容                                         |
| ------------------ | -------------------------------------------- |
| `notes/00-plan.md` | 問い・構成・計測設計                         |
| `scripts/bench.sh` | 構成 × プロンプト × 反復の計測(GPU 機で実行) |
| `scripts/data/`    | W4(短い推論 3 本)と合成長文 3 本             |
| `results/raw/`     | JSONL と読み込みログ(commit 対象外)          |
