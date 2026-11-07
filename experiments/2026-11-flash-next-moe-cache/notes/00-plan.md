# 計画: Qwen3.8-Flash-Next を 16 GB VRAM + 32 GB RAM で動かす(expert キャッシュ方式)

## 問い

1. 125B MoE(活性 6B)を、DRAM を増やさずに手元で動かせるか。llama.cpp 本流の `--cpu-moe` + `--moe-cache-mib`(頻出 experts を VRAM に置く)と `--lazy-mode`(51B の n-gram 表を mmap から行単位に読む)で、本体 55 GB のうちどれだけがメモリに乗り、どれだけが SATA SSD から来るか
2. decode / prefill の tok/s と TTFT は常用に耐えるか。比較対象は現行の Qwen3.6-35B-A3B(IQ3_XXS、全部 VRAM)
3. VRAM キャッシュの大きさ(0 / 9 / 12 GiB)で速度がどう変わるか。Strata の比較記事では 16 GB キャッシュでヒット率 96%、20 GB を切ると崖

## 前提と制約

- GPU 機: RTX 4080 SUPER 16 GB、Ryzen 5 3600、DDR4 32 GB(デュアルチャネル、実効 35〜40 GB/s)、システムディスクは SATA SSD(Crucial BX500、0.5 GB/s)。NVMe なし
- モデル: unsloth `Qwen3.8-Flash-Next-GGUF` UD-IQ3_XXS(4 分割 83.6 GB。うち 00004 の 28.8 GB が n-gram 表 IQ4_NL、本体は約 55 GB)
- ランタイム: llama-swap 同梱の llama.cpp build 12182(`qwen4exp` 対応、`--moe-cache-mib`、`--lazy-mode`)。MTP ヘッドは本流未対応なので今回は使わない(unsloth のフォークなら 1.3〜1.7 倍)
- 電力制限 250 W(手動設定、再起動で 320 W に戻る)
- 計測中は ComfyUI コンテナを止める(RAM 22 GB を占有するため)

## 構成(`deploy/llama-swap/config.yaml`)

| 名前                                | experts | VRAM キャッシュ | n-gram 表 | ctx / KV   |
| ----------------------------------- | ------- | --------------- | --------- | ---------- |
| `qwen38-flash-next-iq3xxs`          | CPU     | 9 GiB           | lazy mmap | 32k / q8_0 |
| `qwen38-flash-next-iq3xxs-cache12g` | CPU     | 12 GiB          | 同上      | 同上       |
| `qwen38-flash-next-iq3xxs-nocache`  | CPU     | なし            | 同上      | 同上       |

## 計測

- `scripts/bench.sh`(GPU 機で実行)。構成ごとに llama-swap で読み込み → W4(短い推論 3 本、thinking)と合成長文(1k / 4k / 12k 文字、要約)を 2 反復。1 リクエストごとに `/metrics` 差分から prefill / decode tok/s、`nvidia-smi` で VRAM ピークと電力、`/proc/diskstats` で SATA SSD の読み出し量を記録
- 見るもの: 読み込み時間、decode tok/s(目標 5 以上)、prefill tok/s(2 万トークン入力の TTFT を見積もる)、ディスク読み出し量(ミス率の代理)、RAM のページキャッシュ占有
- 判定: decode 5 tok/s 未満か prefill 50 tok/s 未満なら「NVMe 追加か DRAM 増設が要る」。NVMe の効果は SATA の読み出し量から見積もる

## 範囲外

- 品質評価(別途。IQ3_XXS の 6B 活性がどの程度か)
- Strata / Lean-QwenFlash(NVMe 前提。導入したら別実験)
