# 階層型 KV キャッシュ (VRAM / DRAM / SSD) — Phase 1 事前調査

長 context の Agent セッションを一度停止し、再開したときの TTFT を「再 Prefill」ではなく「KV の復元」で賄えるか。その仮説を検証するために、既存 OSS がどこまでを実装済みかを調べ、本機のハードウェア上限を実測した記録です。

- 調査日: 2026-09-03
- 前提資産: [2026-08-pd-split/notes/01-research.md](../../2026-08-pd-split/notes/01-research.md) / [2026-08-pd-split/notes/04-results.md](../../2026-08-pd-split/notes/04-results.md) (llama.cpp の slot save/restore による PD分離 PoC、実測済み)

## 0. 結論(先に3つ)

1. **本機の SSD は NVMe ではなく SATA(Crucial BX500)で、実測 R/W ともに約 360〜377 MiB/s。** これが SSD tier の性能を一意に決めます。設計はこの数字から逆算するべきで、NVMe を前提にした OSS の売り文句(GPUDirect Storage / NVMe-oF)は本機では効きません。
2. **DRAM tier は全 context 長で圧勝、SSD tier は損益分岐が KV 量子化に強く依存します。** 解析すると f16 で約 111k tokens、q8_0 で約 46k tokens、q4_0 で約 12k tokens が分岐点です (§5)。**KV 量子化が SSD tier を成立させる唯一のレバーです。**
3. **第一候補は Mooncake ではありません。** 単一ノード・SATA・16GB VRAM という制約下では、SGLang HiCache (L1/L2/L3 = GPU/host/file) が最も素直に目的を満たします。Mooncake は「2台目 + 10GbE」に進んだ段階で HiCache の L3 backend として差し替えるのが正しい順序です (§6)。

## 1. ハードウェア実測値

設計の全ての数字がここから出るため、推測ではなく実測しました。

| 項目         | 実測値                                                               | 測定方法           |
| ------------ | -------------------------------------------------------------------- | ------------------ |
| GPU          | GeForce RTX 4080 SUPER / VRAM 16,376 MiB / sm_89                     | `nvidia-smi`       |
| PCIe         | Max Gen4 x16 (Host Max 4 / Device Max 4)。idle 時は Gen1 に降段      | `nvidia-smi -q`    |
| マザーボード | MSI MPG X570 GAMING PLUS                                             | DMI                |
| CPU          | Ryzen 5 3600 (6C/12T, Zen2)                                          | `lscpu`            |
| DRAM         | 30,987 MiB (DDR4)。現時点の空き約 16 GiB                             | `free`             |
| ストレージ   | **Crucial BX500 1TB / SATA / 非NVMe**。他に HDD 2台                  | `lsblk -o TRAN`    |
| SSD 順次読み | **377 MiB/s** (O_DIRECT, 3 GiB)                                      | 自作スクリプト実測 |
| SSD 順次書き | **360 MiB/s** (O_DIRECT, 6 GiB, fsync込。SLC 枯渇による落ち込みなし) | 同上               |
| 空き容量     | 479 GiB (`/` = ext4 on sda2)                                         | `df`               |
| Docker       | 29.7.2 / nvidia runtime + CDI 構成済み                               | `docker info`      |

### 1.1 この構成が意味すること

- **DRAM tier と SSD tier の速度差は約 50〜70 倍**です。PCIe Gen4 x16 の pinned 転送(実効 20 GB/s 級)に対し SATA は 0.36 GB/s。「DRAM と SSD は同じ『GPU外』ではなく、まったく別の階層」として設計する必要があります。
- BX500 は DRAM-less SSD ですが、6 GiB の連続書き込みで速度低下が見られませんでした。**SLC キャッシュ枯渇は本 PoC の規模(1試行あたり最大 15 GiB 程度)では顕在化しない**見込みです。ただし掃引を連続実行する場合は要再確認です。
- PCIe の `Current: 1` は idle 時の降段であり異常ではありません。**実負荷時に Gen4 へ復帰することは実測で確認します** (§9-1)。仮に Gen1 のままなら DRAM tier の速度が 1/8 になり、結論が変わります。

## 2. モデルと KV サイズ (GGUF メタデータ実測)

ローカルの GGUF からメタデータを直接読み出して確認しました。

| モデル                            | 層  | kv_heads | head_dim | native ctx  | f16           | q8_0         | q4_0         |
| --------------------------------- | --- | -------- | -------- | ----------- | ------------- | ------------ | ------------ |
| **Qwen3-4B-Instruct-2507 Q4_K_M** | 36  | 8        | 128      | **262,144** | 144.0 KiB/tok | 76.5 KiB/tok | 40.5 KiB/tok |
| Qwen3-8B Q4_K_M                   | 36  | 8        | 128      | 40,960      | 144.0         | 76.5         | 40.5         |
| Qwen3-14B Q4_K_M                  | 40  | 8        | 128      | 40,960      | 160.0         | 85.0         | 45.0         |

**4B と 8B は KV サイズが完全に同一**です (層数・KV ヘッド数が同じ)。したがって「KV を小さくするために小さいモデルを選ぶ」戦略は Qwen3 系では機能しません。効くのは KV 量子化だけです。

**native ctx 262,144 を持つのは Qwen3-4B-Instruct-2507 のみ**です。目標の 200k tokens を扱えるモデルは、手元にはこれしかありません。

### 2.1 context 長ごとの KV 容量

| tokens | f16       | q8_0      | q4_0     |
| ------ | --------- | --------- | -------- |
| 10k    | 1.37 GiB  | 0.73 GiB  | 0.39 GiB |
| 50k    | 6.87 GiB  | 3.65 GiB  | 1.93 GiB |
| 100k   | 13.73 GiB | 7.30 GiB  | 3.86 GiB |
| 200k   | 27.47 GiB | 14.59 GiB | 7.72 GiB |

### 2.2 VRAM に収容できる上限 — ここが最大の制約

**復元後に Decode するには、そのセッションの KV 全体が VRAM に載っている必要があります。** 階層化しても、アクティブなセッションの上限は VRAM が決めます。

VRAM 予算 = 15.5 GiB (デスクトップ描画ぶんを除く) − 重み − 計算バッファ約 1.0 GiB

| 構成                          | 重み       | KV 予算  | f16       | q8_0                  | q4_0                      |
| ----------------------------- | ---------- | -------- | --------- | --------------------- | ------------------------- |
| llama.cpp Qwen3-4B **Q4_K_M** | 2.33 GiB   | 12.2 GiB | 88.8k tok | **167k tok**          | **262k tok (モデル上限)** |
| SGLang/vLLM Qwen3-4B **FP8**  | 約 4.0 GiB | 10.0 GiB | 72.8k tok | **145k tok** (fp8 KV) | —                         |
| SGLang/vLLM Qwen3-4B **FP16** | 約 8.0 GiB | 6.0 GiB  | 43.7k tok | 87.4k tok (fp8 KV)    | —                         |

**目標の 200k tokens に到達できるのは llama.cpp + q8_0/q4_0 KV だけです。** vLLM / SGLang は重みが GGUF ほど小さくならないため、現実的な上限は 145k tokens 前後になります。これは PoC の測定点設計に直接効きます (§9)。

## 3. 各 OSS の実装状況

### 3.1 SGLang HiCache (v0.5.18, 2026-08)

RadixAttention を 3 階層へ拡張したものです。**本 PoC の要求とほぼ一対一で対応します。**

| 項目               | 実装                                                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| 階層               | L1 = GPU HBM / L2 = host DRAM (インスタンス専有) / L3 = 外部ストレージ                                               |
| L3 backend         | **`file` (ローカルFS)** / `mooncake` / `hf3fs` / `nixl` / `aibrix`                                                   |
| 粒度               | page (連続トークンの束)。`--page-size` で指定                                                                        |
| partial prefix hit | **対応**。HiRadixTree が L1/L2 をまたいで「連続する prefix」を返す                                                   |
| demotion policy    | `--hicache-write-policy` = `write_through` / `write_through_selective` (頻度ベース) / `write_back` (eviction 時のみ) |
| promotion policy   | `--hicache-storage-prefetch-policy` = `best_effort` / `wait_complete` / `timeout`                                    |
| eviction           | LRU 相当 (ドキュメントに明記なし。ソース確認が必要)                                                                  |
| メモリレイアウト   | `layer_first` / `page_first` / `page_first_direct` (CPU-GPU 転送最適化)                                              |
| 転送経路           | `--hicache-io-backend` = `direct` / `kernel` (GPU アシスト)                                                          |
| RDMA / GPUDirect   | **core 機能には不要**。`mooncake` backend を選んだときだけ RDMA が要る                                               |
| 単一ノード         | 想定内。L2 は元々「ノードローカル・インスタンス専有」と定義されている                                                |

`--hicache-write-policy write_back` は、まさに「VRAM から溢れたら DRAM/SSD へ落とす」demotion そのものです。**将来のコストベース policy を差し込む場所として、この write policy と prefetch policy が拡張点になります。**

### 3.2 vLLM native OffloadingConnector (v0.28.0, 2026-08)

vLLM 0.11.0 で導入、0.12.0 で大幅高速化。**外部依存ゼロで CPU tier + fs tier が使えます。**

| 項目     | 実装                                                                                                                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 階層     | CPU (pinned host memory、常設) + secondary tiers = `fs` / object store / P2P                                                         |
| 粒度     | block。0.12.0 で「全層の KV を1つの物理連続ブロック」に再構成し、実効ブロックサイズが KB 単位から **0.5〜2.5 MB** へ拡大             |
| eviction | **LRU / ARC (既定 ARC)** — LRU 以外の policy が既に選べます                                                                          |
| 転送     | `cudaMemcpyAsync` による GPU DMA。0.11.0 以降は非同期 offload                                                                        |
| 永続性   | **fs tier は同一 `root_dir` でプロセス再起動を跨いで再利用可能**。ただし model / block_size / parallelism / dtype が一致していること |
| 制約     | secondary tier は GPU メモリへ直接アクセスできず、**必ず CPU を経由**する                                                            |
| 実測     | H100 / Llama-3.1-8B で TTFT 最大 4x 短縮、スループット 5x                                                                            |

設定例:

```json
{
  "kv_connector": "OffloadingConnector",
  "spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": 10737418240,
  "secondary_tiers": [{ "type": "fs", "root_dir": "/mnt/kv_cache" }]
}
```

### 3.3 LMCache (2026-08 時点で活発に開発中)

機能面では最も豊富です。

- backend: CPU RAM / ローカルディスク(SSD/NVMe) / Redis・Valkey / Mooncake / InfiniStore / S3 / NIXL / GDS / Device-DAX
- **CacheBlend による非 prefix 再利用** — prompt 中の任意位置の KV ブロックを再利用し、品質回復のために一部トークンだけ再計算する。**これは他のどの実装にもありません**
- vLLM とは**別デーモンプロセス**として動くため、推論エンジンがクラッシュしても KV が失われない
- 2026-01 に multi-node P2P CPU 共有が experimental → production へ昇格
- 2026-08 に保存時暗号化、Device-DAX 対応

**将来の Case E (partial prefix reuse) を本気でやるなら LMCache 一択**ですが、構成要素が増えるぶん初手には重いです。

### 3.4 Mooncake Store (v0.3.13, 2026-08)

Kimi の本番基盤、FAST 2025 Best Paper。**production-ready であることは疑いありません。** ただし本機との相性は別問題です。

| 項目           | 実装                                                                                 |
| -------------- | ------------------------------------------------------------------------------------ |
| 階層           | GPU HBM (L1) / DRAM (L2) / **ローカル NVMe SSD (L3)**                                |
| 粒度           | block                                                                                |
| offload policy | `eager` (常時バックグラウンドフラッシュ) / `watermark` (DRAM 使用率が閾値超過時のみ) |
| 単一ノード     | 対応 (`kv_both`)。RDMA は必須ではなく TCP フォールバックあり                         |
| 実測           | DGX A100 で SSD offload により hit ratio 36% → 84%、TTFT 16s → 9.4s                  |
| 成熟度         | early production。GPUDirect Storage / NVMe-oF は「予定」                             |

**本機で採用しない理由は3つです。**

1. Mooncake の中核価値は Transfer Engine (RDMA / zero-copy / cross-node) にあり、**単一ノードではその大半が無効化されます**。
2. SSD tier が **NVMe 前提**の設計です。実測 360 MiB/s の SATA では、NVMe 向けに作られた並列 I/O 設計の投資対効果が出ません。
3. **SGLang HiCache の L3 backend として後から差し替えられます。** まず `file` backend で階層の物理を測り、その上で Mooncake に替えて差分を見るほうが、切り分けとして正しい順序です。

### 3.5 NVIDIA Dynamo KVBM (v1.4.2, 2026-08)

G1 (GPU) → G2 (host pinned) → G3 (local NVMe) → G4 (remote/object) という 4 階層で、**ユーザーが描いた将来像に概念上最も近い**実装です。G3 は NIXL descriptor でファイルオフセットを公開し zero-copy / GDS に対応します。vLLM の `kv_connector_extra_config` から使えます。

ただし Dynamo は etcd / NATS を伴う K8s 志向のオーケストレーション基盤全体であり、1台の検証機には過剰です。**「目指すアーキテクチャの参照実装」として読む価値は高いが、最初の PoC の土台には向きません。**

### 3.6 llama.cpp (b10679 = 既存資産)

既存の PD分離 PoC で使っている構成が、実は**すでに 2 階層を持っています**。ソースを直接確認しました。

| 階層                 | 実装                                                                                         | 確認箇所                        |
| -------------------- | -------------------------------------------------------------------------------------------- | ------------------------------- |
| L1 (VRAM)            | slot の KV                                                                                   | —                               |
| **L2 (host DRAM)**   | `-cram, --cache-ram N` (既定 **8192 MiB**、-1 で無制限、0 で無効)                            | `tools/server/README.md:169`    |
| L2 への自動 demotion | `--cache-idle-slots` (**既定 enabled**)。新規タスク到着時に idle slot を prompt cache へ退避 | `tools/server/README.md:171`    |
| L3 (SSD)             | `/slots/{id}?action=save                                                                     | restore` による手動ファイル保存 | 既存 PoC で実測済み |

L2 の実装 (`tools/server/server-task.cpp:1793` `server_prompt_cache::load` / `:1870` `update`):

- 保持は `std::list<server_prompt_cache_state>`。**eviction は `states.pop_front()` = 挿入順の最古から (FIFO)**。ヒットしたエントリは list から erase されて再登録されるため、実効的には LRU 相当です。
- マッチングは **prompt 全体の LCP (最長共通接頭辞)**。`f_keep = lcp/cached_len`、`f_sim = lcp/new_len` を計算し、`f_keep >= 0.25` を満たすもののうち両方が改善するエントリを選びます。
- **つまり partial prefix reuse は動きますが、page/block 単位の radix ではなく「1エントリ丸ごと復元 → LCP 以降を再 Prefill」という粗い粒度**です。複数エントリの部分結合はできません。
- **プロセス再起動は跨げません** (in-process の `std::vector<uint8_t>`)。永続化するには slot save で明示的にファイルへ書く必要があります。

## 4. 比較表

|                        | Mooncake        | LMCache        | Dynamo KVBM      | vLLM native | SGLang HiCache   | llama.cpp           |
| ---------------------- | --------------- | -------------- | ---------------- | ----------- | ---------------- | ------------------- |
| VRAM/DRAM/SSD の3層    | ○               | ○              | ◎ (4層)          | ○ (CPU+fs)  | ◎                | △ (L3は手動)        |
| 単一ノードでの価値     | △               | ○              | △                | ◎           | ◎                | ◎                   |
| SATA SSD との相性      | × (NVMe前提)    | ○              | × (NVMe/GDS前提) | ○           | ○                | ○                   |
| partial prefix hit     | ○               | ◎ (非prefixも) | ○                | ○           | ◎                | △ (LCP粗粒度)       |
| eviction policy の選択 | eager/watermark | 複数           | 複数             | **LRU/ARC** | write policy 3種 | FIFO/LRU相当のみ    |
| 再起動を跨ぐ永続性     | ○               | ◎ (別デーモン) | ○                | ○ (fs tier) | ○ (file backend) | × (手動保存のみ)    |
| 導入コスト             | 高              | 中             | 高               | 低          | **低〜中**       | **ゼロ (導入済み)** |
| 16GB VRAM で 200k 到達 | ×               | ×              | ×                | ×           | ×                | **○**               |
| 既存資産の再利用       | ×               | ×              | ×                | ×           | ×                | **◎**               |

## 5. 損益分岐の事前計算 — 実測で置き換える

### 5.1 full prefill の推定モデル

[2026-08-pd-split/notes/04-results.md](../../2026-08-pd-split/notes/04-results.md) §4.2 の 4080 SUPER 実測 (1k〜16k) から、トークンあたり Prefill 時間を1次で近似しました。

```
T(N) = N × (0.0758 + 2.84e-6 × N)  [ms]
```

| tokens | full prefill 推定 |
| ------ | ----------------- |
| 10k    | 1.0 s             |
| 50k    | 10.9 s            |
| 100k   | 36.0 s            |
| 200k   | 128.8 s           |

**16k からの外挿であり、実測で置き換えます。** アテンションが二次で効くため長いほど不確実です。

### 5.2 復元コスト

- DRAM tier: llama.cpp の実測から **restore 2.66 GiB/s / save 2.13 GiB/s** (1,196.5 MiB を 450 ms / 548 ms、[2026-08-pd-split/notes/04-results.md](../../2026-08-pd-split/notes/04-results.md) §5.3)。PCIe Gen4 の理論値より1桁遅く、`llama_io` が一時バッファを経由する実装 (`llama-context.cpp` の `write_tensor`) に律速されています。
- SSD tier: **360 MiB/s** (§1 実測) が上限。

### 5.3 q8_0 KV での比較

| tokens | KV        | full prefill | DRAM restore | 比        | SSD restore | 比               |
| ------ | --------- | ------------ | ------------ | --------- | ----------- | ---------------- |
| 10k    | 0.73 GiB  | 1.0 s        | 0.27 s       | 3.9x      | 2.1 s       | **0.50x (負け)** |
| 50k    | 3.65 GiB  | 10.9 s       | 1.37 s       | 8.0x      | 10.4 s      | 1.05x (ほぼ同着) |
| 100k   | 7.30 GiB  | 36.0 s       | 2.74 s       | 13.1x     | 20.8 s      | 1.73x            |
| 200k   | 14.59 GiB | 128.8 s      | 5.48 s       | **23.5x** | 41.5 s      | **3.10x**        |

### 5.4 SSD tier の損益分岐点(解析解)

`0.0758 + 2.84e-6·N = KV_bytes_per_token / 360 MiB/s` を解いたものです。

| KV 型 | 1 token   | SSD 復元     | **損益分岐**       |
| ----- | --------- | ------------ | ------------------ |
| f16   | 144.0 KiB | 0.391 ms/tok | **約 111k tokens** |
| q8_0  | 76.5 KiB  | 0.208 ms/tok | **約 46k tokens**  |
| q4_0  | 40.5 KiB  | 0.110 ms/tok | **約 12k tokens**  |

**これが本調査で最も重要な予測です。** SATA SSD であっても、KV を q4_0 まで落とせば 12k tokens から SSD tier が再計算より速くなります。逆に f16 のままだと 111k tokens まで SSD tier は損です。

DRAM tier は全長で勝ちます (10k で 3.9x、200k で 23.5x)。分岐点は存在しません。

## 6. 単一ノードでの構成案

```
       Qwen3-4B-Instruct-2507 (native ctx 262,144)
                    │
  L1  VRAM  12.2 GiB  ← q8_0 で 167k tok / q4_0 で 262k tok
       ↑↓  実測 2.1〜2.7 GiB/s (llama.cpp) / 理論 20 GB/s (PCIe Gen4 x16)
  L2  DRAM  12〜16 GiB  ← q8_0 で 1 セッション分 (200k) がやっと
       ↑↓  実測 0.36 GiB/s (SATA)  ← ここが 50〜70 倍の壁
  L3  SATA SSD  479 GiB 空き  ← q8_0 なら 200k セッションを 32 本
```

この図から出る設計上の含意:

- **L2 は「1セッション分の一時退避先」にしかなりません。** 30 GiB しかない DRAM に、200k セッションを複数常駐させることはできません。マルチセッションを扱うには L3 が必須です。これが本 PoC で L3 を作る動機です。
- **L2 ↔ L3 の 50〜70 倍差は、promotion / demotion の判断を「単純 LRU では足りない」領域に押し上げます。** ここがユーザーの構想 (コストベース eviction) が実際に効く場所です。逆に L1 ↔ L2 は速すぎて、賢い policy を作る動機が薄いです。

## 7. 15 の設問への回答

| #   | 設問                                              | 回答                                                                                                                                                                     |
| --- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Mooncake Store の production-ready 度             | 高い (Kimi 本番稼働、FAST 2025 Best Paper)。ただし SSD offloading 機能自体は "early production"                                                                          |
| 2   | VRAM/DRAM/SSD の階層化サポート                    | Mooncake: L1/L2/L3 あり。SGLang HiCache も L1/L2/L3。Dynamo KVBM は G1〜G4。vLLM native は CPU+fs。LMCache は多数                                                        |
| 3   | promotion / demotion の粒度                       | いずれも block / page 単位。vLLM 0.12.0 は全層を1物理ブロックに束ねて 0.5〜2.5 MB。llama.cpp のみ「prompt 全体」の粗粒度                                                 |
| 4   | eviction policy                                   | Mooncake: eager / watermark。vLLM: **LRU / ARC**。SGLang: write_through / write_through_selective / write_back + LRU 相当。llama.cpp: FIFO(実効LRU)                      |
| 5   | LRU 以外への拡張性                                | vLLM は ARC が既にある。SGLang は write policy と prefetch policy が明示的な拡張点。**いずれもプラグイン境界が存在する**                                                 |
| 6   | cache hit 時に prefill をどこまで skip できるか   | ヒットした prefix 全体を skip。llama.cpp 実測では残 Prefill 1 token まで落ちる ([2026-08-pd-split/notes/04-results.md](../../2026-08-pd-split/notes/04-results.md) §5.3) |
| 7   | partial prefix hit                                | SGLang HiCache ◎ (L1/L2 をまたぐ連続 prefix)、LMCache ◎ (非 prefix も)、vLLM ○、llama.cpp △ (LCP による1エントリ丸ごと)                                                  |
| 8   | cache block 単位の再利用                          | SGLang / vLLM / LMCache / Mooncake は可。llama.cpp は不可                                                                                                                |
| 9   | vLLM / SGLang 統合方法                            | vLLM: `kv_transfer_config` の KV connector (OffloadingConnector / LMCacheConnector / NixlConnector)。SGLang: `--hicache-storage-backend`                                 |
| 10  | 単一ノードでの利用価値                            | vLLM native / SGLang HiCache / LMCache は高い。Mooncake / Dynamo は低い(分散前提の投資が回収できない)                                                                    |
| 11  | SSD を KV tier として使えるか                     | 使える。SGLang `file`、vLLM `fs`、LMCache ローカルディスク、Mooncake NVMe。**ただし本機は SATA で 360 MiB/s**                                                            |
| 12  | SSD → DRAM → VRAM の copy 回数                    | **最低 2 回**。vLLM は「secondary tier は GPU に直接アクセスできず必ず CPU 経由」と明記。GPUDirect Storage を使えば 1 回にできるが、**本機は SATA のため GDS の対象外**  |
| 13  | zero-copy / GPUDirect / RDMA は単一ノードで効くか | **ほぼ効きません。** RDMA はノード間、GDS は NVMe が前提。本機で効く最適化は pinned memory と `cudaMemcpyAsync` のみ                                                     |
| 14  | プロセス再起動を跨ぐ永続性                        | vLLM fs tier ○ / SGLang file backend ○ (要実測) / LMCache ◎ (別デーモン) / llama.cpp × (手動 slot save のみ)                                                             |
| 15  | session 単位でなく prefix 単位で再利用できるか    | SGLang / vLLM / LMCache は prefix (page/block) 単位。llama.cpp は session 単位に近い                                                                                     |

## 8. 制約(本機固有)

1. **SSD が SATA である。** NVMe 前提の OSS の最適化 (GDS / NVMe-oF / 高並列 I/O) は無効。360 MiB/s が天井。
2. **VRAM 16 GiB。** 200k tokens に到達するには KV 量子化が必須で、vLLM / SGLang では 145k 前後が現実的上限 (§2.2)。
3. **DRAM 30 GiB。** L2 tier に割けるのは 12〜16 GiB 程度。しかも現在 llama-server が RSS 9.4 GiB を占有しており、**PoC 前に停止が必要**です。
4. **VRAM も現在 8.3 GiB を llama-server が占有中。** 同上。
5. **DDR4 デュアルチャネル (Ryzen 3600)** の帯域は約 45 GB/s。L2 の pinned memory 確保と GPU 転送が同じ帯域を食い合います。
6. **単一 GPU。** テンソル並列の検証はできません。

## 9. 未確定事項(実測で解消する)

1. **PCIe が実負荷時に Gen4 へ復帰するか。** 転送中に `nvidia-smi --query-gpu=pcie.link.gen.current` を取得して確認します。Gen1 のままなら DRAM tier の結論が変わります。
2. **§5.1 の prefill 外挿が 50k〜200k で成立するか。** 16k からの外挿であり、最も不確実な仮定です。
3. **SGLang HiCache の `file` backend が再起動を跨いで実際に再利用されるか。** ドキュメントに明記がありません。
4. **SGLang HiCache の L2 eviction が本当に LRU か。** ソース確認が必要です。
5. **Case D の測定で page cache が混入しないか。** 15 GiB の KV を書いた直後に読むと、DRAM 30 GiB のうち相当量が page cache に残り、**Case D が実質 Case C になります**。`drop_caches` か O_DIRECT による切り分けが必須です (§10)。
6. **q4_0 KV の品質劣化。** 損益分岐を 12k まで下げる最大のレバーですが、品質は未評価です。既存の [docs/evaluation.md](../../../docs/evaluation.md) の採点系で測れます。
7. **Qwen3-4B-Instruct-2507 の FP8 / AWQ チェックポイントの入手性。** vLLM / SGLang を使う場合に必要です。
8. **BX500 の SLC キャッシュが掃引の連続実行で枯渇しないか。**

## 10. 計測上の落とし穴(先に潰しておく)

| 罠                                 | 症状                                                                                                                           | 対策                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **page cache 混入**                | Case D (SSD) が Case C (DRAM) と同じ速度に見える                                                                               | 各試行前に `sync && echo 3 > /proc/sys/vm/drop_caches`。または O_DIRECT を使う backend を選ぶ                    |
| **PCIe の idle 降段**              | 最初の1回だけ転送が遅い                                                                                                        | 計測前にウォームアップ転送を挟み、`pcie.link.gen.current` を記録                                                 |
| **既存 llama-server の占有**       | VRAM 8.3 GiB / DRAM 9.4 GiB が失われる                                                                                         | PoC 前に停止する                                                                                                 |
| **HTTP 転送の単一 TCP フロー問題** | 既存 PD分離 PoC で 3〜4 倍の差が出た ([2026-08-pd-split/notes/04-results.md](../../2026-08-pd-split/notes/04-results.md) §5.4) | 本 PoC はローカル FS なので該当しないが、将来のリモート tier では再発する                                        |
| **KV 型の不一致**                  | restore が黙って失敗、または出力だけ壊れる                                                                                     | [2026-08-pd-split/notes/01-research.md](../../2026-08-pd-split/notes/01-research.md) §3 の照合条件をそのまま適用 |
| **`-fa auto`**                     | バックエンドごとに解決結果が変わり restore が失敗                                                                              | `-fa on` を明示                                                                                                  |

## 参考

- [SGLang HiCache 設計ドキュメント](https://docs.sglang.io/advanced_features/hicache_design.html) / [LMSYS ブログ](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [vLLM KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) / [vLLM ブログ: Inside vLLM's New KV Offloading Connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- [LMCache](https://github.com/lmcache/lmcache) / [LMCache ブログ](https://blog.lmcache.ai/)
- [Mooncake](https://kvcache-ai.github.io/Mooncake/) / [Scaling KV Cache Beyond Memory with Mooncake SSD Offloading](https://kvcache.ai/blog/scaling-kv-cache-beyond-memory/) / [Mooncake x SGLang HiCache 設計](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
- [NVIDIA Dynamo KVBM components](https://docs.nvidia.com/dynamo/latest/architecture/kvbm_components.html)
- [llama.cpp: Mastering Host-Memory Prompt Caching in llama-server (Discussion #20574)](https://github.com/ggml-org/llama.cpp/discussions/20574) / [PR #16391 (--cache-ram)](https://github.com/ggml-org/llama.cpp/pull/16391)
