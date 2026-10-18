# 階層型 KV キャッシュ — Phase 2 PoC 設計

[01-research.md](01-research.md) の調査結果と本機の実測値をもとに、何を・どの順で・どう測るかを確定させたものです。

- 作成日: 2026-09-03
- 検証したい仮説: **長 context の Agent セッションを停止・再開するとき、KV を VRAM 外の階層から復元するほうが、再 Prefill するより速い。**

## 0. 二段構えにする

|                  | Track 0                         | Track 1                                                 |
| ---------------- | ------------------------------- | ------------------------------------------------------- |
| エンジン         | **llama.cpp b10679** (導入済み) | **SGLang v0.5.18 HiCache**                              |
| 目的             | **階層の物理を確定させる**      | **粒度と policy を得る**                                |
| 測るもの         | 帯域・復元時間・損益分岐点      | page 単位 partial hit、write/prefetch policy、hit ratio |
| 到達可能 context | **200k tokens** (q4_0 KV)       | 約 145k tokens (fp8 KV)                                 |
| 新規依存         | なし                            | SGLang / FP8 or AWQ チェックポイント                    |
| 見積り           | 1〜2 日                         | 3〜5 日                                                 |

### 0.1 なぜ Track 0 を先にやるのか

「まず Mooncake + vLLM/SGLang の最小構成を動かす」という当初の方針から外れる判断なので、理由を明示します。

1. **最重要比較 (cache restore time vs full prefill time) は、エンジンではなく物理で決まります。** PCIe Gen4 の帯域、SATA の 360 MiB/s、4080 の prefill スループット — この3つが分岐点を決めており、llama.cpp で測っても SGLang で測っても同じ結論になります。**先に物理を確定させれば、その後のあらゆる設計判断の土台になります。**
2. **既存資産がそのまま使えます。** [2026-08-pd-split/config/](../../2026-08-pd-split/config/) の CUDA llama-server コンテナ、[2026-08-pd-split/scripts/pd_run.py](../../2026-08-pd-split/scripts/pd_run.py) の計測ロジック、`timings.cache_n` / `prompt_n` による再 Prefill 判定手法 ([2026-08-pd-split/notes/01-research.md](../../2026-08-pd-split/notes/01-research.md) §4.2) は、すべて本 PoC でも有効です。**セットアップに時間を使わず、初日に数字が出ます。**
3. **200k tokens に到達できるのは llama.cpp だけです** ([01-research.md](01-research.md) §2.2)。vLLM / SGLang は重みが大きく、16 GiB VRAM では 145k 前後が上限になります。目標の 200k を1度でも実測しておく価値があります。
4. **[01-research.md](01-research.md) §5.4 の予測を反証できます。** 損益分岐が f16 で 111k / q8_0 で 46k / q4_0 で 12k という予測が外れるなら、Track 1 の設計前提そのものを組み直す必要があります。**外れる可能性のある仮定を、最も安く検証できる場所で先に潰します。**
5. **llama.cpp はすでに L2 (host DRAM) を持っています。** `--cache-ram` (既定 8192 MiB) と `--cache-idle-slots` (既定 enabled) により、idle slot の DRAM への自動 demotion が実装済みです ([01-research.md](01-research.md) §3.6)。Case C は**新規実装なしで測れます**。

Track 1 は Track 0 が足りない部分 — page 単位の partial prefix hit、設定可能な write/prefetch policy、L3 の自動化 — を埋めるものであり、置き換えではありません。

### 0.2 Mooncake を初手で使わない理由(再掲)

単一ノード・SATA SSD という本機の条件下では、Mooncake の中核価値である Transfer Engine (RDMA / GPUDirect / cross-node zero-copy) が丸ごと無効化されます ([01-research.md](01-research.md) §3.4)。かつ Mooncake は **SGLang HiCache の L3 backend として後から `--hicache-storage-backend file` → `mooncake` へ差し替えられます**。`file` で物理を測ってから差分を見るほうが、切り分けとして厳密です。

**Mooncake の出番は、2台目 + 10GbE を導入して PD分離に戻る段階です。** そこでは既存の PD分離 PoC ([2026-08-pd-split/notes/04-results.md](../../2026-08-pd-split/notes/04-results.md)) で確認した「転送が支配的」という結論に対して、Mooncake の Transfer Engine が直接効きます。

## 1. Track 0 の構成

### 1.1 固定するもの

| 項目            | 値                                                                               | 理由                                                                                                                                                                             |
| --------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| エンジン        | llama.cpp `b10679` = `50f068ff`                                                  | 既存 PD分離 PoC と同一。結果を横に並べられる                                                                                                                                     |
| 実行形態        | Docker ([2026-08-pd-split/config/](../../2026-08-pd-split/config/) を流用・改変) | 再現性。GPU は CDI 経由で問題なく通っている                                                                                                                                      |
| モデル          | **Qwen3-4B-Instruct-2507 Q4_K_M** (sha256 `3605803b...`)                         | native ctx **262,144**。手元で 200k を扱える唯一のモデル。既知 regression に該当しない ([2026-08-pd-split/notes/01-research.md](../../2026-08-pd-split/notes/01-research.md) §6) |
| Flash Attention | **`-fa on` を明示**                                                              | `auto` は禁止。KV レイアウト (`v_trans`) が変わり restore が失敗する                                                                                                             |
| 並列度          | `-np 1`                                                                          | KV が並列数倍に膨らむと比較が壊れる (既存 README の方針と同じ)                                                                                                                   |
| context shift   | `--no-context-shift`                                                             | 途中でトークンが落ちると `cache_n` の解釈が壊れる                                                                                                                                |
| GPU 配置        | `-ngl 99`                                                                        | CPU オフロード禁止 (既存 README の方針)                                                                                                                                          |

### 1.2 変動させる軸

| 軸         | 値                                                |
| ---------- | ------------------------------------------------- |
| KV 量子化  | `q8_0` (主系列) / `f16` / `q4_0`                  |
| context 長 | 10k / 50k / 100k / 150k (q8_0) 、200k (q4_0 のみ) |
| tier       | VRAM / DRAM / SSD                                 |

**q8_0 で 200k が測れないのは VRAM 制約です** ([01-research.md](01-research.md) §2.2: q8_0 の VRAM 上限は約 167k tokens)。200k は q4_0 でのみ到達できます。この非対称は結果表に明記します。

### 1.3 tier の作り方

| tier    | 実装              | 起動オプション                                                  |
| ------- | ----------------- | --------------------------------------------------------------- |
| L1 VRAM | slot の KV        | `-c <n_ctx>`                                                    |
| L2 DRAM | prompt cache      | `--cache-ram 16384 --cache-idle-slots --kv-unified`             |
| L3 SSD  | slot save/restore | `--slot-save-path /kv` (SATA SSD 上のディレクトリを bind mount) |

`--kv-unified` は必須です。これがないと prompt cache が黙って無効化されます。

## 2. 測定ケース

既存の Case A〜E を、本機の階層構成に合わせて再定義します。

| Case                    | 手順                                                                                         | 期待                  | 判定指標                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------- |
| **A: cache なし**       | 未見の prompt を投入                                                                         | full prefill          | `prompt_n` ≈ 全長、`cache_n` ≈ 0                                     |
| **B: VRAM hit**         | 同一 prompt を連続投入                                                                       | prefill skip          | `cache_n` ≈ 全長、`prompt_n` = 1                                     |
| **C: DRAM 退避 → 再開** | ①session1 を prefill → ②session2 (別 prefix) を投入し slot を奪わせる → ③session1 を再投入   | prompt cache から復元 | `cache_n` ≈ 全長、サーバログに `found better prompt with f_keep=...` |
| **D: SSD 退避 → 再開**  | ①prefill → ②`/slots/0?action=save` → ③**`drop_caches`** → ④`/slots/0?action=restore` → ⑤生成 | ファイルから復元      | `restore_ms`、`cache_n` ≈ 全長                                       |
| **E: partial prefix**   | prefix N tokens をキャッシュ済みにし、tail を差し替えて投入                                  | tail のみ prefill     | `cache_n` ≈ N、`prompt_n` ≈ tail 長                                  |
| **D': 再起動を跨ぐ**    | ②の後にサーバを再起動してから ④                                                              | 永続性の確認          | 同上                                                                 |

### 2.1 Case D で page cache を必ず落とす

**これを守らないと Case D は Case C になります。** DRAM 30 GiB のうち十数 GiB が page cache として使えるため、書いた直後の KV ファイルはほぼ確実にメモリ上に残っています。

各試行の ③ で以下を実行します。

```bash
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

**対照として `drop_caches` なしの値も記録します。** 両者の差がそのまま「L3 の実コスト」と「page cache に助けられた場合」の差になり、これ自体が有用なデータです。

### 2.2 Case C を確実に発生させる

`--cache-idle-slots` は「新規タスク到着時に idle slot を prompt cache へ退避する」挙動なので、session2 の投入が必要です。session2 の prefix は session1 と共通接頭辞を持たないようにします (`f_keep < 0.25` の枝刈りに引っかからないよう、独立した prompt にする)。

サーバを `LLAMA_LOG_VERBOSITY=5` 相当で起動し、`server_prompt_cache::update()` が出す `- cache state: N prompts, X MiB` を採取すると、**L2 の実占有量が直接読めます**。

## 3. テスト context の作り方

synthetic な繰り返し文字列では、prefix 共有と圧縮率が実運用と乖離します。Coding Agent の実データ構造に寄せます。

```
[system prompt]           約 2k tokens   固定・全セッション共通 (= 最も再利用されるべき prefix)
[repository context]      約 40〜60%      本リポジトリの実ソース (scripts/, docs/) を連結
[tool output]             約 20〜30%      ls / grep / cat / test 実行結果を模した構造化テキスト
[conversation history]    約 10〜20%      user / assistant の往復
[tail: 新しい user 発話]   約 0.5k tokens  Case E で差し替える部分
```

- **本リポジトリ自身をソースにします。** ライセンス問題がなく、実際の Coding Agent が読む対象そのものです。
- 生成器は `scripts/make_context.py` として作り、`--tokens` で目標長を指定、実トークン数を llama.cpp の `/tokenize` で確定させて記録します。
- **セッション間で system prompt + repository context を共有し、tail だけ変える**構成にします。これが Case E と、将来の prefix 共有の評価軸になります。

## 4. 取得メトリクス

要求された全項目について、取得元を確定させます。

| メトリクス                    | 取得元 (Track 0)                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| TTFT                          | `/completion` の `timings.prompt_ms` + 初トークンまでの wall time                     |
| prefill time                  | `timings.prompt_ms`                                                                   |
| prefill tokens/sec            | `timings.prompt_per_second`                                                           |
| decode tokens/sec             | `timings.predicted_per_second`                                                        |
| **cache hit ratio**           | `cache_n / (cache_n + prompt_n)`                                                      |
| **full recompute time**       | Case A の `prompt_ms`                                                                 |
| **cache restore time**        | Case C: 再投入時の wall time − `prompt_ms` / Case D: `/slots` の `timings.restore_ms` |
| cache lookup time             | サーバログの `looking for better prompt` 〜 `found better prompt` の区間              |
| KV cache size                 | `/slots` の `n_written` / `n_read` (実測)。§理論値と突き合わせる                      |
| VRAM usage                    | `nvidia-smi` (既存 [tools/bench/vram.py](../../../tools/bench/vram.py) を流用)        |
| DRAM usage                    | `/proc/meminfo` + サーバログの `cache state: N prompts, X MiB`                        |
| SSD usage                     | `du -sb <slot-save-path>`                                                             |
| **SSD → DRAM transfer time**  | Case D の `restore_ms` (drop_caches 後)                                               |
| **DRAM → VRAM transfer time** | Case D の `restore_ms` (drop_caches なし) − 上記の差分、および Case C の復元時間      |
| PCIe link gen                 | 転送中の `nvidia-smi --query-gpu=pcie.link.gen.current`                               |
| SSD 実効帯域                  | `n_read / restore_ms`。§1 の 377 MiB/s と突き合わせる                                 |

出力は既存の慣習に合わせ `results/kvtier_runs.jsonl` (生データ) → `results/kvtier.csv` (集計) → `results/kvtier_breakeven.csv` (損益分岐) の3段にします。

### 4.1 最重要の1枚

最終的に出したいのはこの表です。

| tokens | KV型 | A: full prefill | C: DRAM 復元 | C の倍率 | D: SSD 復元 | D の倍率 |
| ------ | ---- | --------------- | ------------ | -------- | ----------- | -------- |

そして **D の倍率が 1.0 を跨ぐ点**が SSD tier の損益分岐です。[01-research.md](01-research.md) §5.4 の予測 (f16: 111k / q8_0: 46k / q4_0: 12k) をこの表で検証します。

## 5. Track 1 の構成 (Track 0 の結果を見てから着手)

| 項目     | 値                                                                                |
| -------- | --------------------------------------------------------------------------------- |
| エンジン | SGLang v0.5.18 (`lmsysorg/sglang` Docker イメージ)                                |
| モデル   | Qwen3-4B-Instruct-2507 の **FP8 または AWQ-Int4** チェックポイント (要入手性確認) |
| KV       | fp8 (Ada sm_89 は FP8 対応)                                                       |
| context  | 10k / 50k / 100k / 145k                                                           |

```
--enable-hierarchical-cache
--hicache-ratio 2
--hicache-size 12                       # L2 = host DRAM 12 GiB
--hicache-io-backend kernel
--hicache-mem-layout page_first
--hicache-storage-backend file          # L3 = SATA SSD 上のディレクトリ
--hicache-storage-prefetch-policy wait_complete
--hicache-write-policy write_through_selective
--page-size 64
```

### 5.1 Track 1 でしか測れないもの

1. **page 単位の partial prefix hit** — L1/L2 をまたぐ連続 prefix の再利用。llama.cpp の LCP 粗粒度では測れません。
2. **write policy の比較** — `write_through` / `write_through_selective` / `write_back` の3通りで、SSD 書き込み量と hit ratio のトレードオフを測ります。**本機は SSD 書き込みが 360 MiB/s しかないため、`write_through` は高確率で足を引っ張ります。ここが最も面白い測定点です。**
3. **prefetch policy の比較** — `best_effort` / `wait_complete` / `timeout`。将来の predictive prefetch の土台になります。
4. **マルチセッション時の hit ratio** — Prometheus `/metrics` から取得。DRAM 12 GiB に 200k セッションが1本しか入らない状況で、L3 がどれだけ hit ratio を押し上げるか。

### 5.2 Track 1 の主要リスク

| リスク                                         | 影響                                | 緩和                                                           |
| ---------------------------------------------- | ----------------------------------- | -------------------------------------------------------------- |
| FP8/AWQ の Qwen3-4B-Instruct-2507 が存在しない | 重み 8 GiB → 実質 87k tokens が上限 | FP16 で走らせ、測定点を 10k/30k/60k/87k に下げる               |
| pinned memory 12 GiB の確保に失敗              | 起動不能                            | `--hicache-size` を 8 に落とす。llama-server を確実に停止      |
| `file` backend が再起動を跨がない              | Case D' が測れない                  | 実測で確認。跨がないならファイルの存在を直接検証する           |
| SGLang が sm_89 + 16GB で不安定                | Track 1 全体が停止                  | Track 0 の結果は独立して残る。これが二段構えにする理由でもある |

## 6. やらないこと(今回)

- **PD分離**。[2026-08-pd-split/](../../2026-08-pd-split/) で実測済み。2台目 + 10GbE を導入する段階で再開します。
- **自作 scheduler**。既存の write policy / prefetch policy / eviction policy の比較までに留めます。
- **Mooncake / Dynamo の導入**。§0.2 の理由により後回し。
- **セマンティック hotness (RUNNING / WAITING_TOOL / …)**。Track 0/1 で「どの tier がどれだけ速いか」が確定してから、そのコスト関数の入力として設計します。**現時点では、コスト関数に入れるべき定数がまだ測れていません。**
- **マルチノード / remote tier**。

## 7. 仮説と反証条件

**測る前に「何が出たら仮説が否定されるか」を書いておきます。**

| #   | 仮説                                                                                   | 反証条件                                                                                                          |
| --- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| H1  | DRAM tier は全 context 長で full prefill より速い                                      | 10k で DRAM 復元が full prefill より遅い。復元スループットが 1 GiB/s を大きく下回る場合に起こりうる               |
| H2  | SSD tier の損益分岐は q8_0 で約 46k tokens                                             | 実測分岐点が予測の ±50% を外れる。prefill の外挿 ([01-research.md](01-research.md) §5.1) が崩れる場合に起こりうる |
| H3  | 200k セッションの SSD 復元 (予測 41.5 s) は full prefill (予測 128.8 s) より明確に速い | 復元が 100 s を超える。SSD のランダム I/O やファイル分割で実効帯域が落ちる場合                                    |
| H4  | 階層化により、DRAM に収まらない本数のセッションを実用速度で扱える                      | L3 からの復元がセッション切替のたびに発生し、体感がむしろ悪化する                                                 |
| H5  | KV 量子化が SSD tier 成立の主レバーである                                              | q4_0 にしても分岐点が下がらない。復元が帯域ではなく固定オーバーヘッドに律速されている場合                         |

**H1 が否定された場合、この PoC の前提そのものが崩れます。** 最初に測るのは H1 です。

## 8. 実施順序

| Phase | 内容                                                                                                                                         | 完了条件                                                            |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 3-a   | 既存 llama-server (VRAM 8.3 GiB / DRAM 9.4 GiB 占有) を停止し、[2026-08-pd-split/config/](../../2026-08-pd-split/config/) を階層計測用に改変 | `--cache-ram` / `--slot-save-path` 付きで起動し `/slots` が応答する |
| 3-b   | `scripts/make_context.py` で 10k〜200k のテスト context を生成                                                                               | 実トークン数が `/tokenize` で確定している                           |
| 3-c   | Case A / B / C を q8_0 で計測                                                                                                                | **H1 の判定**                                                       |
| 4     | Case D (drop_caches あり/なし)、Case D' を追加                                                                                               | **H2 / H3 の判定**                                                  |
| 5     | KV 型 3 種 × context 長の掃引、Case E                                                                                                        | **H5 の判定**、損益分岐表の完成                                     |
| 6     | 結果を `docs/kv-tier/03-results.md` にまとめ、Track 1 の要否を判断                                                                           | Track 1 へ進むかの意思決定                                          |
| 7     | (条件付き) Track 1: SGLang HiCache                                                                                                           | page 単位 hit と write policy の比較                                |

## 9. 成果物の予定

| パス                                                  | 内容                                   |
| ----------------------------------------------------- | -------------------------------------- |
| `config/`                                             | 階層計測用の llama-server Docker 構成  |
| `scripts/make_context.py`                             | Coding Agent 相当のテスト context 生成 |
| `scripts/run_case.py`                                 | Case A〜E の1試行実行と計測            |
| `scripts/sweep.sh`                                    | KV 型 × context 長の掃引               |
| `scripts/aggregate.py`                                | CSV 出力と損益分岐の算出               |
| `results/kvtier_runs.jsonl`                           | 生データ                               |
| `results/kvtier.csv` / `results/kvtier_breakeven.csv` | 集計                                   |
| `docs/kv-tier/03-results.md`                          | 実測結果                               |
