# Qwen3.8-Flash-Next を 16 GB + 32 GB RAM で回す(2026-11-flash-next-moe-cache)

125B MoE(活性 6B、n-gram 表 51B)を DRAM を増やさずに動かせるか。llama.cpp 本流の expert キャッシュ(`--cpu-moe --moe-cache-mib`)と n-gram 表の遅延読み込み(`--lazy-mode`)で、SATA SSD しか無い機でどこまで出るかを測る。

**完了(2026-11-08)。計画は [notes/00-plan.md](notes/00-plan.md)。**

| 項目       | 値                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------ |
| 期間       | 2026-11-07 〜                                                                              |
| 機材       | RTX 4080 SUPER 16 GB / Ryzen 5 3600 / DDR4 32 GB / SATA SSD(BX500)/ Ubuntu 26.04 / 250 W   |
| ランタイム | llama-swap 同梱 llama.cpp build 12182(commit fd27b2034)                                    |
| 方法論     | `/metrics` 差分で tok/s(speculative-decoding 実験の `spec_bench.py` を流用)                |
| 関連実験   | [2026-10-speculative-decoding](../2026-10-speculative-decoding/)(Qwen3.6-35B-A3B の基準値) |

## 結論

**今の機(SATA SSD、RAM 32 GB)では動くが常用は無理。原因はストレージ帯域で、DRAM ではない。** 9 GiB の expert キャッシュ構成で、初見のプロンプトは prefill 中央値 3.8 tok/s・decode 6.9 tok/s。生成中は SATA SSD の読み出しが 230〜290 MB/s(実効上限)に張り付き、GPU 使用率は 0〜17%。同じ文の再送(experts がキャッシュ済み)なら 41.6 tok/s 出るので、計算側には余裕がある。キャッシュ無しだと decode 3.2 tok/s で、9 GiB のキャッシュは SSD 読み出しを 24% 減らして速度を 2.2 倍にするが、VRAM 16 GB では 9〜10 GiB が上限(12 GiB は起動しない)。

| 構成(IQ3_XXS、32k ctx)              | VRAM    | 初見 prefill | 初見 decode   | SSD 読み出し / 100 tok | 再送 decode  |
| ----------------------------------- | ------- | ------------ | ------------- | ---------------------- | ------------ |
| experts CPU + VRAM キャッシュ 9 GiB | 15.1 GB | 3.8 tok/s    | **6.9 tok/s** | 4.2 GiB                | 15〜42 tok/s |
| experts CPU、キャッシュ無し         | 5.9 GB  | 2.4 tok/s    | 3.2 tok/s     | 5.5 GiB                | 3.2 tok/s    |

100 トークン生成するたびに 4〜5 GB を SSD から読む。NVMe(3〜5 GB/s)なら同じ読み出しが 1〜2 秒で済み、decode は 20〜30 tok/s 圏に入る見込み(この読み出し量が律速の全て)。DRAM を 64 GB にしても本体 55 GB は収まりきらず、SSD 読み出しを減らせるだけで消せない。**買うなら 1 TB の Gen4 NVMe(2 万円台)が先で、DRAM は後**。ただし prefill は NVMe でも数十 tok/s 止まりと見込まれ、Claude Code の長い入力には向かない。チャット・要約用途なら NVMe 追加で実用圏。

## 計測条件

- モデル: unsloth UD-IQ3_XXS(83.6 GB、n-gram 表 28.8 GB 込み)。llama-cache ボリュームの `llama.cpp/flash-next/` に `hf download --local-dir` で配置
- 構成 3 本は `deploy/llama-swap/config.yaml` の `qwen38-flash-next-*`。計測は `scripts/bench.sh`

## 結果

### 9 GiB キャッシュ(`qwen38-flash-next-iq3xxs`)、W4 3 本 + 1k 文字の要約、2 反復

読み込み 205 s(ページキャッシュが冷えた状態)。VRAM 15.1 GB、プロセスの RAM 使用 1.9 GB(本体は mmap で、ページキャッシュ 29 GB に乗る)。

| リクエスト                             | prompt_n            | prefill tok/s | decode tok/s | 300 tok あたりの SSD 読み出し | 備考                                    |
| -------------------------------------- | ------------------- | ------------- | ------------ | ----------------------------- | --------------------------------------- |
| W4-01 3 回目(experts がキャッシュ済み) | 4(接頭辞キャッシュ) | 13.9          | **41.6**     | 52 MiB                        | ヒットすればこの速さ                    |
| W4-02 初見                             | 112                 | 4.0           | 6.2          | 18.2 GiB                      | SATA の実効上限 237 MB/s に張り付き     |
| W4-03 初見                             | 40                  | 2.8           | 7.2          | 12.7 GiB                      |                                         |
| L1-1k 初見(要約)                       | 775                 | **9.3**       | 6.0          | 30.5 GiB                      | 775 トークンの prefill に 83 s          |
| W4-01 4 回目(間に 3 本挟む)            | 44                  | 3.8           | 6.9          | 12.4 GiB                      | 間の 3 本で追い出され、初見と同じに戻る |
| 2 反復目 W4-02 / W4-03 / L1            | 4〜40               | 2.6〜5.6      | 5.7〜7.3     | 9.8〜12.2 GiB                 | 反復しても改善しない                    |

- **律速は SSD**。初見のプロンプトでは生成中ずっと 230〜290 MB/s の読み出しが続き(`vmstat` の bi、wa 6〜15%)、GPU 使用率は 0〜17%。300 トークンごとに 12〜18 GB を読むのは、1 トークンあたり 40〜60 MB の experts をキャッシュ外から引いている計算
- **キャッシュ(VRAM 9 GB + ページキャッシュ 29 GB ≒ 本体 55 GB の 7 割)でも、プロンプトをまたぐと効かない**。同じ文の再送だけ 41.6 tok/s で、別の 3 本を挟むと 6.9 tok/s に戻る。expert の参照はプロンプトごとに十分ばらつき、作業セットがキャッシュを超える
- **prefill は decode より遅い**(3〜9 tok/s)。1 トークンごとに違う experts を踏むので、ミスが最も多い局面。2 万トークンの Claude Code の入力なら 40 分前後で、常用は不可
- 比較: Qwen3.6-35B-A3B IQ3_XXS(全部 VRAM)は decode 70 tok/s 台、prefill 数百 tok/s(2026-10 の実験)

### キャッシュ無し(`qwen38-flash-next-iq3xxs-nocache`)と 12 GiB

- キャッシュ無し: 読み込み 412 s、VRAM 5.9 GB。初見 prefill 1.0〜2.5 tok/s、decode 2.5〜4.4 tok/s、300 tok で SSD 13〜22 GB(1k 要約は 42 GB)。2 反復目も改善しない
- 12 GiB キャッシュ: llama-server が起動直後に落ちる(VRAM 超過。9 GiB 構成のピークが 15.1 GB)。上限は 9.5〜10 GiB
- 集計は `results/summary.md`(`scripts/aggregate.py`、初見 = prompt_n ≥ 10、再送 = 接頭辞キャッシュで prompt_n < 10)

## 落とし穴

| 罠                                           | 症状                                                                                                                 | 対処                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `hf download` の既定キャッシュ               | 新しい hf CLI は hub キャッシュを別レイアウト(xet)で書き、llama-server の `-hf` が拾えない                           | `--local-dir` で直置きし `-m` で指す                             |
| compose のプロジェクト名でボリュームが変わる | `deploy/llama-swap` で `up` すると `llama-swap_llama-cache` が新規に作られ、220 GB の `llama_llama-cache` が見えない | compose の `volumes:` に `name:` + `external: true` を書いて固定 |
| `--moe-cache-mib` の上限                     | 12 GiB で起動せず(エラーは llama-swap のログに出ない)                                                                | 9 GiB 構成の VRAM ピーク(15.1 GB)から逆算。10 GiB 前後が限界     |
| `pkill -f` が自分に当たる                    | ssh で渡したコマンド文字列自体がパターンに一致し、ssh が 255 で落ちる                                                | `bash -s` のヒアドキュメントで実行する                           |
| llama-swap のアンロード                      | `POST /unload` は効かず `GET /unload`                                                                                | 計測の切替は llama-swap に任せ、終わったら `GET /unload`         |

## ファイル

| パス                   | 内容                                         |
| ---------------------- | -------------------------------------------- |
| `notes/00-plan.md`     | 問い・構成・計測設計                         |
| `scripts/bench.sh`     | 構成 × プロンプト × 反復の計測(GPU 機で実行) |
| `scripts/data/`        | W4(短い推論 3 本)と合成長文 3 本             |
| `results/raw/`         | JSONL と読み込みログ(commit 対象外)          |
| `scripts/aggregate.py` | JSONL を構成ごとに集計(初見 / 再送を分ける)  |
| `results/summary.md`   | 集計表                                       |
