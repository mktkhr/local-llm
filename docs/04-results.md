# 04. 動作実績

GPU別・モデル別の実測リファレンス。安定して動作した組み合わせを蓄積する。

## 記載ルール

- 1行 = 1計測条件
- `100% GPU` 動作したものを基本とする。CPUオフロード発生分は **参考値** として別表
- `VRAM 使用` は `nvidia-smi` 実測、`SIZE` は `ollama ps` の想定確保量
- `生成` は `eval_count / eval_duration` の換算値
- `Ollama` バージョンを併記(挙動が変わるため)

## 共通計測条件

特記なき限り以下:

- `OLLAMA_MAX_LOADED_MODELS=1`
- `OLLAMA_NUM_PARALLEL=1`
- `OLLAMA_FLASH_ATTENTION=1`(`FA` 列の値による)

---

## NVIDIA GeForce GTX 1080 (8 GB, Compute 6.1, Pascal)

実機: Intel i7-3770 / DDR3 16GB / Ubuntu 22.04 / Driver 535.288.01 (CUDA 12.2)

### 100% GPU 動作(採用候補)

| モデル         | 重み量子化 | num_ctx    | FA     | KV       | SIZE       | VRAM使用      | 残VRAM      | 生成 tok/s | プロンプト tok/s |
| -------------- | ---------- | ---------- | ------ | -------- | ---------- | ------------- | ----------- | ---------- | ---------------- |
| qwen3.5:2b     | Q8_0       | 4,096      | off    | f16      | 4.2 GB     | 3,712 MiB     | 4,399 MiB   | —          | —                |
| qwen3.5:2b     | Q8_0       | 131,072    | off    | f16      | 7.3 GB     | 6,640 MiB     | 1,471 MiB   | 54.8       | 402.3            |
| qwen3.5:2b     | Q8_0       | 131,072    | on     | q8_0     | 6.7 GB     | 6,134 MiB     | 1,977 MiB   | 53.9       | —                |
| qwen3.5:2b     | Q8_0       | 131,072    | on     | q4_0     | 6.3 GB     | 5,750 MiB     | 2,361 MiB   | 53.5       | —                |
| qwen3.5:4b     | Q4_K_M     | 4,096      | off    | f16      | 5.9 GB     | 5,372 MiB     | 2,739 MiB   | —          | —                |
| **qwen3.5:4b** | Q4_K_M     | **65,536** | **on** | **q8_0** | **7.8 GB** | **7,204 MiB** | **907 MiB** | **35.3**   | 251.5            |

**所見**

- `qwen3.5:2b @ num_ctx=131,072` は素の f16 KV でも 100% GPU に収まる。`q8_0` / `q4_0` でVRAM を更に削減できるが、生成速度は誤差範囲
- `qwen3.5:4b @ num_ctx=65,536` は FA有効 + KV `q8_0` でちょうど 100% GPU に収まる(残VRAM 907 MiB と境界)。128k は不可(下表参照)
- 量子化の選び方:
  - 2b: 品質優先なら `q8_0`、VRAM 余裕優先なら `q4_0`(速度差は誤差)
  - 4b: `q8_0` でちょうど境界なので、これより緩めると即オフロードに転落する

### CPUオフロード発生(参考値、運用非推奨)

| モデル     | num_ctx | FA  | KV   | SIZE   | VRAM使用  | PROCESSOR         | 生成 tok/s |
| ---------- | ------- | --- | ---- | ------ | --------- | ----------------- | ---------- |
| qwen3.5:4b | 131,072 | off | f16  | 12 GB  | 7,714 MiB | 38% CPU / 62% GPU | 7.5        |
| qwen3.5:4b | 131,072 | on  | q8_0 | 10 GB  | 7,748 MiB | 26% CPU / 74% GPU | 12.4       |
| qwen3.5:4b | 131,072 | on  | q4_0 | 9.1 GB | 7,150 MiB | 20% CPU / 80% GPU | 19.6       |

**所見**

- 4b @ 128k は最強チューニング(FA + q4_0)でも GTX 1080 (8 GB) には乗り切らない
- 4b で長コンテキストを使いたい場合は **64k に縮める** ことで 100% GPU 動作が可能(上表参照)

### 動作不可(モデル候補から除外)

| モデル     | num_ctx | 状況                                                                                                           |
| ---------- | ------- | -------------------------------------------------------------------------------------------------------------- |
| gemma4:e2b | 4,096   | VRAM 7,392 MiB(残 719 MiB)。100% GPU には乗るが 128k 化の余地なし。「有効2.3B / 総5.1B」の総パラメータ分が必要 |
| gemma4:e4b | 4,096   | 4k ですでに 67% CPU / 33% GPU。オフロード前提                                                                  |

### 採用推奨

GTX 1080 (8 GB) では用途に応じて以下の2構成を使い分ける(共通設定: `OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`)。

| 用途                   | モデル       | num_ctx | 速度       | 残VRAM    | コメント                      |
| ---------------------- | ------------ | ------- | ---------- | --------- | ----------------------------- |
| **長コンテキスト優先** | `qwen3.5:2b` | 131,072 | 53.9 tok/s | 1,977 MiB | 128k フル動作、速度も最速     |
| **品質優先**           | `qwen3.5:4b` | 65,536  | 35.3 tok/s | 907 MiB   | 64k で 100% GPU、4Bの推論品質 |

備考:

- 2b は KV を `q4_0` に下げれば残VRAM 2,361 MiB まで広がる(長文参照精度との交換)
- 4b の 64k 構成は残VRAM が薄い(907 MiB)。`OLLAMA_NUM_PARALLEL` を上げる場合は再計測必須
- 重み量子化は Ollama 公式タグのデフォルト(2b は `Q8_0`、4b は `Q4_K_M`)を使用。より小さい量子化に興味がある場合は Unsloth GGUF(UD-Q3_K_XL 等)の取り込みを検討

### クライアント適性(本機固有の結論)

GTX 1080 (8GB) の応答速度では、クライアントとの相性が以下のように分かれる。

| クライアント                | 経由パス                                     | 本機での実用性 | 理由                                                                                                                                                                |
| --------------------------- | -------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 直接 curl(`/api/chat`)      | `/api/*`                                     | ◎              | injector の `think:false` が効いて 1〜3秒応答                                                                                                                       |
| opencode(`baseURL=.../api`) | `/api/chat`                                  | ○              | 同上。ただしエージェント機能はモデル能力に依存                                                                                                                      |
| opencode(`baseURL=.../v1`)  | `/v1/chat/completions`                       | △              | thinking ON で応答が分単位に。実用厳しい                                                                                                                            |
| Cline                       | `/v1/messages` または `/v1/chat/completions` | ×              | 重いシステムプロンプト+ thinking ON で詰まる                                                                                                                        |
| **Claude Code**             | `/v1/messages`                               | **×**          | thinking ON で応答が分単位、Claude Code の **10回固定リトライ** が発火してリクエスト多重化(詳細は [05-auth-gate.md](05-auth-gate.md) の Claude Code セクション参照) |

GTX 1080 での結論: **コード補完・短文要約・翻訳など軽量用途は実用範囲**、**Cline/Claude Code のような自律エージェント運用は GPU 性能不足で非推奨**。同じソフトウェア構成でも RTX 4080 SUPER 以上の GPU に換装すれば Claude Code 経由でも普通に動作する見込み(秒オーダー応答でリトライ発火しない)。

---

## NVIDIA GeForce RTX 4080 SUPER (16 GB, Compute 8.9, Ada Lovelace)

実機: AMD Ryzen 5 3600 / DDR4 32GB / Ubuntu 24.04.4 LTS / Driver 580.142 (CUDA 12.x)

評価方法は [docs/06-evaluation.md](06-evaluation.md) に準拠。計測時の Ollama バージョン・スクリプト hash・モデル digest は各テーブル直前のヘッダーブロックに記録する。

### 評価対象選定(2026-05-25 時点 Ollama Library 調査)

Qwen3 系 / Gemma 系 / DeepSeek 系を対象とする(ユーザー選択)。各モデル系列について、16GB に乗る量子化タグを実在確認した結果を以下に示す。

#### 対象モデル

##### Qwen3.5 系(汎用、9B 中心)

| タグ                | 重み   | 既定 ctx | ねらい                               |
| ------------------- | ------ | -------- | ------------------------------------ |
| `qwen3.5:4b-q4_K_M` | 3.4 GB | 256K     | 速度上限・長 ctx 主力                |
| `qwen3.5:4b-q8_0`   | 5.3 GB | 256K     | 4B 品質上限                          |
| `qwen3.5:9b-q4_K_M` | 6.6 GB | 256K     | 9B 起点                              |
| `qwen3.5:9b-q8_0`   | 11 GB  | 256K     | 9B 品質上限。長 ctx は KV 量子化必須 |

##### Gemma 4 系(マルチモーダル、e4b 中心)

| タグ                   | 重み   | 既定 ctx | ねらい                  |
| ---------------------- | ------ | -------- | ----------------------- |
| `gemma4:e2b-it-q4_K_M` | 7.2 GB | 128K     | 小型、速度比較          |
| `gemma4:e4b-it-q4_K_M` | 9.6 GB | 128K     | Gemma 4 系主力(実効 4B) |
| `gemma4:e4b-it-q8_0`   | 12 GB  | 128K     | e4b 品質上限            |

##### DeepSeek-R1 系(思考特化 distill)

| タグ                                  | 重み   | 既定 ctx | ねらい           |
| ------------------------------------- | ------ | -------- | ---------------- |
| `deepseek-r1:7b-qwen-distill-q4_K_M`  | 4.7 GB | 128K     | 7B 起点          |
| `deepseek-r1:7b-qwen-distill-q8_0`    | 8.1 GB | 128K     | 7B 品質寄り      |
| `deepseek-r1:8b-0528-qwen3-q4_K_M`    | 5.2 GB | 128K     | 0528 改良版      |
| `deepseek-r1:14b-qwen-distill-q4_K_M` | 9.0 GB | 128K     | 14B 品質上限狙い |

##### DeepSeek-Coder-V2 系(コーディング MoE)

| タグ                                         | 重み   | 既定 ctx | ねらい                                  |
| -------------------------------------------- | ------ | -------- | --------------------------------------- |
| `deepseek-coder-v2:16b-lite-instruct-q4_0`   | 8.9 GB | **160K** | 長 ctx 対応の唯一タグ、コーディング主力 |
| `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | 10 GB  | 4K       | `num_ctx` 拡張が効くかは要検証          |

> DeepSeek-Coder-V2 は **タグごとに既定 ctx が異なる**。長 ctx 評価では `q4_0` を採用。`q4_K_M` 等は Modelfile の `PARAMETER num_ctx` を override できるかを Pass 1 で確認。

#### 対象外と理由

| モデル                              | 量子化 / 重みサイズ | 理由                                                                |
| ----------------------------------- | ------------------- | ------------------------------------------------------------------- |
| `qwen3.5:27b-int4`                  | 16 GB               | 残 VRAM ほぼゼロで実用性低、量子化バリエーション無し                |
| `qwen3.5:27b-q4_K_M`                | 17 GB               | 16GB に乗らない                                                     |
| `qwen3.6:*` (27b / 35b-a3b)         | 17+ GB              | 16GB CUDA で乗る量子化タグが存在しない(`nvfp4` / `mxfp8` は MLX 用) |
| `gemma4:26b` / `gemma4:31b`         | 18〜20 GB           | 16GB に乗らない                                                     |
| `deepseek-r1:32b`                   | 20 GB               | 16GB に乗らない                                                     |
| `deepseek-r1:14b-qwen-distill-q8_0` | 16 GB               | 残 VRAM ほぼゼロで KV 量子化の余地が無い                            |

> Qwen3.5 / Qwen3.6 系には Q3_K_M / IQ3_XS タグが存在しないため、27B 級を 16GB に押し込む方法が `int4` しかなく、それも残 VRAM ほぼゼロのため対象外とした。

### 測定環境(2026-05-25 〜 26 計測)

- GPU: NVIDIA RTX 4080 SUPER (16 GB, Compute 8.9, Ada Lovelace)
- Driver: 580.142 / CUDA 13.0
- OS: Ubuntu 24.04.4 LTS, Linux 6.17.0-23-generic
- Ollama: **v0.24.0**
- 計測スクリプト: `scripts/bench/` @ git 3d64098(+作業ブランチ未コミット)
- 共通設定: `OLLAMA_FLASH_ATTENTION=1` / `OLLAMA_NUM_PARALLEL=1` / `OLLAMA_MAX_LOADED_MODELS=1` / **KV キャッシュ量子化 = q8_0**

### 100% GPU 動作

長コンテキスト・100% GPU を共通条件として、各モデルの **二分探索で求めた最大 ctx での速度** を以下にまとめる。`Max ctx` 列の `*` は Ollama がモデル既定上限で要求 ctx をキャップしていることを示す(VRAM 制約ではなくモデルの context_length が天井)。

| Model                                      | Max ctx    | Decode tok/s | Prefill tok/s | TTFT (s) | VRAM peak (MiB) | VRAM free (MiB) |
| ------------------------------------------ | ---------- | ------------ | ------------- | -------- | --------------- | --------------- |
| qwen3.5:4b-q4_K_M                          | 262144     | **134.6**    | 1704.9        | 0.29     | 14201           | 1742            |
| qwen3.5:4b-q8_0                            | 241664     | 97.5         | 1470.9        | 0.30     | 15335           | 608             |
| qwen3.5:9b-q4_K_M                          | 221184     | 88.9         | 1363.7        | 0.30     | 15421           | 522             |
| qwen3.5:9b-q8_0                            | 116736     | 59.1         | 1265.3        | 0.31     | 15442           | 501             |
| gemma4:e2b-it-q4_K_M                       | 131072 `*` | **213.0**    | **3048.4**    | 0.38     | 8880            | 7063            |
| gemma4:e4b-it-q4_K_M                       | 131072 `*` | 130.9        | 2214.3        | 0.38     | 12284           | 3659            |
| gemma4:e4b-it-q8_0                         | 131072 `*` | 91.0         | 1783.3        | 0.38     | 14216           | 1727            |
| deepseek-r1:7b-qwen-distill-q4_K_M         | 104448     | 128.7        | 2732.3        | 0.16     | 8672            | 7271            |
| deepseek-r1:7b-qwen-distill-q8_0           | 75776      | 79.2         | 2071.4        | 0.17     | 10864           | 5079            |
| deepseek-r1:8b-0528-qwen3-q4_K_M           | 116736     | 106.7        | 1860.3        | 0.15     | 15204           | 739             |
| deepseek-r1:14b-qwen-distill-q4_K_M        | 31744      | 66.1         | 1398.7        | 0.16     | 12686           | 3257            |
| deepseek-coder-v2:16b-lite-instruct-q4_0   | 19456      | **268.2**    | 2335.0        | **0.10** | 15400           | 543             |
| deepseek-coder-v2:16b-lite-instruct-q4_K_M | 11264      | 246.8        | 2205.4        | 0.14     | 14230           | 1713            |

**読みどころ**

- **最速**は MoE の `deepseek-coder-v2:16b-lite-instruct-q4_0`(decode **268 tok/s**)。次点も同シリーズの q4_K_M(247 tok/s)。MoE は実効計算量が小さく、生成速度で他を引き離す
- **2 位グループ**は Gemma 4 e2b(213 tok/s)と Qwen3.5 4B q4_K_M(135 tok/s)、Gemma 4 e4b q4_K_M(131 tok/s)、DeepSeek-R1 7B distill q4_K_M(129 tok/s)
- **最遅**は qwen3.5:9b-q8_0(59 tok/s)と deepseek-r1:14b(66 tok/s)。重み量子化を緩めるかパラメータを増やすと素直に速度が下がる
- **TTFT** は DeepSeek 系が顕著に速い(0.10〜0.17 秒)。Qwen / Gemma の 0.29〜0.38 秒と比べて 2 倍以上速く反応する
- **Ctx 上限**: Qwen3.5 4B q4_K_M が **262K**(モデルの最大)まで 100% GPU で乗る唯一の構成。Qwen3.5 9B q4_K_M も 221K と長い
- **Ctx キャップ**: Gemma 4 系 3 タグはすべて 131072 でキャップ(モデル既定が 128K)。`ctx_search` は 262K を試して成功扱いになるが、`ollama ps` の `context_length` で実際は 131072 と判明したため再記載

### CPU オフロード発生(参考、対象外の裏取り)

「評価対象選定」で **16GB に乗らない** と判断した 27B 級モデルを ctx=16384 で実測し、本当に CPU オフロード前提でしか動かないことを確認した。

| モデル | num_ctx | FA | KV | SIZE | VRAM 使用 | CPU offload | PROCESSOR | Decode tok/s | Prefill tok/s | TTFT (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.6:27b | 16384 | on | q8_0 | 22.2 GB | 13.0 GiB | 9.2 GiB | **58% GPU / 42% CPU** | **3.6** | 64.8 | 0.91 |
| gemma4:26b | 16384 | on | q8_0 | 18.4 GB | 14.5 GiB | 3.9 GiB | **79% GPU / 21% CPU** | 26.2 | 278.6 | 0.42 |

**所見**

- `qwen3.6:27b` は重みだけで 22 GB(Q4_K_M)、16 GB GPU には 58% 分しか乗らず **decode 3.6 tok/s**。対話 UX に耐えない遅さで、評価対象から外した判断は妥当
- `gemma4:26b` はオフロード比率が小さい(21%)ものの decode 26 tok/s。緊急用途で「動かないよりはマシ」レベル。やはり本評価では対象外の妥当性が裏付けられた
- 同条件でも qwen3.6 と gemma4 で乗り方が違うのは、モデルアーキ(レイヤー数 / KV ヘッド構成)と GGUF のメモリレイアウトの差。Ollama 0.24.0 の自動配置は VRAM を最大限使う方向に振っている

### 品質評価:Needle / Coding / Summary

各モデルの最大 ctx で Needle(深度 100% × 位置 50%、`needle_id=603904`)、ctx 16K で Coding(6 タスク自動採点)と Summary(2 会議 × n-gram 3 で被覆率)を計測。

| Model                                      | Needle | Coding   | Summary  | 備考                                                                                              |
| ------------------------------------------ | ------ | -------- | -------- | ------------------------------------------------------------------------------------------------- |
| qwen3.5:4b-q4_K_M                          | ✓      | **1.00** | **0.86** | 速度・長 ctx・品質のバランス筆頭                                                                  |
| qwen3.5:4b-q8_0                            | ✓      | 1.00     | 0.58     | q8 のメリットは観察できず、むしろ Summary 悪化                                                    |
| qwen3.5:9b-q4_K_M                          | ✓      | 1.00     | 0.78     | 9B クラスの標準                                                                                   |
| qwen3.5:9b-q8_0                            | ✓      | 1.00     | 0.64     | q8 化で速度・ctx・Summary 全て劣化、メリットなし                                                  |
| gemma4:e2b-it-q4_K_M                       | ✓      | 0.83     | 0.78     | Coding の bugfix が 0.5 と弱い                                                                    |
| gemma4:e4b-it-q4_K_M                       | ✓      | **1.00** | **0.86** | Coding/Summary 両方高い、汎用候補                                                                 |
| gemma4:e4b-it-q8_0                         | ✓      | 1.00     | 0.81     | q4_K_M とほぼ同等で速度だけ遅い                                                                   |
| deepseek-r1:7b-qwen-distill-q4_K_M         | **✗**  | 0.92     | 0.53     | Needle 失敗(数字捏造)、Summary も低い                                                             |
| deepseek-r1:7b-qwen-distill-q8_0           | **✗**  | 0.98     | 0.53     | 同上                                                                                              |
| deepseek-r1:8b-0528-qwen3-q4_K_M           | ✓      | 1.00     | 0.75     | 0528 リビジョン(Qwen3 ベース蒸留)、7B 蒸留より明らかに優秀                                        |
| deepseek-r1:14b-qwen-distill-q4_K_M        | ✓      | 1.00     | 0.72     | max_ctx が 31K に圧縮されるが品質は安定                                                           |
| deepseek-coder-v2:16b-lite-instruct-q4_0   | **✗**  | 1.00     | 0.81     | コーディングは満点、Needle は「秘密の数字は存在しません」と否定。コード特化で日本語長文検索は苦手 |
| deepseek-coder-v2:16b-lite-instruct-q4_K_M | **✗**  | 1.00     | 0.75     | 同上                                                                                              |

**Coding 種別内訳**(impl / bugfix / refactor)

| Model                                             | Overall | impl     | bugfix   | refactor |
| ------------------------------------------------- | ------- | -------- | -------- | -------- |
| qwen3.5:4b-q4_K_M / 4b-q8_0 / 9b-q4_K_M / 9b-q8_0 | 1.00    | 1.00     | 1.00     | 1.00     |
| gemma4:e2b-it-q4_K_M                              | 0.83    | 1.00     | **0.50** | 1.00     |
| gemma4:e4b-it-q4_K_M / e4b-it-q8_0                | 1.00    | 1.00     | 1.00     | 1.00     |
| deepseek-r1:7b-qwen-distill-q4_K_M                | 0.92    | **0.75** | 1.00     | 1.00     |
| deepseek-r1:7b-qwen-distill-q8_0                  | 0.98    | 1.00     | 1.00     | 0.93     |
| deepseek-r1:8b-0528-qwen3-q4_K_M / 14b-q4_K_M     | 1.00    | 1.00     | 1.00     | 1.00     |
| deepseek-coder-v2:16b-lite-instruct-q4_0 / q4_K_M | 1.00    | 1.00     | 1.00     | 1.00     |

**読みどころ**

- **Coding**: 13 タグ中 10 タグが満点。落としたのは Gemma 4 e2b(bugfix 0.5)と DeepSeek-R1 7B(impl 0.75 / refactor 0.93)。サンプル 6 タスクの難度から考えると小型モデルでも実用域
- **Needle 失敗の解釈**:
  - DeepSeek-R1 7B distill は think=false 時に応答が崩壊するか数字を捏造する。同シリーズでも 8B-0528(Qwen3 ベース)と 14B は問題なし
  - DeepSeek-Coder-V2 は明確に「秘密の数字は存在しません」と否定する。**日本語長文中からの情報抽出は訓練範囲外** と推測。コード補完専用と割り切るのが妥当
  - Gemma 4 系は初回 ctx=262144 要求で 128K にキャップされた結果 needle がコンテキスト外に落ちて失敗していたが、再計測(ctx=131072、chars=120000)では全タグ成功した
- **Summary** は全モデルで decisions(決定事項)の取りこぼしが目立つ。会議内で明示されない暗黙的な決定の抽出が一様に弱い。これは 4B〜14B クラスの限界か、本評価の合成データの設計が「決定事項」を抽出しにくくしているかのどちらかで、実データへの差し替え後に再検証する余地あり

### KV キャッシュ量子化の比較(q8_0 / q4_0 / f16)

同 13 モデルを **KV q4_0** と **f16** で再計測し、本評価既定の **q8_0** と並べた表を以下に示す。f16 は Ollama の既定(`OLLAMA_KV_CACHE_TYPE` 未設定相当)、q4_0 は最も攻めた KV 量子化。

| Model | KV | Max ctx | Decode tok/s | Prefill tok/s | TTFT (s) | VRAM peak (MiB) | Needle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5:4b-q4_K_M | q8_0 | 262144 | 134.6 | 1704.9 | 0.29 | 14201 | ✓ |
| qwen3.5:4b-q4_K_M | q4_0 | 262144 | 125.9 | 1493.0 | 0.30 | 12164 | ✓ |
| qwen3.5:4b-q4_K_M | f16 | 225280 | 133.0 | 1670.9 | 0.29 | 15416 | ✓ |
| qwen3.5:4b-q8_0 | q8_0 | 241664 | 97.5 | 1470.9 | 0.30 | 15335 | ✓ |
| qwen3.5:4b-q8_0 | q4_0 | 262144 | 95.5 | 1481.0 | 0.30 | 13960 | ✓ |
| qwen3.5:4b-q8_0 | f16 | 181248 | 91.0 | 1391.6 | 0.30 | 15324 | ✓ |
| qwen3.5:9b-q4_K_M | q8_0 | 221184 | 88.9 | 1363.7 | 0.30 | 15421 | ✓ |
| qwen3.5:9b-q4_K_M | q4_0 | 262144 | 87.2 | 1274.2 | 0.30 | 14670 | ✓ |
| qwen3.5:9b-q4_K_M | f16 | 164864 | 87.5 | 1473.2 | 0.30 | 15359 | ✓ |
| qwen3.5:9b-q8_0 | q8_0 | 116736 | 59.1 | 1265.3 | 0.31 | 15442 | ✓ |
| qwen3.5:9b-q8_0 | q4_0 | 152576 | 60.2 | 1127.6 | 0.31 | 15391 | ✓ |
| qwen3.5:9b-q8_0 | f16 | 83968 | 59.7 | 1126.1 | 0.32 | 15290 | ✓ |
| gemma4:e2b-it-q4_K_M | q8_0 | 131072 | 213.0 | 3048.4 | 0.38 | 8880 | ✓ |
| gemma4:e2b-it-q4_K_M | q4_0 | 131072 | 210.3 | 3112.3 | 0.37 | 8644 | ✓ |
| gemma4:e2b-it-q4_K_M | f16 | 131072 | 218.5 | 3170.4 | 0.38 | 9016 | ✓ |
| gemma4:e4b-it-q4_K_M | q8_0 | 131072 | 130.9 | 2214.3 | 0.38 | 12284 | ✓ |
| gemma4:e4b-it-q4_K_M | q4_0 | 131072 | 128.0 | 2278.2 | 0.38 | 11676 | ✓ |
| gemma4:e4b-it-q4_K_M | f16 | 131072 | 136.5 | 2323.7 | 0.39 | 12792 | ✓ |
| gemma4:e4b-it-q8_0 | q8_0 | 131072 | 91.0 | 1783.3 | 0.38 | 14216 | ✓ |
| gemma4:e4b-it-q8_0 | q4_0 | 131072 | 88.6 | 1762.1 | 0.38 | 13608 | ✓ |
| gemma4:e4b-it-q8_0 | f16 | 131072 | 90.7 | 1785.0 | 0.38 | 14726 | ✓ |
| deepseek-r1:7b-qwen-distill-q4_K_M | q8_0 | 104448 | 128.7 | 2732.3 | 0.16 | 8672 | ✗ |
| deepseek-r1:7b-qwen-distill-q4_K_M | q4_0 | 124928 | 132.5 | 2724.5 | 0.16 | 7644 | ✗ |
| deepseek-r1:7b-qwen-distill-q4_K_M | f16 | 79872 | 134.1 | 2855.4 | 0.16 | 9868 | ✗ |
| deepseek-r1:7b-qwen-distill-q8_0 | q8_0 | 75776 | 79.2 | 2071.4 | 0.17 | 10864 | ✗ |
| deepseek-r1:7b-qwen-distill-q8_0 | q4_0 | 88064 | 83.2 | 2078.5 | 0.17 | 10042 | ✗ |
| deepseek-r1:7b-qwen-distill-q8_0 | f16 | 56320 | 81.4 | 2128.8 | 0.16 | 11458 | ✗ |
| deepseek-r1:8b-0528-qwen3-q4_K_M | q8_0 | 116736 | 106.7 | 1860.3 | 0.15 | 15204 | ✓ |
| deepseek-r1:8b-0528-qwen3-q4_K_M | q4_0 | 131072 | 108.2 | 1957.8 | 0.16 | 11834 | ✓ |
| deepseek-r1:8b-0528-qwen3-q4_K_M | f16 | 68608 | 111.8 | 1859.2 | 0.15 | 15440 | ✓ |
| deepseek-r1:14b-qwen-distill-q4_K_M | q8_0 | 31744 | 66.1 | 1398.7 | 0.16 | 12686 | ✓ |
| deepseek-r1:14b-qwen-distill-q4_K_M | q4_0 | 44032 | 66.4 | 1477.5 | 0.17 | 11882 | ✓ |
| deepseek-r1:14b-qwen-distill-q4_K_M | f16 | 19456 | 66.9 | 1416.2 | 0.16 | 13028 | ✓ |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | q8_0 | 19456 | 268.2 | 2335.0 | 0.10 | 15400 | ✗ |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | q4_0 | 19456 | 256.3 | 2234.9 | 0.11 | 15400 | ✗ |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | f16 | 19456 | 252.6 | 2066.8 | 0.11 | 15389 | ✗ |
| deepseek-coder-v2:16b-lite-instruct-q4_K_M | q8_0 | 11264 | 246.8 | 2205.4 | 0.14 | 14230 | ✗ |
| deepseek-coder-v2:16b-lite-instruct-q4_K_M | q4_0 | 11264 | 242.4 | 2064.2 | 0.11 | 14230 | ✗ |
| deepseek-coder-v2:16b-lite-instruct-q4_K_M | f16 | 11264 | 242.5 | 2011.0 | 0.11 | 14233 | ✗ |

#### KV による max_ctx の伸縮(q8_0 を基準)

| Model | q8_0 | q4_0 (vs q8) | f16 (vs q8) | KV 感受性 |
| --- | --- | --- | --- | --- |
| qwen3.5:4b-q4_K_M | 262144 | 同じ(モデル上限) | -14% | 中 |
| qwen3.5:4b-q8_0 | 241664 | **+8.5%** (262144、上限) | -25% | 中 |
| qwen3.5:9b-q4_K_M | 221184 | **+18.5%** (262144、上限) | -25% | 中〜大 |
| qwen3.5:9b-q8_0 | 116736 | **+30.7%** (152576) | -28% | **大** |
| gemma4:e2b / e4b 全 3 | 131072 | 同じ(モデル上限) | 同じ | なし |
| deepseek-r1:7b-q4_K_M | 104448 | **+19.6%** (124928) | -23% | 中 |
| deepseek-r1:7b-q8_0 | 75776 | **+16.2%** (88064) | -26% | 中 |
| deepseek-r1:8b-0528 | 116736 | **+12.3%** (131072、上限) | -41% | 中〜大 |
| deepseek-r1:14b | 31744 | **+38.7%** (44032) | -39% | **大** |
| deepseek-coder-v2:q4_0 / q4_K_M | 19456 / 11264 | 同じ(VRAM 制約) | 同じ | なし(VRAM 律速) |

**読みどころ**

- **q4_0 はほぼフリーランチ**: 14 タグ中 7 タグで **q8_0 比 +10〜39% の max ctx 拡張**、残りはモデル/VRAM 上限で頭打ち。速度低下は ≤6%(誤差範囲)、Needle 品質は q8_0 と同パターン(失敗するモデルはどの KV でも失敗、成功するモデルはどの KV でも成功)
- **f16 は不利な選択**: max ctx が q8_0 比 -14〜-41% 縮む。速度メリットは無く、Flash Attention 有効下では q8_0/q4_0 と速度同等。**長 ctx を狙うなら f16 を選ぶ理由は無い**
- **KV 感受性の最大は deepseek-r1:14b**: 32K(q8_0)→ 44K(q4_0)→ 19K(f16)と 2 倍以上の幅。weight-heavy なモデルほど KV 量子化の影響を強く受ける
- **Gemma 4 と DeepSeek-Coder-V2 は KV を変えても max ctx 不変**: Gemma 4 は **モデル既定 ctx 131072 の天井**、Coder-V2 は **MoE の VRAM 消費** が頭打ちを作っている
- **本評価の主力 KV(q8_0)を q4_0 に置き換えても支障なし**: 速度・品質ともに変わらず、長 ctx だけ伸びる。記事化時の運用推奨は `OLLAMA_KV_CACHE_TYPE=q4_0` に倒すのが妥当(本評価結果を踏まえた今後の既定値変更候補)

> **注意**: q4_0 の Needle 初回 sweep では Gemma 4 全 3 / deepseek-r1:8b-0528 で偽の失敗が出ていた(`ctx_search` がモデル上限キャップを検出していなかったため、262144 文字の文書で needle が context overflow で落ちていた)。`ctx_search` の修正と chars=ctx×0.9 への変更後、すべて成功を再確認している。表の数値は修正後の値。

### Thinking モードの比較(主要 3 モデル、ctx=16384)

`think=true` / `think=false` を主要 3 モデルで切り替え、speed・coding・summary をそれぞれ計測した。Speed は `num_predict=512`(false は 128 → 512 で揃え)、Coding/Summary は `num_predict=2048`(false は 1024 → 2048 で揃え)で実施。

| Model | think | Decode tok/s | Speed eval_count | TTFT (s) | Total elapsed (s) | Coding | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5:9b-q4_K_M | false | 89.5 | 74 | 0.30 | 1.21 | **1.00** | 0.67 |
| qwen3.5:9b-q4_K_M | **true** | 87.0 | 512 | 6.44 | 6.44 | **0.67** | **0.00** |
| gemma4:e4b-it-q4_K_M | false | 123.2 | 72 | 0.37 | 1.21 | 1.00 | 0.67 |
| gemma4:e4b-it-q4_K_M | **true** | 128.6 | 512 | 2.44 | 3.44 | 1.00 | **0.81** |
| deepseek-r1:8b-0528-qwen3-q4_K_M | false | 111.2 | 61 | 0.15 | 0.95 | 1.00 | 0.72 |
| deepseek-r1:8b-0528-qwen3-q4_K_M | **true** | 107.1 | 357 | 2.41 | 3.58 | 1.00 | 0.75 |

**観察**

- **Decode tok/s は think に依らずほぼ不変**(±5% 以内)。トークン当たりの生成速度は同じで、差は出力トークン数の多寡から生まれる
- **eval_count(生成トークン数)は 5〜7 倍に膨らむ**: think=true で `<think>...</think>` ブロックを大量に挟むため。Speed テスト(num_predict=512)では Qwen3.5:9b / Gemma 4 が両方とも上限 512 まで使い切った
- **Total elapsed は 3〜7 倍**: TTFT は同じでも、思考分のトークンを全て生成するまでストリームは終わらない。**対話 UX としては明らかに重い**
- **品質はモデル依存で大きく分岐**:
  - **Gemma 4 e4b**: Summary が 0.67 → **0.81** に向上。思考が品質に効くタイプ
  - **DeepSeek-R1:8B-0528**: Summary が 0.72 → 0.75 と微増。Coding は両モードとも 1.00 で差なし
  - **Qwen3.5:9B**: `num_predict=2048` 時点では **Coding 1.00 → 0.67、Summary 0.67 → 0.00** と大幅劣化していた(思考だけで budget を使い切り `code_extracted=false` / 応答完全に空)。**追試で `num_predict=4096` まで上げたところ、Coding 1.00 / Summary 0.78 まで回復** — これは think=false ベースライン(0.67〜0.78)と同等以上。**think 失敗の原因は budget であってモデル特性ではない** ことが確定した
- **示唆**: thinking 系モデルは **num_predict を最低 4096、複雑タスクでは 8192 以上** 取る前提で運用する。「思考分 + 回答分」の両方が収まるよう余裕を持たせること。**budget 不足での think=true は think=false より露骨に悪い結果になる**

**運用提言**

- 対話型用途では **think=false が標準**。Total elapsed の差(1 秒 vs 6 秒)はユーザー体感に直結する
- 要約・複雑な分析で品質を取りたい場合のみ **think=true を限定的に**。ただし **Qwen3.5 系は num_predict 不足で逆に劣化** するリスクがあるので、有効化時は予算を十分に取る
- DeepSeek-R1 系は元々思考前提に設計されているが、本評価の合成タスク程度では think=false でも十分機能する

> [docs/05-auth-gate.md](05-auth-gate.md) §「think-injector の役割」で言及している通り、本評価では `/api/*` 経路で `think:false` を主測定値としている。本セクションはそのスコープ外でモデル本来の挙動を観察した結果。

### Needle In A Haystack 網羅スイープ(深度 × 位置)

主要 4 モデル(2 全勝 + 2 落第)を **4 深度 × 3 位置 = 12 試行** で計測。試行は `KV=q8_0`、`think=false`、`chars=ctx×0.9`。`✓` は needle_id を応答に含む、`✗` は含まない。

| Model | depth | pos 10% | pos 50% | pos 90% |
| --- | --- | --- | --- | --- |
| qwen3.5:4b-q4_K_M | 4096 | ✓ | ✓ | ✓ |
| qwen3.5:4b-q4_K_M | 16384 | ✓ | ✓ | ✓ |
| qwen3.5:4b-q4_K_M | 65536 | ✓ | ✓ | ✓ |
| qwen3.5:4b-q4_K_M | 262144 | ✓ | ✓ | ✓ |
| gemma4:e4b-it-q4_K_M | 4096 | ✓ | ✓ | ✓ |
| gemma4:e4b-it-q4_K_M | 16384 | ✓ | ✓ | ✓ |
| gemma4:e4b-it-q4_K_M | 65536 | ✓ | ✓ | ✓ |
| gemma4:e4b-it-q4_K_M | 131072 | ✓ | ✓ | ✓ |
| deepseek-r1:7b-qwen-distill-q4_K_M | 4096 | ✓ | ✓ | ✓ |
| deepseek-r1:7b-qwen-distill-q4_K_M | 16384 | ✗ | ✗ | ✓ |
| deepseek-r1:7b-qwen-distill-q4_K_M | 65536 | ✗ | ✗ | ✗ |
| deepseek-r1:7b-qwen-distill-q4_K_M | 104448 | ✗ | ✗ | ✗ |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | 4096 | ✗ | ✗ | ✓ |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | 16384 | ✗ | ✗ | ✓ |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | 19456 | ✗ | ✗ | ✓ |

**読みどころ**

- **qwen3.5:4b-q4_K_M / gemma4:e4b-it-q4_K_M**: **12/12 全勝**。4K から各々の上限まで、3 位置すべてで needle を抽出。本評価の主力候補として相応しい長文耐性
- **deepseek-r1:7b-qwen-distill-q4_K_M = ctx スケーリング問題**:
  - 4K では 3/3 ✓(短文では普通に動く)
  - 16K で 1/3(pos 90% のみ)に崩れ、65K 以上では 0/3
  - **ctx が伸びるほど検索精度が崩壊する**。蒸留 7B クラスの長文能力の限界
  - 元の think=false 計測でも長 ctx で needle を捏造していたのと整合
- **deepseek-coder-v2:16b-lite-instruct-q4_0 = 位置依存問題**:
  - どの ctx でも **pos 90% のみ ✓、pos 10% / 50% は全て ✗**(9 試行中 3 ✓)
  - ctx 長は関係なく、**末尾近くしか参照しない**。長文中の「検索」ではなく「直近文脈の継続」しかしていない
  - コード補完モデルとして「いま書いている関数のすぐ上を参照する」用途には適合、文書からの情報抽出には不向きという結論を強く裏付ける
- **示唆**: Needle が「pos 50%」1 点だけだと deepseek-coder-v2 のような **位置依存型の弱さを見落とす**。pos 10% と 90% を併せて測ることで、モデルの長文性質が立体的に見える

> 上の表は本評価方法論(`scripts/bench/run_needle_sweep.sh`)で再現可能。深度・位置は `DEPTHS` / `POSITIONS` 配列で変更できる。

### 実コード相当の難タスク(`tasks_hard.json`)

合成 6 タスク(`tasks.json`、impl=add/fib、bugfix=factorial/count_vowels、refactor=classify_age/squares_of_evens)では 13 タグ中 10 タグが 100% で、モデル間の差が出なかった。OIDC 用途も意識した難度を上げた 6 タスク(`tasks_hard.json`)を別途設計し、主力 6 モデルで再計測した。

**追加タスク一覧**

| Task | 種別 | 内容 |
| --- | --- | --- |
| `impl_flatten_dict` | impl | ネストした dict を `"."` 区切りキーに平坦化(値が dict のもののみ再帰) |
| `impl_balanced_brackets` | impl | `()`/`[]`/`{}` の対応をチェック。空文字列は True、その他は混在文字を含めて判定 |
| `impl_parse_bearer_token` | impl | `Authorization: Bearer <token>` 形式から token を抽出(case sensitive、空文字や不正形式は None) |
| `bugfix_two_sum` | bugfix | LeetCode 風 two-sum の seen dict 検索条件が間違っている(`n in seen` ではなく `target - n in seen` であるべき) |
| `bugfix_max_subarray` | bugfix | Kadane アルゴリズムの `best = current`(`best = max(best, current)` 漏れ) |
| `refactor_data_validator` | refactor | 6 段ネストした if 文の検証関数を早期 return / ガード節で書き直す(7 種の戻り値分岐) |

`num_predict=3072`(easy の 2 倍)、ctx=16384、KV=q8_0、think=false。

#### 合成タスク vs 難タスクの全体比較

| Model | Easy overall | Hard overall | Hard impl | Hard bugfix | Hard refactor |
| --- | --- | --- | --- | --- | --- |
| qwen3.5:4b-q4_K_M | 1.00 | 0.963 | 0.926 | 1.00 | 1.00 |
| qwen3.5:9b-q4_K_M | 1.00 | **0.815** | 0.963 | **0.50** | 1.00 |
| **gemma4:e4b-it-q4_K_M** | 1.00 | **0.982** | 0.963 | 1.00 | 1.00 |
| deepseek-r1:8b-0528-qwen3-q4_K_M | 1.00 | 0.945 | 0.889 | 1.00 | 1.00 |
| **deepseek-r1:14b-qwen-distill-q4_K_M** | 1.00 | **0.852** | **0.704** | 1.00 | 1.00 |
| deepseek-coder-v2:16b-lite-instruct-q4_0 | 1.00 | 0.963 | 0.926 | 1.00 | 1.00 |

#### 難タスクのタスク別正答数

| Task | qwen 4b | qwen 9b | gemma e4b | r1 8b-0528 | r1 14b | coder-v2 q4_0 |
| --- | --- | --- | --- | --- | --- | --- |
| impl_flatten_dict (max 6) | 6 | 6 | 6 | 6 | **2** | 6 |
| impl_balanced_brackets (max 10) | 10 | 10 | 10 | 10 | 10 | 10 |
| impl_parse_bearer_token (max 9) | 7 | 8 | 8 | 6 | 7 | 7 |
| bugfix_two_sum (max 6) | 6 | **err** | 6 | 6 | 6 | 6 |
| bugfix_max_subarray (max 6) | 6 | 6 | 6 | 6 | 6 | 6 |
| refactor_data_validator (max 9) | 9 | 9 | 9 | 9 | 9 | 9 |

**読みどころ**

- **合成タスクは差を出せていなかった**: 6 モデル全てが 1.00 で並ぶ。`add(a, b)` レベルでは差別化困難。**評価難度は意図的に上げる必要がある**
- **難タスクのトップは Gemma 4 e4b(0.982)**: マルチモーダル指向の小型モデルが、コード特化の DeepSeek-Coder-V2 と並ぶか上回る
- **DeepSeek-R1:14B は impl で 0.704 と急落、特に `impl_flatten_dict` で 2/6**: 大きな失敗パターンは「キーを `"."` で連結する仕様を読み落とし、`"ab"` のように直接連結」。**最大モデル = 最高品質とは限らない**ことの強い証拠
- **Qwen3.5:9B は `bugfix_two_sum` で「バグはない」と主張**: バグ報告を受けても「このコードは LeetCode 標準解と整合的で論理的に正しく機能します」と返し、結果として `code_extracted=true` だが構文不全なコード(num_predict=3072 でも改善せず再現)。**LLM の自信過剰が debug 用途で致命的になりうる**
- **`impl_parse_bearer_token`(OIDC ドメイン)で全モデルが減点**: 主に `"bearer abc"` の小文字プレフィックスを誤って受理 / `"Bearer "` の空トークンを誤って受理 / `"Bearer  spaced"` の二重空白の扱いが揺れる、など。本評価のスコープを超えるが、**プロトコル準拠コードでは LLM 全般に注意が要る**
- **`refactor_data_validator` は全モデル 9/9**: 6 段ネスト → 早期 return への書き換えは、7 つの戻り値分岐込みでも全モデルが正しく機能保存できた。**意味保存リファクタは LLM が比較的得意な領域**

**示唆**

- 評価データは難度勾配を意図して設計しないと、モデル選定で差が出ない
- 「**サイズが大きい = 賢い**」は本評価で **明確に成り立たない**。DeepSeek-R1:14B は同シリーズ 8B(0528 改良版)よりも低スコア
- 推奨運用構成では `gemma4:e4b-it-q4_K_M` を **コーディング寄りでも候補に上げて良い**(Coding 0.982、Summary 0.86、Needle 全勝)
- 同じ「Coding 完全合格」と表示されている easy 結果は、評価難度の問題で **モデル選定の決め手にしてはいけない**

> 難タスクの再現は `scripts/bench/data/coding/tasks_hard.json` を `--tasks` に渡すだけ。`uv run python run_coding.py --model <tag> --ctx 16384 --num-predict 3072 --tasks data/coding/tasks_hard.json`。

### 注目すべき所見一覧

各セクションで観察した「直感に反する / 記事化したい」発見を 1 か所に集約。詳細は各セクション本文。

| # | 所見 | 由来 |
| --- | --- | --- |
| 1 | **KV `q4_0` はほぼフリーランチ**: q8_0 比で max ctx +10〜39%、速度劣化 ≤6%、Needle 品質維持。**運用既定値は q4_0 に振っていい** | KV キャッシュ量子化の比較 |
| 2 | **KV `f16` は不利のみ**: max ctx -14〜-41%、Flash Attention 有効下では速度メリット無し。**選ぶ理由が無い** | KV キャッシュ量子化の比較 |
| 3 | **`think=true` は num_predict を最低 4096 取らないと逆効果**: Qwen3.5:9b は num_predict=2048 で Coding 1.00 → 0.67、Summary 0.67 → 0.00 と崩壊するが、4096 に増やせば 1.00 / 0.78 まで回復。**think は思考分の予算を見込む必要がある**。応答時間は 3〜7 倍 | Thinking モードの比較 |
| 4 | **Needle 失敗には 2 種類ある**: (a) **ctx スケーリング型**(deepseek-r1:7b は 4K で 3/3 → 16K で 1/3 → 65K 以降で 0/3、長くなるほど崩壊)、(b) **位置依存型**(deepseek-coder-v2 は ctx に依らず pos 90% のみ ✓、末尾しか参照しない検索能力の欠如) | Needle 網羅スイープ |
| 5 | **easy タスクはモデル選定の決め手にならない**: `add(a,b)` 等の合成 6 タスクは 6 モデル全てが 100%。難タスクに差し替えて初めて 0.815〜0.982 に分散 | 実コード相当の難タスク |
| 6 | **「サイズが大きい = 賢い」は明確に成り立たない**: DeepSeek-R1:14B は同シリーズの 8B(0528)よりスコアが低く、impl_flatten_dict では 2/6 と急落。最大モデルが勝つとは限らない | 実コード相当の難タスク |
| 7 | **Gemma 4 e4b は意外な総合トップ**: 難 Coding 0.982(最高)+ Summary 0.86 + Needle 全勝。**マルチモーダル指向の小型モデルがコード特化 MoE に並ぶ** | 実コード相当の難タスク |
| 8 | **Qwen3.5:9B には「バグはない」と主張する debug 盲点**: 明らかな two_sum バグを「論理的に正しい」と言い切り、num_predict を増やしても直らない。**LLM の自信過剰が debug 用途で致命的** | 実コード相当の難タスク |
| 9 | **MoE は decode が速いが日本語長文検索を苦手とする**: deepseek-coder-v2 は **decode 268 tok/s と全モデル中最速**、しかし needle は全 ctx で pos 90% しか拾えない。**コード補完専用と割り切る**のが妥当 | 100% GPU 動作 + Needle 網羅スイープ |
| 10 | **Ollama はモデル上限で要求 ctx を黙ってキャップする**: Gemma 4 で `--ctx 262144` を要求しても実効は `131072`。`/api/ps` の `context_length` でしか判明しない罠。**ctx_search.py で検出してログに残す** ようにしてある | ctx_search 実装 |
| 11 | **q8_0 化のメリットが見えない**: Qwen3.5 では q4_K_M → q8_0 で速度・max ctx・Summary すべて劣化。**VRAM を使って何も得ていない** | 100% GPU 動作 + 品質評価 |
| 12 | **DeepSeek-R1:7B 蒸留は think=false で不安定**: Needle で数字捏造 / 応答崩壊。同シリーズの 8B-0528(Qwen3 ベース)と 14B では問題なし。**think=false の安定度はリビジョンの新しさに依存** | 品質評価 |
| 13 | **`refactor_data_validator` は全モデル 9/9**: 6 段ネストの早期 return 化は LLM が得意な領域。**意味保存リファクタは小型モデルでも信頼できる** | 実コード相当の難タスク |

### 採用推奨(用途別)

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

### 最終分析

1. **最大搭載可能モデル**: `deepseek-r1:14b-qwen-distill-q4_K_M`(9 GB、ctx 31744)。CPU オフロードなしの 14B 級は q4_K_M なら 32K ctx までなら載る。重みを 16 GB に押し込もうとすると残 VRAM が KV を圧迫する
2. **最大搭載可能 ctx**: `qwen3.5:4b-q4_K_M` の **262144**。モデル自体の context_length 上限と一致。VRAM は 14.2 GB / 16 GB で 1.7 GB の余裕がある
3. **最速構成**: `deepseek-coder-v2:16b-lite-instruct-q4_0` の **268 tok/s**(MoE による実効パラメータの小ささが効く)。TTFT も 0.10 秒で対話 UX も最速
4. **最高品質構成**: Coding は 13 タグ中 10 タグが満点なので絞れない。Coding × Summary × Needle の積で比較すると `qwen3.5:4b-q4_K_M` と `gemma4:e4b-it-q4_K_M`(両方 Coding 1.00 / Summary 0.86 / Needle ✓)が同点トップ。速度・ctx で qwen3.5:4b-q4_K_M に軍配
5. **コーディング用途最適**: `deepseek-coder-v2:16b-lite-instruct-q4_0`(Coding 1.00 + 268 tok/s + TTFT 0.10s)。短中ファイルの編集なら 19K ctx で十分
6. **会議要約用途最適**: `qwen3.5:4b-q4_K_M`(Summary 0.86、長 ctx、135 tok/s)。マルチモーダルが要るなら `gemma4:e4b-it-q4_K_M`(Summary 0.86、128K ctx、131 tok/s)
7. **長コンテキスト用途最適**: `qwen3.5:4b-q4_K_M`(262K、Needle ✓)。256K で実用速度 135 tok/s が出る唯一の構成
8. **推奨運用構成**: 用途で 2 モデルを使い分け
   - **汎用 + 長文**: `qwen3.5:4b-q4_K_M`(全 13 タグで最もバランスが良い)
   - **コーディング特化**: `deepseek-coder-v2:16b-lite-instruct-q4_0`(速度と Coding 品質)
9. **推奨しない構成**: `deepseek-r1:7b-qwen-distill-*`(Needle 失敗 + Summary 0.53)、`qwen3.5:*-q8_0`(q4_K_M に対して全指標で劣化)。`deepseek-coder-v2:*` はコード以外の用途には選ばない
10. **トレードオフ分析**:
    - **重み量子化(q4 vs q8)**: Qwen3.5 では q8 にすると VRAM を 1.7 GB → 0.5 GB に削るだけで Summary は逆に下がる(0.86 → 0.58、0.78 → 0.64)。**q8 化は本評価の範囲ではメリットなし**。低 ctx で品質を狙うなら別だが、長 ctx 運用では q4_K_M 固定が合理的
    - **重みサイズ(4B vs 9B / 14B)**: 4B Qwen は Coding/Summary とも 9B と同等以上。14B は max_ctx が 31K まで縮むので長文用途に向かない。**16 GB GPU の最適点はおおむね 4〜8B 級**
    - **MoE vs Dense**: DeepSeek-Coder-V2 16B-MoE は実効 2.4B で 268 tok/s と Dense より明らかに速い。代わりに **長文検索能力が低い**(日本語長文 Needle 失敗)。用途特化なら強いが汎用には選ばない
    - **思考系モデル(R1)の think=false**: 7B distill は応答が破綻するか捏造する。8B-0528(Qwen3 ベース)・14B では問題なし。**think=false で安定するかはモデルの新しさに依存**
    - **マルチモーダル**: Gemma 4 はテキスト系の Coding/Summary でも上位。画像要件があれば優先度を上げる選択肢になる

---

## Apple M4 Pro (48 GB Unified Memory, MLX)

実機: MacBook Pro (Mac16,8) / Apple M4 Pro (8P + 4E) / 48GB Unified / macOS 26.3.1 / Darwin 25.3.0 / Driver: Metal (applegpu_g16s)

評価方法は [docs/06-evaluation.md](06-evaluation.md) に準拠。本セクションのみ推論ランタイムが **mlx-lm** で、Linux + Ollama 版とは合否判定ロジック・KV 量子化 API が異なる。詳細は [docs/mac/01-setup.md](mac/01-setup.md) / [docs/mac/02-benchmark.md](mac/02-benchmark.md) / [scripts/bench-mlx/README.md](../scripts/bench-mlx/README.md) を参照。

### 実効 GPU 上限

`mx.device_info()` から得られる値:

| 指標                                | 値        | 役割                                                                          |
| ----------------------------------- | --------- | ----------------------------------------------------------------------------- |
| `memory_size`                       | 48.00 GiB | ユニファイドメモリ総量                                                        |
| `max_recommended_working_set_size`  | 37.44 GiB | **実効 VRAM 相当**。これを超えると macOS が swap に逃がして全体が遅くなる     |
| `max_buffer_length`                 | 28.08 GiB | 単一バッファ上限。極端に大きなモデル重みの単一テンソル割当で効く              |

本評価では `safety_margin = 2048 MiB` を採用し、ピーク使用量が `37.44 - 2 = 35.4 GiB ≈ 36247 MiB` 以下に収まる構成のみ「100% GPU 動作」とみなす。

### 評価対象選定(2026-05-26 時点 mlx-community 調査)

[huggingface.co/mlx-community](https://huggingface.co/mlx-community) で実在確認済みのタグから、RTX 4080 SUPER (16GB) と直接比較できる Qwen3.5 / Gemma 4 / DeepSeek-R1 / DeepSeek-Coder-V2 系列 +、48GB なら載る 27B〜32B 級を追加した。

#### 対象モデル

##### Qwen3.5 系(汎用)

| HF タグ                                | 重み量子化 | 既定 ctx | ねらい                              |
| -------------------------------------- | ---------- | -------- | ----------------------------------- |
| `mlx-community/Qwen3.5-4B-MLX-4bit`    | 4bit       | (確認後) | 4080 の `qwen3.5:4b-q4_K_M` 相当    |
| `mlx-community/Qwen3.5-4B-MLX-8bit`    | 8bit       | (確認後) | 4080 の `qwen3.5:4b-q8_0` 相当      |
| `mlx-community/Qwen3.5-9B-MLX-4bit`    | 4bit       | (確認後) | 4080 の `qwen3.5:9b-q4_K_M` 相当    |
| `mlx-community/Qwen3.5-9B-MLX-8bit`    | 8bit       | (確認後) | 4080 の `qwen3.5:9b-q8_0` 相当      |
| `mlx-community/Qwen3.5-27B-4bit`       | 4bit       | (確認後) | **48GB 追加**: 4080 不可だった 27B  |
| `mlx-community/Qwen3.5-27B-8bit`       | 8bit       | (確認後) | **48GB 追加**: 27B 品質上限(KV 余地は薄い、計測で確認) |

##### Qwen3.6 系(最新 dense)

| HF タグ                          | 重み量子化 | 既定 ctx | ねらい                                                          |
| -------------------------------- | ---------- | -------- | --------------------------------------------------------------- |
| `mlx-community/Qwen3.6-27B-4bit` | 4bit       | (確認後) | **48GB 追加**: 4080 で 16GB 量子化タグが無く対象外だった Qwen3.6 |
| `mlx-community/Qwen3.6-27B-8bit` | 8bit       | (確認後) | **48GB 追加**: 27B 品質上限(KV 余地は薄い、計測で確認)         |

##### Gemma 4 系(マルチモーダル / MoE)

| HF タグ                                  | 重み量子化 | 既定 ctx | ねらい                                                          |
| ---------------------------------------- | ---------- | -------- | --------------------------------------------------------------- |
| `mlx-community/gemma-4-e2b-it-4bit`      | 4bit       | (確認後) | 4080 の `gemma4:e2b-it-q4_K_M` 相当                              |
| `mlx-community/gemma-4-e4b-it-4bit`      | 4bit       | (確認後) | 4080 の `gemma4:e4b-it-q4_K_M` 相当(マルチモーダル主力)         |
| `mlx-community/gemma-4-e4b-it-8bit`      | 8bit       | (確認後) | 4080 の `gemma4:e4b-it-q8_0` 相当                                |
| `mlx-community/gemma-4-26b-a4b-it-4bit`  | 4bit (MoE) | (確認後) | **48GB 追加**: Gemma 4 26B MoE (実効 4B)、4080 で対象外だった   |
| `mlx-community/gemma-4-31b-it-4bit`      | 4bit dense | (確認後) | **48GB 追加**: Gemma 4 31B dense、4080 で対象外だった           |

##### DeepSeek-R1 系(思考特化 distill)

| HF タグ                                              | 重み量子化 | 既定 ctx | ねらい                                                |
| ---------------------------------------------------- | ---------- | -------- | ----------------------------------------------------- |
| `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`     | 4bit       | (確認後) | 4080 の `deepseek-r1:7b-qwen-distill-q4_K_M` 相当      |
| `mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit`     | 8bit       | (確認後) | 4080 の `deepseek-r1:7b-qwen-distill-q8_0` 相当        |
| `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit`       | 4bit       | (確認後) | 4080 の `deepseek-r1:8b-0528-qwen3-q4_K_M` 相当        |
| `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit`    | 4bit       | (確認後) | 4080 の `deepseek-r1:14b-qwen-distill-q4_K_M` 相当     |
| `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit`    | 4bit       | (確認後) | **48GB 追加**: 4080 で対象外だった 32B                 |

##### DeepSeek-Coder-V2 系(コーディング MoE)

| HF タグ                                                       | 重み量子化 | 既定 ctx | ねらい                                                 |
| ------------------------------------------------------------- | ---------- | -------- | ------------------------------------------------------ |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`      | 4bit       | (確認後) | 4080 の `deepseek-coder-v2:16b-lite-instruct-q4_0` 相当 |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit`          | 8bit       | (確認後) | **48GB 追加**: 4bit との品質差を見る                    |

#### 対象外と理由(暫定)

| HF タグ / 重み                                                | 重みサイズ (実測)  | 理由                                                                                            |
| ------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| `mlx-community/Qwen3.5-122B-A10B-4bit` / `Qwen3.5-397B-A17B-4bit` | 60+ GB          | 48GB ユニファイドメモリに乗らない                                                               |
| `mlx-community/Qwen2.5-72B-Instruct-4bit`                     | 約 40 GB           | working_set 37GB を超過、KV 余地ほぼゼロ                                                        |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                          | 約 18 GB           | MoE、興味深いが本評価のスコープ拡大を避けて見送り                                               |
| `mlx-community/gemma-4-e4b-it-8bit`                           | 8.4 GB             | Gemma 4 `e*` は multimodal、e2b / e4b で既に失敗確認済み                                        |
| `mlx-community/Qwen3.5-9B-MLX-8bit`                           | 9.7 GB             | 4bit 版で MLX の品質傾向が確認できたため省略                                                    |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit`          | 15.6 GB            | 4bit 版で十分傾向が見える                                                                       |
| `mlx-community/Qwen3.5-27B-8bit` / `Qwen3.6-27B-8bit`         | 各 27.5 GB         | 重みだけで working_set の 75% を占有、KV 余地ほぼゼロで結果が自明                                |

### 測定環境(2026-05-26 計測)

- Chip: Apple M4 Pro (8P + 4E) / 48GB Unified Memory
- Effective GPU limit: 37.44 GiB (`max_recommended_working_set_size`)
- OS: macOS 26.3.1 (Build 25D771280a), Darwin 25.3.0
- mlx: 0.31.2 / mlx-lm: 0.31.3
- 計測スクリプト: `scripts/bench-mlx/` @ git 4a879ca(+作業ブランチ未コミット)
- 共通設定: `max_kv_size=<ctx_search の最大値>` / `kv_bits=None`(主力 pass) / `prefill_step_size=2048` / `safety_margin_mib=2048`
- 進行: `eval_cycle_mlx.sh` で 1 モデルずつ pull → 評価 → 削除 のサイクル(計測中)

### 100% GPU 動作

長コンテキスト・peak メモリが `working_set − 2GB` (≒ 35.4 GB) 以内、を共通条件として、各モデルの **二分探索で求めた最大 ctx での速度** を以下にまとめる。`Max ctx` 列の `*` はモデル既定 ctx 上限でキャップを示す。

> ⚠️ **計測中の暫定値**。`prefill_tps` は `run_speed_mlx.py` の既定プロンプトが ~60 token と短いため値が不安定(短プロンプトは前処理の固定コストが支配的)。長プロンプトでの本当の prefill 速度は **Needle テストの `prompt_tps`** を別途参照すること。

| Model                                | Max ctx       | Decode tok/s | Prefill tok/s(short) | TTFT (s) | Peak mem (MiB) | Needle Prefill (Mtok/s) |
| ------------------------------------ | ------------- | ------------ | -------------------- | -------- | -------------- | ----------------------- |
| Qwen3.5-4B-MLX-4bit                  | **262144**    | 77.09        | 100.83               | 0.49     | 2,448          | 270.5 @ 235K chars      |
| Qwen3.5-4B-MLX-8bit                  | **262144**    | 47.07        | 203.91               | 0.30     | 4,448          | 244.4 @ 235K chars      |
| Qwen3.5-9B-MLX-4bit                  | **262144**    | 48.32        | 142.36               | 0.37     | 5,009          | 178.1 @ 235K chars      |
| DeepSeek-R1-Distill-Qwen-7B-4bit     | 131072 `*`    | 57.21        | 121.21               | 0.35     | 4,205          | (集計中)                |
| DeepSeek-R1-Distill-Qwen-7B-8bit     | 131072 `*`    | 31.75        | 124.69               | 0.35     | 7,797          | 127.0 @ 117K chars      |
| DeepSeek-R1-0528-Qwen3-8B-4bit       | 131072 `*`    | 54.18        | 22.04                | 0.37     | 4,451          | (集計中)                |

### 品質評価:Needle / Coding / Summary

| Model                              | Needle | Coding | Summary | 備考                                                          |
| ---------------------------------- | ------ | ------ | ------- | ------------------------------------------------------------- |
| Qwen3.5-4B-MLX-4bit                | ✓      | 1.00   | 0.81    | 4080 と同等(decode は 4080 の 57%)                            |
| Qwen3.5-4B-MLX-8bit                | ✓      | 1.00   | 0.67    | 4080 8bit に概ね近い(4080: 1.00/0.58)。decode は 4bit の 61%   |
| Qwen3.5-9B-MLX-4bit                | ✓      | 1.00   | 0.70    | 4080 (1.00/0.78) と近い、Summary は若干下振れ                  |
| DeepSeek-R1-Distill-Qwen-7B-4bit   | ✗      | 0.50   | 0.20    | 4080: Needle ✗ / Coding 0.92 / Summary 0.53、MLX 4bit で大幅低下 |
| DeepSeek-R1-Distill-Qwen-7B-8bit   | ✗      | 1.00   | 0.14    | 4080: Needle ✗ / Coding 0.98 / Summary 0.53、8bit で Coding 回復・Summary は下振れ継続 |
| DeepSeek-R1-0528-Qwen3-8B-4bit     | ✗      | 0.67   | 0.20    | 4080: Needle ✓ / Coding 1.00 / Summary 0.75、MLX 4bit で大幅低下 |

#### Coding 種別内訳

| Model                              | Overall | impl | bugfix | refactor |
| ---------------------------------- | ------- | ---- | ------ | -------- |
| Qwen3.5-4B-MLX-4bit                | 1.00    | 1.00 | 1.00   | 1.00     |
| Qwen3.5-4B-MLX-8bit                | 1.00    | 1.00 | 1.00   | 1.00     |
| Qwen3.5-9B-MLX-4bit                | 1.00    | 1.00 | 1.00   | 1.00     |
| DeepSeek-R1-Distill-Qwen-7B-4bit   | 0.50    | 0.50 | 0.50   | 0.50     |
| DeepSeek-R1-Distill-Qwen-7B-8bit   | 1.00    | 1.00 | 1.00   | 1.00     |
| DeepSeek-R1-0528-Qwen3-8B-4bit     | 0.67    | 1.00 | 0.50   | 0.50     |

### 計測失敗(モデル / mlx-lm 非互換)

| Model                       | エラー                                                                   | 推測される原因                                                                                            |
| --------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `mlx-community/gemma-4-e2b-it-4bit` | `ValueError: Received 140 parameters not in model: language_model.model.layers.X...` | Gemma 4 の `e2b` / `e4b` はマルチモーダル(`language_model.*` 名前空間)。mlx-lm 0.31.3 の標準テキスト LM loader が拒絶。`mlx-vlm` 必要 |
| `mlx-community/gemma-4-e4b-it-4bit` | 同上                                                                     | 予測通り e4b も同じ構造で失敗                                                                             |

> Gemma 4 `e2b` / `e4b` 全タグ(`e2b-4bit` / `e4b-4bit` / `e4b-8bit`)はマルチモーダルのため mlx-lm では読めない。Gemma 4 `26b-a4b`(MoE Dense) と `31b`(Dense) はマルチモーダル名前空間を持たない可能性が高く別途試行する。

### 観察された MLX vs Ollama(4080)の品質差

中間集計の重要な傾向:

- **MLX 4bit は GGUF Q4_K_M より品質が落ちる傾向**:DSR1 7B distill / 0528-8B で Coding / Summary とも顕著に低下した。Qwen3.5-4B-MLX-4bit は 4080 と同等を維持
- **Needle の `think=false` 挙動**:DSR1 系列(7B / 0528-8B 含む)で Needle 失敗が増加。4080 でも 7B distill は失敗していたが、0528-8B は成功していた → MLX 4bit + think=false の組み合わせで悪化
- **decode 速度**:Qwen3.5-4B-MLX-4bit が 77 tok/s(4080 は 134.6 tok/s)。M4 Pro の decode は 4080 の **約 57%**(メモリ帯域比 ~270 GB/s vs ~700 GB/s に概ね一致)

これらは中間結果。全モデル測定後に最終分析として確定させる。

### CPU offload 相当(参考)

> Apple Silicon では明示的な「CPU offload」は存在しないが、`peak_memory > effective_gpu_limit` で macOS が swap に逃がしたケースをここに記録する。現時点では発生していない。

### 採用推奨(用途別)

> 全モデル測定完了後に確定する。

### 最終分析

> 全モデル測定完了後に [docs/06-evaluation.md §9.4](06-evaluation.md) の 10 項目で執筆する。

---

## (テンプレート)別GPUでの追加記載例

別環境で計測したら、以下のフォーマットで追記する。

```
## NVIDIA <GPU名> (<VRAM>GB, Compute <X.Y>, <世代>)

実機: <CPU> / <RAM> / <OS> / Driver <ver> (CUDA <ver>)

### 100% GPU 動作

| モデル | 重み量子化 | num_ctx | FA | KV | SIZE | VRAM使用 | 残VRAM | 生成 tok/s | プロンプト tok/s |
| ... |

### CPUオフロード発生(参考)

| ... |

### 採用推奨

- ...
```
