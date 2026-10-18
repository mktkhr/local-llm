# MLX ベンチ: Apple M4 Pro (48 GB Unified Memory)

macOS + mlx-lm で、16 GB GPU に載らない大型モデルを 48 GB Mac がどれだけの速度で動かせるかを計測した記録。RTX 4080 SUPER との同タグ直接比較を含む。

| 項目     | 値                                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------- |
| 期間     | 2026-05-27                                                                                                        |
| 方法論   | [docs/evaluation.md](../../docs/evaluation.md) / [docs/benchmark-mac.md](../../docs/benchmark-mac.md)             |
| harness  | [tools/bench-mlx](../../tools/bench-mlx/README.md)                                                                |
| 比較対象 | [2026-05-ollama-bench](../2026-05-ollama-bench/README.md)(RTX 4080 SUPER)                                         |
| 結論     | 本文末尾の「採用推奨」「最終分析」。横断の推奨表は [docs/hardware-reference.md](../../docs/hardware-reference.md) |

## Apple M4 Pro (48 GB Unified Memory, MLX)

実機: MacBook Pro (Mac16,8) / Apple M4 Pro (8P + 4E) / 48GB Unified / macOS 26.3.1 / Darwin 25.3.0 / Driver: Metal (applegpu_g16s)

評価方法は [docs/evaluation.md](../../docs/evaluation.md) に準拠。本セクションのみ推論ランタイムが **mlx-lm** で、Linux + Ollama 版とは合否判定ロジック・KV 量子化 API が異なる。詳細は [docs/setup-mac.md](../../docs/setup-mac.md) / [docs/benchmark-mac.md](../../docs/benchmark-mac.md) / [tools/bench-mlx/README.md](../../tools/bench-mlx/README.md) を参照。

### 実効 GPU 上限

`mx.device_info()` から得られる値:

| 指標                               | 値        | 役割                                                                      |
| ---------------------------------- | --------- | ------------------------------------------------------------------------- |
| `memory_size`                      | 48.00 GiB | ユニファイドメモリ総量                                                    |
| `max_recommended_working_set_size` | 37.44 GiB | **実効 VRAM 相当**。これを超えると macOS が swap に逃がして全体が遅くなる |
| `max_buffer_length`                | 28.08 GiB | 単一バッファ上限。極端に大きなモデル重みの単一テンソル割当で効く          |

本評価では `safety_margin = 2048 MiB` を採用し、ピーク使用量が `37.44 - 2 = 35.4 GiB ≈ 36247 MiB` 以下に収まる構成のみ「100% GPU 動作」とみなす。

### 評価対象選定(2026-05-27 改訂)

本セクションの問いは **「48GB の Mac は 16GB GPU に載らない大型モデルを動かせるが、どれだけ遅いのか / なぜ遅いのか」**。それを最小手数で語るため、Qwen3.5 dense のサイズ階段(4B→9B→27B)でアーキを固定した速度カーブを取り、両端に「最大 dense (32B)」と「大きいのに速い MoE」を置いた 5 モデルに絞った。重み量子化は **全て 4bit**(8bit は帯域を食って decode が遅くなるだけで、RTX 評価でも q8 は無益と確認済み)。KV は **q8**(RTX 主力 pass と同条件)。

#### 対象モデル(5 本)

| HF タグ                                                  | 重み     | 役割                                       | RTX 対応                 |
| -------------------------------------------------------- | -------- | ------------------------------------------ | ------------------------ |
| `mlx-community/Qwen3.5-4B-MLX-4bit`                      | 2.85 GB  | サイズ階段の底、256K ctx                   | 同タグ相当あり(直接比較) |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                      | 5.57 GB  | サイズ階段の中                             | 同タグ相当あり(直接比較) |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | 8.24 GB  | MoE 16B(実効 2.4B)。「大きいのに速い」対照 | 同タグ相当あり(直接比較) |
| `mlx-community/Qwen3.5-27B-4bit`                         | 14.98 GB | サイズ階段の上。**16GB GPU には載らない**  | 4080 不可                |
| `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit`        | 17.18 GB | 最大 dense。**遅さの極北**                 | 4080 不可                |

#### 対象外と理由

| HF タグ / 重み                                                          | 理由                                                                                                      |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `mlx-community/gemma-4-e2b-it-4bit` / `gemma-4-e4b-it-*`                | Gemma 4 `e*` はマルチモーダル(`language_model.*` 名前空間)で mlx-lm 0.31.3 が load 不可(実測で確認、後述) |
| 全 `*-8bit` 変種(Qwen3.5 4B/9B、DSC-V2-Lite ほか)                       | 4bit より重み帯域を食って decode が遅くなるだけ。RTX 評価でも q8 重み量子化は無益と判明済み               |
| `mlx-community/Qwen3.6-27B-*` / `gemma-4-26b-a4b` / `gemma-4-31b`       | まず Qwen 軸を綺麗に取る方針。必要なら後から追加 pull する                                                |
| `mlx-community/Qwen3.5-122B-A10B-4bit` / `Qwen2.5-72B-Instruct-4bit` 等 | 重み 40〜60GB+ で 48GB ユニファイドメモリに乗らない / KV 余地ゼロ                                         |

> **2026-05-26 の初回計測について**: KV を f16(非量子化)、warmup 無しで 6 モデルを測ったが、(1) RTX 主力 pass の KV q8 と条件が揃わない、(2) Metal shader の初回 JIT コンパイルが TTFT に混入する、の 2 点で比較に使えないことが判明。**KV q8 + warmup に揃えて全モデル再計測**する(`tools/bench-mlx/` の修正コミット参照)。以下の表は再計測後に確定。

### 測定環境(2026-05-27 計測)

- Chip: Apple M4 Pro (8P + 4E) / 48GB Unified Memory
- Effective GPU limit: 37.44 GiB (`max_recommended_working_set_size`)
- OS: macOS 26.3.1 (Build 25D771280a), Darwin 25.3.0
- mlx: 0.31.2 / mlx-lm: 0.31.3
- 計測スクリプト: `tools/bench-mlx/` @ git(計測時 hash)
- 共通設定: **KV q8(`--kv-bits 8`、RTX 主力 pass と同条件)** / warmup 1 token / `prefill_step_size=2048` / `safety_margin_mib=2048`
- 進行: `eval_cycle_mlx.sh` で 1 モデルずつ pull → 評価 → 削除 のサイクル

### 計測中に発生した事象:大型モデルの長 ctx プローブによる OS クラッシュ / OOM

長 ctx プローブ(`ctx_search` 高側、256K 相当のフィラーを実際に prefill)で以下が発生した。**いずれも weight が大きいモデルで起きており**、ctx_search の上限設定がクラッシュ要因になることが分かった。

| 事象      | モデル / 設定                       | 結果                                               | 証跡                                           |
| --------- | ----------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| 1 回目    | DSC-V2-Lite、`--high 262144`        | **OS クラッシュ → 強制再起動**(計測中、ユーザ報告) | クラッシュでログ途絶、JSON 未生成              |
| 2 回目    | DSR1-32B (17GB)、`--high 131072`    | **OS クラッシュ → 強制再起動**(計測中、ユーザ報告) | 同上(`[probe] high=131072` の直後でログ途絶)   |
| Metal OOM | DSC-V2-Lite、`--high 131072`        | プロセス内例外で停止、**OS は無事**                | `[METAL] ... Insufficient Memory` をログに記録 |
| 成功      | Qwen3.5-27B (15GB)、`--high 131072` | peak 32.9GB で **完走**(OS 落ちず)                 | ctx_search JSON あり                           |

**読み筋**:

- OS クラッシュは **直接ログが残らない**(再起動でプロセスごと消える)ため、上記 1・2 回目は **ユーザ報告 + ログ途絶位置からの推定**。Metal OOM(DSC@131K)と 27B@131K 成功は **JSON / ログで確認済み**
- 同じ `--high 131072` でも **27B(weight 15GB)は完走 / 32B(weight 17GB)は OS クラッシュ**。weight 2GB の差で、長 ctx prefill 時の peak が working_set (37GB) の安全域を超えるかどうかが分かれた
- Qwen3.5-4B の ctx スイープでは 256K で peak 25GB + swap delta 8.5GB を実測(swap が走っている。下表参照)。大型モデルではこれが OS を巻き込む規模になる

**対策 / 再開時の設定**:

- `eval_cycle_mlx.sh` に `HIGH_CTX` 環境変数を導入。**weight が大きいモデルは `HIGH_CTX` を下げて起動**(27B: 131072 で完走 / 32B: 16384 まで下げて安全確保 / DSC-V2-Lite: 49152 で完走)。二分探索は安全側から上限を探すため、危険な高 ctx プローブだけが消える
- 27B / 大型モデルの `run_ctx_sweep_mlx.py` は省略(4B で decode/prefill の ctx 依存は鮮明に取れており、大型の max_ctx 近傍 sweep は同じ swap リスクを抱えるため割に合わない)
- 計測中は重い常駐アプリ(Outlook / Teams / IDE 等)を停止することを推奨

> **Mac/MLX 環境固有のリスク**: NVIDIA GPU は OOM で **プロセスがエラー終了**するだけで OS は無事だが、Apple Silicon のユニファイドメモリは **OS と推論ランタイムが同じプールを取り合う**ため、過度な確保は OS 全体の安定性を損なう。`safety_margin_mib` は安全側に振っておくべき(本評価は 2048 MiB)。

**DSC-V2-Lite の Metal OOM について**: `--high 131072` で次の例外が出た(OS は無事):

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Insufficient Memory
```

131K の prefill で Metal アロケータが失敗。weight 8.2GB と軽い割に長 ctx で OOM するのは、**このモデルの KV 形状(層数 × KV ヘッド数 × head_dim × ctx)とランタイム実装上のメモリ要求が、weight の割に大きい**ため(MoE/dense を問わず KV 量は注意機構の形状で決まる)。OS 落ちはせず MLX 例外で停止し、`--high 49152` で再 probe して完走した。

### 速度(decode / 長文 prefill)

`Decode tok/s(短)` は短プロンプト後の生成速度(decode 上限)、`Needle TTFT` は max_ctx 近くまで詰めた長文 prefill の所要時間(対話 UX の長文初動)。

| Model                                       | Max ctx    | Peak mem @max (GB) | Decode tok/s(短) | Needle prefill tok/s | Needle TTFT (s) |
| ------------------------------------------- | ---------- | ------------------ | ---------------- | -------------------- | --------------- |
| Qwen3.5-4B-MLX-4bit                         | **262144** | 24.3               | 83.1             | 216.9                | 639             |
| Qwen3.5-9B-MLX-4bit                         | **262144** | 26.9               | 49.0             | 177.9                | 779             |
| DeepSeek-Coder-V2-Lite-Instruct-4bit (MoE)  | 49152 `§`  | 21.1               | **114.0**        | 286.6                | 190             |
| Qwen3.5-27B-4bit                            | 131072 `†` | 32.9               | 15.3             | 65.0                 | 1,062           |
| DeepSeek-R1-Distill-Qwen-32B-4bit ⚠思考強制 | 16384 `‡`  | 22.6 (@16K)        | 12.2             | 60.2 (@16K)          | 181 (@16K)      |

> `Peak mem @max` は各モデルの ctx_search 高側プローブ(= max_ctx ロード時)の `mx.get_peak_memory()`。出所を揃えてある(32B のみ max_ctx=16K 時点の値)。

> `Decode tok/s(短)` は短プロンプト直後の生成速度(decode の上限値)。実際は ctx が伸びると大きく落ちる(下の ctx スイープ参照)。
> `†` = 安全のため `HIGH_CTX=131072` で探索を打ち切った値(クラッシュ対策、上述)。27B は 131072 で既に peak 33.7GB(working_set 35.4GB の安全域上限)なので、実上限もほぼこの近辺。
> `‡` = 32B は 131072 プローブで **OS クラッシュ**(上表の 2 回目)したため、`HIGH_CTX=16384` まで下げて安全側のみ計測(値は実上限ではなく安全 cap)。16K でも needle TTFT 181 秒・decode 9.2 tok/s。**48GB Mac で 32B は長 ctx 不可・短 ctx でも実用ギリ**で、実用構成として成立しない。coding は途中まで走ったが完走 JSON が残る前に停止したため、本文では不採用とした。
> `§` = DSC-V2-Lite は当初 `HIGH_CTX=131072` で Metal OOM(weight 8.2GB の割に長 ctx の KV メモリ要求が大きく、131K で確保失敗)。`HIGH_CTX=49152` で再計測し 49152 まで安全に確認(peak 21.1GB)。実上限はもう少し高い可能性があるが、安全側で打ち切った値。

### 品質(Needle / Coding hard / Summary)

> Coding は RTX と同じ難タスク (`tasks_hard.json`)。num_predict は **needle 2048 / coding 8192 / summary 4096**(後述の DSR1-32B 強制思考に対応するため RTX より大きく取るが、非思考モデルは EOS で早期停止するため出力は RTX と同一)。各結果に `truncated`(finish_reason==length)を記録し、打ち切りが無いことを検証する。

| Model                                       | Needle  | Coding hard    | Summary | truncated | 備考                                                                                                                                   |
| ------------------------------------------- | ------- | -------------- | ------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.5-4B-MLX-4bit                         | ✓       | 0.945          | 0.75    | 0         | RTX 4b (Coding 0.963) と同等。impl 0.963 / bugfix 1.0 / refactor 0.778                                                                 |
| Qwen3.5-9B-MLX-4bit                         | ✓       | 0.982          | 0.70    | coding 1  | RTX 9b (0.815〜0.904) より上振れ。truncated 1 件は合格後に冗長生成継続したもので採点は有効                                             |
| DeepSeek-Coder-V2-Lite-Instruct-4bit (MoE)  | ✓       | 0.963          | 0.722   | 0         | 16B 総 / 実効 2.4B。impl 0.926 / bugfix 1.0 / refactor 1.0。RTX では Needle ✗(pos50 不可)だったが MLX 版は pos50 で成功                |
| Qwen3.5-27B-4bit                            | ✓       | **1.00**       | 0.666   | 0         | Qwen3.5 dense で Coding 最高。サイズ順に 0.945→0.982→1.00。Summary はサイズ非依存で頭打ち                                              |
| DeepSeek-R1-Distill-Qwen-32B-4bit ⚠思考強制 | ✓ (16K) | 完走 JSON なし | 未計測  | —         | 16K で needle ✓ のみ確認。coding は途中まで走ったが停止し完走結果が残らず、本文では不採用。実用不可判断は速度・TTFT で十分(速度表 `‡`) |

> **DSR1-32B は always-think**: chat template 末尾が `<｜Assistant｜><think>\n` で思考を強制し、`enable_thinking=False` を無視する(実測確認)。Ollama の `think:false` で測った RTX の DSR1 系とは **別タスク**(think=true 相当)。num_predict を大きく取って打ち切りを防ぐが、TTFT・総出力長・採点への思考の影響は残る。Mac の DSR1-32B 行は「思考込み」の値として読むこと。

### ctx 依存スイープ(decode は帯域律速 / prefill は compute 律速の分離)

> 代表 2 モデル(Qwen3.5-4B-4bit と Qwen3.5-27B-4bit)で ctx ∈ {4K, 16K, 64K, max} の decode / prefill / TTFT を計測。`run_ctx_sweep_mlx.py`。「なぜ遅いか」の核心データ。

| Model           | ctx     | Prefill tok/s | TTFT (s)  | Decode tok/s | Peak mem (MiB) | swap Δ (MiB) |
| --------------- | ------- | ------------- | --------- | ------------ | -------------- | ------------ |
| Qwen3.5-4B-4bit | 4,096   | 602.5         | 6.7       | 74.6         | 4,078          | 0            |
| Qwen3.5-4B-4bit | 16,384  | 541.7         | 30.2      | 58.1         | 5,013          | 0            |
| Qwen3.5-4B-4bit | 65,536  | 375.0         | 174.6     | 27.2         | 8,850          | 0            |
| Qwen3.5-4B-4bit | 262,144 | 148.0         | **1,770** | **7.0**      | 24,957         | 8,543        |

**読みどころ(Qwen3.5-4B、ctx を 64 倍に伸ばした影響)**:

- **decode は 83→7 tok/s と約 1/12**。KV キャッシュが ctx に比例して膨らみ、1 トークン生成ごとに全 KV を読むため帯域律速で落ちる。256K では peak 25GB の大半が KV で、読み出しだけで律速
- **TTFT は 6.7→1,770 秒(約 264 倍)と超線形**。プロンプト長を 4 倍にするたび TTFT が 5〜10 倍に膨らむ(6.7→30→175→1,770)。アテンションが O(ctx²) の compute 律速だから。**256K 一括投入は初動 29 分** で対話用途は完全に不成立
- **256K で swap が発生**(swap Δ 8,543)。peak 25GB 自体は working_set 37GB 内だが、他プロセス込みで macOS がページアウトを始める。decode 崩壊(7 tok/s)はこの影響も含む
- → **decode 曲線(緩やか)と TTFT 曲線(急峻)の傾き差**が「短文 decode は実用 / 長文 prefill は致命的」を定量的に裏づける

### KV 量子化対照(q4 vs q8、Qwen3.5-9B-4bit)

> RTX 評価では KV `q4_0` が「ほぼフリーランチ(max ctx +10〜39%、速度劣化 ≤6%、品質維持)」と判明。Mac/MLX でも同様か 1 モデルで確認。

| KV       | Max ctx    | Decode tok/s(短) | Decode tok/s(256K, needle) | Needle prefill tok/s | peak @256K (MiB) | Needle | 備考                                            |
| -------- | ---------- | ---------------- | -------------------------- | -------------------- | ---------------- | ------ | ----------------------------------------------- |
| q8(主力) | 262144 `*` | 49.0             | 14.0                       | 177.9                | 17,338           | ✓      | —                                               |
| q4       | 262144 `*` | 50.5             | **19.4**                   | **194.9**            | **16,196**       | ✓      | 長 ctx decode +39% / prefill +10% / peak -1.1GB |

`*` = モデル既定 256K 上限でキャップ(q4 でも ctx は伸びない=メモリではなくモデルが天井)。

**読みどころ**: RTX の「KV q4 はほぼフリーランチ」が Mac/MLX でも成立。**同 ctx なら q4 のほうが速く(KV バイト数が半分で読み出しが軽い)、メモリも減り、Needle 品質は維持**。ただし 9B は 48GB では 256K でモデル上限に当たるため、RTX で見られた「q4 で max ctx が伸びる」効果は出ない(メモリに余裕があり ctx 律速がモデル側)。**長 ctx 運用は q4 を既定にしてよい**。

### vs RTX 4080 SUPER(同タグ直接比較)

decode の比率が「Mac はどれだけ遅いか」の核心。

| Model             | 4080 decode     | M4 Pro decode(短) | Mac 比率    |
| ----------------- | --------------- | ----------------- | ----------- |
| Qwen3.5 4B        | 134.6           | 83.1              | 62%         |
| Qwen3.5 9B        | 88.9            | 49.0              | 55%         |
| Qwen3.5 27B       | (16GB に載らず) | 15.3              | —(RTX 不可) |
| DSC-V2-Lite (MoE) | 268.2           | 114.0             | 42%         |

> **サイズ → decode(M4 Pro、短 ctx、KV q8)**: dense は 4B 83.1 → 9B 49.0 → 27B 15.3 tok/s。重みが大きいほど 1 トークンあたりのメモリ読み出しが増え、帯域律速で decode がサイズに対し強い負相関で低下。**27B は 16GB GPU には載らないが Mac なら動く — ただし 15 tok/s で対話はギリギリ、128K 投入時は decode 7.8 tok/s / TTFT 18 分で長文は非現実的**。
>
> **MoE が階段を破る**: DSC-V2-Lite は総 16B(9B より大きい)にもかかわらず **decode 114 tok/s と 9B(49)の 2.3 倍速い**。実効アクティブが 2.4B しかないため。**「decode 速度を決めるのは総パラメータではなくアクティブパラメータ × メモリ帯域」**を実証。「載るが遅い」の本質は総サイズではなく、毎トークン読むメモリ量にある。
>
> RTX 比は 4B 62% / 9B 55% / MoE 42%。**メモリ帯域比(~273/736 ≈ 37%)と同じオーダーで整合的**だが、dense 2 本は 55〜62% と上振れしており「帯域比そのもの」ではない。差は量子化方式(K-quants vs group quant)・ランタイム・アーキの寄与とみられる。「帯域が主因」は言えるが「帯域比に張り付く」とまでは言えない。

### 計測失敗(mlx-lm 非互換)

| Model                                   | 原因                                                                                                                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gemma-4-e2b-it-*` / `gemma-4-e4b-it-*` | Gemma 4 `e*` はマルチモーダル(`language_model.*` 名前空間)。mlx-lm 0.31.3 の標準テキスト LM loader が `ValueError: Received N parameters not in model` で拒絶。テキスト用途でも `mlx-vlm` が必要 |

### 考察:なぜ Mac は RTX より遅いのか

実測で次の 4 つの要因に分解できた。

1. **decode はメモリ帯域律速**。decode は 1 トークンごとに全重み + 全 KV をメモリから読むため、速度はメモリ帯域に強く支配される。M4 Pro ~273 GB/s / RTX 4080 SUPER ~736 GB/s ≈ **37%**。実測の decode 比は 4B 62% / 9B 55% / MoE 42% で、**帯域差と整合的だが帯域比そのものには張り付いていない**(dense 2 本は 55〜62% と上振れ)。差分は量子化方式(K-quants vs MLX group quant)・ランタイム実装・アーキの寄与とみられる。それでも dense のサイズ階段 **4B 83 → 9B 49 → 27B 15 tok/s** という強い負相関は「重い=毎トークン読む量が多い=遅い」を明確に裏づける。

2. **MoE がその階段を破る**。DSC-V2-Lite は総 16B(9B より大)だが **decode 114 tok/s と 9B の 2.3 倍**。decode で読むのは活性化した実効 2.4B 分の重みだけだから。**decode 速度を決めるのは総パラメータではなくアクティブパラメータ × 帯域**。「大きいのに速い」は MoE の構造的帰結(なお KV キャッシュ量は MoE/dense を問わず注意機構の形状で決まり、expert 数では増えない — DSC-V2-Lite が長 ctx で OOM したのは KV 形状と実装上の要求が weight の割に大きいため)。

3. **長文 prefill は ctx² で悪化し、RTX との差が decode 以上に開く**。アテンションの計算量は O(ctx²)。Qwen3.5-4B の ctx スイープで **TTFT 6.7s(4K)→ 1,770s(256K、29 分)と 264 倍** に膨張(4080 は同条件で秒オーダー)。decode が緩やかに落ちるのに対し prefill TTFT が急峻に悪化する(同じ 4B 内で decode は ×12、TTFT は ×264)ことから、**prefill は decode と別の要因(行列演算スループット)に支配されている**と読める。NVIDIA は低精度 GEMM / Flash Attention の最適化が成熟しており、その差が長文で顕在化する。**長文一括投入は Mac では対話として成立しない**。

4. **大型は「載る」が decode が実用域を割り、長文は物理的に不可**。27B(decode 15、128K で TTFT 18 分)/ 32B(decode 12、長 ctx 不可)は 16GB GPU には載らないが Mac なら動く — しかし速度・ctx の両面で実用に届かない。さらに長 ctx プローブで **ユニファイドメモリの取り合いにより OS ごとクラッシュ**した(DSC-V2-Lite@256K と 32B@131K の 2 回、いずれも計測中に強制再起動。直接ログは残らないが weight 15GB の 27B@131K は完走しており、weight 差が境界)。NVIDIA は OOM がプロセスエラーで済むのに対し、**Apple Silicon は OS 安定性まで巻き込む**のが構造的リスク。

**揃えられない限界(比較の注意)**:

- **量子化アルゴリズム**: Ollama の `Q4_K_M`(K-quants)と MLX の `4bit`(group quant)は別物。同じ「4bit」でも品質・速度特性が異なる。トークナイザも GGUF と HF で別
- **モデルアーキの差**: 4B/9B/27B/MoE で GQA の KV ヘッド数・head_dim・層数が異なり、decode/KV メモリを「総パラメータ」だけで語るのは粗い。サイズ階段の傾向は明確だが、厳密には KV 形状の差も効く
- **実装パラメータ**: `prefill_step_size=2048`(MLX 既定)、MLX の attention カーネル成熟度、macOS のメモリ圧縮・swap・前景アプリ負荷も長文 prefill と OOM 挙動に影響する

本比較は「同一モデル系列・同一評価データ・同一採点・KV q8 統一・warmup 統一」までは揃えたが、上記は残差として残る。

### 採用推奨(用途別)

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

### 最終分析

1. **最大搭載可能モデル**: `DeepSeek-R1-Distill-Qwen-32B-4bit`(weight 17GB)。48GB ユニファイドメモリなら 32B-4bit まで load 自体は可能。ただし long ctx は OS クラッシュ、実用 ctx は ~16-32K に制限。**「載る」と「使える」は別**。
2. **最大搭載可能 ctx**: `Qwen3.5-4B-MLX-4bit` / `9B-MLX-4bit` の **262144**(モデル既定上限)。9B でも 256K が peak 17GB で乗り、48GB の余裕は大きい。ctx の天井は **メモリではなくモデルの context_length** 側。
3. **最速構成**: `DeepSeek-Coder-V2-Lite-Instruct-4bit`(MoE)の **decode 114 tok/s**。実効 2.4B が効く。
4. **最高品質構成**: `Qwen3.5-27B-4bit`(Coding hard 1.00)。ただし decode 15 tok/s。dense はサイズ順に品質向上(Coding 0.945→0.982→1.00)。
5. **コーディング用途最適**: 速度重視なら `DSC-V2-Lite`(114 tok/s, Coding 0.963)、品質重視なら `Qwen3.5-27B`(15 tok/s, 1.00)。短中文なら速度を取って MoE。
6. **会議要約用途最適**: `Qwen3.5-9B-MLX-4bit`(Summary 0.70、長 ctx、decode 49)。Summary は全サイズで 0.67〜0.75 と頭打ちでサイズを上げる旨味が薄いため、速度と ctx で 9B。
7. **長コンテキスト用途**: dense の 4B/9B が 256K まで載る唯一の選択肢。ただし **prefill TTFT が ctx² で悪化**するため、実用上は数万トークンまで。100K 超はバッチ処理前提。
8. **推奨運用構成**: 汎用 `Qwen3.5-9B-MLX-4bit` + KV q4 を主力、コード高速用途に `DSC-V2-Lite` を併用。
9. **推奨しない構成**: 32B(実用不可)、8bit 重み(無益)、Mac での長文一括投入。
10. **トレードオフ分析**:
    - **decode 速度 vs サイズ**: dense は帯域律速で総パラメータと強い負相関(4B 83 → 27B 15)。**16GB GPU なら 4〜9B が最適点という結論は Mac でも同じ**だが、Mac は絶対速度が 4080 の 5〜6 割。
    - **MoE vs dense**: MoE は decode が速い(総 16B で 114 tok/s、活性 2.4B だから)が、本モデルは **長 ctx で OOM し max ctx が伸びなかった**(131K で OOM、49K まで確認)。KV 量はアーキ形状で決まり MoE 固有ではないが、本モデルは weight の割に KV 要求が大きい。長文は dense、速度は MoE。
    - **KV q4 vs q8**: q4 が長 ctx decode +39% / メモリ -1.1GB / 品質維持で **フリーランチ**。RTX と同結論。
    - **長文 prefill**: Mac 最大の弱点。TTFT が ctx² で爆発し、256K で 29 分。**Mac でローカル LLM を使うなら「短い prompt で回す」設計に倒すのが必須**。
    - **大型モデルの価値**: 48GB は 16GB GPU が載せられない 27B/32B を載せられるのが唯一の優位点。だが decode が実用域を割り、長文は OS クラッシュ。**「大きいモデルが動く」ことと「実用になる」ことは Mac では一致しない**。
