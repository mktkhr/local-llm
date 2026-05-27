# scripts/bench-mlx/

[docs/06-evaluation.md](../../docs/06-evaluation.md) の評価方法論を **Apple Silicon / MLX 環境向け** に実装した計測スクリプト群。Linux + CUDA + Ollama 版は [scripts/bench/](../bench/)。

機能は bench/ と同じ:**速度計測 / 最大 ctx 二分探索 / KV キャッシュ切替 / Needle In A Haystack / Coding 評価 / Summary 評価**。加えて Mac の「なぜ遅いか」分析用に **ctx 依存スイープ(`run_ctx_sweep_mlx.py`)** を持つ。

> **計測の妥当性メモ**:
>
> - 全ランナーは load 直後に warmup(1 token)を入れ、Metal shader 初回 JIT を本計測から外す
> - KV は `--kv-bits 8` を主力(RTX 主力 pass の q8_0 と同条件)。`eval_cycle_mlx.sh` は Qwen3.5-9B のみ q4 対照も取る
> - needle/coding/summary は `finish_reason` / `truncated` を保存。思考強制モデル(DeepSeek-R1 distill は chat template が `<think>` を強制)が num_predict を食い潰して途中打ち切りしていないかを検証できる
> - `aggregate_mlx.py` は `--kv-bits`(既定 8)で metadata の kv_bits を見て集計対象を絞り、q4/q8 混成行を防ぐ

## 前提環境

| 項目            | バージョン / 内容                  | 用途                                                  |
| --------------- | ---------------------------------- | ----------------------------------------------------- |
| OS              | **macOS 14 以上(動作確認は 26.3)** | mlx-metal / unified memory 前提                       |
| Chip            | **Apple Silicon (M1 〜 M4)**       | MLX は ARM64 専用、Intel Mac では動かない             |
| Python          | **3.11 以上**                      | `pyproject.toml` で指定                               |
| uv              | 0.8 以上                           | 依存管理と仮想環境                                    |
| mlx             | **0.31 以上**                      | Apple MLX フレームワーク                              |
| mlx-lm          | **0.31 以上**                      | LLM 用ラッパー、ストリーミング生成・KV 量子化サポート |
| huggingface_hub | 0.24 以上                          | `mlx-community/*` モデル取得                          |
| git             | 任意                               | `metadata_mlx.py` がコミット hash を採取するため      |

[docs/mac/01-setup.md](../../docs/mac/01-setup.md) でセットアップ手順、ハードウェア事前確認、ストアアプリ停止などを確認すること。

## セットアップ

```bash
cd scripts/bench-mlx
uv sync
```

これで `.venv/` が作られて依存(mlx, mlx-lm, huggingface_hub)が解決される。以降は `uv run python <script>.py ...` で実行する。

## 構成

```
scripts/bench-mlx/
├── client_mlx.py         # mlx-lm のラッパー(streaming TTFT 計測込み)
├── metadata_mlx.py       # macOS / Apple Silicon / mlx version / git hash
├── memory_mlx.py         # mx.get_peak_memory / device_info / vm_stat ベースの計測
├── ctx_search_mlx.py     # 最大 max_kv_size の二分探索
├── run_speed_mlx.py      # 速度計測ランナー
├── run_needle_mlx.py     # Needle In A Haystack ランナー
├── run_coding_mlx.py     # Coding 評価ランナー
├── run_summary_mlx.py    # Summary 評価ランナー
├── pull_models_mlx.py    # mlx-community のモデルを一括 huggingface-hub 取得
├── run_all_mlx.sh        # 全モデルに対して ctx_search → speed → needle → coding → summary を回す
├── aggregate_mlx.py      # results/ を Markdown 表に集計(docs/04-results.md 用)
├── scorer.py -> ../bench/scorer.py    # scoring ロジックは bench/ と共有 (symlink)
├── data/ -> ../bench/data             # coding/summary/needle データは bench/ と共有 (symlink)
├── results/              # 計測結果(gitignore)
├── pyproject.toml
├── uv.lock
└── README.md
```

> **データと scorer は bench/ から symlink** している。これにより Linux 版と Mac 版で「同一の評価データ + 同一の採点ロジック」を保ったまま、ランタイムだけ差し替えた比較ができる。

## 使い方

### 1. 速度計測

```bash
uv run python run_speed_mlx.py --model mlx-community/Qwen3.5-9B-MLX-4bit --ctx 32768
```

オプション:

- `--model` (必須): HuggingFace の `mlx-community/*` repo id またはローカルパス
- `--ctx`: `max_kv_size`。未指定なら無制限(モデル既定)
- `--num-predict`: 生成上限トークン数(既定 128)
- `--prompt`: 任意のプロンプトに差し替え
- `--think true|false`: thinking モデル向け (Qwen3 系の `enable_thinking`)。代表値は `false`
- `--kv-bits {4,8}`: KV キャッシュ量子化 bit 数。未指定なら非量子化
- `--kv-group-size`: KV 量子化のグループサイズ(既定 64)
- `--prefill-step`: `prefill_step_size`(既定 2048)

出力: `results/<timestamp>/speed_<model>_ctx<ctx>.json` + `metadata.json`

### 2. Needle テストデータ生成

bench/ と同じ。`data/needle/generate.py` は symlink 経由で参照される:

```bash
uv run python data/needle/generate.py \
    --chars 50000 --position-pct 0.5 \
    --output data/needle/32k_p50.json
```

### 3. Needle テスト実行

```bash
uv run python run_needle_mlx.py --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --ctx 32768 --needle data/needle/32k_p50.json
```

### 4. KV キャッシュ量子化の切替

bench/ の `kv.py` のように systemd を経由する必要はない。`--kv-bits {4,8}` をランナーに渡すだけ。

```bash
uv run python run_speed_mlx.py --model "$MODEL" --ctx 131072 --kv-bits 4
uv run python run_speed_mlx.py --model "$MODEL" --ctx 131072 --kv-bits 8
uv run python run_speed_mlx.py --model "$MODEL" --ctx 131072            # 非量子化
```

> mlx-lm はリクエストごとに KV を量子化できるため、Ollama のような「サーバ再起動 + 比較」フローは不要。`ctx_search_mlx.py` も `--kv-bits` を受け取る。

### 5. 最大 ctx の二分探索

CPU オフロードに該当する概念は MLX には無いため、代わりに **「ピークメモリが `effective_gpu_limit_mib − safety_margin` に収まるか」** で OK/NG を判定する。

```bash
uv run python ctx_search_mlx.py --model mlx-community/Qwen3.5-9B-MLX-4bit
uv run python ctx_search_mlx.py --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --low 4096 --high 262144 --tolerance 4096 --safety-margin-mib 2048
```

オプション:

- `--low`: 探索下限(既定 4096)
- `--high`: 探索上限(既定 262144)
- `--tolerance`: 探索打ち切り粒度(既定 4096)
- `--safety-margin-mib`: `effective_gpu_limit_mib` から差し引くマージン(MiB)。これ以下にピークが収まれば OK(既定 1024)
- `--kv-bits {4,8}`: KV 量子化を有効にして探索

実装上の留意点:

- mlx-lm は KV を **lazy allocate** する(Ollama のような事前確保はしない)。そのため `ctx_search_mlx.py` は「ctx 相当の長さのフィラーを実際に prefill して」ピークメモリを計測する。長 ctx のプローブには時間がかかる(M4 Pro で 200K トークンの prefill ≒ 100 秒)
- モデル自身の `max_position_embeddings` を超える ctx 要求は **ok_capped** として扱い、それ以上は探索しない

出力: `results/<timestamp>/ctx_search_<model>.json` に探索履歴と `max_ctx` を保存。

### 6. モデル一括 pull

```bash
./pull_models_mlx.py                                  # 既定の Mac 評価対象リスト
./pull_models_mlx.py mlx-community/foo mlx-community/bar  # 指定タグだけ pull
```

ログは `results/pull-<unixtime>.log` に残る。

> モデルサイズは事前にチェックされない。SSD 残量に注意(13 モデル全部 pull すると 100GB を超える)。

### 7. Coding 評価

```bash
uv run python run_coding_mlx.py --model mlx-community/Qwen3.5-9B-MLX-4bit --ctx 8192 \
    --tasks data/coding/tasks.json
```

サンプルデータは `data/coding/tasks.json`(bench/ と共有、**6 タスク**:impl × 2、bugfix × 2、refactor × 2)。

出力: `results/<timestamp>/coding_<model>_ctx<ctx>.json`(タスク別合格率 + 種別ごとの平均)。

### 8. Summary 評価

```bash
uv run python run_summary_mlx.py --model mlx-community/Qwen3.5-9B-MLX-4bit --ctx 8192 \
    --meetings data/summary/meetings.json
```

サンプルデータは `data/summary/meetings.json`(bench/ と共有、**2 会議**:プロジェクト進捗 + 技術選定)。

オプション: `--ngram` / `--threshold` で被覆率判定の挙動を調整できる。

出力: `results/<timestamp>/summary_<model>_ctx<ctx>.json`(カテゴリ別 matched 率 + 全体平均)。

### 9. 一括実行 (run_all_mlx.sh)

```bash
./run_all_mlx.sh                                     # 既定 17 モデル(事前 pull 済み前提)
./run_all_mlx.sh mlx-community/Qwen3.5-9B-MLX-4bit   # 指定モデルだけ
```

各モデルに対して ctx_search → speed → needle → coding → summary を順に実行する。ログは `results/run_all-<timestamp>.log`。

> **前提**: モデルは pull 済みであること(`pull_models_mlx.py` 後に使う)。事前 pull 全部分のディスクが要る点に注意。

### 10. ディスク逼迫環境向け: pull → 評価 → 削除 サイクル (eval_cycle_mlx.sh)

ディスク容量が全モデル分(約 220GB)無い場合に使う。

```bash
./eval_cycle_mlx.sh                                  # 既定 20 モデル(全タグ)
./eval_cycle_mlx.sh mlx-community/Qwen3.5-9B-MLX-4bit
```

各モデルに対して以下を順番に実行する:

1. **pull**: HuggingFace から該当モデルを取得
2. **ctx_search → speed → needle → coding → summary**: `run_all_mlx.sh` と同じ
3. **cleanup**: `~/.cache/huggingface/hub/models--<org>--<repo>` を削除して次のモデルへ

最大瞬間ディスク使用量は「次に評価するモデルの重みサイズ」(最大 ~28 GB)で抑えられる。総時間は `run_all_mlx.sh` より pull の分長い(全タグ pull で約 +30〜60 分)。ログは `results/eval_cycle-<timestamp>.log`。

> 評価データ(`results/*/speed_*.json` 等)は削除されないため、累積する集計結果は完全に残る。

### 11. 集計

```bash
uv run python aggregate_mlx.py | tee /tmp/results_table.md
```

`results/` 配下から最新の各計測を拾って docs/04-results.md 用の Markdown 表を生成する。

## 採点ポリシー

[docs/06-evaluation.md §7](../../docs/06-evaluation.md) と同じ。**LLM-as-a-judge を使わない**。`scorer.py` は bench/ から symlink しているため、Linux 版と完全に同一ロジック。

- Coding は実行ベース(pytest 不要、stdlib のみ)で pass/fail を取る
- Summary は n-gram 被覆率で文字列ベースに判定する
- Needle は needle_id の含有チェック

## 既知の制約 / 注意点

| 項目                               | 内容                                                                                    | 対処                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **mlx-lm のバージョン依存**        | KV 量子化 API は 0.20 以降。`peak_memory` フィールドは 0.21 以降                        | `pyproject.toml` で `mlx-lm>=0.20` を固定。出力 JSON にバージョン記録       |
| **同一プロセス再ロード**           | model オブジェクトを del + `mx.clear_cache()` してもピーク値は残る                      | 計測は **必ず別プロセス**で実行する(bench-mlx スクリプトはこれを守っている) |
| **ctx_search の prefill コスト**   | 長 ctx プローブは時間が線形にかかる                                                     | Pass 1 では `--tolerance` を粗めにする(8192 など)                           |
| **macOS の swap 動作**             | OOM ではなく swap に逃げる挙動。`peak_memory` が `working_set` 超過しても例外にならない | `effective_gpu_limit_mib - safety_margin` の閾値で NG 判定する              |
| **iogpu.wired_limit_mb の sysctl** | 設定すると OS 全体に効く                                                                | 計測終了後は 0 に戻すか、再起動で消えるのを許容                             |
| **HF rate limit**                  | 未認証では 5GB/h 程度で詰まることあり                                                   | `HF_TOKEN` を設定                                                           |

## 新規 Mac 環境で評価を一通り実施する手順

ハードウェアが変わった時、最初から最後まで何を順に実行すれば `docs/04-results.md` にエントリが 1 つ完成するか、を以下にまとめる。所要時間は GPU と回線次第で 4〜10 時間程度を見込む。

### 0. 前提確認

```bash
sw_vers
sysctl -n machdep.cpu.brand_string hw.memsize
uv --version           # 0.8 以上を確認
python3 -c "import mlx.core as mx; print(mx.device_info())"
```

`device_name` / `memory_size` / `max_recommended_working_set_size` を控える(後で `docs/04-results.md` のヘッダーに記載)。

### 1. ストアアプリの停止

```bash
~/.lmstudio/bin/lms server stop 2>/dev/null
killall ollama 2>/dev/null
```

### 2. 評価対象モデルの選定(unified memory 制約)

[ollama.com/library](https://ollama.com/library) ではなく [huggingface.co/mlx-community](https://huggingface.co/mlx-community) を参照。Qwen3.5 / Gemma 4 / DeepSeek-R1 / DeepSeek-Coder-V2 系列の最新タグを確認し、対象 unified memory に収まる量子化に絞る。

目安:

| 重みサイズ vs `max_recommended_working_set` | 判断                                 |
| ------------------------------------------- | ------------------------------------ |
| 重み ≤ 上限 × 0.5                           | 余裕。8bit も含めて評価対象          |
| 重み = 上限 − 2〜5 GB                       | 境界。KV 量子化必須、長 ctx は厳しい |
| 重み > 上限                                 | 4bit でも乗らないなら対象外          |

### 3. docs/04-results.md に新規 Mac セクションを起こす

既存 GTX 1080 / RTX 4080 SUPER セクションをテンプレートに、新規 Mac 用のセクションを追加する。**この時点では結果テーブルは空のままで、評価対象選定の根拠だけを書く**(後で結果を埋める)。

### 4. モデル取得 → 評価(2 通り)

#### A. ディスクに余裕がある場合: 一括 pull → 一括評価

```bash
./pull_models_mlx.py            # 既定 20 モデルを全て取得(約 220 GB)
./run_all_mlx.sh                # ctx_search → speed → needle → coding → summary
```

#### B. ディスクが厳しい場合: pull → 評価 → 削除 サイクル

```bash
./eval_cycle_mlx.sh             # モデル単位で取得・評価・削除を繰り返す
```

最大瞬間ディスク使用量は ~28 GB(最大モデルの重み)で抑えられる。

どちらでも各モデルに対して ctx_search → speed → needle(深度100%×pos50%)→ coding(ctx 16k)→ summary(ctx 16k)を実行する。

### 5. KV 量子化を振りたい場合の追加 pass

主力 pass で目標 ctx に届かなかったモデルがあれば、`--kv-bits 8` や `--kv-bits 4` を付けて個別に再 ctx_search → speed する。

```bash
MODEL=mlx-community/Qwen3.5-27B-8bit
uv run python ctx_search_mlx.py --model "$MODEL" --kv-bits 8
uv run python run_speed_mlx.py --model "$MODEL" --kv-bits 8 --ctx <max_ctx_from_search>
```

### 6. 結果を docs/04-results.md にまとめる

```bash
uv run python aggregate_mlx.py
```

の出力を docs/04-results.md の該当 Mac セクションに転記。続いて[docs/06-evaluation.md §9.4](../../docs/06-evaluation.md) の 10 項目で最終分析を執筆。
