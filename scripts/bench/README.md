# scripts/bench/

[docs/06-evaluation.md](../../docs/06-evaluation.md) の評価方法論を実装した計測スクリプト群。**速度計測 / 最大 ctx 二分探索 / KV キャッシュ切替 / Needle In A Haystack / Coding 評価 / Summary 評価** を提供する。

## 前提環境

| 項目          | バージョン / 内容                         | 用途                                                                                                          |
| ------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| OS            | Linux(本リポジトリは Ubuntu 24.04 で開発) | systemd 連携 と `signal.SIGALRM` を使うため Unix 系前提                                                       |
| Python        | **3.12 以上**                             | `pyproject.toml` で指定。`uv sync` 時に未満なら自動で適切な版を取得する                                       |
| uv            | 0.11 以上(動作確認は 0.11.16)             | 依存管理と仮想環境。未インストールなら `curl -LsSf https://astral.sh/uv/install.sh \| sh`                     |
| Ollama        | **v0.24.0 以上**(計測時の最新 stable)     | 推論ランタイム。[docs/01-setup.md](../../docs/01-setup.md) でセットアップ                                     |
| NVIDIA Driver | 計測時の最新                              | GPU 計測時に必須。`nvidia-smi` が PATH に通っていること                                                       |
| CUDA          | 12.x 以上                                 | Ollama の CUDA バックエンドが要求するもの                                                                     |
| git           | 任意                                      | `metadata.py` がコミット hash を採取するため(無くても動作はする)                                              |
| sudo 権限     | 必要                                      | `kv.py` が systemd `override.conf` を書き換えるため。事前に `sudo -v` で credentials をキャッシュしておくこと |
| bash          | 任意                                      | `pull_models.sh` を使う場合                                                                                   |

## セットアップ

```bash
cd scripts/bench
uv sync
```

これで `.venv/` が作られて依存(httpx)が解決される。以降は `uv run python <script>.py ...` で実行する。

## 構成

```
scripts/bench/
├── client.py         # Ollama HTTP クライアント(streaming TTFT 計測込み)
├── metadata.py       # Ollama / GPU / OS / git hash 等の収集
├── vram.py           # nvidia-smi ポーリングで VRAM ピーク取得
├── kv.py             # KV キャッシュタイプを systemd 経由で切替(要 sudo)
├── ctx_search.py     # 最大 num_ctx の二分探索
├── scorer.py         # Coding / Summary / Needle の採点ロジック
├── run_speed.py      # 速度計測ランナー
├── run_needle.py     # Needle In A Haystack ランナー
├── run_coding.py     # Coding 評価ランナー(pytest 不要、stdlib のみ)
├── run_summary.py    # Summary 評価ランナー(n-gram 被覆率)
├── pull_models.sh    # 評価対象モデルを一括 pull するヘルパー
├── data/
│   ├── needle/
│   │   └── generate.py    # Needle 合成データ生成(出力は gitignore)
│   ├── coding/
│   │   └── tasks.json     # Coding 評価のサンプルタスク 6 件
│   └── summary/
│       └── meetings.json  # Summary 評価のサンプル会議 2 件
├── results/          # 計測結果(gitignore)
├── pyproject.toml
├── uv.lock
└── README.md
```

## 使い方

### 1. 速度計測

```bash
uv run python run_speed.py --model qwen3.5:9b --ctx 32768
```

オプション:

- `--model` (必須): Ollama タグ
- `--ctx`: `num_ctx`。未指定なら Ollama 既定値
- `--num-predict`: 生成上限トークン数(既定 128)
- `--prompt`: 任意のプロンプトに差し替え
- `--think true|false`: thinking モデル向け。代表値は `false`

出力: `results/<timestamp>/speed_<model>_ctx<ctx>.json` + `metadata.json`

### 2. Needle テストデータ生成

```bash
uv run python data/needle/generate.py \
    --chars 50000 --position-pct 0.5 \
    --output data/needle/32k_p50.json
```

オプション:

- `--chars` (必須): 目標文書文字数(日本語で概ね 1 文字 1〜2 トークン)
- `--position-pct`: needle 挿入位置(0.0〜1.0、既定 0.5)
- `--needle-id`: 固定 6 桁 ID(未指定ならランダム生成)
- `--seed`: フィラー生成シード(既定 42)

### 3. Needle テスト実行

```bash
uv run python run_needle.py --model qwen3.5:9b --ctx 32768 \
    --needle data/needle/32k_p50.json
```

出力: `results/<timestamp>/needle_<model>_ctx<ctx>_pos<pct>.json` + `metadata.json`。`success: true/false` で成功判定。

## 計測の前提

[../../README.md の「評価の前提」](../../README.md#評価の前提なぜこの既定値なのか) の通り、以下は systemd の `override.conf` で固定しておくこと:

```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
```

`OLLAMA_KV_CACHE_TYPE` は評価軸として変動させるため、計測ごとに systemd を経由して切り替える(本ディレクトリの `kv.py` を使う)。

### 4. KV キャッシュタイプ切替

`OLLAMA_KV_CACHE_TYPE` は起動時環境変数なので、変更には Ollama 再起動が必要。本スクリプトが systemd `override.conf` を書き換えて再起動まで行う(sudo 必要)。

評価セッション開始時に `sudo -v` で credentials をキャッシュしてから:

```bash
uv run python kv.py --type q8_0   # KV を q8_0 に設定して再起動
uv run python kv.py --type q4_0   # KV を q4_0 に設定して再起動
uv run python kv.py --type f16    # 明示的に f16
uv run python kv.py --type none   # OLLAMA_KV_CACHE_TYPE 行を削除(= Ollama 既定 f16)
```

### 5. 最大 ctx の二分探索

CPU オフロードなしで動作する最大 `num_ctx` を求める。各プローブで「load → /api/ps と nvidia-smi で確認 → unload」を実行し、`size == size_vram`(100% GPU)かつ残 VRAM ≥ `--min-free-mib`(既定 500)を OK 条件とする。

```bash
uv run python ctx_search.py --model qwen3.5:9b-q4_K_M
uv run python ctx_search.py --model qwen3.5:9b-q4_K_M --low 4096 --high 262144 --tolerance 4096
```

オプション:

- `--low`: 探索下限(既定 4096)
- `--high`: 探索上限(既定 262144)
- `--tolerance`: 探索打ち切り粒度(既定 4096)
- `--min-free-mib`: OK と判定する残 VRAM 下限(既定 500)

出力: `results/<timestamp>/ctx_search_<model>.json` に探索履歴と `max_ctx` を保存。

### 6. モデル一括 pull

評価対象モデルを一括 pull するヘルパースクリプト。

```bash
./pull_models.sh                  # 既定の RTX 4080 SUPER 対象リスト
./pull_models.sh foo:tag bar:tag  # 指定タグだけ pull
```

ログは `results/pull-<timestamp>.log` に残る。

### 7. Coding 評価

タスクファイル(JSON 配列)を読んで各タスクをモデルに投げ、応答から ```python フェンスを抽出 → `exec` → 関数を呼び出してテストケースを通すかを判定する。

```bash
uv run python run_coding.py --model qwen3.5:9b-q4_K_M --ctx 8192 \
    --tasks data/coding/tasks.json
```

サンプルデータ(`data/coding/tasks.json`)は **6 タスク** 入り(impl × 2、bugfix × 2、refactor × 2)。各タスクは `function_name` と `tests` を持ち、`tests[].args / kwargs / expected` で入出力を定義する。

出力: `results/<timestamp>/coding_<model>_ctx<ctx>.json`(タスク別合格率 + 種別ごとの平均)。

### 8. Summary 評価

会議文字起こし(JSON 配列)を読んで固定の要約プロンプトでモデルに投げ、応答を `expected_keypoints` / `expected_decisions` / `expected_todos` との **n-gram 被覆率** で採点する(既定 3-gram、閾値 0.5)。

```bash
uv run python run_summary.py --model qwen3.5:9b-q4_K_M --ctx 8192 \
    --meetings data/summary/meetings.json
```

サンプルデータ(`data/summary/meetings.json`)は **2 会議** 入り(プロジェクト進捗 + 技術選定)。

オプション: `--ngram` / `--threshold` で被覆率判定の挙動を調整できる。

出力: `results/<timestamp>/summary_<model>_ctx<ctx>.json`(カテゴリ別 matched 率 + 全体平均)。

## 採点ポリシー

[docs/06-evaluation.md §7](../../docs/06-evaluation.md) の通り、本評価は **LLM-as-a-judge を使わない**。

- Coding は実行ベース(pytest 不要、stdlib のみ)で pass/fail を取る
- Summary は n-gram 被覆率で文字列ベースに判定する
- Needle は needle_id の含有チェック

再現性が最優先。実データに差し替えた時に主観的妥当性が必要になれば、`scorer.py` にオプションとして追加可能な構造で書いてある。

## 新規 GPU 環境で評価を一通り実施する手順

ハードウェアが変わった時、最初から最後まで何を順に実行すれば `docs/04-results.md` にエントリが 1 つ完成するか、を以下にまとめる。所要時間は GPU と回線次第で 4〜10 時間程度を見込む。

### 0. 前提確認

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
ollama --version            # v0.24.0 以上を確認
uv --version                # 0.11 以上を確認
```

GPU 名 / VRAM / Driver / CUDA / Ollama version を控える(後で `docs/04-results.md` の測定環境ヘッダーに記載)。

### 1. systemd override を本評価仕様に固定

```bash
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
systemctl show ollama -p Environment   # 4 つが反映されているか確認
```

> `OLLAMA_KV_CACHE_TYPE` は意図的に書かない(評価軸として `kv.py` で動的に切り替えるため)。

### 2. 評価対象モデルの選定(VRAM 制約)

[ollama.com/library](https://ollama.com/library) で Qwen3.5 / Gemma 4 / DeepSeek-R1 / DeepSeek-Coder-V2 系列の最新タグを確認し、対象 GPU の VRAM に収まる量子化に絞る。目安:

| 重みサイズ vs VRAM      | 判断                                                      |
| ----------------------- | --------------------------------------------------------- |
| 重み ≤ VRAM × 0.6       | 余裕。Q8_0 も含めて評価対象                               |
| 重み = VRAM − 1 〜 2 GB | 境界。KV 量子化必須、長 ctx は厳しい                      |
| 重み > VRAM             | 強量子化(IQ3 / int4 等)タグがあるかを確認、無ければ対象外 |

加えて、**速度の下限を引き出す参照用に各系列の最小バリアントも 1〜2 個含める** ことを推奨(Qwen3.5 の 0.8b / 2b 等)。これらは小さい分高速で、「**サイズと品質のトレードオフ曲線**」を完成させる。VRAM に余裕があるので評価コストもほぼ無視できる。

### 3. docs/04-results.md に新規 GPU セクションを起こす

既存 GTX 1080 / RTX 4080 SUPER セクションをテンプレートに、新規 GPU 用のセクションを追加する。**この時点では結果テーブルは空のままで、評価対象選定の根拠だけを書く**(後で結果を埋める)。

書く内容:

- 実機構成(CPU / RAM / OS / Driver)
- 対象モデルテーブル(系列ごと)
- **対象外モデルと理由**(VRAM 超過 / 量子化タグなし / 残 VRAM ほぼゼロ等)
- 結果プレースホルダ:測定環境 / 100% GPU 動作 / Needle・Coding・Summary 詳細 / 採用推奨 / 最終分析

### 4. モデル一括 pull

`pull_models.sh` の `DEFAULT_MODELS` 配列を新 GPU 用に書き換えて実行するか、個別タグを引数で渡す:

```bash
./pull_models.sh \
    qwen3.5:4b-q4_K_M \
    qwen3.5:9b-q4_K_M \
    gemma4:e4b-it-q4_K_M \
    deepseek-r1:7b-qwen-distill-q4_K_M \
    # ...
```

> `ollama pull` は内部で複数の HTTP 接続を張る。回線を圧迫したくない場合は、別ターミナルで 1 個ずつ手動 pull する。`pull_models.sh` 自体はモデル単位では直列。

### 5. sudo 資格情報をキャッシュして KV を起点設定に

```bash
sudo -v
uv run python kv.py --type q8_0   # 主力起点
```

### 6. 各モデルで主力 pass(KV=q8_0)を回す

1 モデルあたりの典型フロー。`MODEL` を入れ替えながらループ:

```bash
MODEL=qwen3.5:9b-q4_K_M

# 6-1. 最大 ctx を二分探索
uv run python ctx_search.py --model "$MODEL" --high 262144

# 結果ファイルの max_ctx を読み取って以降の CTX に使う(または手動で決める)
CTX=131072   # ctx_search の結果から

# 6-2. 速度計測(think:false が代表値)
uv run python run_speed.py --model "$MODEL" --ctx "$CTX"
uv run python run_speed.py --model "$MODEL" --ctx "$CTX" --think true

# 6-3. Needle(4 深度 × 3 位置 = 12 試行)
# モデルは常に最大 CTX で load、needle プロンプト長を depth で変える
for depth_pct in 25 50 75 100; do
  target_chars=$(( CTX * depth_pct / 100 ))
  for pos in 0.1 0.5 0.9; do
    NEEDLE="data/needle/$(echo "$MODEL" | tr ':' _)_d${depth_pct}_p${pos}.json"
    uv run python data/needle/generate.py \
        --chars "$target_chars" --position-pct "$pos" --output "$NEEDLE"
    uv run python run_needle.py --model "$MODEL" --ctx "$CTX" --needle "$NEEDLE"
  done
done

# 6-4. Coding 評価(ctx は内容に応じて 8k〜16k で十分)
uv run python run_coding.py --model "$MODEL" --ctx 16384 \
    --tasks data/coding/tasks.json

# 6-5. Summary 評価
uv run python run_summary.py --model "$MODEL" --ctx 16384 \
    --meetings data/summary/meetings.json
```

これを全モデル分繰り返す。所要時間の目安(中型 GPU): ctx_search 5 分 + speed 1 分 + needle 5 分 + coding 3 分 + summary 1 分 = **約 15 分 / モデル**。13 モデルなら 3〜4 時間。

### 7. KV 量子化を振りたい場合の追加 pass

主力 pass で目標 ctx に届かなかったモデルがあれば、KV を切り替えて再度 ctx_search する:

```bash
uv run python kv.py --type q4_0
uv run python ctx_search.py --model "$MODEL" --high 262144

uv run python kv.py --type f16   # 比較の基準として
uv run python ctx_search.py --model "$MODEL" --high 262144
```

KV を変えたデータは「KV=XXX」を明示して別表として `docs/04-results.md` に追記する。

### 7.5. 合成タスクで差が出なかった時は難タスクへ差し替え

`data/coding/tasks.json` の合成タスク(add / fib / factorial / ...)は **小型モデルでも 100% 取りやすく、モデル間の差別化に弱い**。本評価でも 13 タグ中 10 タグが満点で頭打ちになった。

差別化の必要があれば、難度を上げた `data/coding/tasks_hard.json`(impl_flatten_dict / impl_balanced_brackets / impl_parse_bearer_token / bugfix_two_sum / bugfix_max_subarray / refactor_data_validator の 6 タスク、OIDC ドメインを含む)に差し替えて再走する。**num_predict は 3072 以上** を推奨(モデルが解説を出した後にコードを書く分の余裕)。

```bash
uv run python run_coding.py --model "$MODEL" --ctx 16384 \
    --num-predict 3072 \
    --tasks data/coding/tasks_hard.json
```

本評価では、easy → hard で 1.00 → 0.815〜0.982 まで差が開いた。最大モデル(R1:14B)が最低スコアだったり、bugfix で「バグはない」と LLM が拒否する挙動など、easy では見えなかった特性が浮き出る。

同じ要領で Summary 用にも難セット `data/summary/meetings_hard.json` を将来追加可能(現状は 2 件のみで未整備)。

### 7.6. think モード比較(任意、thinking 対応モデルのみ)

Qwen3.5 / Gemma 4 / DeepSeek-R1 系で thinking モードが品質に効くかを見る場合は、`run_think_compare.sh` を使うか、個別に `--think true` で再走する。

**重要: `num_predict` を最低 4096 取ること**。本評価で Qwen3.5:9b を `num_predict=2048` think=true で走らせたら **思考だけで budget を使い切って応答が空**(Coding 0.67、Summary 0.00)、4096 に上げたら 1.00 / 0.78 まで回復した。思考分の予算を見込まないと **think=false より露骨に悪い結果**になる。

```bash
uv run python run_coding.py --model "$MODEL" --ctx 16384 \
    --think true --num-predict 4096 \
    --tasks data/coding/tasks.json
```

### 7.7. 多人数同時利用想定 pass(任意、num_parallel)

`OLLAMA_NUM_PARALLEL=1` を 2 / 4 に上げて再計測する場合の手順:

```bash
sudo sed -i 's/OLLAMA_NUM_PARALLEL=1/OLLAMA_NUM_PARALLEL=2/' \
    /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
uv run python ctx_search.py --model "$MODEL" --high 262144
uv run python run_speed.py --model "$MODEL" --ctx "$CTX"
```

NP=4 でも同様(`s/=2/=4/` に変更)。終わったら `s/=4/=1/` で戻す。

**Ollama 0.24.0 時点の挙動はモデル依存**(本評価の RTX 4080 SUPER 観察):

- **per-slot ctx が 1/N に縮むタイプ**(例: deepseek-r1:14b): NP=1 で max 31K → NP=4 で max 7K。KV を NP 数分 pre-allocate
- **NP に依らず単発 max が不変なタイプ**(例: qwen3.5:9b): NP=1/2/4 すべて max 221K。遅延割り当て挙動

**`decode tok/s` は NP に関わらず不変**(per-request の処理速度には影響しない)。多人数同時実利用時のスループット倍率は本評価では未測定(2 並行ストレスを別途流す必要あり)。

社内多人数共有(API として複数ユーザにサーブ)を想定する時に意味のあるパス。1 ユーザ専有なら本パスは不要。

### 7.8. 再現性 / 揺らぎ確認(任意)

`temperature=1`(Ollama 既定)では出力に揺らぎが出る。記事や正式レポート向けに数値を出す場合、主力 3 モデル程度を **3 回繰り返し** て stddev を出すことを推奨。tok/s や VRAM は安定するが、Coding / Summary スコアにはバラつきが乗ることがある。

簡易ループ:

```bash
for i in 1 2 3; do
  uv run python run_speed.py --model qwen3.5:9b-q4_K_M --ctx 131072
  uv run python run_coding.py --model qwen3.5:9b-q4_K_M --ctx 16384 \
      --tasks data/coding/tasks_hard.json --num-predict 3072
done
```

### 8. 結果を docs/04-results.md にまとめる

`results/<timestamp>/` 配下の JSON から数値を拾って表に転記。docs/04 既存セクションのフォーマットに合わせて:

1. **測定環境ヘッダー**(日付 / Ollama version / Driver / Kernel / git hash)
2. **100% GPU 動作テーブル**(モデル × 重み × KV × ctx × FA × VRAM × 速度)
3. **採用推奨表**(用途別 1〜数行)
4. **Needle / Coding / Summary 詳細表**
5. **最終分析**(docs/06-evaluation.md §9.4 の 10 項目)

### 9. KV を起動時の既定に戻す(任意)

評価終わったら、運用想定の KV(通常 `q8_0`)に固定しておく:

```bash
uv run python kv.py --type q8_0
```

または `kv.py --type none` で完全に未設定(= Ollama 既定 `f16`)に戻す。
