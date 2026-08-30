# PD分離 PoC — 再現手順書

本リポジトリの PD分離 (Prefill/Decode 分離) の計測を、別のマシンで最初から再現するための手順です。

計測結果は [04-results.md](04-results.md) / [05-model-scaling.md](05-model-scaling.md) / [06-kv-quantization.md](06-kv-quantization.md) / [07-kv-quality.md](07-kv-quality.md) を参照してください。背景となる調査は [01-research.md](01-research.md) です。

## 0. 表記と前提

### 0.1 2 台のマシンの呼び分け

本書では 2 台を次のように呼び分けます。**各手順の見出しに、どちらで実行するかを明記します。**

| 呼称 | 役割 | 本 PoC での実機 |
| --- | --- | --- |
| **GPU 側** | Prefill を担当。NVIDIA GPU + CUDA。Linux | Ryzen 5 3600 / RTX 4080 SUPER 16GB / Ubuntu |
| **Mac 側** | Decode を担当。Apple Silicon + Metal | Apple M1 / Unified Memory 16GB / macOS |

**GPU 側は Docker で、Mac 側はホストビルドで動かします。** GPU 側を Docker にする理由は [03-phase2-procedure.md](03-phase2-procedure.md) §1 を参照してください。

### 0.2 ssh は前提にしません

**本書のコマンドは、すべて「そのマシンの前に座って実行する」前提で書いています。** GPU 側と Mac 側で別々の端末を開いてください。

2 台の間でやり取りするのは次の 2 つだけです。いずれもネットワーク越しで、ssh は不要です。

- Mac 側から GPU 側の llama-server を HTTP で操作する
- Mac 側から GPU 側の KV ファイルを HTTP で取得する

**計測の指示は Mac 側から出します。** GPU 側で行うのは環境構築とサーバ起動までです。

### 0.3 事前に必要なもの

| | GPU 側 | Mac 側 |
| --- | --- | --- |
| OS | Linux (Ubuntu で確認) | macOS (Apple Silicon) |
| 必須 | Docker / Docker Compose / NVIDIA ドライバ | Xcode Command Line Tools / Homebrew |
| GPU | NVIDIA (CUDA 12.8 が動くドライバ) | Apple Silicon |
| 空きディスク | 30 GB 以上 | 30 GB 以上 |
| Python | (不要。Docker 内で完結) | 3.10 以上 |

GPU 側にホストの CUDA Toolkit (nvcc) は**不要**です。Docker イメージ内でビルドします。

### 0.4 記入する値

手順の中で次の値を使います。先に確認して控えてください。

| 記号 | 意味 | 確認方法 |
| --- | --- | --- |
| `<GPU_IP>` | GPU 側の LAN IP アドレス | GPU 側で `ip -4 addr show \| grep inet` |
| `<MAC_KV_DIR>` | Mac 側の KV 保存先の絶対パス | 手順 8 で作成。例 `/Users/YOUR_NAME/pd-kv` |
| `<REPO>` | 本リポジトリの作業コピー | 両側でクローンします |

---

## 手順 1 【GPU 側】リポジトリを取得する

```bash
git clone https://github.com/mktkhr/local-llm.git
cd local-llm
```

## 手順 2 【GPU 側】IP アドレスを確認する

Mac 側から接続するため、LAN の IP アドレスを控えます。

```bash
ip -4 addr show | grep inet
```

`192.168.x.x` などの値が `<GPU_IP>` です。

## 手順 3 【GPU 側】モデルを取得する

**Mac 側とバイト単位で同一のファイルが必須です。** slot 保存ファイルはモデルの同一性を照合しないため、異なるモデルでも restore は「成功」し、出力だけが壊れます ([01-research.md](01-research.md) §3.2)。

```bash
mkdir -p ~/models/gguf && cd ~/models/gguf
curl -L --fail -o Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
sha256sum Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

期待値は次のとおりです。**一致しない場合は先に進まないでください。**

| モデル | サイズ | sha256 |
| --- | --- | --- |
| Qwen3-4B-Instruct-2507-Q4_K_M | 2,497,281,120 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| Qwen3-8B-Q4_K_M | 5,027,784,512 | `120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4` |
| Qwen3-14B-Q4_K_M | 9,001,753,984 | `5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66` |

8B / 14B も測る場合は同じ手順で取得します。URL は `unsloth/Qwen3-8B-GGUF` / `unsloth/Qwen3-14B-GGUF` です。

## 手順 4 【GPU 側】設定ファイルを作る

```bash
cd <REPO>/deploy/pd
cp .env.example .env
```

`.env` を編集します。**`MODEL_DIR` は自分の環境の絶対パスに書き換えてください。**

| 項目 | 説明 |
| --- | --- |
| `MODEL_DIR` | GGUF を置いたディレクトリの絶対パス (必須) |
| `MODEL_FILE` | 使う GGUF のファイル名 |
| `LLAMA_TAG` | llama.cpp のタグ。**Mac 側と必ず一致させます** |
| `CUDA_ARCH` | GPU の compute capability。RTX 40 系は `89`、30 系は `86` |
| `PD_N_CTX` | コンテキスト長。**Mac 側と一致させます** |
| `PD_KV_TYPE` | KV 量子化型。**Mac 側と一致させます** |

## 手順 5 【GPU 側】サーバを起動する

```bash
cd <REPO>/deploy/pd
docker compose up -d --build
```

初回はコンテナ内で llama.cpp をビルドするため 10〜20 分かかります。

起動するサービスは 2 つです。

| サービス | ポート | 役割 |
| --- | --- | --- |
| `llama-pd` | 18080 | Prefill 側の llama-server |
| `kv-http` | 18081 | 保存された KV ファイルの配信 (nginx。Range 対応) |

## 手順 6 【GPU 側】起動結果を検証する

**この検証を飛ばさないでください。** 特に `flash_attn` が Mac 側と食い違うと、restore は `incompatible V transposition` で必ず失敗します ([01-research.md](01-research.md) §3.1)。

```bash
docker compose logs llama-pd | grep -E "arch |n_layer |n_embd_k_gqa|flash_attn|n_ctx |KV buffer"
docker compose exec llama-pd cat /app/commit.txt
```

確認する項目です。

| 項目 | 期待値 |
| --- | --- |
| `flash_attn` | **`enabled`** (Mac 側と一致していること) |
| `n_ctx` | `.env` の `PD_N_CTX` と同じ |
| `KV buffer size` | 手順 12 の Mac 側と同じ値になること |
| `commit.txt` | `LLAMA_TAG` に対応する commit |

Qwen3-4B / n_ctx 16384 の場合の `KV buffer size` の期待値です。

| KV 型 | KV buffer size |
| --- | --- |
| f16 | 2,304.00 MiB |
| q8_0 | 1,224.00 MiB |
| q4_0 | 648.00 MiB |

## 手順 7 【Mac 側】リポジトリと必要なツールを用意する

```bash
git clone https://github.com/mktkhr/local-llm.git
cd local-llm

brew install cmake ninja iperf3
```

`iperf3` はネットワークの実効速度を測るために使います。

## 手順 8 【Mac 側】llama.cpp をビルドする

**GPU 側の `LLAMA_TAG` と同じタグを使います。** バージョン差は KV ファイルの互換性を壊す可能性があります。

```bash
mkdir -p ~/ghq/github.com/ggml-org && cd ~/ghq/github.com/ggml-org
git clone --depth 1 --branch b10679 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8 --target llama-server

mkdir -p ~/pd-kv
```

`cmake` の出力に `-- Including METAL backend` があることを確認してください。

`~/pd-kv` が `<MAC_KV_DIR>` です。絶対パスを控えてください (`echo ~/pd-kv` で確認できます)。

## 手順 9 【Mac 側】モデルを取得する

**手順 3 と同一のファイルです。sha256 の一致を必ず確認してください。**

```bash
mkdir -p ~/models/gguf && cd ~/models/gguf
curl -L --fail -o Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
shasum -a 256 Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

手順 3 の表と突き合わせます。

## 手順 10 【Mac 側】GPU 側への疎通を確認する

```bash
curl -s http://<GPU_IP>:18080/health
curl -s -o /dev/null -w "%{http_code}\n" http://<GPU_IP>:18081/
```

`/health` が応答し、18081 が `200` を返せば疎通しています。

届かない場合、Linux 側のファイアウォールを疑う前に **Docker の publish が効いているか**を確認してください。Docker の publish は iptables の `DOCKER` チェーンに入るため、ufw が `DEFAULT_INPUT_POLICY=DROP` でも LAN に出ます。

## 手順 11 【Mac 側】ネットワークの実効速度を測る

損益分岐の算出に使います。GPU 側で iperf3 サーバを起動してから測ります。

**GPU 側で実行:**

```bash
docker run --rm -d --name iperf3-pd -p 5201:5201 networkstatic/iperf3 -s
```

**Mac 側で実行:**

```bash
iperf3 -c <GPU_IP> -t 10 -f m         # Mac -> GPU
iperf3 -c <GPU_IP> -t 10 -f m -R      # GPU -> Mac (PD分離が使う向き)
iperf3 -c <GPU_IP> -t 10 -f m -R -P 4 # 並列 4 フロー
```

**単一フローと並列フローの両方を測ってください。** Wi-Fi では単一フローがリンク容量の半分以下しか出ないことがあります ([04-results.md](04-results.md) §3.2)。

測り終えたら GPU 側で片付けます。

```bash
docker rm -f iperf3-pd
```

## 手順 12 【Mac 側】サーバを起動して検証する

**GPU 側 (手順 5) とフラグを完全に一致させます。** 特に `-fa on` は必須です。既定の `auto` はバックエンドごとに解決されるため、CUDA と Metal で食い違います。

```bash
cd ~/ghq/github.com/ggml-org/llama.cpp
./build/bin/llama-server \
  -m ~/models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 \
  -c 16384 -np 1 -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --slots --slot-save-path <MAC_KV_DIR>/ \
  --no-context-shift --no-warmup -lv 5
```

`-lv 5` は `flash_attn` の実効値を出力させるために必要です。通常の verbosity では出ません。

**別の端末で**、次を確認します。

```bash
grep -E "arch  |n_layer  |n_embd_k_gqa|flash_attn|n_ctx  |KV buffer" <ログ>
```

| 項目 | 期待値 |
| --- | --- |
| `flash_attn` | **`enabled`** |
| `KV buffer size` | **手順 6 の GPU 側と同じ値** |
| `n_ctx` | GPU 側と同じ |

**`KV buffer size` が両側で一致しない場合、設定がずれています。先に進まないでください。**

## 手順 13 【Mac 側】単体構成のベースラインを測る

PD分離の比較対象です。**構成 A (Mac 単体) と構成 B (GPU 単体) の両方を、Mac 側から測ります。**

```bash
cd <REPO>/scripts/pd

# 構成 B: GPU 単体
python3 baseline.py --url http://<GPU_IP>:18080 \
  --tokens 1024 2048 4096 8192 16000 --n-predict 32 \
  --label "B-gpu-only" --model "qwen3-4b-instruct-2507-q4km" --kv-type q8_0 \
  --out ../../results/baseline.jsonl

# 構成 A: Mac 単体
python3 baseline.py --url http://127.0.0.1:18080 \
  --tokens 1024 2048 4096 8192 16000 --n-predict 32 \
  --label "A-mac-only" --model "qwen3-4b-instruct-2507-q4km" --kv-type q8_0 \
  --out ../../results/baseline.jsonl
```

`--label` は `A-` / `B-` で始めてください。集計時に構成を判別する鍵になります。

**構成 A は時間がかかります。** Qwen3-4B / 16k で M1 なら約 140 秒、14B なら約 400 秒です。

## 手順 14 【Mac 側】PD分離を測る

```bash
cd <REPO>
MODEL=qwen3-4b-instruct-2507-q4km KVTYPE=q8_0 CONNS=8 \
  bash scripts/pd/sweep_pd.sh \
    "C-nginx-8conn" \
    http://<GPU_IP>:18080 http://127.0.0.1:18080 \
    http://<GPU_IP>:18081 <MAC_KV_DIR> \
    1024 2048 4096 8192 16000
```

引数の順序は `<label> <prefill_url> <decode_url> <kv_url> <decode_kv_dir> <tokens...>` です。`--label` は `C` で始めてください。

各試行で次の行が出ます。

```
tokens=2053 pd_ok=True prefill=176ms save=61ms xfer=1761ms(87.1MiB/s) restore=20ms residual=1tok
```

### 14.1 成否の判定基準

**`pd_ok=True` と `residual=1tok` の両方を確認してください。HTTP ステータスでは判定しません。**

`residual` は Decode 側で実際に再処理されたトークン数です。llama.cpp は最終トークンを必ず再処理するため 0 にはなりません。**`residual` がプロンプト全長になっていれば、restore は成功していても再 Prefill が起きており、PD分離は成立していません。**

判定に使っている値は llama-server の `/completion` が返す `timings.cache_n` (キャッシュ再利用) と `timings.prompt_n` (実処理) です ([01-research.md](01-research.md) §4.2)。

## 手順 15 【Mac 側】集計する

```bash
cd <REPO>
python3 scripts/pd/aggregate.py
```

次の 2 つが出力されます。

| ファイル | 内容 |
| --- | --- |
| `results/benchmark.csv` | 全試行の一覧 |
| `results/breakeven.csv` | ネットワーク速度別の損益分岐 |

`breakeven.csv` の `network` 列には次が入ります。

| 値 | 意味 |
| --- | --- |
| `wifi-measured` | 手順 14 の転送実測値をそのまま使った結果 |
| `1gbe-est` | 実効 112 MB/s と仮定した推定 |
| `10gbe-est` | 実効 1,100 MB/s と仮定した推定 |

推定値の前提は `scripts/pd/aggregate.py` の `NETWORKS` にあります。**手順 11 で実測した値に差し替えると、自分の環境に即した推定になります。**

## 手順 16 【Mac 側・任意】条件を変えて繰り返す

軸を変えるときは、**両側のサーバを同じ設定で起動し直してから**測ります。

### 16.1 KV 量子化型を変える

**GPU 側:**

```bash
cd <REPO>/deploy/pd
sed -i 's/^PD_KV_TYPE=.*/PD_KV_TYPE=q4_0/' .env
docker compose up -d --force-recreate
docker compose logs llama-pd | grep "KV buffer"
```

**Mac 側:** 手順 12 の `--cache-type-k` / `--cache-type-v` を `q4_0` に変えて起動し直します。

そのうえで手順 13〜15 を `--kv-type q4_0` / `KVTYPE=q4_0` で実行します。**`--kv-type` を変え忘れると、集計時に別の KV 型の結果と突き合わされて誤った損益分岐が出ます。**

### 16.2 モデルを変える

**GPU 側:** `.env` の `MODEL_FILE` を変えて `docker compose up -d --force-recreate`。
**Mac 側:** 手順 12 の `-m` を変えて起動し直します。

そのうえで手順 13〜15 を `--model` / `MODEL` を変えて実行します。

### 16.3 転送の並列数を変える

Mac 側だけの変更です。`CONNS=1` にすると単一接続になります。

```bash
MODEL=... KVTYPE=... CONNS=1 bash scripts/pd/sweep_pd.sh "C-nginx-1conn" ...
```

## 手順 17 【Mac 側・任意】KV 量子化の品質影響を測る

Needle In A Haystack で、KV 量子化が長文からの情報抽出能力を損なわないかを測ります。

### 17.1 データを生成する

```bash
cd <REPO>
mkdir -p scripts/pd/data/needle
```

文字数とトークン数の比は言語とトークナイザに依存します。**まず 1 件作って実測してください。**

```bash
python3 scripts/bench/data/needle/generate.py --chars 6000 --position-pct 0.5 \
  --output /tmp/calib.json
curl -s http://127.0.0.1:18080/tokenize \
  -H 'Content-Type: application/json' \
  -d "{\"content\": $(python3 -c 'import json;print(json.dumps(json.load(open("/tmp/calib.json"))["prompt"]))')}" \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["tokens"]), "トークン")'
```

得られた比から目標トークン数に対応する `--chars` を決め、シード 3 種 × 文脈長 3 種 × 位置 3 種の 27 件を生成します。本 PoC では日本語で 1 トークン = 1.36 文字でした。

### 17.2 測る

```bash
bash scripts/pd/sweep_needle.sh http://<GPU_IP>:18080 \
  qwen3-4b-instruct-2507-q4km q4_0 cuda
```

引数は `<url> <model> <kv_type> <backend>` です。結果は `results/needle.jsonl` に追記されます。

**f16 を基準として必ず測ってください。** f16 は KV を量子化しない条件であり、これとの差が量子化の影響です。

### 17.3 注意点

- **`n_predict` は 256 以上にしてください。** 64 ではモデルが前置きを生成している途中で打ち切られ、品質ではなく生成長の不足で不正解と判定されます ([07-kv-quality.md](07-kv-quality.md) §2.3)
- **試行数を確保してください。** 27 試行でも、量子化して f16 より良くなるという原理的にありえない結果が出る程度のばらつきがあります ([07-kv-quality.md](07-kv-quality.md) §5)

## 手順 18 【両側】片付ける

**GPU 側:**

```bash
cd <REPO>/deploy/pd
docker compose down
rm -f kv/*.bin
```

**Mac 側:**

```bash
# llama-server を Ctrl+C で停止してから
rm -f <MAC_KV_DIR>/*.bin
```

**KV ファイルは大きいので必ず消してください。** Qwen3-4B / 16k / q8_0 で 1 件 1.2 GiB、掃引 1 回で合計 2.3 GiB になります。

`deploy/pd/kv/` と `deploy/pd/.env` は `.gitignore` 済みのため、消し忘れてもコミットはされません。

---

## 付録 A. GPU 側と Mac 側で一致が必要な設定

**一致していないと restore が失敗するか、出力が壊れます。**

| 項目 | 不一致のときに起きること |
| --- | --- |
| **GGUF ファイル (sha256)** | **エラーにならず出力だけが壊れます。** seq state ファイルはモデルの同一性を照合しません |
| **`-fa` (Flash Attention)** | `incompatible V transposition` で restore 失敗 |
| **`--cache-type-k` / `-v`** | `mismatched key type` / `mismatched value type` で restore 失敗 |
| llama.cpp のバージョン | ファイル形式が変わっていれば restore 失敗 |
| `-c` (n_ctx) | Decode 側が Prefill 済みトークン数より小さいと restore 失敗 |

判定ロジックの根拠は [01-research.md](01-research.md) §3 にあります。

## 付録 B. うまくいかないとき

| 症状 | 原因と対処 |
| --- | --- |
| restore が `incompatible V transposition` | `-fa` の実効値が不一致。**両側で `-fa on` を明示** (`auto` はバックエンドごとに解決される) |
| restore が `mismatched key type` | `--cache-type-k` / `-v` が不一致 |
| `residual` がプロンプト全長 | Decode リクエストの `id_slot` が restore 先と違う。プロンプトが完全一致していない |
| 出力が意味不明 | **GGUF が両側で異なる可能性が高い。** sha256 を再確認 |
| `Invalid action` (400) | llama-swap などの router 経由で叩いている。**素の llama-server を直接叩く** |
| 転送が遅い | 単一 TCP フローの頭打ち。`CONNS=8` を使う。配信側が Range 非対応だと単一接続に退避する |
| Mac 側でメモリ不足 | モデルと KV の合計がユニファイドメモリを圧迫。`--cache-type-*` を `q4_0` にするか `-c` を下げる |
| GPU 側の VRAM 不足 | 他の推論サーバが常駐していないか確認。本 PoC では既存の llama-swap を一時停止しました |

## 付録 C. 逆方向 (Mac で Prefill → GPU で Decode)

参考用の構成です。**Mac 側は macOS のファイアウォールが着信を遮断するため、GPU 側が HTTP で取りに行く形にできません。** そのため `scp` で送り出す方式を用意しています。

```bash
python3 scripts/pd/pd_run.py \
  --prefill-url http://127.0.0.1:18080 \
  --decode-url  http://<GPU_IP>:18080 \
  --transfer-mode scp-push \
  --prefill-kv-dir <MAC_KV_DIR> \
  --push-dest "USER@<GPU_IP>:<GPU側のdeploy/pd/kvの絶対パス>" \
  --tokens 4096 --n-predict 32 \
  --label "D-reverse" --model "..." --kv-type q8_0
```

**この構成だけは ssh が必要です。** また ssh の暗号化ぶん不利になるため、HTTP pull の転送速度と直接比較しないでください。

実用上の価値はありません (遅い側で Prefill するため)。クロスバックエンド restore が双方向で成立することの確認と、逆方向のネットワーク速度の測定が目的です ([04-results.md](04-results.md) §6)。
