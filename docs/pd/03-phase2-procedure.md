# PD分離 PoC — Phase 2: 4080 SUPER (CUDA) → M1 (Metal) 手順書

CUDA 側で保存した KV キャッシュを Metal 側で restore できるかを検証します。**本 PoC の中核となる判定です。**

- 前提: [02-poc-phase1.md](02-poc-phase1.md) が PASS していること
- 4080 側の操作は手動で実施します。M1 側は自動で実行します

## 0. 役割分担

| 実行者 | 作業 |
| --- | --- |
| 手動 (4080 / Ubuntu) | §2〜§4 (設定・起動・確認) |
| 自動 (M1 / macOS) | §6 以降 (Prefill 指示・KV 取得・restore・Decode・計測) |

M1 から 4080 の llama-server を HTTP で直接叩くため、**4080 側で起動さえしてもらえれば、以降の操作は M1 側から実行できます。**

## 1. なぜ Docker でビルドするか

当初はホスト上での直接ビルドを想定していましたが、**4080 機のホストには CUDA Toolkit (nvcc) が入っていません。**

```
-- Could not find `nvcc` executable in any searched paths, please set CUDAToolkit_ROOT
CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:268 (message):
  CUDA Toolkit not found
```

既存の llama-swap 構成が CUDA を同梱した Docker イメージを使っているため、ホストには不要だったものです。

Docker を採用すると次の利点があります。

- ホストに CUDA Toolkit (約 3GB) を入れずに済む
- **ufw の設定変更が不要**。Docker の publish は iptables の `DOCKER` チェーンに入るため、`DEFAULT_INPUT_POLICY=DROP` のままでも LAN へ出せる ([deploy/llama/README.md](../../deploy/llama/README.md) と同じ理由)
- llama.cpp のタグを Dockerfile で固定でき、Decode 側とのバージョン一致を保証しやすい

llama.cpp 公式の CUDA イメージ (`ghcr.io/ggml-org/llama.cpp:server-cuda-<build>`) も検討しましたが、**commit 単位のタグが b5343 世代までしか公開されておらず b10679 が存在しない**ため、自前ビルドを採用します。

構成は [deploy/pd/](../../deploy/pd/) にあります。

## 2. 設定 (4080 側)

リポジトリを最新にしてから `.env` を作成します。

```bash
cd <このリポジトリ>/deploy/pd
cp .env.example .env
vi .env      # MODEL_DIR を自分の環境の絶対パスに書き換える
```

`.env` の必須項目は `MODEL_DIR` (GGUF を置くディレクトリの絶対パス) です。その他は既定値のままで動きます。

## 3. モデルの取得 (4080 側)

**M1 側とバイト単位で同一のファイルが必須です。** [01-research.md](01-research.md) §3.2 のとおり、seq state ファイルはモデルの同一性を照合しません。異なるモデルでも restore は「成功」し、出力だけが壊れます。

```bash
mkdir -p ~/models/gguf && cd ~/models/gguf
curl -L --fail -o Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

sha256sum Qwen3-4B-Instruct-2507-Q4_K_M.gguf
# 期待値: 3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597
```

**ハッシュが一致しない場合は先に進まないでください。**

## 4. 起動 (4080 側)

```bash
cd <このリポジトリ>/deploy/pd
docker compose up -d --build
```

初回はコンテナ内で llama.cpp をビルドするため時間がかかります (Ryzen 5 3600 で 10〜20 分程度の見込み)。`CUDA_ARCH=89` により RTX 4080 SUPER 向けの単一アーキのみをビルドします。

起動するサービスは 2 つです。

| サービス | ポート | 役割 |
| --- | --- | --- |
| `llama-pd` | 18080 | Prefill 側の llama-server |
| `kv-http` | 18081 | 保存された KV ファイルの配信 |

既存の llama-swap (11435) / ccshim (11436-11438) / Ollama (11434) とはポートが衝突しません。

### 4.0 ビルドで遭遇した問題

#### CUDA Driver API のリンクエラー

素直に `-DGGML_CUDA=ON` だけを指定すると、リンク段階で失敗します。

```
/usr/bin/ld: bin/libggml-cuda.so.0.22.0: undefined reference to `cuGetErrorString'
collect2: error: ld returned 1 exit status
FAILED: bin/llama-server
```

`libggml-cuda.so` は CUDA Driver API (`libcuda.so`) を参照しますが、`nvidia/cuda:*-devel` イメージに含まれるのは stub のみです。実行時にホストの本物のドライバへ解決させる必要があります。

llama.cpp 公式の `.devops/cuda.Dockerfile` と同じく、リンカに未解決シンボルを許可させることで解決します。

```cmake
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
```

この指定は [deploy/pd/Dockerfile](../../deploy/pd/Dockerfile) に入っています。

#### Web UI のダウンロード失敗は無視して構いません

ビルドログに次の警告が出ますが、致命的ではありません。本 PoC は HTTP API のみを使うため影響しません。

```
-- UI: download dist.tar.gz from b1 failed: "HTTP response code said error"
```

### 4.1 起動後に必ず確認する項目

**`flash_attn` の実効値が最重要です。**

```bash
docker compose logs llama-pd | grep -E "arch |n_layer |n_embd_k_gqa|n_embd_v_gqa|flash_attn|n_ctx "
```

期待される出力です。

```
print_info: arch                  = qwen3
print_info: n_layer               = 36
print_info: n_embd_k_gqa          = 1024
print_info: n_embd_v_gqa          = 1024
llama_context: n_ctx                 = 16384
llama_context: flash_attn            = enabled     ← M1 側と一致すること
```

`flash_attn` が `disabled` の場合、`v_trans` が M1 側と食い違い、restore は `incompatible V transposition` で必ず失敗します ([01-research.md](01-research.md) §3.1)。

ビルドされた llama.cpp の commit も確認できます。

```bash
docker compose exec llama-pd cat /app/commit.txt
# 期待値: 50f068ffffc3e0e4c9c2e4139281c6075224f429
```

### 4.2 ネットワーク実効速度の計測 (推奨)

break-even の算出に必要です。ufw を迂回するため Docker で起動します。

```bash
docker run --rm -d --name iperf3-pd -p 5201:5201 networkstatic/iperf3 -s
```

計測後は `docker rm -f iperf3-pd` で削除します。

## 5. 疎通確認 (M1 側から自動実行)

```bash
curl -s http://<4080のIP>:18080/health
curl -s http://<4080のIP>:18081/
```

## 6. 検証シーケンス (M1 側から自動実行)

1. 4080 の `/completion` に長プロンプトを投げて Prefill (`n_predict=1`)
2. 4080 の `/slots/0?action=save` で KV を保存。`n_written` / `save_ms` を記録
3. M1 から `curl` で `http://<4080のIP>:18081/<filename>` を取得。**転送時間と実効 MB/s を計測**
4. M1 の `--slot-save-path` へ配置
5. M1 の `/slots/0?action=erase` の後、`/slots/0?action=restore`
6. M1 の `/completion` に**同一プロンプト**を投げて Decode

## 7. 判定基準

Phase 1 と同一です。**HTTP 200 では判定しません。**

- restore が成功する (`incompatible V transposition` 等で失敗しない)
- M1 側 `/completion` の `timings.cache_n` が Prefill 済みトークン数とほぼ一致する
- M1 側 `/completion` の `timings.prompt_n` が 1〜数トークンに収まる
- 生成テキストが破綻していない (モデル不一致の検出には出力の目視が唯一の手段)

## 8. 必要な情報

M1 側の自動実行を開始するために、以下が必要です。

- 4080 機の LAN IP アドレス
- §4.1 の `flash_attn` の実際の値
- §3 の sha256 が一致したかどうか
