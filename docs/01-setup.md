# 01. 環境構築

Ollama を計測しやすい状態に整えるところまで。

## 1. ハードウェア事前確認

何を選び、どう量子化するかは GPU・CPU・RAM の構成で決まります。最初に以下を採取しておくと、後の判断材料が揃います。

```bash
# OS / カーネル
uname -a
lsb_release -a

# CPU(AVX2/FMA の有無は llama.cpp 自前ビルド時に効く)
lscpu | grep -E "Model name|Socket|Core|Thread|MHz"
lscpu | grep -oE 'avx[0-9]*|fma|sse4_[12]' | sort -u

# メモリ
free -h

# GPU(VRAM・Compute Capability・ドライバ)
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv

# ストレージ(モデル保管、SSD/HDD)
df -h
cat /sys/block/sda/queue/rotational  # 0=SSD, 1=HDD

# 既存ランタイム
which ollama llama-server vllm 2>/dev/null
```

**注目するポイント**

| 項目                | 影響                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| VRAM 合計           | モデル重み + KVキャッシュの上限。実用上は GUI 等の常駐分(数百MiB)を引いた値で考える                                  |
| Compute Capability  | Pascal(6.x)世代では Flash Attention・KVキャッシュ量子化のサポートが限定的(動くが恩恵限定)。Ampere(8.x)以降は本領発揮 |
| CPU AVX2/FMA の有無 | Ollama 公式バイナリは内部で振り分けるため問題なし。llama.cpp を自前ビルドする場合は対応ビルドオプションが必要        |
| プロキシ            | 社内ネットでは `https_proxy` 設定が必要なケースあり                                                                  |

## 2. Ollama インストール

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
systemctl is-active ollama
```

期待: バージョン表示 + `active`。GPU は自動検出されます。確認は以下。

```bash
journalctl -u ollama --no-pager -n 50 | grep -iE "cuda|gpu|compute"
```

## 3. 計測用 systemd ドロップイン

計測の再現性を確保するため、以下を **固定** します。

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
```

```bash
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_KEEP_ALIVE=-1"
EOF
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3
systemctl show ollama -p Environment
```

## 4. 主要環境変数

| 変数                       | デフォルト           | 計測時の推奨                      | 役割                                                                                                     |
| -------------------------- | -------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `OLLAMA_MAX_LOADED_MODELS` | 自動(VRAM次第で複数) | **1**                             | 同時にメモリ常駐するモデル数。計測中は固定しないと前モデルが残って汚染される                             |
| `OLLAMA_NUM_PARALLEL`      | 1〜4(版による)       | **1**                             | 並行リクエスト数。各並行スロットがKVキャッシュを倍々で確保するため、計測中は1に固定                      |
| `OLLAMA_FLASH_ATTENTION`   | 0(無効)              | **1**                             | Flash Attention 有効化。KVキャッシュ量子化を効かせるための前提。Pascal以前は効果限定的だが有効化は可     |
| `OLLAMA_KV_CACHE_TYPE`     | `f16`                | `q8_0` または `q4_0`              | KVキャッシュの量子化。長コンテキスト時のVRAM消費を大きく削減する                                         |
| `OLLAMA_KEEP_ALIVE`        | `5m`                 | **`-1`**(常駐運用)/ 計測時は `5m` | モデルが何もしない時にアンロードされるまでの時間。`-1` で自動アンロードを無効化(専用機で1人運用なら推奨) |
| `OLLAMA_NUM_GPU`           | 自動                 | 通常未指定                        | レイヤーをGPUに何層載せるか。Ollamaは自動配分するが、強制したい場合に明示                                |

実運用に移すときは `OLLAMA_NUM_PARALLEL` を上げて並行リクエストを捌けるようにしますが、その分 KVキャッシュは並列数倍に膨らみます。コンテキスト長と並行数のトレードオフ。

## 5. 動作確認

軽量モデルで一発スモーク。

```bash
ollama pull qwen3.5:2b
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen3.5:2b",
  "prompt": "日本語で自己紹介してください。",
  "stream": false
}' | grep -oE '"response":"[^"]*"' | head -c 400
ollama ps
```

`ollama ps` で `100% GPU` と表示されていれば CPU オフロードなしでロードできています。

次は [docs/02-benchmark.md](02-benchmark.md) で本格計測へ。
