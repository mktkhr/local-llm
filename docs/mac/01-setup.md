# 01. 環境構築 (Apple Silicon / MLX)

mlx-lm を計測しやすい状態に整えるところまで。Linux + CUDA + Ollama の手順は [docs/01-setup.md](../01-setup.md)。

## 0. 大前提

本書は **Apple Silicon (M シリーズ) + mlx-lm** での計測手順をまとめる。Mac 上で Ollama を Metal バックエンドで動かす道もあるが、本リポジトリは Apple Silicon の最大性能を測る目的で **MLX** を採用している。

Mac/MLX 環境の特徴(Linux + CUDA との主な差分):

| 観点 | Linux + CUDA (Ollama) | Mac + MLX (mlx-lm) |
| --- | --- | --- |
| 推論ランタイム | Ollama (llama.cpp + CUDA) | mlx-lm (MLX, Apple 純正) |
| メモリ | 専用 VRAM | ユニファイドメモリ(CPU と共用) |
| メモリ計測 | `nvidia-smi` | `mlx.get_peak_memory()` |
| KV キャッシュ量子化 | systemd 環境変数 → サーバ再起動 | リクエスト時引数 `kv_bits=4|8` |
| Flash Attention | `OLLAMA_FLASH_ATTENTION=1` | mlx-lm 既定 SDPA(明示ノブなし) |
| サービス常駐 | `systemctl ollama` | 不要(Python プロセスで完結) |
| モデルフォーマット | GGUF | MLX safetensors (HF `mlx-community/*`) |

## 1. ハードウェア事前確認

何を選び、どう量子化するかは Apple Silicon の世代と RAM 量で決まります。最初に以下を採取しておくと、後の判断材料が揃います。

```bash
# OS / カーネル
sw_vers
uname -a

# Chip / コア構成
sysctl -n machdep.cpu.brand_string
sysctl -n hw.ncpu
sysctl -n hw.perflevel0.physicalcpu    # Performance コア
sysctl -n hw.perflevel1.physicalcpu    # Efficiency コア

# メモリ(ユニファイドメモリ総量)
sysctl -n hw.memsize

# GPU(Metal の上限。「実効 VRAM」相当)
python3 -c "import mlx.core as mx; print(mx.device_info())"

# ストレージ
df -h
```

**注目するポイント**

| 項目 | 影響 |
| --- | --- |
| ユニファイドメモリ総量 | モデル重み + KV キャッシュ + OS / アプリ常駐分が同じプールから取られる |
| `max_recommended_working_set_size` | Metal が同時確保すべきでない上限(byte)。**これが「実効 VRAM」**として機能する |
| `max_buffer_length` | 単一バッファ上限。極端に大きなモデル重みを 1 テンソルで割り当てる時に効く |
| Apple GPU 世代 | M1 (G13G) → M2 (G14G) → M3 (G15G) → M4 (G16S)。世代で MLX の最適化サポートが変わる |
| 既存ランタイム | LM Studio / Ollama Metal が常駐していると VRAM を奪うため、計測時は停止 |

参考(M4 Pro 48GB の場合):

```
{'device_name': 'Apple M4 Pro',
 'memory_size': 51539607552,                      # 48 GiB
 'max_recommended_working_set_size': 40200896512, # 37.4 GiB ← 実効 VRAM
 'max_buffer_length': 30150672384,                # 28.1 GiB
 'architecture': 'applegpu_g16s'}
```

## 2. uv インストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version    # 0.8 以上を確認
```

Homebrew 経由でも可: `brew install uv`。

## 3. bench-mlx の依存導入

```bash
cd scripts/bench-mlx
uv sync
```

これで `.venv/` が作られ、`mlx` / `mlx-lm` / `huggingface_hub` が解決される。以降は `uv run python <script>.py ...` で実行する。

動作確認:

```bash
uv run python -c "
import mlx.core as mx
import mlx_lm
print('mlx', mx.__version__ if hasattr(mx, '__version__') else 'ok')
print('mlx_lm', mlx_lm.__version__)
print('device', mx.default_device())
print('info', mx.device_info())
"
```

## 4. ストアアプリの停止

LM Studio / Ollama を併用している環境では、計測中は止めておく(VRAM を奪い、結果が汚染される):

```bash
# LM Studio サーバが動いていれば停止
~/.lmstudio/bin/lms server stop

# Ollama (Mac 版) を停止
killall ollama 2>/dev/null
launchctl unload ~/Library/LaunchAgents/com.ollama.* 2>/dev/null || true
```

## 5. 主要な MLX パラメータ

Linux 版の OLLAMA_* 環境変数に相当する MLX 側のノブ。Linux では systemd で固定したが、**MLX ではリクエストごとに渡せる**ため起動時の固定は不要。

| パラメータ | 既定 | 計測時の推奨 | 役割 |
| --- | --- | --- | --- |
| `max_kv_size` | None (無制限) | テスト対象の ctx 値 | KV キャッシュ最大長。超えた分は古い側から捨てられる |
| `prefill_step_size` | 2048 | 2048 | プロンプト処理のチャンクサイズ。長プロンプト時の中間メモリに効く |
| `kv_bits` | None (非量子化) | `4` または `8` | KV キャッシュの量子化。長 ctx の VRAM 削減に効く |
| `kv_group_size` | 64 | 64 | KV 量子化のグループサイズ |
| `quantized_kv_start` | 0 | 0 | 何トークン目から KV を量子化するか |

これらは `bench-mlx/client_mlx.py` の `generate_with_metrics` で全て引数として渡せる。

## 6. 動作確認(スモーク)

軽量モデルで一発スモーク:

```bash
cd scripts/bench-mlx
uv run python run_speed_mlx.py \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --ctx 8192 --num-predict 64
```

期待出力例:

```
[load] mlx-community/Qwen2.5-0.5B-Instruct-4bit
  active=268 MiB, peak=268 MiB
[measure] generate ctx_max=8192 kv_bits=None
[done] results/<ts>/speed_*.json
  decode  : 421.5 tok/s
  prefill : 773.5 tok/s
  TTFT    : 0.126s
  peak mem: 365 MiB
```

`decode tok/s` が出ていれば成功。`active=*` がモデル重みのおおよその常駐量、`peak=*` が KV を含むピーク。

## 7. wired_limit を引き上げる(任意・上級者向け)

macOS は既定で「ユニファイドメモリの ~75%」を Metal に割り当てる。`iogpu.wired_limit_mb` を引き上げると、より大型のモデルを GPU で動かせるが、システム側がスワップに追いやられて全体が遅くなるリスクがある。

```bash
# 現在値を確認(0 = OS が動的に決める)
sysctl iogpu.wired_limit_mb

# 例: 44 GiB まで GPU に割り当てる(48 GiB マシンで上限近く)
sudo sysctl iogpu.wired_limit_mb=45056
```

本評価は **既定(動的)で運用** することを推奨する。`bench-mlx/memory_mlx.py` の `effective_gpu_limit_mib()` は wired_limit 設定値があればそちらを優先する。

## 8. 環境変数

mlx-lm 側で挙動を変える環境変数はほとんどない。HuggingFace 経由でモデルを取得するため、以下があれば設定する:

```bash
export HF_TOKEN="..."           # private repo や rate limit 緩和用
export HF_HOME=~/.cache/huggingface
```

次は [docs/mac/02-benchmark.md](02-benchmark.md) で本格計測へ。
