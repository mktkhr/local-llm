# 02. 計測 (Apple Silicon / MLX)

mlx-lm で「このモデルがこの設定で安定して動くか」を見るための計測手順と、結果のどこを読むか。Linux + Ollama 版は [docs/02-benchmark.md](../02-benchmark.md)。

## 1. 計測の原則

| 原則                                                       | 理由                                                                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **1 プロセスに 1 モデルだけロード**                        | MLX は同一プロセス内に複数モデルを抱えると `mx.get_peak_memory()` が干渉する。計測ごとにプロセスを終わらせる |
| **`max_kv_size` を明示する**                               | 既定では無制限。これを指定しないと KV キャッシュが伸び続ける                                                 |
| **計測の前に `mx.reset_peak_memory()` を呼ぶ**             | bench-mlx スクリプトは自動で呼んでいる。Ad-hoc 検証時は注意                                                  |
| **`max_recommended_working_set_size` を 1 つの目安にする** | Metal が安全に確保できる上限。超えると swap に逃げて全体が遅くなる                                           |
| **他の Metal アプリ(LM Studio / Ollama Metal)を止める**    | 同じユニファイドメモリプールを取り合うため、計測値が汚染される                                               |

Linux + Ollama 版との差分:

- **「`PROCESSOR=100% GPU` でないとオフロード」という概念はない**。Apple Silicon ではユニファイドメモリ上で動くため、CPU/GPU の境界が曖昧。代わりに「ピーク使用量 vs 実効 GPU 上限」で判定する
- **`keep_alive` は不要**。Python プロセスを終わらせれば即アンロード(`mx.clear_cache()` を併用)
- **systemd の override は不要**。すべて Python の引数で制御する

## 2. 計測コマンド(汎用テンプレ)

`bench-mlx/run_speed_mlx.py` を直接呼ぶのが正攻法。`<MODEL>` と `<CTX>` を埋めるだけ:

```bash
cd scripts/bench-mlx

MODEL=mlx-community/Qwen3.5-9B-MLX-4bit
CTX=32768

uv run python run_speed_mlx.py --model "$MODEL" --ctx "$CTX"
```

KV 量子化を試す場合:

```bash
uv run python run_speed_mlx.py --model "$MODEL" --ctx 131072 --kv-bits 8
uv run python run_speed_mlx.py --model "$MODEL" --ctx 131072 --kv-bits 4
```

直接 Python から使う(REPL や Jupyter で):

```python
from client_mlx import load_model, apply_chat_template, generate_with_metrics
import mlx.core as mx

model, tok = load_model("mlx-community/Qwen3.5-9B-MLX-4bit")
prompt = apply_chat_template(tok, "日本語で自己紹介してください。")

mx.reset_peak_memory()
result = generate_with_metrics(model, tok, prompt, max_tokens=128, max_kv_size=32768)
print(f"decode {result.generation_tps:.1f} tok/s")
print(f"prefill {result.prompt_tps:.1f} tok/s")
print(f"TTFT {result.ttft_sec:.2f}s")
print(f"peak {result.peak_memory_mib_mx:.0f} MiB")
```

## 3. 何を読むか

### `run_speed_mlx.py` の JSON 出力

`results/<timestamp>/speed_<model>_ctx<ctx>.json`:

```json
{
  "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
  "max_kv_size": 32768,
  "num_predict": 128,
  "prompt_tokens": 60,
  "prompt_tps": 1350.4,
  "generation_tokens": 128,
  "generation_tps": 92.1,
  "ttft_sec": 0.31,
  "total_elapsed_sec": 1.42,
  "peak_memory_mib": 6730.2,
  "active_memory_mib_after_load": 5380.1,
  "finish_reason": "length"
}
```

| フィールド                     | 意味                              | 判断                                                     |
| ------------------------------ | --------------------------------- | -------------------------------------------------------- |
| `prompt_tps`                   | prefill (プロンプト処理) の tok/s | 長プロンプトのスループット                               |
| `generation_tps`               | decode (生成) の tok/s            | 対話 UX に直結                                           |
| `ttft_sec`                     | 最初のトークンが出るまでの実時間  | 対話 UX の体感速度                                       |
| `peak_memory_mib`              | `mx.get_peak_memory()` のピーク   | これが `effective_gpu_limit_mib` の 90% 以下に収まるべき |
| `active_memory_mib_after_load` | モデル重みだけのおおよその常駐量  | weight だけのサイズ確認                                  |
| `finish_reason`                | `"length"` / `"stop"` / etc       | `length` なら `--num-predict` が小さすぎただけ           |

### `mx.device_info()` との突き合わせ

ロード後・生成後にデバイス情報と比較:

```python
import mlx.core as mx
info = mx.device_info()
limit_mib = info["max_recommended_working_set_size"] / (1024 * 1024)
peak_mib = mx.get_peak_memory() / (1024 * 1024)
print(f"peak={peak_mib:.0f} / limit={limit_mib:.0f} MiB ({peak_mib/limit_mib*100:.1f}%)")
```

`peak / limit` が 90% を超えていたら、長 ctx 化や並列度を上げると swap に追い出される可能性が高い。

### `top` / `vm_stat` 補足

Python プロセス側からの観測。Activity Monitor の "GPU" タブも参考になるが、サンプリングが粗い。

```bash
# 別ターミナルで watch するパターン
top -pid $(pgrep -f "uv run python run_speed_mlx") -stats pid,command,mem,rsize,vsize -l 0 -s 1
```

## 4. オフロード相当の判定

Apple Silicon ではユニファイドメモリ上で動くため Linux のような「CPU offload」は発生しないが、以下のいずれかが起きると **実用にならない**:

| 症状                                          | 観察方法                                                                                                          |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **swap への逃避**                             | `peak_memory_mib > max_recommended_working_set` を超えた時に起きやすい。`vm_stat` で `Pages swapped out` が増える |
| **`generation_tps` が極端に低い (< 5 tok/s)** | 重みが swap されていると桁で落ちる                                                                                |
| **`max_buffer_length` 超過で例外**            | 単一テンソル割当がデバイス上限を超えた場合。極端に大きなモデルで発生                                              |

→ どれかに該当したら [docs/03-tuning.md](../03-tuning.md) の判定フローを Mac/MLX 用に読み替えて適用する(KV 量子化 → `max_kv_size` を下げる → 重み量子化を強める)。

## 5. 計測結果の残し方

[docs/04-results.md](../04-results.md) の該当 Mac セクションに 1 行追記。必要な情報は以下:

- ハードウェア(`Apple M4 Pro (48GB Unified, MLX)` 等。`max_recommended_working_set` の値も併記)
- モデル名(`mlx-community/<repo>` フル ID)と重み量子化レベル
- 設定(`max_kv_size`, `kv_bits`, `prefill_step_size`)
- 観測値(`peak_memory_mib`, `generation_tps`, `prompt_tps`, `ttft_sec`)
- 計測時の `mlx` / `mlx-lm` バージョン

> Mac 計測結果ヘッダーには nvidia driver / CUDA バージョンの代わりに `mlx` / `mlx-lm` / macOS / カーネルバージョン を記録する。`bench-mlx/metadata_mlx.py` が自動採取する。
