#!/usr/bin/env bash
# 評価対象 MLX モデル群に対して ctx_search → speed → needle → coding → summary を順に実行する。
#
# 引数でリポジトリ ID を渡せばそれだけを対象にする。省略時は DEFAULT_MODELS を使う。
#
# ログは results/run_all-<timestamp>.log に追記。

set -uo pipefail

# 小さい順。途中で中断しても評価済みモデル数が最大化されるように。
# 27B-8bit / DSC-Lite-8bit は重みだけで working_set の大半を占めるため初期 pass では除外。
DEFAULT_MODELS=(
  # 小型 (< 5 GB)
  "mlx-community/Qwen3.5-4B-MLX-4bit"
  "mlx-community/gemma-4-e2b-it-4bit"
  "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
  "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
  "mlx-community/Qwen3.5-4B-MLX-8bit"
  "mlx-community/gemma-4-e4b-it-4bit"
  # 中型 (5〜10 GB)
  "mlx-community/Qwen3.5-9B-MLX-4bit"
  "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit"
  "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit"
  "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
  "mlx-community/gemma-4-e4b-it-8bit"
  "mlx-community/Qwen3.5-9B-MLX-8bit"
  # 大型 (10〜20 GB)
  "mlx-community/gemma-4-26b-a4b-it-4bit"
  "mlx-community/Qwen3.5-27B-4bit"
  "mlx-community/Qwen3.6-27B-4bit"
  "mlx-community/gemma-4-31b-it-4bit"
  "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
)

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/run_all-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

echo "Started at $TS"
echo "Models: ${#MODELS[@]}"
for M in "${MODELS[@]}"; do echo "  - $M"; done

for M in "${MODELS[@]}"; do
  SANITIZED=$(echo "$M" | tr ':' _ | tr '/' _)
  echo
  echo "==================================================================="
  echo "===== $M  (start $(date +%H:%M:%S))"
  echo "==================================================================="

  echo "[1/5] ctx_search"
  uv run python ctx_search_mlx.py --model "$M" --low 4096 --high 262144 \
      --tolerance 4096 --safety-margin-mib 2048

  CTX_FILE=$(find results/ -name "ctx_search_${SANITIZED}.json" \
      -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -1)
  if [ -z "$CTX_FILE" ] || [ ! -f "$CTX_FILE" ]; then
    echo "  ERROR: ctx_search result not found, skipping rest"
    continue
  fi
  CTX=$(python3 -c "import json; print(json.load(open('$CTX_FILE'))['max_ctx'])")
  if [ "$CTX" = "0" ]; then
    echo "  ERROR: max_ctx=0 (probe failed at low bound), skipping"
    continue
  fi
  echo "  max_ctx=$CTX"

  echo
  echo "[2/5] speed @ ctx=$CTX"
  uv run python run_speed_mlx.py --model "$M" --ctx "$CTX"

  echo
  echo "[3/5] needle @ depth=100% pos=50%"
  NEEDLE="data/needle/${SANITIZED}_d100_p50.json"
  NEEDLE_CHARS=$(( CTX * 9 / 10 ))
  uv run python data/needle/generate.py --chars "$NEEDLE_CHARS" --position-pct 0.5 \
      --output "$NEEDLE"
  uv run python run_needle_mlx.py --model "$M" --ctx "$CTX" --needle "$NEEDLE"

  echo
  echo "[4/5] coding @ ctx=16384"
  uv run python run_coding_mlx.py --model "$M" --ctx 16384 \
      --tasks data/coding/tasks.json

  echo
  echo "[5/5] summary @ ctx=16384"
  uv run python run_summary_mlx.py --model "$M" --ctx 16384 \
      --meetings data/summary/meetings.json

  echo "===== $M  (end $(date +%H:%M:%S))"
done

echo
echo "All done at $(date +%H:%M:%S)"
