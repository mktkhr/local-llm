#!/usr/bin/env bash
# 評価対象 MLX モデル群を「1 つずつ pull → 評価 → 削除」のサイクルで回す。
#
# `run_all_mlx.sh` は事前に全モデルを pull 済みであることを前提にするが、
# 本スクリプトはディスク容量制約がある環境向けに、サイクルでローテーションする。
#
# 引数でリポジトリ ID を渡せばそれだけを対象にする。省略時は DEFAULT_MODELS を使う。
# ログは results/eval_cycle-<timestamp>.log に追記。

set -uo pipefail

# Python stdout/stderr をラインバッファで吐かせる。
# パイプ (tee) 経由だと既定でブロックバッファになり、長時間プローブ中の
# [probe] / max_ctx= が見えなくなって進捗判別ができないため。
export PYTHONUNBUFFERED=1

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
  "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit"
  "mlx-community/gemma-4-26b-a4b-it-4bit"
  "mlx-community/Qwen3.5-27B-4bit"
  "mlx-community/Qwen3.6-27B-4bit"
  "mlx-community/gemma-4-31b-it-4bit"
  "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
  # 極大 (~30 GB)、KV 余地が狭いがとりあえず測る
  "mlx-community/Qwen3.5-27B-8bit"
  "mlx-community/Qwen3.6-27B-8bit"
)

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/eval_cycle-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

# モデル ID から HF cache のディレクトリ名を作る (org/repo → models--org--repo)
hf_cache_dir() {
  local repo="$1"
  local sanitized="${repo//\//--}"
  echo "$HOME/.cache/huggingface/hub/models--${sanitized}"
}

free_disk_gb() {
  df -g . | awk 'NR==2 {print $4}'
}

echo "Started at $TS"
echo "Models: ${#MODELS[@]}"
echo "Free disk: $(free_disk_gb) GB"
for M in "${MODELS[@]}"; do echo "  - $M"; done

OK_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

for M in "${MODELS[@]}"; do
  SANITIZED=$(echo "$M" | tr ':' _ | tr '/' _)
  CACHE_DIR="$(hf_cache_dir "$M")"
  echo
  echo "==================================================================="
  echo "===== $M  (start $(date +%H:%M:%S), free $(free_disk_gb) GB)"
  echo "==================================================================="

  # ----------------- pull -----------------
  echo "[0/5] pull"
  if ! uv run python pull_models_mlx.py "$M"; then
    echo "  ERROR: pull failed, skip"
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILED_MODELS+=("$M:pull")
    continue
  fi
  echo "  free after pull: $(free_disk_gb) GB"

  # ----------------- ctx_search -----------------
  echo "[1/5] ctx_search"
  if ! uv run python ctx_search_mlx.py --model "$M" --low 4096 --high 262144 \
        --tolerance 4096 --safety-margin-mib 2048; then
    echo "  ERROR: ctx_search failed"
    FAILED_MODELS+=("$M:ctx_search")
  fi

  CTX_FILE=$(ls -t "results"/*/"ctx_search_${SANITIZED}.json" 2>/dev/null | head -1)
  if [ -z "$CTX_FILE" ] || [ ! -f "$CTX_FILE" ]; then
    echo "  ERROR: ctx_search result not found, cleanup & skip"
    rm -rf "$CACHE_DIR" 2>/dev/null
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILED_MODELS+=("$M:ctx_no_result")
    continue
  fi
  CTX=$(python3 -c "import json; print(json.load(open('$CTX_FILE'))['max_ctx'])")
  if [ "$CTX" = "0" ]; then
    echo "  ERROR: max_ctx=0 (probe failed at low bound), cleanup & skip"
    rm -rf "$CACHE_DIR" 2>/dev/null
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILED_MODELS+=("$M:max_ctx_zero")
    continue
  fi
  echo "  max_ctx=$CTX"

  # ----------------- speed -----------------
  echo
  echo "[2/5] speed @ ctx=$CTX"
  uv run python run_speed_mlx.py --model "$M" --ctx "$CTX" || \
    FAILED_MODELS+=("$M:speed")

  # ----------------- needle -----------------
  echo
  echo "[3/5] needle @ depth=100% pos=50%"
  NEEDLE="data/needle/${SANITIZED}_d100_p50.json"
  # KV ヘッダ等で末尾が削れることを見越して chars は ctx の 90% を狙う
  NEEDLE_CHARS=$(( CTX * 9 / 10 ))
  uv run python data/needle/generate.py --chars "$NEEDLE_CHARS" --position-pct 0.5 \
      --output "$NEEDLE"
  uv run python run_needle_mlx.py --model "$M" --ctx "$CTX" --needle "$NEEDLE" || \
    FAILED_MODELS+=("$M:needle")

  # ----------------- coding -----------------
  echo
  echo "[4/5] coding @ ctx=16384"
  uv run python run_coding_mlx.py --model "$M" --ctx 16384 \
      --tasks data/coding/tasks.json || FAILED_MODELS+=("$M:coding")

  # ----------------- summary -----------------
  echo
  echo "[5/5] summary @ ctx=16384"
  uv run python run_summary_mlx.py --model "$M" --ctx 16384 \
      --meetings data/summary/meetings.json || FAILED_MODELS+=("$M:summary")

  # ----------------- cleanup -----------------
  echo
  echo "[cleanup] $CACHE_DIR"
  if [ -d "$CACHE_DIR" ]; then
    DU=$(du -sh "$CACHE_DIR" 2>/dev/null | awk '{print $1}')
    rm -rf "$CACHE_DIR"
    echo "  removed ${DU} (free now $(free_disk_gb) GB)"
  else
    echo "  cache dir not found, skip"
  fi

  echo "===== $M  (end $(date +%H:%M:%S))"
  OK_COUNT=$((OK_COUNT+1))
done

echo
echo "==================================================================="
echo "All done at $(date +%H:%M:%S)"
echo "  OK: $OK_COUNT"
echo "  Issues: ${#FAILED_MODELS[@]}"
for M in "${FAILED_MODELS[@]}"; do echo "    - $M"; done
echo "Log: $LOG"
