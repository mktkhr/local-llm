#!/usr/bin/env bash
# Needle In A Haystack の網羅スイープ。
#
# 各モデルについて 4 つの depth(ctx 長)× 3 つの needle 位置 = 12 試行。
# docs/06-evaluation.md §7.3 の仕様にほぼ合わせている(25/50/75/100% の depth は
# モデルごとに max_ctx が違うので、本スクリプトでは絶対値で揃える方が比較しやすく
# 4096 / 16384 / 65536 / max_ctx を採用)。
#
# 結果は results/<timestamp>/ 配下に通常通り保存される。

set -uo pipefail

# モデルと既知の max_ctx(KV=q8_0、main sweep の値)
declare -A MAX_CTX
MAX_CTX["qwen3.5:4b-q4_K_M"]=262144
MAX_CTX["gemma4:e4b-it-q4_K_M"]=131072
MAX_CTX["deepseek-r1:7b-qwen-distill-q4_K_M"]=104448
MAX_CTX["deepseek-coder-v2:16b-lite-instruct-q4_0"]=19456

MODELS=(
  "qwen3.5:4b-q4_K_M"
  "gemma4:e4b-it-q4_K_M"
  "deepseek-r1:7b-qwen-distill-q4_K_M"
  "deepseek-coder-v2:16b-lite-instruct-q4_0"
)

DEPTHS=(4096 16384 65536)  # plus max_ctx per model
POSITIONS=(0.1 0.5 0.9)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/run_needle_sweep-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

echo "Started at $TS"

run_one() {
  local M="$1"
  local CTX="$2"
  local POS="$3"
  local SANITIZED=$(echo "$M" | tr ':' _ | tr '/' _)
  local CHARS=$(( CTX * 9 / 10 ))
  local POS_INT=$(printf "%02.0f" "$(echo "$POS * 100" | bc)")
  local NEEDLE="data/needle/sweep_${SANITIZED}_d${CTX}_p${POS_INT}.json"

  echo "--- $M ctx=$CTX pos=$POS ---"
  uv run python data/needle/generate.py --chars "$CHARS" --position-pct "$POS" \
      --output "$NEEDLE"
  uv run python run_needle.py --model "$M" --ctx "$CTX" --needle "$NEEDLE"
}

for M in "${MODELS[@]}"; do
  CTX_MAX=${MAX_CTX[$M]}
  echo
  echo "==================================================================="
  echo "===== $M (max_ctx=$CTX_MAX, start $(date +%H:%M:%S))"
  echo "==================================================================="

  for D in "${DEPTHS[@]}"; do
    if [ "$D" -ge "$CTX_MAX" ]; then
      echo "  skip ctx=$D >= max_ctx=$CTX_MAX"
      continue
    fi
    for P in "${POSITIONS[@]}"; do
      run_one "$M" "$D" "$P"
    done
  done

  # max_ctx でもテスト
  for P in "${POSITIONS[@]}"; do
    run_one "$M" "$CTX_MAX" "$P"
  done

  echo "===== $M (end $(date +%H:%M:%S))"
done

echo
echo "All done at $(date +%H:%M:%S)"
