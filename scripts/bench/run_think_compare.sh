#!/usr/bin/env bash
# think=true / think=false を比較するためのスクリプト。
#
# ctx=16384 を固定して、Speed / Coding / Summary を両モードで実行する。
# Needle は max_ctx 依存なのでこの sweep からは外す。
#
# num_predict は think=true で「思考トークン + 回答」の余白を持たせるため
# 既定より大きめ(speed 512、coding/summary 2048)に設定。

set -uo pipefail

MODELS=(
  "qwen3.5:9b-q4_K_M"
  "gemma4:e4b-it-q4_K_M"
  "deepseek-r1:8b-0528-qwen3-q4_K_M"
)

CTX=16384

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/run_think_compare-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

echo "Started at $TS with ctx=$CTX"
echo "Models: ${#MODELS[@]}"

for M in "${MODELS[@]}"; do
  for T in false true; do
    NP_SPEED=128
    NP_QUALITY=1024
    if [ "$T" = "true" ]; then
      NP_SPEED=512
      NP_QUALITY=2048
    fi
    echo
    echo "==================================================================="
    echo "===== $M  think=$T  (start $(date +%H:%M:%S))"
    echo "==================================================================="

    echo "[1/3] speed @ ctx=$CTX think=$T num_predict=$NP_SPEED"
    uv run python run_speed.py --model "$M" --ctx "$CTX" \
        --think "$T" --num-predict "$NP_SPEED"

    echo
    echo "[2/3] coding @ ctx=$CTX think=$T num_predict=$NP_QUALITY"
    uv run python run_coding.py --model "$M" --ctx "$CTX" \
        --think "$T" --num-predict "$NP_QUALITY" \
        --tasks data/coding/tasks.json

    echo
    echo "[3/3] summary @ ctx=$CTX think=$T num_predict=$NP_QUALITY"
    uv run python run_summary.py --model "$M" --ctx "$CTX" \
        --think "$T" --num-predict "$NP_QUALITY" \
        --meetings data/summary/meetings.json

    echo "===== $M  think=$T  (end $(date +%H:%M:%S))"
  done
done

echo
echo "All done at $(date +%H:%M:%S)"
