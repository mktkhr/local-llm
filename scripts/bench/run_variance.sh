#!/usr/bin/env bash
# 同一構成で複数回計測して再現性 / 揺らぎを見るスクリプト。
#
# Coding と Summary は temperature=1 既定で揺らぎが出るので、3 回繰り返して
# mean / stddev を求めるためのデータを取る。
#
# Speed の決定速度は揺らぎが小さいので本スクリプトでは対象外。

set -uo pipefail

MODELS=(
  "qwen3.5:9b-q4_K_M"
  "gemma4:e4b-it-q4_K_M"
  "deepseek-r1:8b-0528-qwen3-q4_K_M"
)

CTX=16384
ITERATIONS=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/run_variance-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

echo "Started at $TS"

for ITER in $(seq 1 "$ITERATIONS"); do
  for M in "${MODELS[@]}"; do
    echo
    echo "==================================================================="
    echo "===== iter=$ITER  $M  (start $(date +%H:%M:%S))"
    echo "==================================================================="

    echo "[coding hard]"
    uv run python run_coding.py --model "$M" --ctx "$CTX" \
        --num-predict 3072 \
        --tasks data/coding/tasks_hard.json

    echo
    echo "[summary]"
    uv run python run_summary.py --model "$M" --ctx "$CTX" \
        --num-predict 2048 \
        --meetings data/summary/meetings.json

    echo "===== iter=$ITER  $M  (end $(date +%H:%M:%S))"
  done
done

echo
echo "All done at $(date +%H:%M:%S)"
