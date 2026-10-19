#!/usr/bin/env bash
# Phase 2: Claude Code の実タスク M1 / M2 を Laguna と Qwen3.6 で回す(cc_sweep.sh を流用)。
# 使い方(GPU 機、この実験ディレクトリで): bash scripts/phase2.sh
# 環境変数: CONDITIONS(既定 "current-C:none L-iq3xxs:none") RUNS(既定 3)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
SPEC_EXP="$(cd ../2026-10-speculative-decoding && pwd)"
( cd "$SPEC_EXP" && MODEL_ARGS_DIR="$HERE/../config/model" CONDITIONS="${CONDITIONS:-current-C:none L-iq3xxs:none}" \
    RUNS="${RUNS:-3}" TASKS="${TASKS:-M1 M2}" bash scripts/cc_sweep.sh )
# この実験の試行だけを results/raw/cc に集める(ラベルに L- か current-C を含み、cc_sweep の本日分)
mkdir -p results/raw/cc
for d in "$SPEC_EXP"/results/raw/cc/*/; do
  n="$(basename "$d")"
  case "$n" in L-*|current-C-none-run[0-9]*-M[12]) ;; *) continue ;; esac
  [ "$(find "$d" -maxdepth 1 -name summary.json -newermt "$(date +%Y-%m-%d)" | wc -l)" -gt 0 ] || continue
  rm -rf "results/raw/cc/$n" && cp -r "$d" "results/raw/cc/$n"
done
python3 scripts/aggregate_laguna.py
