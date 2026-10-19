#!/usr/bin/env bash
# Phase 1: Laguna と Qwen3.6 の速度を同じワークロードで測る(2026-10-speculative-decoding の sweep.sh を流用)。
# 使い方(GPU 機、この実験ディレクトリで): bash scripts/phase1.sh
# 環境変数: CONFIGS(既定 "L-iq3xxs current-C") BLOCKS(既定 2) WORKLOADS(既定 "W1 W3 W4")
# プロンプトは scripts/data/private(Laguna の 64k に収まるよう --max-bytes 230000 で抽出し直したもの)を使う
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
SPEC_EXP="$(cd ../2026-10-speculative-decoding && pwd)"
TS="$(date +%Y%m%d)-laguna"
( cd "$SPEC_EXP" && MODEL_ARGS_DIR="$HERE/../config/model" PRIVATE_DIR="$HERE/data/private" RESUME_TS="$TS" BLOCKS="${BLOCKS:-2}" \
    CONFIGS="${CONFIGS:-L-iq3xxs current-C}" SPECS=none WORKLOADS="${WORKLOADS:-W1 W3 W4}" bash scripts/sweep.sh )
mkdir -p results/raw && cp "$SPEC_EXP/results/raw/sweep-${TS}.jsonl" results/raw/
python3 scripts/aggregate_laguna.py
