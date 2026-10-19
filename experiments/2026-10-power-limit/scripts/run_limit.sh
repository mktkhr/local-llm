#!/usr/bin/env bash
# 電力リミット 1 水準ぶんの掃引を回す。リミットの変更自体は sudo が要るので、利用者が
#   sudo nvidia-smi -pl <W>
# を実行したあとに本スクリプトを呼ぶ。現在値が引数と一致しなければ何もせず終了する。
#
# 使い方(GPU 機、この実験ディレクトリで): bash scripts/run_limit.sh 150
# 環境変数: BLOCKS(既定 2) SPECS(既定 "none mtp2") WORKLOADS(既定 "W1 W3 W4") CONFIGS(既定 C-iq3xxs)
set -euo pipefail
LIMIT="${1:?電力リミット(W)を指定}"
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
SPEC_EXP="$(cd ../2026-10-speculative-decoding && pwd)"
cur="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1 | tr -d ' ')"
if [ "$cur" != "$LIMIT" ]; then
  echo "現在の電力リミットは ${cur} W です。先に GPU 機で次を実行してください:  sudo nvidia-smi -pl ${LIMIT}" >&2
  exit 2
fi
TS="$(date +%Y%m%d)-pl${LIMIT}"
echo "=== power limit ${LIMIT} W: sweep ${TS} (blocks=${BLOCKS:-2})"
( cd "$SPEC_EXP" && RESUME_TS="$TS" BLOCKS="${BLOCKS:-2}" CONFIGS="${CONFIGS:-C-iq3xxs}" SPECS="${SPECS:-none mtp2}" \
    WORKLOADS="${WORKLOADS:-W1 W3 W4}" bash scripts/sweep.sh )
mkdir -p results/raw
cp "$SPEC_EXP/results/raw/sweep-${TS}.jsonl" results/raw/
cp "$SPEC_EXP/results/raw/primer-${TS}.jsonl" results/raw/ 2>/dev/null || true
echo "=== copied results/raw/sweep-${TS}.jsonl ($(wc -l < results/raw/sweep-${TS}.jsonl) rows)"
python3 scripts/aggregate_power.py || true
