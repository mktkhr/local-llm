#!/usr/bin/env bash
# 評価対象モデル群に対して ctx_search → speed → needle → coding → summary を順に実行する。
#
# 既定リストは qwen3.5:9b-q4_K_M を除く 12 モデル(同モデルは事前に手動実施済み想定)。
# 引数でモデルタグを渡せばそれだけを対象にする。
#
# ログは results/run_all-<timestamp>.log に追記。

set -uo pipefail

DEFAULT_MODELS=(
  "qwen3.5:4b-q4_K_M"
  "qwen3.5:4b-q8_0"
  "qwen3.5:9b-q8_0"
  "gemma4:e2b-it-q4_K_M"
  "gemma4:e4b-it-q4_K_M"
  "gemma4:e4b-it-q8_0"
  "deepseek-r1:7b-qwen-distill-q4_K_M"
  "deepseek-r1:7b-qwen-distill-q8_0"
  "deepseek-r1:8b-0528-qwen3-q4_K_M"
  "deepseek-r1:14b-qwen-distill-q4_K_M"
  "deepseek-coder-v2:16b-lite-instruct-q4_0"
  "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
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
  uv run python ctx_search.py --model "$M" --low 4096 --high 262144 \
      --tolerance 4096 --min-free-mib 500

  CTX_FILE=$(find results/ -name "ctx_search_${SANITIZED}.json" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
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
  uv run python run_speed.py --model "$M" --ctx "$CTX"

  echo
  echo "[3/5] needle @ depth=100% pos=50%"
  NEEDLE="data/needle/${SANITIZED}_d100_p50.json"
  # 質問文 + トークナイズ余白で末尾が削れるため chars は ctx の 90% を狙う
  NEEDLE_CHARS=$(( CTX * 9 / 10 ))
  uv run python data/needle/generate.py --chars "$NEEDLE_CHARS" --position-pct 0.5 \
      --output "$NEEDLE"
  uv run python run_needle.py --model "$M" --ctx "$CTX" --needle "$NEEDLE"

  echo
  echo "[4/5] coding @ ctx=16384"
  uv run python run_coding.py --model "$M" --ctx 16384 \
      --tasks data/coding/tasks.json

  echo
  echo "[5/5] summary @ ctx=16384"
  uv run python run_summary.py --model "$M" --ctx 16384 \
      --meetings data/summary/meetings.json

  echo "===== $M  (end $(date +%H:%M:%S))"
done

echo
echo "All done at $(date +%H:%M:%S)"
