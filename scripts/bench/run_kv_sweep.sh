#!/usr/bin/env bash
# KV キャッシュタイプを振って計測するためのスクリプト。
#
# 現在の OLLAMA_KV_CACHE_TYPE で全 13 モデルに対して
# ctx_search → speed → needle を回す。Coding/Summary は KV 量子化の
# 影響が小さい(ctx=16K で全モデル収まる範囲のため)スキップ。
#
# 事前に kv.py --type <q4_0|f16|q8_0> で KV を切り替えてから本スクリプトを起動する。

set -uo pipefail

MODELS=(
  "qwen3.5:4b-q4_K_M"
  "qwen3.5:4b-q8_0"
  "qwen3.5:9b-q4_K_M"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 現在の KV を取得(ファイル名に使うため)
KV=$(systemctl show ollama -p Environment | grep -oE 'OLLAMA_KV_CACHE_TYPE=[^ ]*' | cut -d= -f2)
if [ -z "$KV" ]; then
  KV="f16"  # 未設定なら Ollama 既定の f16
fi

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/run_kv_sweep-${KV}-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

echo "Started at $TS with KV=$KV"
echo "Models: ${#MODELS[@]}"
echo "Stages: ctx_search + speed + needle (coding/summary skipped)"

for M in "${MODELS[@]}"; do
  SANITIZED=$(echo "$M" | tr ':' _ | tr '/' _)
  echo
  echo "==================================================================="
  echo "===== $M  KV=$KV  (start $(date +%H:%M:%S))"
  echo "==================================================================="

  echo "[1/3] ctx_search"
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
    echo "  ERROR: max_ctx=0, skipping"
    continue
  fi
  echo "  max_ctx=$CTX"

  echo
  echo "[2/3] speed @ ctx=$CTX"
  uv run python run_speed.py --model "$M" --ctx "$CTX"

  echo
  echo "[3/3] needle @ depth=100% pos=50%"
  NEEDLE="data/needle/${SANITIZED}_kv${KV}_d100_p50.json"
  # 質問文 + トークナイズの余白を持たせるため ctx の 90% を狙う(chars==ctx だと
  # 末尾が切り詰められて needle が落ちることがある)
  NEEDLE_CHARS=$(( CTX * 9 / 10 ))
  uv run python data/needle/generate.py --chars "$NEEDLE_CHARS" --position-pct 0.5 \
      --output "$NEEDLE"
  uv run python run_needle.py --model "$M" --ctx "$CTX" --needle "$NEEDLE"

  echo "===== $M  KV=$KV  (end $(date +%H:%M:%S))"
done

echo
echo "All done at $(date +%H:%M:%S)"
