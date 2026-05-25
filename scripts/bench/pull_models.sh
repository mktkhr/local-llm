#!/usr/bin/env bash
# RTX 4080 SUPER 評価対象モデル(docs/04-results.md の選定)をまとめて pull する。
#
# 小さいモデルから順に pull することで、ダウンロード完了次第すぐ計測を始められる
# ようにしている。Ollama は既存タグがあればスキップするので、再実行しても安全。
#
# 進捗はこのスクリプトの stdout と、results/pull-<timestamp>.log の双方に出る。
#
# 使い方:
#   ./pull_models.sh                  # 既定の対象リストを pull
#   ./pull_models.sh foo:tag bar:tag  # 指定タグだけ pull

set -uo pipefail

DEFAULT_MODELS=(
  # Qwen3.5 系(汎用、主力)
  "qwen3.5:4b-q4_K_M"                            # 3.4 GB
  "qwen3.5:4b-q8_0"                              # 5.3 GB
  "qwen3.5:9b-q4_K_M"                            # 6.6 GB(既存 qwen3.5:9b と同一の見込み)
  "qwen3.5:9b-q8_0"                              # 11 GB

  # Gemma 4 系(マルチモーダル、e4b 中心)
  "gemma4:e2b-it-q4_K_M"                         # 7.2 GB
  "gemma4:e4b-it-q4_K_M"                         # 9.6 GB
  "gemma4:e4b-it-q8_0"                           # 12 GB

  # DeepSeek-R1 系(思考特化 distill)
  "deepseek-r1:7b-qwen-distill-q4_K_M"           # 4.7 GB
  "deepseek-r1:7b-qwen-distill-q8_0"             # 8.1 GB
  "deepseek-r1:8b-0528-qwen3-q4_K_M"             # 5.2 GB
  "deepseek-r1:14b-qwen-distill-q4_K_M"          # 9.0 GB

  # DeepSeek-Coder-V2 系(コーディング MoE)
  "deepseek-coder-v2:16b-lite-instruct-q4_0"     # 8.9 GB(既定 ctx 160K)
  "deepseek-coder-v2:16b-lite-instruct-q4_K_M"   # 10 GB
)

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/pull-$TS.log"

log() {
  local msg="$*"
  echo "[$(date +%H:%M:%S)] $msg" | tee -a "$LOG_FILE"
}

log "pull start: ${#MODELS[@]} models"
log "log file: $LOG_FILE"
echo

declare -a failed
SECONDS_TOTAL=0

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  idx=$((i + 1))
  log "[$idx/${#MODELS[@]}] $model"
  start=$SECONDS
  if ollama pull "$model" 2>&1 | tee -a "$LOG_FILE"; then
    elapsed=$((SECONDS - start))
    SECONDS_TOTAL=$((SECONDS_TOTAL + elapsed))
    log "  ok ($elapsed s)"
  else
    log "  FAILED"
    failed+=("$model")
  fi
  echo
done

log "pull finished. total: ${SECONDS_TOTAL}s"
if [ "${#failed[@]}" -gt 0 ]; then
  log "FAILED models:"
  for m in "${failed[@]}"; do
    log "  - $m"
  done
  exit 1
fi

log "all models pulled successfully."
