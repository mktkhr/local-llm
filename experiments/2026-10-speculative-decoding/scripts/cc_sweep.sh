#!/usr/bin/env bash
# Phase 2: 条件(config:spec)ごとに llama-spec + ccshim-spec を立て、Claude Code の実タスクを cc_task.sh で回す。
#
# 使い方(実験ディレクトリ、GPU 機で):
#   bash scripts/cc_sweep.sh
#   CONDITIONS="C-iq3xxs:none C-iq3xxs:mtp2" RUNS=1 TASKS=M1 bash scripts/cc_sweep.sh
#
# 環境変数:
#   CONDITIONS  "config:spec" の空白区切り(既定: 現行構成 / 同一 GGUF の none / mtp2 / mtp1)
#   RUNS        各条件 × 課題の試行数(既定 2)。run ごとに全条件を一巡させ、時間ドリフトを条件間で均す
#   TASKS       既定 "M1 M2"
#   DRY_RUN     1 なら docker / cc_task を echo
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
CONDITIONS="${CONDITIONS:-current-C:none C-iq3xxs:none C-iq3xxs:mtp2 C-iq3xxs:mtp1}"
RUNS="${RUNS:-2}"; TASKS="${TASKS:-M1 M2}"; DRY_RUN="${DRY_RUN:-0}"
URL="http://127.0.0.1:${SPEC_SERVER_PORT:-18081}"
COMPOSE="docker compose -f config/compose.yml"
run() { if [ "$DRY_RUN" = "1" ]; then echo "$*"; else "$@"; fi; }
TS="$(date +%Y%m%d-%H%M%S)"; LOG="results/raw/cc-sweep-${TS}.log"; mkdir -p results/raw results/raw/ccshim-log
trap 'echo "中断: 停止します" >&2; $COMPOSE down >/dev/null 2>&1 || true' INT TERM

wait_health() {
  local waited=0
  while true; do
    if [ "$DRY_RUN" = "1" ]; then run curl -sf "$URL/health"; return 0; fi
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    if [ "$(docker inspect -f '{{.State.Running}}' llama-spec 2>/dev/null)" != "true" ] \
       || docker logs llama-spec 2>&1 | grep -qE "ggml_abort|CUDA error|exiting due to model loading error|failed to allocate"; then
      echo "llama-spec の起動に失敗" | tee -a "$LOG" >&2; docker logs --tail 20 llama-spec 2>&1 | tee -a "$LOG" >&2 || true; return 1
    fi
    sleep 5; waited=$((waited + 5)); [ "$waited" -ge 1800 ] && { echo "起動タイムアウト" | tee -a "$LOG" >&2; return 1; }
  done
}

echo "=== cc sweep ${TS}: conditions=[${CONDITIONS}] runs=${RUNS} tasks=[${TASKS}]" | tee -a "$LOG"
for ((i = 1; i <= RUNS; i++)); do
  for cond in $CONDITIONS; do
    config="${cond%%:*}"; spec="${cond##*:}"
    export LLAMA_ARGS="$(cat config/common.args "config/model/${config}.args" "config/spec/${spec}.args" | tr '\n' ' ')"
    echo "--- run=$i $config/$spec" | tee -a "$LOG"
    run $COMPOSE up -d --force-recreate
    wait_health || { run $COMPOSE down; continue; }
    # ウォームアップ(モデル切替直後の /v1/messages は切られるため)
    run curl -s -m 600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":1}' -o /dev/null
    for task in $TASKS; do
      run bash scripts/cc_task.sh "$task" "${config}-${spec}-run${i}" 2>&1 | tee -a "$LOG" || true
    done
    run $COMPOSE down
  done
done
run python3 scripts/aggregate.py
echo "=== done" | tee -a "$LOG"
