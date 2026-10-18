#!/usr/bin/env bash
# KV 型と n_ctx を切り替えながら Case A〜E を掃引する。
#
# cache-type-k/v と n_ctx は llama-server の起動時オプションなので、
# KV 型を変えるにはコンテナごと入れ替える必要がある。
#
# 使い方:
#   bash experiments/2026-09-kv-tier/scripts/sweep.sh q8_0 155648 14336 10000 25000 48000 100000 150000
set -euo pipefail

KV_TYPE="$1"; N_CTX="$2"; CACHE_RAM="$3"; shift 3
TOKENS=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/config/compose.yml"
ENVFILE="$ROOT/config/.env"

echo "=== KV=$KV_TYPE n_ctx=$N_CTX cache-ram=${CACHE_RAM}MiB tokens=${TOKENS[*]} ==="

KVT_KV_TYPE="$KV_TYPE" KVT_N_CTX="$N_CTX" KVT_CACHE_RAM="$CACHE_RAM" \
  docker compose -f "$COMPOSE" --env-file "$ENVFILE" up -d --force-recreate >/dev/null 2>&1

for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:18090/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:18090/health >/dev/null 2>&1 || { echo "サーバが起動しませんでした"; docker logs --tail 30 llama-kvtier; exit 1; }

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

python3 "$ROOT/scripts/run_cases.py" --tokens "${TOKENS[@]}" --kv-type "$KV_TYPE" --n-ctx "$N_CTX"

# 次の KV 型の測定に page cache を持ち越さないよう、KV ダンプを消す
rm -f "$ROOT/config/kv"/*.bin
