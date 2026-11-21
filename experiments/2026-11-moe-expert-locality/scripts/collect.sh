#!/usr/bin/env bash
# キャッシュサイズごとに -cmoe のサーバを立て、同じプロンプト列を greedy で流してトレースと速度を残す(GPU 機で実行)。
#   CACHES="11000 8192 4096" PROMPTS=<prompts.json> bash scripts/collect.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
IMAGE="${IMAGE:-llama-swap:b12464}"; PORT=18093; OUT="${OUT:-results/raw/runs.jsonl}"
PROMPTS="${PROMPTS:?}"; CACHES="${CACHES:?}"; N="${N:-400}"; EXTRA="${EXTRA:-}"; LABEL="${LABEL:-}"
mkdir -p results/raw
for c in $CACHES; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
  name="c${c}${LABEL:+-$LABEL}"
  docker rm -f moe >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  docker run -d --name moe --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache -v "$PWD/results/raw:/out" \
    -e LLAMA_MOE_CACHE_TRACE="/out/trace-${name}.trace" --entrypoint /app/llama-server "$IMAGE" \
    -m "$MODEL" --host 0.0.0.0 --port 8080 -ngl 99 -cmoe --moe-cache-mib "$c" --flash-attn on -np 1 -c 16384 \
    -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --no-mmproj --metrics $EXTRA >/dev/null
  for i in $(seq 1 180); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
  echo "== ${name} $(date +%T) vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  python3 scripts/decode_run.py --url http://127.0.0.1:${PORT} --prompts "$PROMPTS" --n "$N" \
    --tag "cache_mib=${c},label=${LABEL:-default}" --out "$OUT"
  docker stop -t 60 moe >/dev/null
  docker logs moe > "results/raw/server-${name}.log" 2>&1
  docker rm moe >/dev/null
done
echo "== done $(date +%T)"
