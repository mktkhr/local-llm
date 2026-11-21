#!/usr/bin/env bash
# build ごとに本番 C 構成相当(IQ3_S、128k、KV q8_0)のサーバを立て、ツール軸とメッセージ軸を測る(GPU 機で実行)。
#   SRC=<private ディレクトリ> bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
SRC="${SRC:?SRC に Claude Code リクエストのディレクトリを指定}"
OUT="${OUT:-results/raw/turns.jsonl}"; PORT=18092; mkdir -p results/raw
BUILDS="${BUILDS:-b12464 b12182}"
for b in $BUILDS; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
  docker rm -f turn >/dev/null 2>&1 || true
  docker run -d --name turn --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache --entrypoint /app/llama-server "llama-swap:$b" \
    -m "$MODEL" --host 0.0.0.0 --port 8080 --jinja -ngl 99 --flash-attn on -np 1 -c 131072 --kv-unified \
    -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --cache-reuse 256 --no-mmproj --metrics >/dev/null
  for i in $(seq 1 120); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
  for spec in "W1-02:tools" "W1-03:msgs"; do
    echo "== $b ${spec} $(date +%T)"
    python3 scripts/turn_probe.py --url http://127.0.0.1:${PORT} --src "$SRC/${spec%%:*}.json" --axis "${spec##*:}" \
      --tag "build=$b" --out "$OUT" --reps "${REPS:-5}"
  done
  docker logs turn > "results/raw/server-$b.log" 2>&1 || true
  docker rm -f turn >/dev/null
done
echo "== done $(date +%T)"
