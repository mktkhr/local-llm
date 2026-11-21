#!/usr/bin/env bash
# build × spec ごとに単独サーバを立て、全プロンプトを greedy で生成する(GPU 機で実行)。
#   bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
LABEL="${LABEL:-}"
OUT="${OUT:-results/raw/runs.jsonl}"; PORT=18091; mkdir -p results/raw
BUILDS="${BUILDS:-b12464 b12182}"; SPECS="${SPECS:-none none2 ngram-simple ngram-mod}"
for b in $BUILDS; do
  for s in $SPECS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
    spec=(); case "$s" in none|none2) ;; *) spec=(--spec-type "$s");; esac
    docker rm -f div >/dev/null 2>&1 || true
    docker run -d --name div --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache --entrypoint /app/llama-server "llama-swap:$b" \
      -m "$MODEL" --host 0.0.0.0 --port 8080 -ngl 99 --flash-attn on -np 1 -c 16384 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 \
      --no-mmproj --metrics "${spec[@]}" >/dev/null
    for i in $(seq 1 120); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
    echo "== $b $s $(date +%T)"
    python3 scripts/gen_probe.py --url http://127.0.0.1:${PORT} --prompts scripts/data/prompts.json --n "${N:-400}" --tag "build=$b${LABEL:+-$LABEL},spec=$s" --out "$OUT"
    docker rm -f div >/dev/null
  done
done
echo "== done $(date +%T)"
