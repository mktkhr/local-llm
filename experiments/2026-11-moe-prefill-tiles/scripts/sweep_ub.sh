#!/usr/bin/env bash
# build × ubatch ごとに全部 VRAM のサーバを立て、同じ 32k トークンの prefill を 3 回測る(GPU 機で実行)。
#   SRC=<W1-03.json> TURN=<cc-turn-overhead の scripts> bash scripts/sweep_ub.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
BUILDS="${BUILDS:-b12464 b12182}"; UBS="${UBS:-64 128 256 512 1024 2048 4096}"; PORT=18094
OUT="${OUT:-results/raw/prefill.jsonl}"; TOK=results/raw/prompt-tokens.json; mkdir -p results/raw
for b in $BUILDS; do
  for ub in $UBS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
    bs=$(( ub > 2048 ? ub : 2048 ))
    docker rm -f pf >/dev/null 2>&1 || true
    docker run -d --name pf --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache --entrypoint /app/llama-server "llama-swap:$b" \
      -m "$MODEL" --host 0.0.0.0 --port 8080 --jinja -ngl 99 --flash-attn on -np 1 -c 36864 -ctk q8_0 -ctv q8_0 \
      -b "$bs" -ub "$ub" --no-mmproj >/dev/null
    ok=0; for i in $(seq 1 120); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && { ok=1; break; }; sleep 2; done
    if [ "$ok" = 0 ]; then echo "== $b ub=$ub failed to start"; docker logs pf 2>&1 | tail -3; docker rm -f pf >/dev/null; continue; fi
    echo "== $b ub=$ub $(date +%T) vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    python3 scripts/prefill_probe.py --url http://127.0.0.1:${PORT} --tokens "$TOK" --src "${SRC:-}" --turn-scripts "${TURN:-}" \
      --reps 3 --tag "build=$b,ub=$ub" --out "$OUT" || echo "== $b ub=$ub probe failed"
    docker rm -f pf >/dev/null
  done
done
echo "== done $(date +%T)"
