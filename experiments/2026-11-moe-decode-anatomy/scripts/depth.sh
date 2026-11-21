#!/usr/bin/env bash
# decode 時間の文脈長依存(c·ctx)を llama bench の -d で測る(k は既定の 8)。GPU 機で実行。
#   IMAGE=llama-swap:b12464 BUILD=b12464 bash scripts/depth.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-llama-swap:b12182}"; BUILD="${BUILD:-b12182}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
OUT="${OUT:-results/raw/depth.jsonl}"; mkdir -p results/raw
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
docker run --rm --gpus all -v llama_llama-cache:/root/.cache --entrypoint /app/llama "$IMAGE" bench \
  -m "$MODEL" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384,32768,65536 -r 3 -o jsonl 2>/dev/null \
  | python3 -c "
import json,sys
for l in sys.stdin:
    l=l.strip()
    if not l.startswith('{'): continue
    r=json.loads(l); r['build']='$BUILD'; print(json.dumps(r))" >> "$OUT"
tail -5 "$OUT" | python3 -c "
import json,sys
for l in sys.stdin:
    r=json.loads(l); print(r['build'], 'd', r.get('n_depth'), 'tg', round(r.get('avg_ts',0),2), 'tok/s')"
