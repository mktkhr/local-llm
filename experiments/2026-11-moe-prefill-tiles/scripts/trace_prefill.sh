#!/usr/bin/env bash
# 同じ 32k トークンを -ub 32 で prefill し、全 ubatch のルーティングを MoE cache のトレースに残す(GPU 機で実行)。
# 32 トークン以下のバッチだけがトレースされるので -ub 32。-cmoe と全 expert が入るキャッシュはトレースを有効にするためだけ。
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
PORT=18094; mkdir -p results/raw
docker rm -f pf >/dev/null 2>&1 || true
docker run -d --name pf --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache -v "$PWD/results/raw:/out" \
  -e LLAMA_MOE_CACHE_TRACE=/out/trace-prefill.trace --entrypoint /app/llama-server llama-swap:b12464 \
  -m "$MODEL" --host 0.0.0.0 --port 8080 -ngl 99 -cmoe --moe-cache-mib 11100 --flash-attn on -np 1 -c 36864 \
  -ctk q8_0 -ctv q8_0 -b 32 -ub 32 --no-mmproj >/dev/null
for i in $(seq 1 120); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
python3 scripts/prefill_probe.py --url http://127.0.0.1:${PORT} --tokens results/raw/prompt-tokens.json --reps 1 \
  --tag "build=b12464,ub=32,trace=1" --out results/raw/trace-run.jsonl
docker stop -t 60 pf >/dev/null; docker rm pf >/dev/null
ls -la results/raw/trace-prefill.trace
