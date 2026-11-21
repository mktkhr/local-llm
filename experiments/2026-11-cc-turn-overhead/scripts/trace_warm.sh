#!/usr/bin/env bash
# 再送 1 回の残差(スロット側の約 50 ms)の中身を見るため、-lv 4(trace)で同一本文を 3 回ずつ送ってログを残す(GPU 機で実行)。
#   SRC=<private ディレクトリ> bash scripts/trace_warm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
SRC="${SRC:?}"; B="${B:-b12464}"; PORT=18092
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
docker rm -f turn >/dev/null 2>&1 || true
docker run -d --name turn --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache --entrypoint /app/llama-server "llama-swap:$B" \
  -m "$MODEL" --host 0.0.0.0 --port 8080 --jinja -ngl 99 --flash-attn on -np 1 -c 131072 --kv-unified \
  -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --no-mmproj --metrics -lv 4 >/dev/null
for i in $(seq 1 120); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
python3 - "$SRC/W1-02.json" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from turn_probe import call, variant
src = json.load(open(sys.argv[1]))
for n in (0, 137):
    body = variant(src, n_tools=n, n_msgs=None, max_tokens=1)
    for i in range(4):
        st, _, wall, err = call("http://127.0.0.1:18092", "/v1/messages", body)
        print(f"tools={n} rep={i} status={st} wall={wall*1000:.1f} ms {err}", flush=True)
PY
docker logs turn > "results/raw/trace-$B.log" 2>&1 || true
docker rm -f turn >/dev/null
