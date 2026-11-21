#!/usr/bin/env bash
# k(活性 experts 数)× CUDA graph の on/off を、単独の llama-server コンテナで掃引する(GPU 機で実行)。
#   IMAGE=llama-swap:b12182 BUILD=b12182 bash scripts/sweep_k.sh
# llama-swap が GPU を掴んでいないこと(VRAM 1 GB 未満)を毎回確かめる。
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-llama-swap:b12182}"; BUILD="${BUILD:-b12182}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
ARCH="${ARCH:-qwen35moe}"; KS="${KS:-1 2 4 6 8 10 12 16}"; GRAPHS="${GRAPHS:-on off}"
LABEL="${LABEL:-qwen36-iq3s}"; OUT="${OUT:-results/raw/k-sweep.jsonl}"; PORT=18090
mkdir -p results/raw
for g in $GRAPHS; do
  for k in $KS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$used" -gt 1000 ] && { echo "GPU busy (${used} MiB), abort"; exit 1; }
    env_args=(); [ "$g" = off ] && env_args=(-e GGML_CUDA_DISABLE_GRAPHS=1); [ "$g" = opt ] && env_args=(-e GGML_CUDA_GRAPH_OPT=1)
    ov=(); [ "$k" != "-" ] && ov=(--override-kv "${ARCH}.expert_used_count=int:${k}")
    docker rm -f anatomy >/dev/null 2>&1 || true
    docker run -d --name anatomy --gpus all -p ${PORT}:8080 -v llama_llama-cache:/root/.cache "${env_args[@]}" \
      --entrypoint /app/llama-server "$IMAGE" -m "$MODEL" --host 0.0.0.0 --port 8080 \
      -ngl 99 --flash-attn on -np 1 -c 8192 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --no-mmproj "${ov[@]}" ${NODES:+-lv 4} >/dev/null
    for i in $(seq 1 120); do curl -sf 127.0.0.1:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
    # ノード数は -lv 4 のときだけ出る(sched_reserve: graph: nodes = N, splits = M)。NODES=1 で取得する
    nodes=$(docker logs anatomy 2>&1 | grep -oE "graph: nodes = [0-9]+" | tail -1 | grep -oE "[0-9]+$" || echo "")
    splits=$(docker logs anatomy 2>&1 | grep -oE "splits = [0-9]+" | tail -1 | grep -oE "[0-9]+$" || echo "")
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    python3 scripts/decode_probe.py --url http://127.0.0.1:${PORT} --reps 5 --n 256 --out "$OUT" \
      --tag "model=${LABEL},build=${BUILD},k=${k},graphs=${g},graph_nodes=${nodes},graph_splits=${splits},vram_mib=${vram},power_limit=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)"
    docker rm -f anatomy >/dev/null
  done
done
