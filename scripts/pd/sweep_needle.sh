#!/usr/bin/env bash
# KV 量子化が長文からの情報抽出能力に与える影響を測る。
#
# 使い方:
#   ./sweep_needle.sh <url> <model> <kv_type> <backend>
#
# n_predict を 256 にしているのは、64 だと前置きの生成中に打ち切られ、
# 品質ではなく打ち切りで FAIL と判定されるため。
set -euo pipefail

URL="$1"; MODEL="$2"; KVTYPE="$3"; BACKEND="$4"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../results/needle.jsonl"
mkdir -p "$(dirname "$OUT")"

echo "=== $MODEL / KV=$KVTYPE / $BACKEND ===" >&2
# シードを 3 種に分けているのは、9 試行では KV 量子化の影響と
# 境界事例のばらつきを区別できなかったため。
for seed in 42 7 123; do
 for tag in 4k 8k 16k; do
  for pos in 10 50 90; do
    python3 "$HERE/needle_llamacpp.py" --url "$URL" \
      --needle "$HERE/data/needle/s${seed}_${tag}_p${pos}.json" \
      --model "$MODEL" --kv-type "$KVTYPE" --backend "$BACKEND" \
      --n-predict 256 --out "$OUT" || true
  done
 done
done
