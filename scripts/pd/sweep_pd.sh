#!/usr/bin/env bash
# PD分離の構成を複数のプロンプト長で掃く。結果は results/pd_runs.jsonl に追記される。
#
# 使い方:
#   ./sweep_pd.sh <label> <prefill_url> <decode_url> <kv_url> <decode_kv_dir> <tokens...>
#
# 環境変数:
#   CONNS  KV 取得の並列接続数(既定 8)。1 にすると単一接続。
#          配信側が Range に対応している必要がある。
#   MODEL  モデル識別子(必須)。集計時の突き合わせ鍵。
set -euo pipefail

LABEL="$1"; PREFILL_URL="$2"; DECODE_URL="$3"; KV_URL="$4"; DECODE_KV_DIR="$5"
shift 5

CONNS="${CONNS:-8}"
MODEL="${MODEL:?MODEL を指定してください}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../results/pd_runs.jsonl"
mkdir -p "$(dirname "$OUT")"

for t in "$@"; do
  echo "=== $LABEL / $t tokens ===" >&2
  python3 "$HERE/pd_run.py" \
    --prefill-url "$PREFILL_URL" \
    --decode-url  "$DECODE_URL" \
    --kv-url      "$KV_URL" \
    --decode-kv-dir "$DECODE_KV_DIR" \
    --tokens "$t" --n-predict 32 --conns "$CONNS" \
    --label "$LABEL" --model "$MODEL" --out "$OUT" > /dev/null
done
echo "wrote: $OUT" >&2
