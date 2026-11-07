#!/usr/bin/env bash
# Qwen3.8-Flash-Next を llama-swap 経由で回し、構成 × プロンプト × 反復の速度を JSONL に残す。
#
# GPU 機で実行する(nvidia-smi と vmstat を採る):
#   bash scripts/bench.sh                                   # 既定: 3 構成 × (W4 3 本 + 長文 3 本) × 2 反復
#   CONFIGS="qwen38-flash-next-iq3xxs" REPEATS=1 bash scripts/bench.sh
#
# 1 リクエストごとに experiments/2026-10-speculative-decoding/scripts/spec_bench.py を使う(/metrics 差分で
# prompt/decode の tok/s を取る)。llama-swap の /upstream/<model> を基底 URL にすると llama-server の
# /metrics と /v1/chat/completions がそのまま通る。構成の切替は llama-swap に任せる(前のモデルは自動でアンロード)。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
SPEC_BENCH="${SPEC_BENCH:-$ROOT/../2026-10-speculative-decoding/scripts/spec_bench.py}"
[ -f "$SPEC_BENCH" ] || SPEC_BENCH="$HERE/spec_bench.py"
SWAP="${SWAP:-http://127.0.0.1:11435}"
CONFIGS="${CONFIGS:-qwen38-flash-next-iq3xxs qwen38-flash-next-iq3xxs-cache12g qwen38-flash-next-iq3xxs-nocache}"
REPEATS="${REPEATS:-2}"
PROMPT_FILES="${PROMPT_FILES:-scripts/data/w4_prompts.json scripts/data/prompts_long.json}"
ONLY_IDS="${ONLY_IDS:-.*}"   # prompt id の正規表現(例: "W4-.*|L1-.*")。長文は prefill が遅いと 1 本で数十分かかる
N_PREDICT="${N_PREDICT:-300}"
TS="${RESUME_TS:-$(date +%Y%m%d-%H%M%S)}"
OUT="results/raw/bench-${TS}.jsonl"; mkdir -p results/raw
LOADLOG="results/raw/load-${TS}.log"

ids_of() { python3 -c 'import json,sys; print(" ".join(p["id"] for p in json.load(open(sys.argv[1]))))' "$1"; }
one() {  # one <prompt-file> <id>  → 単一プロンプトの一時ファイルを作って spec_bench に渡す
  python3 - "$1" "$2" <<'PY'
import json,sys,pathlib
ps=[p for p in json.load(open(sys.argv[1])) if p["id"]==sys.argv[2]]
# spec_bench.py は 1 ファイル 1 プロンプト(dict)を期待する
tmp=pathlib.Path("results/raw/_one.json"); tmp.write_text(json.dumps(ps[0],ensure_ascii=False)); print(tmp)
PY
}

for config in $CONFIGS; do
  url="$SWAP/upstream/$config"
  echo "== $config: load $(date +%H:%M:%S)" | tee -a "$LOADLOG"
  # 前のモデルを明示的に外してから読み込み時間を測る(ページキャッシュは温まったまま)
  curl -s -X POST "$SWAP/unload" -o /dev/null || true
  t0=$(date +%s)
  until curl -sf "$url/health" >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 2400 ] && { echo "load timeout" | tee -a "$LOADLOG"; break; }; done
  echo "   loaded in $(( $(date +%s) - t0 )) s; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB; $(free -m | awk '/Mem:/{print "ram_used="$3"MiB cache="$6"MiB"}')" | tee -a "$LOADLOG"
  { docker logs llama-swap --since 40m 2>&1 | grep -iE "moe|cache|lazy|offload|CPU buffer|CUDA0 buffer|KV self size|compute buffer" || true; } | tail -12 | sed 's/^/   /' | tee -a "$LOADLOG"
  for block in $(seq 1 "$REPEATS"); do
    for pf in $PROMPT_FILES; do
      wl=$(basename "$pf" .json)
      for id in $(ids_of "$pf"); do
        echo "$id" | grep -qE "^(${ONLY_IDS})$" || continue
        tmp=$(one "$pf" "$id")
        # 長文は要約なので 300 tok、W4 は思考込みで 300 tok(元実験と同じ)
        vm_before=$(awk '/^sda /{print $6}' /proc/diskstats)
        python3 "$SPEC_BENCH" --url "$url" --endpoint chat --prompt-file "$tmp" --n-predict "$N_PREDICT" \
          --out "$OUT" --config "$config" --spec none --workload "$wl" --prompt-id "$id" --block "$block" \
          --timeout 3600 --nvidia-smi 1 >/dev/null || echo "   FAIL $config $id"
        vm_after=$(awk '/^sda /{print $6}' /proc/diskstats)
        # 直前の行に SATA SSD の読み出しセクタ数(512 B)を足す
        python3 - "$OUT" "$vm_before" "$vm_after" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])
if not p.exists(): sys.exit(0)
lines=p.read_text().splitlines()
r=json.loads(lines[-1]); r["sda_read_mib"]=round((int(sys.argv[3])-int(sys.argv[2]))*512/2**20,1)
lines[-1]=json.dumps(r,ensure_ascii=False); p.write_text("\n".join(lines)+"\n")
PY
        [ -f "$OUT" ] && tail -1 "$OUT" | python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); m=r.get("metrics",{}); print(f"   {r[\"config\"]} {r[\"prompt_id\"]} b{r[\"block\"]}: prompt {m.get(\"prefill_tok_s\")} tok/s, decode {m.get(\"decode_tok_s\")} tok/s, wall {r.get(\"wall_s\")} s, sda_read {r.get(\"sda_read_mib\")} MiB, vram {r.get(\"gpu\",{}).get(\"vram_peak_mib\")}")'
      done
    done
  done
done
echo "== done $(date +%H:%M:%S) → $OUT"
