#!/usr/bin/env bash
# 投機デコードの設定(spec)× モデル構成(config)× ワークロードをブロック設計で掃引する。
#
# 1 ブロック = 「spec 設定ごとにサーバを起動し、全プロンプトをシャッフルして流す」。
# ブロック間で spec の順序を回転させ、ブロック内に必ず none(baseline)を含める。
# spec の切替は llama-server の起動オプションなのでコンテナごと入れ替える。
#
# 使い方(実験ディレクトリ experiments/2026-10-speculative-decoding で):
#   bash scripts/sweep.sh
#   BLOCKS=1 CONFIGS=C SPECS="none mtp2" WORKLOADS=W4 bash scripts/sweep.sh
#   CONFIGS=current-C SPECS=none BLOCKS=1 bash scripts/sweep.sh   # 比較対象 (a) 現行構成
#
# 環境変数:
#   BLOCKS          ブロック数(既定 3)
#   CONFIGS         config/model/<name>.args の名前(既定 "C B")
#   SPECS           config/spec/<name>.args の名前(既定は 7 設定)
#   WORKLOADS       W1 W2 W3 W4 のうち流すもの(既定 全部)
#   HEALTH_TIMEOUT  /health 待ちの上限秒(既定 2400。初回は GGUF 取得で 30 分超える)
#   DRY_RUN         1 なら docker / curl / python3 を echo に置き換える
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

BLOCKS="${BLOCKS:-3}"
CONFIGS="${CONFIGS:-C B}"
SPECS="${SPECS:-none mtp1 mtp2 mtp3 ngram-simple ngram-mod mtp2-ngram-simple}"
WORKLOADS="${WORKLOADS:-W1 W2 W3 W4}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
DRY_RUN="${DRY_RUN:-0}"
PORT="${SPEC_SERVER_PORT:-18081}"
URL="http://127.0.0.1:${PORT}"
COMPOSE="docker compose -f config/compose.yml"

# ワークロードごとの既定(00-plan.md Global Constraints)
n_predict_of() { case "$1" in W1) echo 256;; W2) echo 768;; W3) echo 512;; W4) echo 300;; *) echo 256;; esac; }
endpoint_of()  { case "$1" in W1|W2) echo messages;; *) echo chat;; esac; }
think_flag_of(){ case "$1" in W4) echo "--think";; *) echo "";; esac; }
# W1〜W3 は抽出物(commit しない)、W4 は手書き(commit 対象)
prompt_files_of() {
  case "$1" in
    W4) ls scripts/data/w4_prompts.json ;;
    *)  ls scripts/data/private/"$1"-*.json 2>/dev/null || true ;;
  esac
}

run() { if [ "$DRY_RUN" = "1" ]; then echo "$*"; else "$@"; fi; }

[ -f config/.env ] || { echo "config/.env がありません。config/.env.example をコピーしてください" >&2; exit 1; }
mkdir -p results/raw
TS="$(date +%Y%m%d-%H%M%S)"
OUT="results/raw/sweep-${TS}.jsonl"
LOG="results/raw/sweep-${TS}.log"
echo "=== sweep ${TS}: blocks=${BLOCKS} configs=[${CONFIGS}] specs=[${SPECS}] workloads=[${WORKLOADS}] ===" | tee -a "$LOG"

# W4 を含む展開(w4_prompts.json は 1 ファイルに 3 本入っているので個別ファイルに割る)
W4_DIR="results/raw/w4"
if [[ " $WORKLOADS " == *" W4 "* ]]; then
  mkdir -p "$W4_DIR"
  if [ "$DRY_RUN" != "1" ]; then
    python3 - "$W4_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for p in json.load(open("scripts/data/w4_prompts.json")):
    (d / f"{p['id']}.json").write_text(json.dumps(p, ensure_ascii=False))
PY
  fi
fi

# プロンプト一覧(W4 は展開先から)
all_prompts() {
  for w in $WORKLOADS; do
    if [ "$w" = "W4" ]; then ls "$W4_DIR"/W4-*.json 2>/dev/null || echo "$W4_DIR/W4-01.json $W4_DIR/W4-02.json $W4_DIR/W4-03.json" | tr ' ' '\n'
    else prompt_files_of "$w"; fi
  done
}

wait_health() {
  local waited=0
  while true; do
    if [ "$DRY_RUN" = "1" ]; then run curl -sf "$URL/health"; return 0; fi
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      echo "サーバが ${HEALTH_TIMEOUT} 秒以内に起動しませんでした" | tee -a "$LOG" >&2
      docker logs --tail 50 llama-spec 2>&1 | tee -a "$LOG" >&2 || true
      return 1
    fi
  done
}

rotate() { # rotate "<list>" <k> : 先頭 k 個を末尾へ
  local list=($1) k=$2 n; n=${#list[@]}; [ "$n" -eq 0 ] && return
  k=$((k % n)); echo "${list[@]:$k} ${list[@]:0:$k}"
}

for ((b = 1; b <= BLOCKS; b++)); do
  specs_b="$(rotate "$SPECS" $((b - 1)))"
  for config in $CONFIGS; do
    for spec in $specs_b; do
      [ -f "config/model/${config}.args" ] || { echo "config/model/${config}.args がありません" >&2; exit 1; }
      [ -f "config/spec/${spec}.args" ] || { echo "config/spec/${spec}.args がありません" >&2; exit 1; }
      LLAMA_ARGS="$(cat config/common.args "config/model/${config}.args" "config/spec/${spec}.args" | tr '\n' ' ')"
      echo "--- block=$b config=$config spec=$spec" | tee -a "$LOG"
      echo "    args: $LLAMA_ARGS" >> "$LOG"
      export LLAMA_ARGS
      run $COMPOSE up -d --force-recreate llama-spec
      wait_health || { run $COMPOSE down llama-spec; exit 1; }
      run curl -s "$URL/props" -o "results/raw/props-${config}-${spec}-${b}.json"

      # 捨てリクエスト(W1 の先頭)。プレフィックスを温め、W1 を全て warm で測る
      primer="$(prompt_files_of W1 | head -1)"
      if [ -n "$primer" ]; then
        run python3 scripts/spec_bench.py --url "$URL" --endpoint messages --prompt-file "$primer" \
          --n-predict 8 --out /dev/null --config "$config" --spec "$spec" --workload W1 \
          --prompt-id primer --block "$b" --nvidia-smi 0 || true
      fi

      # ブロック番号を seed にした決定的シャッフル(macOS にも shuf が無いので python で)
  for f in $(all_prompts | python3 -c 'import random,sys; l=sys.stdin.read().split(); random.Random(int(sys.argv[1])).shuffle(l); print("\n".join(l))' "$b"); do
        pid="$(basename "$f" .json)"; w="${pid%%-*}"
        run python3 scripts/spec_bench.py --url "$URL" --endpoint "$(endpoint_of "$w")" --prompt-file "$f" \
          --n-predict "$(n_predict_of "$w")" --out "$OUT" --config "$config" --spec "$spec" \
          --workload "$w" --prompt-id "$pid" --block "$b" $(think_flag_of "$w") 2>&1 | tee -a "$LOG" || true
      done
      run $COMPOSE down llama-spec
    done
  done
done
echo "=== done: $OUT ===" | tee -a "$LOG"
