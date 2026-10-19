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
#   CONFIGS         config/model/<name>.args の名前(既定 "C-iq3xxs B"。IQ3_S は MTP ヘッド分で 16 GB に収まらない。00-plan.md §4)
#   SPECS           config/spec/<name>.args の名前(既定は 7 設定)
#   WORKLOADS       W1 W2 W3 W4 のうち流すもの(既定 全部)
#   HEALTH_TIMEOUT  /health 待ちの上限秒(既定 2400。初回は GGUF 取得で 30 分超える)
#   PRIVATE_DIR     W1〜W3 の抽出プロンプトの置き場(既定 scripts/data/private。別実験で別のサイズ上限で抽出したものを使うとき)
#   MODEL_ARGS_DIR  model args の置き場(既定 config/model。別実験のモデル定義を使うときに絶対パスで指定)
#   DRY_RUN         1 なら docker / curl / python3 を echo に置き換える
#   RESUME_TS       途中で止まった掃引の TS(例 20261018-115236)。同じ JSONL に追記して再開する
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

BLOCKS="${BLOCKS:-3}"
CONFIGS="${CONFIGS:-C-iq3xxs B}"
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
    *)  ls "${PRIVATE_DIR:-scripts/data/private}"/"$1"-*.json 2>/dev/null || true ;;
  esac
}

run() { if [ "$DRY_RUN" = "1" ]; then echo "$*"; else "$@"; fi; }
shuffle_by() { python3 -c 'import random,sys; l=sys.stdin.read().split(); random.Random(int(sys.argv[1])).shuffle(l); print("\n".join(l))' "$1"; }

[ -f config/.env ] || { echo "config/.env がありません。config/.env.example をコピーしてください" >&2; exit 1; }
for config in $CONFIGS; do [ -f "${MODEL_ARGS_DIR:-config/model}/${config}.args" ] || { echo "${MODEL_ARGS_DIR:-config/model}/${config}.args がありません" >&2; exit 1; }; done
for spec in $SPECS; do [ -f "config/spec/${spec}.args" ] || { echo "config/spec/${spec}.args がありません" >&2; exit 1; }; done
# 本番の llama-swap など、別プロセスが VRAM を掴んだままだと OOM か計測汚染になる(00-plan.md Phase 0 手順 6)
if [ "$DRY_RUN" != "1" ]; then
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$used" ] && [ "$used" -gt "${VRAM_IDLE_MAX_MIB:-1000}" ]; then
    echo "GPU が使用中です(memory.used=${used} MiB)。本番スタックを止めてから実行してください: docker compose -f ../../deploy/llama-swap/compose.yml stop" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    exit 1
  fi
fi
trap 'echo "中断: llama-spec を停止します" >&2; $COMPOSE down llama-spec >/dev/null 2>&1 || true' INT TERM
mkdir -p results/raw
# RESUME_TS=<既存の TS> を与えると、その掃引の JSONL / LOG に追記して再開する(途中で止まった掃引の続き)。
# aggregate は sweep=ファイル名で対付けするので、同じファイルに書かないと baseline と対にならない
TS="${RESUME_TS:-$(date +%Y%m%d-%H%M%S)}"
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

# 戻り値: 0 = healthy / 2 = 起動時に失敗(その設定を飛ばす)/ 1 = タイムアウト(掃引を止める)
wait_health() {
  local waited=0
  while true; do
    if [ "$DRY_RUN" = "1" ]; then run curl -sf "$URL/health"; return 0; fi
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    # コンテナが落ちた、または ggml_abort / CUDA error で止まったまま(abort 後にプロセスが残り Running のままになる)
    if [ "$(docker inspect -f '{{.State.Running}}' llama-spec 2>/dev/null)" != "true" ] \
       || docker logs llama-spec 2>&1 | grep -qE "ggml_abort|CUDA error|exiting due to model loading error|failed to allocate"; then
      echo "llama-spec が起動中に失敗しました(${waited}s)" | tee -a "$LOG" >&2
      docker logs --tail 30 llama-spec 2>&1 | tee -a "$LOG" >&2 || true
      return 2
    fi
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
      [ -f "${MODEL_ARGS_DIR:-config/model}/${config}.args" ] || { echo "${MODEL_ARGS_DIR:-config/model}/${config}.args がありません" >&2; exit 1; }
      [ -f "config/spec/${spec}.args" ] || { echo "config/spec/${spec}.args がありません" >&2; exit 1; }
      LLAMA_ARGS="$(cat config/common.args "${MODEL_ARGS_DIR:-config/model}/${config}.args" "config/spec/${spec}.args" | tr '\n' ' ')"
      echo "--- block=$b config=$config spec=$spec" | tee -a "$LOG"
      echo "    args: $LLAMA_ARGS" >> "$LOG"
      export LLAMA_ARGS
      run $COMPOSE up -d --force-recreate llama-spec
      set +e; wait_health; hc=$?; set -e
      if [ "$hc" -eq 2 ]; then
        echo "${config} ${spec} block=${b}: 起動失敗のため飛ばします" | tee -a "$LOG" "results/raw/sweep-${TS}-skipped.txt" >&2
        run $COMPOSE down llama-spec; continue
      elif [ "$hc" -ne 0 ]; then
        run $COMPOSE down llama-spec; exit 1
      fi
      run curl -s "$URL/props" -o "results/raw/props-${TS}-${config}-${spec}-${b}.json"

      # 捨てリクエスト(W1 の先頭)。cold の TTFT として primer-<TS>.jsonl に残す(sweep-*.jsonl には混ぜない)
      primer=""; [[ " $WORKLOADS " == *" W1 "* ]] && primer="$(prompt_files_of W1 | head -1)"
      if [ -n "$primer" ]; then
        run python3 scripts/spec_bench.py --url "$URL" --endpoint messages --prompt-file "$primer" \
          --n-predict 8 --out "results/raw/primer-${TS}.jsonl" --config "$config" --spec "$spec" --workload W1 \
          --prompt-id primer --block "$b" --nvidia-smi 0 || true
      fi

      # W1 は primer 直後に連続して流し、全て warm prefix で測る。その後に W2〜W4。
      # 各群の中だけをブロック番号 seed で決定的にシャッフルする(macOS に shuf が無いので python で)
      w1_files=""; [[ " $WORKLOADS " == *" W1 "* ]] && w1_files="$(prompt_files_of W1)"
      for f in $( { echo "$w1_files" | grep -v "^$" | shuffle_by "$b"; all_prompts | grep -v "/W1-" | shuffle_by "$b"; } ); do
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
