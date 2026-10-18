#!/usr/bin/env bash
# Claude Code の実タスク(M1 / M2)を無人で実行し、機械的に合否判定する(Phase 2)。
#
# GPU 機で実行する前提(claude は ~/.local/bin、go / node は ~/.asdf/shims)。
# 事前に config/compose.yml の llama-spec と ccshim-spec を、測りたい spec 設定で起動しておく:
#   LLAMA_ARGS="$(cat config/common.args config/model/C.args config/spec/mtp2.args | tr '\n' ' ')" \
#     docker compose -f config/compose.yml up -d
#
# 使い方(実験ディレクトリで):
#   bash scripts/cc_task.sh M1 C-mtp2-run1
#
# 出力: results/raw/cc/<label>-<task>/{claude.json,before.txt,after.txt,diff.txt,cc_metrics.json,summary.json}
#
# 環境変数:
#   RAG_REPO    対象リポジトリ(既定 ~/ghq/github.com/mktkhr/rag)
#   RAG_REV     対象コミット(既定 c2181d0)
#   SHIM_URL    実験用 ccshim(既定 http://127.0.0.1:11439)
#   SERVER_URL  実験用 llama-server(既定 http://127.0.0.1:18081、/metrics 用)
#   TASK_TIMEOUT  秒(既定 900)
#   DRY_RUN     1 なら claude / git worktree / judge を echo に置き換える
set -euo pipefail

TASK="${1:?M1 または M2}"; LABEL="${2:?ラベル(例 C-mtp2-run1)}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.asdf/shims:$PATH"
# asdf の shim は .tool-versions が無いディレクトリ(作業用 worktree)で "No version is set for command go" になり
# 判定が全滅する。インストール済みの最新版を明示する(環境変数 GO_VERSION で上書き可)
export ASDF_GOLANG_VERSION="${GO_VERSION:-$(asdf list golang 2>/dev/null | tr -d ' *' | tail -1)}"

RAG_REPO="${RAG_REPO:-$HOME/ghq/github.com/mktkhr/rag}"
RAG_REV="${RAG_REV:-c2181d0}"
SHIM_URL="${SHIM_URL:-http://127.0.0.1:11439}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:18081}"
TASK_TIMEOUT="${TASK_TIMEOUT:-900}"
DRY_RUN="${DRY_RUN:-0}"
LOGDIR="results/raw/ccshim-log"

run() { if [ "$DRY_RUN" = "1" ]; then echo "$*"; else "$@"; fi; }

PROMPT="$(python3 -c 'import json,sys; t=[x for x in json.load(open(sys.argv[1])) if x["id"]==sys.argv[2]]; print(t[0]["prompt"] if t else "")' scripts/data/cc_tasks.json "$TASK")"
JUDGE="$(python3 -c 'import json,sys; t=[x for x in json.load(open(sys.argv[1])) if x["id"]==sys.argv[2]]; print(t[0]["judge"] if t else "")' scripts/data/cc_tasks.json "$TASK")"
[ -n "$PROMPT" ] || { echo "cc_tasks.json に $TASK がありません" >&2; exit 2; }

OUT="results/raw/cc/${LABEL}-${TASK}"; mkdir -p "$OUT"
WT="$(pwd)/$OUT/wt"

trace_lines() { [ -f "$LOGDIR/trace.jsonl" ] && wc -l < "$LOGDIR/trace.jsonl" || echo 0; }
trace_last_idx() { # trace.jsonl の末尾行の idx(無ければ 0)
  [ -f "$LOGDIR/trace.jsonl" ] && tail -1 "$LOGDIR/trace.jsonl" | python3 -c 'import json,sys; l=sys.stdin.read().strip(); print(json.loads(l).get("idx",0) if l else 0)' || echo 0
}

echo "=== $TASK / $LABEL ==="
run git -C "$RAG_REPO" worktree add --detach "$WT" "$RAG_REV"
# 途中で失敗しても worktree を残さない(同じラベルで再実行できるように)
trap 'git -C "$RAG_REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
# ccshim は再起動すると idx を 1 から振り直すので、範囲は trace.jsonl の行番号で持つ
LINE_FROM=$(( $(trace_lines) + 1 ))
IDX_FROM=$(( $(trace_last_idx) + 1 ))
run curl -s "$SERVER_URL/metrics" -o "$OUT/before.txt"

T0=$(date +%s)
set +e
if [ "$DRY_RUN" = "1" ]; then
  echo "claude -p <prompt:$TASK> --dangerously-skip-permissions --output-format json  (cwd=$WT, timeout=$TASK_TIMEOUT)"
  EXIT=0
else
  ( cd "$WT" && timeout "$TASK_TIMEOUT" env \
      ANTHROPIC_BASE_URL="$SHIM_URL" ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN=dummy \
      ANTHROPIC_MODEL=spec-bench API_TIMEOUT_MS=1800000 CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 \
      claude -p "$PROMPT" --dangerously-skip-permissions --output-format json > "$ROOT/$OUT/claude.json" 2> "$ROOT/$OUT/claude.err" )
  EXIT=$?
fi
set -e
WALL=$(( $(date +%s) - T0 ))
run curl -s "$SERVER_URL/metrics" -o "$OUT/after.txt"
IDX_TO=$(trace_last_idx)
LINE_TO=$(trace_lines)
# この範囲の resp.bin は次に ccshim が再起動すると上書きされるので、いま退避する
if [ "$DRY_RUN" != "1" ] && [ -f "$LOGDIR/trace.jsonl" ]; then
  mkdir -p "$OUT/ccshim"
  sed -n "${LINE_FROM},${LINE_TO}p" "$LOGDIR/trace.jsonl" > "$OUT/ccshim/trace.jsonl"
  python3 -c 'import json,sys; [print("%03d" % json.loads(l)["idx"]) for l in open(sys.argv[1]) if l.strip()]' "$OUT/ccshim/trace.jsonl" \
    | while read -r n; do for f in "$LOGDIR/$n-req.json" "$LOGDIR/$n-resp.bin"; do [ -f "$f" ] && cp "$f" "$OUT/ccshim/"; done; done
fi

# 判定: 変更 0 件 / timeout(124) は FAIL。それ以外は judge コマンドで決める
CHANGED=0; PASS=false; JUDGE_EXIT=
if [ "$DRY_RUN" = "1" ]; then
  echo "judge: $JUDGE  (cwd=$WT)"
else
  git -C "$WT" add -A >/dev/null 2>&1 || true
  git -C "$WT" diff --cached --stat > "$OUT/diff.txt" || true
  git -C "$WT" diff --cached > "$OUT/diff.patch" || true
  CHANGED=$(git -C "$WT" diff --cached --name-only | wc -l | tr -d ' ')
  if [ "$EXIT" -ne 124 ] && [ "$CHANGED" -gt 0 ]; then
    set +e; ( cd "$WT" && bash -c "$JUDGE" ) > "$OUT/judge.log" 2>&1; JUDGE_EXIT=$?; set -e
    [ "$JUDGE_EXIT" -eq 0 ] && PASS=true
  fi
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "python3 scripts/cc_metrics.py --logdir $LOGDIR --line-from $LINE_FROM --line-to $LINE_TO --out $OUT/cc_metrics.json"
else
  set +e
  python3 scripts/cc_metrics.py --logdir "$LOGDIR" --line-from "$LINE_FROM" --line-to "$LINE_TO" \
    --metrics-before "$OUT/before.txt" --metrics-after "$OUT/after.txt" --out "$OUT/cc_metrics.json" > /dev/null
  [ $? -eq 0 ] || echo "警告: cc_metrics が範囲内のリクエストを見つけられませんでした(trace 行 ${LINE_FROM}-${LINE_TO})" >&2
  set -e
fi

NUM_TURNS="$( [ -f "$OUT/claude.json" ] && python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("num_turns",""))
except Exception: print("")' "$OUT/claude.json" || echo "")"

python3 - "$OUT/summary.json" "$TASK" "$LABEL" "$PASS" "$WALL" "$EXIT" "${NUM_TURNS}" "$CHANGED" "$IDX_FROM" "$IDX_TO" "$LINE_FROM" "$LINE_TO" "$OUT/cc_metrics.json" <<'PY'
import json, pathlib, sys
out, task, label, ok, wall, exit_code, turns, changed, i0, i1, l0, l1, ccm = sys.argv[1:]
cc = json.load(open(ccm)) if pathlib.Path(ccm).exists() else None
pathlib.Path(out).write_text(json.dumps({
    "task": task, "label": label, "pass": ok == "true", "wall_s": int(wall), "exit_code": int(exit_code),
    "num_turns": int(turns) if turns else None, "changed_files": int(changed),
    "trace_idx": [int(i0), int(i1)], "trace_lines": [int(l0), int(l1)], "cc_metrics": cc}, ensure_ascii=False, indent=2) + "\n")
print(f"{task} {label}: pass={ok} wall={wall}s exit={exit_code} turns={turns or '?'} changed={changed}")
PY

run git -C "$RAG_REPO" worktree remove --force "$WT"
trap - EXIT
