#!/usr/bin/env bash
# cc_task.sh が残した diff.patch を元に、判定だけをやり直す(判定コマンドの不備で全滅した場合など)。
# 対象リポジトリの c2181d0 に patch を当てて judge を実行し、summary.json の pass を更新する。
#
# 使い方(実験ディレクトリ、GPU 機で): bash scripts/cc_rejudge.sh [results/raw/cc/<label>-<task> ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
export PATH="$HOME/.local/bin:$HOME/.asdf/shims:$PATH"
export ASDF_GOLANG_VERSION="${GO_VERSION:-$(asdf list golang 2>/dev/null | tr -d ' *' | tail -1)}"
RAG_REPO="${RAG_REPO:-$HOME/ghq/github.com/mktkhr/rag}"; RAG_REV="${RAG_REV:-c2181d0}"
DIRS=("$@"); [ ${#DIRS[@]} -gt 0 ] || DIRS=(results/raw/cc/*/)
for d in "${DIRS[@]}"; do
  d="${d%/}"; [ -f "$d/summary.json" ] || continue
  task="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$d/summary.json")"
  JUDGE="$(python3 -c 'import json,sys; t=[x for x in json.load(open(sys.argv[1])) if x["id"]==sys.argv[2]]; print(t[0]["judge"])' scripts/data/cc_tasks.json "$task")"
  WT="$(pwd)/$d/rejudge-wt"
  git -C "$RAG_REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true
  git -C "$RAG_REPO" worktree add --detach "$WT" "$RAG_REV" >/dev/null 2>&1
  PASS=false; JEXIT=
  # 旧い diff.patch にバイナリ(go build の成果物)が混ざっていると apply できないので、そのセクションを除く
  python3 - "$d/diff.patch" "$d/diff.nobin.patch" <<'PY'
import re, sys
src, dst = sys.argv[1:]
try:
    text = open(src).read()
except FileNotFoundError:
    text = ""
sections = re.split(r"(?m)^(?=diff --git )", text)
keep = [sec for sec in sections if sec and "Binary files" not in sec and "GIT binary patch" not in sec]
open(dst, "w").write("".join(keep))
PY
  if [ -s "$d/diff.nobin.patch" ] && git -C "$WT" apply "$(pwd)/$d/diff.nobin.patch" 2> "$d/rejudge-apply.log"; then
    set +e; ( cd "$WT" && bash -c "$JUDGE" ) > "$d/judge.log" 2>&1; JEXIT=$?; set -e
    [ "$JEXIT" -eq 0 ] && PASS=true
  else
    echo "patch 適用失敗または変更なし" > "$d/judge.log"
  fi
  python3 - "$d/summary.json" "$PASS" "${JEXIT:-}" <<'PY'
import json, sys
p, ok, jexit = sys.argv[1:]
s = json.load(open(p)); s["pass"] = ok == "true"; s["judge_exit"] = int(jexit) if jexit else None; s["rejudged"] = True
json.dump(s, open(p, "w"), ensure_ascii=False, indent=2)
PY
  git -C "$RAG_REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true
  echo "$d: pass=$PASS judge_exit=${JEXIT:-n/a}"
done
