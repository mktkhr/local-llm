#!/usr/bin/env bash
# 対照用の評価テキストを Mac 側で作る。ja: このリポジトリの実験 README(日本語の技術文書)、en: llama.cpp の docs(英語)。
#   bash scripts/make_docs_eval.sh <llama.cpp のチェックアウト>
set -euo pipefail
cd "$(dirname "$0")/.."
LC="${1:?llama.cpp のチェックアウト}"; mkdir -p results/raw
cat ../*/README.md > results/raw/ja-eval.txt
( cd "$LC" && for f in $(git ls-tree -r --name-only d3c3e242c docs | grep '\.md$' | sort); do git show "d3c3e242c:$f"; echo; done ) > results/raw/en-eval.txt
wc -c results/raw/ja-eval.txt results/raw/en-eval.txt
