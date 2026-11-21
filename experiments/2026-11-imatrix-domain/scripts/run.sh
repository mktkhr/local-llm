#!/usr/bin/env bash
# Qwen3.5-9B Q8_0 を元に、imatrix なし / unsloth 配布 / Claude Code ログ由来の 3 通りで IQ3_S と Q4_K_M を作り、
# Claude Code の評価用会話とリポジトリの文書で Q8_0 に対する KL を測る(GPU 機で実行。段ごとに出力があれば飛ばす)。
#   bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=results/raw; mkdir -p "$R" models
IMG=llama-swap:b12464
SNAP=/root/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5
U=$(id -u); G=$(id -g)
CHUNKS="${CHUNKS:-100}"; EVAL_CHUNKS="${EVAL_CHUNKS:-32}"

# 0) 元モデルをボリュームから取り出す(root 所有のため)
[ -f models/q8_0.gguf ] || docker run --rm -v llama_llama-cache:/root/.cache:ro -v "$PWD/models:/m" alpine \
  sh -c "cp $SNAP/Qwen3.5-9B-Q8_0.gguf /m/q8_0.gguf && chown $U:$G /m/q8_0.gguf"

# 1) コーパス(ccshim の 3 ボリューム、読み取り専用)
[ -f $R/cc-calib.txt ] || docker run --rm -v llama_ccshim-log:/l1:ro -v llama_ccshim-log-c:/l2:ro -v llama_ccshim-log-d:/l3:ro \
  -v "$PWD:/w" -w /w -u $U:$G python:3.12-slim python3 scripts/make_corpus.py --logs /l1 /l2 /l3 --out-dir $R

# 2) imatrix(CPU 版をソースからビルドしたもの)
[ -f $R/imatrix-cc.gguf ] || ./src/build/bin/llama-imatrix -m models/q8_0.gguf -f $R/cc-calib.txt -o $R/imatrix-cc.gguf \
  -c 512 --chunks "$CHUNKS" -t 6 > $R/imatrix-cc.log 2>&1
[ -f $R/imatrix-unsloth.gguf ] || curl -sfL -o $R/imatrix-unsloth.gguf \
  https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/imatrix_unsloth.gguf_file

# 3) 量子化(CPU)
for t in IQ3_S Q4_K_M; do
  for m in none unsloth cc; do
    out=models/${t,,}-$m.gguf
    [ -f "$out" ] && continue
    im=(); [ "$m" != none ] && im=(--imatrix /w/$R/imatrix-$m.gguf)
    docker run --rm -v "$PWD:/w" -u $U:$G --entrypoint /app/llama $IMG quantize --allow-requantize "${im[@]}" \
      /w/models/q8_0.gguf /w/$out $t 6 2>&1 | tail -2
  done
done

# 4) KL(GPU)。基準は Q8_0 の logits
for e in cc ja en; do
  base=$R/base-$e.kld
  [ -f $base ] || docker run --rm --gpus all -v "$PWD:/w" -u $U:$G --entrypoint /app/llama $IMG perplexity \
    -m /w/models/q8_0.gguf -f /w/$R/$e-eval.txt -c 2048 --chunks "$EVAL_CHUNKS" -ngl 99 --kl-divergence-base /w/$base 2>&1 | tail -3
  for q in models/iq3_s-*.gguf models/q4_k_m-*.gguf; do
    n=$(basename "$q" .gguf)
    [ -f $R/kld-$e-$n.log ] && continue
    docker run --rm --gpus all -v "$PWD:/w" -u $U:$G --entrypoint /app/llama $IMG perplexity \
      -m /w/$q -c 2048 --chunks "$EVAL_CHUNKS" -ngl 99 --kl-divergence-base /w/$base --kl-divergence > $R/kld-$e-$n.log 2>&1
    grep -E "Mean    KLD|Same top p" $R/kld-$e-$n.log | sed "s|^|$e $n |"
  done
done
echo "== done $(date +%T)"
