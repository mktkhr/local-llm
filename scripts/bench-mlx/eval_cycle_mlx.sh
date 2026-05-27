#!/usr/bin/env bash
# 評価対象 MLX モデル群を「1 つずつ pull → 評価 → 削除」のサイクルで回す。
#
# `run_all_mlx.sh` は事前に全モデルを pull 済みであることを前提にするが、
# 本スクリプトはディスク容量制約がある環境向けに、サイクルでローテーションする。
#
# 引数でリポジトリ ID を渡せばそれだけを対象にする。省略時は DEFAULT_MODELS を使う。
# ログは results/eval_cycle-<timestamp>.log に追記。

set -uo pipefail

# Python stdout/stderr をラインバッファで吐かせる。
# パイプ (tee) 経由だと既定でブロックバッファになり、長時間プローブ中の
# [probe] / max_ctx= が見えなくなって進捗判別ができないため。
export PYTHONUNBUFFERED=1

# RTX 4080 SUPER (KV q8_0) との公平比較のため、Mac側も KV を 8bit 量子化する。
# 各ランナーの --kv-bits 引数で指定する。
KV_BITS=8

# ctx_search の探索上限。大型モデル(15GB+)で 262144 を要求すると 25〜35GB peak +
# swap を誘発してマシンごと落ちることがあるため、環境変数で下げられるようにする。
# 例: HIGH_CTX=131072 ./eval_cycle_mlx.sh ...(大型モデルの再開時に推奨)
HIGH_CTX="${HIGH_CTX:-262144}"

# 「Mac は大型モデルが載るが遅い、その原因」を語るための最小構成。
# Qwen3.5 dense のサイズ階段(4B→9B→27B)でアーキ固定の size→速度カーブを見せ、
# 32B distill で極端な遅さ、MoE(DSC-V2-Lite)で「大きくても速い」対照を置く。
# 増やす場合は後から引数で追加する方針。
DEFAULT_MODELS=(
  "mlx-community/Qwen3.5-4B-MLX-4bit"                       # 2.85 GB, 256K, RTX 直接比較
  "mlx-community/Qwen3.5-9B-MLX-4bit"                       # 5.57 GB, RTX 直接比較
  "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"  # 8.24 GB, MoE 対照
  "mlx-community/Qwen3.5-27B-4bit"                          # 14.98 GB, RTX 不可
  "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"         # 17.18 GB, 最大 dense
)

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

# ctx 依存スイープ(decode/prefill/TTFT の ctx カーブ)を取る対象。
# 4B で帯域律速 vs compute 律速の分離は十分鮮明に取れたため、27B は対象外。
# 大型モデルの max ctx 近傍 prefill は peak 25GB+ で swap 巻き添えのリスクが高い。
SWEEP_MODELS=(
  "mlx-community/Qwen3.5-4B-MLX-4bit"
)
# KV q4 対照(q8 主力に対する伸縮)を取る対象。
KV4_MODELS=(
  "mlx-community/Qwen3.5-9B-MLX-4bit"
)

in_list() {
  local needle="$1"; shift
  local x
  for x in "$@"; do [ "$x" = "$needle" ] && return 0; done
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG="results/eval_cycle-$TS.log"
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

# モデル ID から HF cache のディレクトリ名を作る (org/repo → models--org--repo)
hf_cache_dir() {
  local repo="$1"
  local sanitized="${repo//\//--}"
  echo "$HOME/.cache/huggingface/hub/models--${sanitized}"
}

free_disk_gb() {
  df -g . | awk 'NR==2 {print $4}'
}

echo "Started at $TS"
echo "Models: ${#MODELS[@]}"
echo "Free disk: $(free_disk_gb) GB"
for M in "${MODELS[@]}"; do echo "  - $M"; done

OK_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

for M in "${MODELS[@]}"; do
  SANITIZED=$(echo "$M" | tr ':' _ | tr '/' _)
  CACHE_DIR="$(hf_cache_dir "$M")"
  echo
  echo "==================================================================="
  echo "===== $M  (start $(date +%H:%M:%S), free $(free_disk_gb) GB)"
  echo "==================================================================="

  # ----------------- pull -----------------
  echo "[0/5] pull"
  if ! uv run python pull_models_mlx.py "$M"; then
    echo "  ERROR: pull failed, skip"
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILED_MODELS+=("$M:pull")
    continue
  fi
  echo "  free after pull: $(free_disk_gb) GB"

  # ----------------- ctx_search -----------------
  echo "[1/5] ctx_search (kv_bits=$KV_BITS)"
  if ! uv run python ctx_search_mlx.py --model "$M" --low 4096 --high "$HIGH_CTX" \
        --tolerance 4096 --safety-margin-mib 2048 --kv-bits "$KV_BITS"; then
    echo "  ERROR: ctx_search failed"
    FAILED_MODELS+=("$M:ctx_search")
  fi

  CTX_FILE=$(ls -t "results"/*/"ctx_search_${SANITIZED}.json" 2>/dev/null | head -1)
  if [ -z "$CTX_FILE" ] || [ ! -f "$CTX_FILE" ]; then
    echo "  ERROR: ctx_search result not found, cleanup & skip"
    rm -rf "$CACHE_DIR" 2>/dev/null
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILED_MODELS+=("$M:ctx_no_result")
    continue
  fi
  CTX=$(python3 -c "import json; print(json.load(open('$CTX_FILE'))['max_ctx'])")
  if [ "$CTX" = "0" ]; then
    echo "  ERROR: max_ctx=0 (probe failed at low bound), cleanup & skip"
    rm -rf "$CACHE_DIR" 2>/dev/null
    FAIL_COUNT=$((FAIL_COUNT+1))
    FAILED_MODELS+=("$M:max_ctx_zero")
    continue
  fi
  echo "  max_ctx=$CTX"

  # ----------------- speed -----------------
  echo
  echo "[2/5] speed @ ctx=$CTX kv_bits=$KV_BITS"
  uv run python run_speed_mlx.py --model "$M" --ctx "$CTX" --kv-bits "$KV_BITS" || \
    FAILED_MODELS+=("$M:speed")

  # ----------------- needle -----------------
  # num_predict は 2048。思考強制モデル(DSR1 系)が <think> を吐いてから needle 回答に
  # 届くだけの予算を確保する。非思考モデルは回答後 EOS で早期停止するので無害。
  echo
  echo "[3/5] needle @ depth=100% pos=50% kv_bits=$KV_BITS"
  NEEDLE="data/needle/${SANITIZED}_d100_p50.json"
  # KV ヘッダ等で末尾が削れることを見越して chars は ctx の 90% を狙う
  NEEDLE_CHARS=$(( CTX * 9 / 10 ))
  uv run python data/needle/generate.py --chars "$NEEDLE_CHARS" --position-pct 0.5 \
      --output "$NEEDLE"
  uv run python run_needle_mlx.py --model "$M" --ctx "$CTX" --needle "$NEEDLE" \
      --kv-bits "$KV_BITS" --num-predict 2048 || FAILED_MODELS+=("$M:needle")

  # ----------------- coding (難タスク tasks_hard) -----------------
  # num_predict 8192: 思考強制モデルが思考 + コードを両方収めるための予算。
  # 非思考モデル(Qwen / DSC-V2)は EOS で早期停止するので RTX(3072)と同一出力。
  echo
  echo "[4/5] coding_hard @ ctx=16384 kv_bits=$KV_BITS"
  uv run python run_coding_mlx.py --model "$M" --ctx 16384 --kv-bits "$KV_BITS" \
      --num-predict 8192 --tasks data/coding/tasks_hard.json || FAILED_MODELS+=("$M:coding")

  # ----------------- summary -----------------
  # num_predict 4096: 同上(RTX think=true 知見で思考は最低 4096 必要)
  echo
  echo "[5/5] summary @ ctx=16384 kv_bits=$KV_BITS"
  uv run python run_summary_mlx.py --model "$M" --ctx 16384 --kv-bits "$KV_BITS" \
      --num-predict 4096 --meetings data/summary/meetings.json || FAILED_MODELS+=("$M:summary")

  # ----------------- ctx 依存スイープ(対象モデルのみ) -----------------
  if in_list "$M" "${SWEEP_MODELS[@]}"; then
    echo
    echo "[6] ctx_sweep (kv_bits=$KV_BITS, max_ctx=$CTX)"
    uv run python run_ctx_sweep_mlx.py --model "$M" --kv-bits "$KV_BITS" \
        --ctx-points 4096,16384,65536 --max-ctx "$CTX" --num-predict 64 || \
      FAILED_MODELS+=("$M:ctx_sweep")
  fi

  # ----------------- KV q4 対照(対象モデルのみ) -----------------
  if in_list "$M" "${KV4_MODELS[@]}"; then
    echo
    echo "[7] KV q4 control: ctx_search + speed + needle"
    uv run python ctx_search_mlx.py --model "$M" --low 4096 --high "$HIGH_CTX" \
        --tolerance 4096 --safety-margin-mib 2048 --kv-bits 4 || \
      FAILED_MODELS+=("$M:ctx_search_q4")
    CTX4_FILE=$(ls -t "results"/*/"ctx_search_${SANITIZED}.json" 2>/dev/null | head -1)
    CTX4=$(python3 -c "import json; print(json.load(open('$CTX4_FILE'))['max_ctx'])" 2>/dev/null || echo 0)
    if [ "$CTX4" != "0" ]; then
      echo "  q4 max_ctx=$CTX4"
      uv run python run_speed_mlx.py --model "$M" --ctx "$CTX4" --kv-bits 4 || \
        FAILED_MODELS+=("$M:speed_q4")
      NEEDLE4="data/needle/${SANITIZED}_q4_d100_p50.json"
      uv run python data/needle/generate.py --chars "$(( CTX4 * 9 / 10 ))" \
          --position-pct 0.5 --output "$NEEDLE4"
      uv run python run_needle_mlx.py --model "$M" --ctx "$CTX4" --needle "$NEEDLE4" \
          --kv-bits 4 --num-predict 2048 || FAILED_MODELS+=("$M:needle_q4")
    fi
  fi

  # ----------------- cleanup -----------------
  echo
  echo "[cleanup] $CACHE_DIR"
  if [ -d "$CACHE_DIR" ]; then
    DU=$(du -sh "$CACHE_DIR" 2>/dev/null | awk '{print $1}')
    rm -rf "$CACHE_DIR"
    echo "  removed ${DU} (free now $(free_disk_gb) GB)"
  else
    echo "  cache dir not found, skip"
  fi

  echo "===== $M  (end $(date +%H:%M:%S))"
  OK_COUNT=$((OK_COUNT+1))
done

echo
echo "==================================================================="
echo "All done at $(date +%H:%M:%S)"
echo "  OK: $OK_COUNT"
echo "  Issues: ${#FAILED_MODELS[@]}"
for M in "${FAILED_MODELS[@]}"; do echo "    - $M"; done
echo "Log: $LOG"
