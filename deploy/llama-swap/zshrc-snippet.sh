# ---------------------------------------------------------------------------
# ローカル LLM (Qwen3.6-35B-A3B) を Claude Code のバックエンドにする
#   ~/.zshrc に読み込む:  source /path/to/zshrc-snippet.sh
#
# llama-swap は model 名を見て自動でモデルを入れ替える(常に 1 つだけ常駐)。
# ただし入れ替え中に /v1/messages を叩くと connection reset で切られるため、
# 先に /v1/chat/completions でウォームアップしてから claude を起動する。
# ---------------------------------------------------------------------------

export LLM_HOST="${LLM_HOST:-192.168.0.144}"
# モデルごとに専用ポート。シム側で model を固定しているため、
# サブエージェント(Explore 等)もそのポートのモデルに束ねられ、
# llama-swap がモデルを往復ロードしない = プレフィックスキャッシュが飛ばない。
export LLM_ENDPOINT_B="http://${LLM_HOST}:11436"   # UD-IQ4_XS
export LLM_ENDPOINT_C="http://${LLM_HOST}:11437"   # UD-IQ3_S
export LLM_ENDPOINT="$LLM_ENDPOINT_B"

# CLAUDE_CODE_MAX_CONTEXT_TOKENS:
#   Claude Code は未知のモデル名に対しコンテキスト長を 200k と仮定する。
#   実際は --ctx-size 131072 なので、教えないと超過して破綻する。

# 指定モデルをロードさせる。ロード完了までブロックする。
_llm_warm() {
  local model="$1" ep="$2"
  curl -s --max-time 1800 "${ep}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
    -o /dev/null -w '%{http_code}'
}

_llm_claude() {
  local model="$1" ep="$2"; shift 2
  printf '\033[2m%s をロード中...\033[0m ' "$model"
  local code
  code="$(_llm_warm "$model" "$ep")"
  if [ "$code" != "200" ]; then
    printf '\033[31m失敗 (http=%s)\033[0m\n' "$code"
    printf 'サーバを確認: curl -sS %s/v1/models\n' "$ep"
    return 1
  fi
  printf '\033[32mOK\033[0m\n'
  ANTHROPIC_BASE_URL="$ep" \
  ANTHROPIC_API_KEY=dummy \
  ANTHROPIC_AUTH_TOKEN=dummy \
  ANTHROPIC_MODEL="$model" \
  API_TIMEOUT_MS=1800000 \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 \
  claude "$@"
}

# B: UD-IQ4_XS / ncmoe=10。実タスクの総所要が最短(推奨)
ccb() { _llm_claude qwen36-35b-iq4xs "$LLM_ENDPOINT_B" "$@"; }
# C: UD-IQ3_S / オフロードなし。初回応答が速い(プレフィル 3.1 倍)
ccc() { _llm_claude qwen36-35b-iq3s "$LLM_ENDPOINT_C" "$@"; }

# いまロードされているモデルと選択肢を表示
ccstat() {
  curl -s --max-time 20 "${LLM_ENDPOINT_B}/v1/models" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print("サーバに到達できません"); raise SystemExit(1)
for m in d["data"]:
    st=m.get("status",{}).get("value","?")
    mark="●" if st=="loaded" else "○"
    mid=m["id"]
    print("  %s %-26s %s" % (mark, mid, st))
'
}
