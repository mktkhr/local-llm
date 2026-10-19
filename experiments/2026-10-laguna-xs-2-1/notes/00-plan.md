# 計画: Laguna XS 2.1(33B-A3B)を Claude Code の実タスクで Qwen3.6-35B-A3B と比較する

作成: 2026-10-19。

## 問い

1. Poolside Laguna XS 2.1(33B-A3B、agentic coding 特化、OpenMDW-1.1)は RTX 4080 SUPER 16 GB に Qwen3.6-35B-A3B と同じ条件(フル GPU・128k)で載るか。VRAM・decode・prefill はどう違うか
2. Claude Code の実タスク(M1 / M2)で、現行バックエンド(Qwen3.6 UD-IQ3_S、フル GPU)より速いか、手数は少ないか、tool_use は壊れないか
3. 置き換える価値があるか。Poolside 自身の表では Qwen3.6-35B-A3B が上(SWE-bench Verified 73.4 vs 70.9、Terminal-Bench 2.0 51.5 vs 37.5)。実タスクでそれが再現するか

## 前提(調査済み)

- llama.cpp は b10087 から Laguna に対応。llama-swap イメージは build 10920 なので対応済み(Phase 0 で実ロードを確認)
- アーキ: 40 層、SWA(512)30 層 + グローバル 10 層の 3:1、256 routed + 1 shared expert、ctx 262,144、思考あり(`enable_thinking` で切替)、tool 呼び出しは vLLM では `poolside_v1` パーサ。llama.cpp 側のネイティブパーサの有無は Phase 0 で確認(Generic フォールバックだと tool_use が壊れやすい)
- GGUF: 公式は BF16 / Q4_K_M(20.3 GB)のみ。bartowski に imatrix 量子化が一通り(IQ3_XXS 14.3 GB / IQ3_XS 15.6 GB / IQ4_XS 18.2 GB)。Qwen3.6 UD-IQ3_S(13.0 GiB)に最も近いのは IQ3_XXS

## 固定するもの

| 項目       | 値                                                                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 共通フラグ | 前実験と同じ(`--jinja -ngl 99 --flash-attn on -np 1 --metrics --temp 0`、`-ctk/-ctv q8_0`、`-b 2048 -ub 512`、`--cache-reuse 256`) |
| ctx        | 131072                                                                                                                             |
| GPU 電力   | 220 W                                                                                                                              |
| 比較対象   | `current-C`(Qwen3.6 UD-IQ3_S、フル GPU)。オフロード同士の比較は `current-B`(Qwen3.6 UD-IQ4_XS ncmoe 10)vs `L-q4km`(Q4_K_M ncmoe N) |
| 実タスク   | M1 / M2(`experiments/2026-10-speculative-decoding/scripts/data/cc_tasks.json`)、各 3 回、判定は機械的                              |

## 動かすもの

- モデル: `L-iq3xxs`(フル GPU 狙い)/ `L-q4km`(オフロード)/ 対照 `current-C` / `current-B`
- Phase 1 ワークロード: W1(長文 tool_use)/ W3(要約)/ W4(思考)。2 ブロック、各セル n=6
- Phase 2: M1 / M2 × 3 回 × {L-iq3xxs, current-C}

## 判定

- 置き換え候補になる条件: M1 + M2 の PASS 率が同等以上、合計所要時間が 15% 以上短い、tool_use 破損 0
- それ以外は「用途別の選択肢」または「見送り」として記録

## 成果物

- `config/model/*.args`(本実験のモデル定義。`MODEL_ARGS_DIR` で前実験の sweep.sh / cc_sweep.sh に渡す)、`scripts/`(薄いラッパ)、`results/*.csv`、README

## 実施時の変更(2026-10-19)

- Laguna の全層 GPU 上限は 64k(KV 約 26 KiB/token)。128k で比較する予定だったが Laguna を 65536、Qwen3.6 は現行どおり 131072 とした(「16 GB で現行どおり使えるか」が問いなので、同 ctx に揃えるより現実の制約を反映する)
- W1 プロンプトは 64k に収まるよう `--max-bytes 230000` で抽出し直した(前実験の W1 は 85〜109k で 400 になる)
- Phase 2 の `CLAUDE_CODE_MAX_CONTEXT_TOKENS` はサーバの `--ctx-size` から自動で渡す(cc_sweep.sh)
