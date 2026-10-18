# 投機デコード: Qwen3.6-35B-A3B の MTP / ngram を 16 GB で測る(2026-10-speculative-decoding)

RTX 4080 SUPER (16 GB) 上の Qwen3.6-35B-A3B に対して、MTP ヘッド / n-gram / 外部 draft による投機デコードが decode を速くするか、Claude Code の実ワークロードで受理率はいくつか、`--n-cpu-moe` と prefix cache の下でも利得が残るかを測る。

**計測中(Phase 0 完了、Phase 1 準備中)。設計は [notes/00-plan.md](notes/00-plan.md)。**

| 項目       | 値                                                                        |
| ---------- | ------------------------------------------------------------------------- |
| 期間       | YYYY-MM-DD 〜 YYYY-MM-DD                                                  |
| 機材       | GPU / CPU / RAM / OS / Driver                                             |
| ランタイム | llama.cpp bNNNN(commit)/ Ollama vX.Y / mlx-lm X.Y                         |
| 方法論     | [docs/evaluation.md](../../docs/evaluation.md) との差分があればここに書く |
| 関連実験   | [experiments/...](../)                                                    |

## 結論

<結論だけ先に。数字は 1〜2 個、表は 1 つまで。詳細は下に>

## なぜそうなるか

<メカニズムの説明。所見が直感に反するならここで理由を示す>

## 計測条件

設計は [notes/00-plan.md](notes/00-plan.md)、ツールの実装は [notes/01-implementation-plan.md](notes/01-implementation-plan.md)。

### Phase 0(環境確認、2026-10-18)で確定したこと

| 項目                                    | 結果                                                                                                                                                                                                                                                                         |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| llama.cpp                               | `ghcr.io/mostlygeek/llama-swap:cuda` 同梱の **build 10920**(commit `eafe15a5e`)。`--spec-type draft-mtp / ngram-simple / ngram-mod` と `--spec-draft-n-max` が使える。deploy と同じイメージをダイジェストでピン                                                              |
| MTP ヘッドの VRAM コスト                | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_S` は **14,635 MiB**。通常版 IQ3_S(13,043 MiB)より **+1,592 MiB**。128k の KV(q8_0、1,360 MiB)より大きい                                                                                                                            |
| C'(主力 C 相当、フル GPU)               | IQ3_S は 128k / 96k / `--n-cpu-moe 2` のいずれも KV 1,360 MiB が確保できず OOM。**UD-IQ3_XXS に落として 128k フル GPU**(ロード後 15,882 MiB、56k prefill 中ピーク 15,914 MiB、推論中の増分 32 MiB)。基準の 15.5 GB は超えるが、KV と compute buffer はロード時確保のため採用 |
| 受理率の取得                            | `/metrics` に `llamacpp:spec_decode_num_draft_tokens_total` / `_num_accepted_tokens_total` / `_num_drafts_total` / `_num_accepted_tokens_per_pos_total{position}` が存在。**`/v1/messages` の応答に `timings` は無い**ため、tok/s も `/metrics` 差分で取る                   |
| スモーク(C' / mtp2 / W1-01、56k tokens) | decode **125 tok/s**、draft 52 / accepted 36(受理率 **0.69**、draft/generated 0.81)、status 200                                                                                                                                                                              |
| プロンプト                              | ccshim ログ 2,278 件から W1 × 3(224〜461 KB、13〜103 メッセージ)、W2 × 3(345〜425 KB)を抽出。`scripts/data/prompts_manifest.json` に形だけ記録(本体は commit しない)                                                                                                         |
| B'(対照、IQ4_XS + ncmoe)                | ncmoe は Phase 0 で再確定(下記)                                                                                                                                                                                                                                              |

## 結果

<表。生データは `results/` の csv / jsonl を指す>

## 落とし穴

| 罠  | 症状 | 対処 |
| --- | ---- | ---- |
|     |      |      |

## 次

<未検証・次に測るべきこと>

## ファイル

| パス       | 内容                                                              |
| ---------- | ----------------------------------------------------------------- |
| `notes/`   | 調査メモ・手順書・途中経過(複数可、番号 prefix で順序づけ)        |
| `scripts/` | この実験専用の計測スクリプト。共通 harness は `tools/` を使う     |
| `config/`  | この実験専用の llama-swap entry / compose / systemd override など |
| `results/` | 集計済みの csv / jsonl。生ログは `results/raw/`(gitignore)        |
