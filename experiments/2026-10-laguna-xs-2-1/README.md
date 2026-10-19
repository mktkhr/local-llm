# Laguna XS 2.1 vs Qwen3.6-35B-A3B: Claude Code バックエンドの置き換え候補か(2026-10-laguna-xs-2-1)

Poolside Laguna XS 2.1(33B-A3B、agentic coding 特化)を RTX 4080 SUPER 16 GB に Qwen3.6-35B-A3B と同じ条件で載せ、速度・VRAM と Claude Code の実タスク(M1 / M2)で比較する。

**計測中。設計は [notes/00-plan.md](notes/00-plan.md)。**

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

<固定したもの / 動かしたもの。コマンドや config は `config/` `scripts/` にあるものをリンクで指す>

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
