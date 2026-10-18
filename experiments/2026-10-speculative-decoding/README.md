# 投機デコード: Qwen3.6-35B-A3B の MTP / ngram を 16 GB で測る(2026-10-speculative-decoding)

RTX 4080 SUPER (16 GB) 上の Qwen3.6-35B-A3B に対して、MTP ヘッド / n-gram / 外部 draft による投機デコードが decode を速くするか、Claude Code の実ワークロードで受理率はいくつか、`--n-cpu-moe` と prefix cache の下でも利得が残るかを測る。

**計画中。設計は [notes/00-plan.md](notes/00-plan.md)。**

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
