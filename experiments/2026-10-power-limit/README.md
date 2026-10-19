# GPU 電力リミットと推論速度・電力効率(2026-10-power-limit)

RTX 4080 SUPER の電力リミット(150 / 220 / 320 W)を変えたとき、Qwen3.6-35B-A3B(IQ3_XXS、フル GPU)の prefill / decode 速度と tok/s/W、そして MTP 投機デコードの利得と電力差がどう変わるかを測る。[2026-10-speculative-decoding](../2026-10-speculative-decoding/README.md) が 220 W 制限下で得た所見の追試。

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
