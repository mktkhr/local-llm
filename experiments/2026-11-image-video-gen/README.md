# ローカル画像・動画生成: RTX 4080 SUPER 16 GB で最高品質を狙う(2026-11-image-video-gen)

16 GB + RAM 32 GB で動く画像生成(Qwen-Image 2.1 / FLUX.2 / Z-Image / Krea 2 / HiDream-O1)と動画生成(Wan 2.2 / HunyuanVideo 1.5 / LTX-2.5 / Kandinsky 6.0 / CogVideoX)を、品質最優先で比較し、常用構成を決める。

**進行中(Phase 0 完了)。調査は [notes/00-research.md](notes/00-research.md)、計画は [notes/01-plan.md](notes/01-plan.md)、取得済みモデルは [notes/02-models.md](notes/02-models.md)。**

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

### Phase 0: 環境(2026-11-03)

- ComfyUI を Docker(`yanwk/comfyui-boot:cu130-slim-v2`)で起動。モデル・ノード・出力はホスト `~/comfy-data/` に bind mount。詳細と落とし穴は [notes/02-models.md](notes/02-models.md)
- 生成と計測は `scripts/comfy_api.py`(API 形式のグラフを `/prompt` に投げ、`/history` で完了を待ち、`nvidia-smi` を 1 秒間隔で採取)。同じプロンプト・同じシードは ComfyUI がキャッシュして 1 秒で返すので、時間を測るときはシードを変える
- GPU の電力制限は計測時に `nvidia-smi --query-gpu=power.limit` で記録する(Phase 0 のスモークは 320 W)

### スモーク: Z-Image Turbo(bf16、8 ステップ、res_multistep / simple、shift 3)

| 条件                              | 所要      | VRAM ピーク    | 平均 / 最大電力  |
| --------------------------------- | --------- | -------------- | ---------------- |
| 1024×1024、初回(モデルロード込み) | 53.8 s    | 15,589 MiB     | 63 / 290 W       |
| 1024×1024、2 回目以降             | **6.0 s** | 15,435 MiB     | 222〜247 / 288 W |
| 2048×2048                         | 30.1 s    | **15,915 MiB** | 271 / 298 W      |

- bf16 の DiT 12.3 GB + エンコーダ 8 GB は同時には載らないが、ComfyUI が順にロード / 退避して 16 GB で回る。ただし 2K は残り 460 MiB で限界。エンコーダを fp8 にするか DiT を fp8 / GGUF にすると余裕ができる
- 文字: チョークボードの「焼きたて パン」が「焼きたて **バ**ン」(半濁点が濁点に)、「OPEN 7:00-19:00」は正確。日本語の濁点・半濁点は文字生成の見どころになる
- 構図・写実は 8 ステップとして十分(作例は `results/raw/img/`、commit 対象外)

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
