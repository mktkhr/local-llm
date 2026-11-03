# 計画: 16 GB で実現できる最高品質の画像・動画生成環境

作成: 2026-11-03。調査は [00-research.md](00-research.md)。

## 目的

RTX 4080 SUPER 16 GB + RAM 32 GB で、**生成品質を最優先**に画像生成・動画生成のモデルを比較し、常用する構成(モデル・量子化・ワークフロー)を決める。速度・VRAM は副次指標。VRAM に収まらないモデルも量子化と CPU オフロードで動くなら対象。

## 問い

1. 画像: プロンプト追従・複雑な構図・文字・写実・美的品質・編集で、16 GB で動く構成のうちどれが最良か
2. 量子化の代償: fp8 / GGUF Q8 / Q6 / Q4 で、同一シードの出力がどこから崩れるか(モデルごとに違うはず)
3. 動画: T2V と I2V で、Wan 2.2 A14B / HunyuanVideo 1.5 / LTX-2.5 / Kandinsky 6.0 Lite / Wan 5B / CogVideoX のどれが最良か。LTX-2.5 は 16 GB + 32 GB RAM に載るか
4. 画像 → 動画: 最良の画像モデルの出力を I2V に渡したとき、どの組み合わせが最も破綻しないか

## 固定するもの

| 項目           | 値                                                                                                                                                                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 環境           | ComfyUI(Docker `yanwk/comfyui-boot` CUDA 12.8、ホストの `models/` `custom_nodes/` `output/` をマウント)+ ComfyUI-Manager + ComfyUI-GGUF。バージョンは `docker image inspect` のダイジェストと ComfyUI のコミットを記録                   |
| GPU            | 電力制限は 320 W(既定)のまま計測する。品質は seed 固定で決定的なので電力に依らず、速度も同一条件の相対比較で足りるため 220 W の再計測はしない(2026-11-03 決定)。各行に `power_limit_w` を記録する                                        |
| 画像プロンプト | 固定 8 本: 文字入りポスター(日本語 + 英語)/ 複数人物の複雑構図 / 写実ポートレート / 風景 / 商品写真 / イラスト / 建築 / 小物の配置指示。各モデルで同一シード 3 個                                                                        |
| 画像の評価     | 1024×1024(モデルの推奨解像度に合わせ 1K と 2K を併記)。**人手の 5 段階評価**(追従・構図・文字・写実/美的・破綻)+ 補助として VLM(Gemma 4 / Qwen3.5-VL をローカル llama-server で)による同一基準の採点。VLM 採点は参考値で、最終判断は人手 |
| 動画プロンプト | 固定 4 本(人物の動き / カメラワーク / 物理(液体・布)/ 文字看板)× T2V、I2V は Phase 1 の最良画像 4 枚                                                                                                                                     |
| 動画の評価     | 480p と 720p、5 秒。人手 5 段階(時間一貫性・動きの自然さ・手と顔・プロンプト追従・破綻)+ 生成時間 + VRAM ピーク                                                                                                                          |
| 記録           | 全生成物を `results/raw/`(commit しない)、評価表と代表画像(縮小版)を `results/` に commit                                                                                                                                                |

## Phase

### Phase 0: 環境(半日)

1. ディスクを 80 GB 以上空ける(docker prune、実験済み LLM GGUF の削除)
2. ComfyUI を Docker で起動、ComfyUI-GGUF を導入、Z-Image Turbo で 1 枚生成してパイプラインを確認
3. モデルは `~/models/comfy/{diffusion_models,text_encoders,vae,loras}` に置き、フェーズごとに入れ替える。ダウンロード台帳(ファイル名・サイズ・sha・出典・ライセンス)を `notes/02-models.md` に残す

### Phase 1: 画像(1〜2 日)

- 実行は ComfyUI 公式テンプレート(`scripts/data/templates/`、MIT)を `scripts/tpl2api.py` で API 形式に変換して使う。公式の推奨設定(サンプラー・ステップ・CFG・shift)をそのまま採用し、変えるのはモデルファイル名・プロンプト・seed・解像度だけ
- 対象(`scripts/data/phase1_models.json`): Z-Image Turbo(bf16)/ Z-Image Base(bf16)/ Qwen-Image 2.1(公式 int8 と bf16)/ FLUX.3 Image Klein(distilled int8・bf16、base bf16)/ FLUX.2 [klein] 9B(fp8)と 4B(bf16)/ Krea 2 Turbo(fp8 / int8 / GGUF Q4_K_M)/ HiDream-O1-Image dev(fp8 / bf16)。FLUX.2 [dev] は Q2_K + fp4 エンコーダで 1 プロンプトだけ試し、載るか・品質がどうかを記録して打ち切る
- FLUX.3 Image Klein と FLUX.2 [klein] 9B は HF の gated repo(ライセンス同意 + トークンが要る)。2026-11-03 の判断で **一旦スキップ**。Qwen-Image 2.1 / Krea 2 / Z-Image で本命が揃ったため。後で入れるなら `black-forest-labs/flux-3-image-klein-comfy` と `black-forest-labs/FLUX.2-klein-base-9b-fp8` に同意し、トークンを GPU 機の `~/.hf_token` に置いて `scripts/data/phase1_models.json` の構成をそのまま回す
- プロンプト拡張(テンプレート同梱の LLM によるプロンプト書き換え。Qwen-Image 2.1・FLUX.3・Krea 2・HiDream にある)は全モデルで **切って** 比較する。同じ文で比べるため。効果は最良モデルで別途 1 回だけ見る
- 量子化の比較は Qwen-Image 2.1(bf16 vs int8)と Krea 2(fp8 vs int8 vs GGUF Q4)、FLUX.3 Klein(bf16 vs int8)で行う(同一シード)
- 編集は Qwen-Image 2.1 と klein 9B で同じ指示(背景差し替え・文字修正・物体追加)

### Phase 2: 動画(2〜3 日)

- T2V: Wan 2.2 TI2V-5B(fp16)、Wan 2.2 T2V-A14B(GGUF Q5 + Lightning 4-step、必要なら Q4)、HunyuanVideo 1.5 480p / 720p(fp8 + オフロード)、LTX-2.5(int8 の低 VRAM ローダ。載らなければ 2.3 fp8)、Kandinsky 6.0 Lite(公式ノード)、CogVideoX 1.5-5B(基準点)
- I2V: 同じモデル群の I2V 版に Phase 1 の最良画像を入れる
- LTX-2.5 は最初に「載るか」だけを確かめる。RAM 32 GB で Gemma 4 12B int8 + DiT int8 のオフロードが成立するかが分岐点

### Phase 3: 画像 → 動画のワークフロー(1 日)

- Phase 1 の最良モデル × Phase 2 の最良 I2V で 4 本。解像度の受け渡し(2K 画像 → 720p 動画の縮小)、アスペクト比、プロンプトの再利用を整理し、ComfyUI のワークフロー JSON を `config/workflows/` に残す

### Phase 4: 判定

- 画像 1 本・動画 1 本(+ 音声が要る場合の代替)を「常用構成」として README に書く。ライセンス(商用可否)も併記

## 見送り・注意

- Wan 2.5 / 2.6 / 3.0、Seedream 等の API 専用モデルは対象外
- ライセンスが非商用のもの(Qwen-Image 2.1、klein 9B、FLUX.2 dev)は評価用途として扱い、公開する作例に使うときは明記する
- HunyuanVideo 1.5 は EU / UK / 韓国を除外する license。日本国内での利用は対象内
- 動画は生成時間が長い(14B で 1 本 3〜10 分)。無人で回せるようにバッチ化する
