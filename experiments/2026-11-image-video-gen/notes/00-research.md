# 調査: RTX 4080 SUPER 16 GB で動く画像・動画生成モデル(2026-11-03 時点)

Web 上の公式モデルカード・ライセンス・ComfyUI 公式ページを中心に確認した。VRAM の数字はブログ間で食い違うものが多く、**実機で測るまでは目安**。ライセンスは一次情報(LICENSE ファイル)を読んだものに ✓ を付ける。

## 0. 前提となる制約

| 項目         | 値                                                                                                                              | 影響                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| VRAM         | 16 GB(16,376 MiB、デスクトップ常駐 ~120 MiB)                                                                                    | fp8 / GGUF Q4〜Q8 が中心。bf16 は 6〜7B まで                                                                            |
| システム RAM | 32 GB(空き ~27 GB)                                                                                                              | **CPU オフロードの天井**。「VRAM に載らない分を RAM に逃がす」が RAM にも載らないモデルがある(FLUX.2 dev + Mistral 24B) |
| ディスク     | **空き 124 GB**(/ 915 GB 中 745 GB 使用)                                                                                        | 全モデルを同時には置けない。LLM の GGUF(docker volume `llama_llama-cache` 220 GB)を整理するか、フェーズごとに入れ替える |
| ランタイム   | Docker + NVIDIA Container Toolkit 動作確認済み。ホストの Python は 3.14(PyTorch 未対応)→ ComfyUI は Docker か uv で 3.12 を使う |                                                                                                                         |

## 1. 画像生成

| モデル                         | 公開       | 規模                                                 | ライセンス                                                                                       | 16 GB での現実解                                                                                                                                                                                  | 評判・特徴                                                                                                                                                  |
| ------------------------------ | ---------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen-Image 2.1**             | 2026-09-20 | DiT 7B(単一ストリーム 32 層)+ Qwen3-VL 8B エンコーダ | **Qwen Research License ✓ 非商用のみ**(「research or evaluation purposes only」、商用は別途契約) | bf16 14.2 GB は厳しい → **GGUF Q8_0(7.6 GB)/ Q6_K(5.9 GB)+ エンコーダ fp8**。ComfyUI-GGUF は leejet fork が必要との報告                                                                           | 生成と編集が 1 モデル、参照画像 10 枚、**透過 PNG(RGBA)直接出力**、ネイティブ 2K。Artificial Analysis のオープン部門で首位(1,035 Elo)。文字生成が最強クラス |
| **FLUX.2 [dev]**               | 2025-11-25 | DiT **32B** + Mistral-Small-3.2-24B エンコーダ       | flux-2-dev non-commercial                                                                        | GGUF Q3_K_M 16 GB / Q2_K 12.9 GB + エンコーダ fp4 12.3 GB(fp8 は 18 GB)→ **VRAM にも RAM(32 GB)にも収まりきらない**。Q2_K + fp4 で約 25 GB をオフロード運用、品質劣化大。**期待薄、1 回だけ試す** | オープン最高品質の基準点だが 16 GB では本領が出ない                                                                                                         |
| **FLUX.2 [klein] 9B**          | 2026-01    | 9B + Qwen3 エンコーダ                                | FLUX Non-Commercial(4B は Apache 2.0)                                                            | fp8 約 15 GB、または unsloth GGUF。エンコーダはオフロード                                                                                                                                         | 生成 + 単一/複数参照の編集。9B KV は編集向きで同品質で速い                                                                                                  |
| **FLUX.2 [klein] 4B**          | 2026-01    | 4B                                                   | **Apache 2.0**                                                                                   | bf16 でも約 13 GB、余裕                                                                                                                                                                           | プロンプト追従が良いという評。商用可の軽量枠                                                                                                                |
| **Z-Image Turbo**              | 2025-11    | 6B(S3-DiT)+ Qwen3-4B エンコーダ                      | Apache 2.0(要 LICENSE 確認)                                                                      | **bf16 がそのまま載る**(14〜16 GB)。fp8 / GGUF も豊富                                                                                                                                             | 8 ステップで写実。16 GB クラスの定番。CFG 無し・LoRA 不可                                                                                                   |
| **Z-Image(Base)**              | 2026-01-28 | 6B                                                   | Apache 2.0(要確認)                                                                               | Turbo と同じ                                                                                                                                                                                      | 非蒸留、28〜50 ステップ + CFG。多様性と微調整向き。Edit / Omni は 2026-08 時点で未公開(要再確認)                                                            |
| **Krea 2 Turbo**               | 2026-06-22 | DiT 12B + Qwen3-VL 4B エンコーダ + Qwen-Image VAE    | Krea 2 Community License(gated。個人・小規模の商用を許容と公式は説明)                            | **GGUF Q4_K_M 7.9 GB / Q6_K 10.6 GB + エンコーダ fp8 5 GB**。bf16 24 GB は不可                                                                                                                    | 8 ステップ CFG 0、ネイティブ 2K。写真表現・美的品質。ComfyUI は `krea2` CLIP type 対応版が必要                                                              |
| **HiDream-O1-Image**(追加候補) | 2026-05-08 | 8B                                                   | **MIT**                                                                                          | fp8 約 10 GB(公式に fp8-scaled / MXFP8 / bf16)。ComfyUI ネイティブ対応                                                                                                                            | アリーナでオープン 2 位の報告。商用自由で品質上位の穴馬                                                                                                     |

所見:

- **品質最優先なら Qwen-Image 2.1 と FLUX.2 [klein] 9B が本命、Z-Image と HiDream-O1 が対抗**。FLUX.2 [dev] は 16 GB + 32 GB RAM では量子化を落としすぎて比較にならない可能性が高い
- 商用可は Z-Image / klein 4B / HiDream-O1 / Krea 2(条件付き)。Qwen-Image 2.1 と klein 9B と FLUX.2 dev は非商用
- 文字生成は Qwen-Image 2.1、写実は Z-Image Turbo、美的は Krea 2、編集は Qwen-Image 2.1 と klein、という棲み分けの評判。実機で同一プロンプト・同一シードで確かめる

## 2. 動画生成

| モデル                                    | 公開       | 規模                                                    | ライセンス                                                                                                       | 16 GB での現実解                                                                                                                                                                                                                              | 評判・特徴                                                                                                          |
| ----------------------------------------- | ---------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **LTX-2.5**(2.3 の後継、現行)             | 2026-08-11 | DiT **22B** + **Gemma 4 12B** エンコーダ                | LTX-2.x Community License(年商 $10M 未満は商用無料)                                                              | 公式は bf16 / int8(ComfyUI 用)/ NVFP4(Blackwell 用)。**int8 でも DiT 約 22 GB + エンコーダ約 12 GB → VRAM 超過、RAM 32 GB も圧迫**。ComfyUI の低 VRAM ローダ + `--reserve-vram` で試す。コミュニティ GGUF は要調査                            | 音声同期・マルチショット・4K・50 fps・蒸留 8 ステップ。速い。手や人物の安定は Wan 2.2 が上という比較報告            |
| **LTX-2.3**                               | 2026-03-05 | 22B + Gemma 3 12B                                       | 同上                                                                                                             | fp8 公式あり(16 GB 向けと紹介されるが公式最小は 32 GB)                                                                                                                                                                                        | 2.5 に置き換わった。2.5 が載らない場合の代替                                                                        |
| **Wan 2.2 TI2V-5B**                       | 2025-07-28 | 5B(高圧縮 VAE、720p 24 fps)                             | **Apache 2.0**                                                                                                   | **fp16 10 GB で余裕**。T2V / I2V 兼用                                                                                                                                                                                                         | 軽量枠。品質は 14B に劣るが安定して回る                                                                             |
| **Wan 2.2 A14B**(T2V / I2V)               | 2025-07-28 | MoE 27B(high / low noise 各 14B、活性 14B)              | **Apache 2.0**                                                                                                   | fp8_scaled が high / low 各 14.3 GB → **2 本を順にロード + UMT5 エンコーダ fp8 オフロード**、または **GGUF Q4_K_M / Q5_K_M(各 9〜10 GB)**。480p なら 16 GB に収まる報告、720p は厳しい。**Lightning 4-step LoRA** で 20 ステップ → 4 ステップ | 写実と物理挙動は現時点のオープン最高評。Wan 2.5 / 2.6 / 3.0 は API のみで重みなし → **ローカルの最新は 2.2 のまま** |
| **HunyuanVideo 1.5**                      | 2025-11-20 | DiT 8.3B + Qwen2.5-VL 7B(fp8 9.4 GB)+ ByT5              | Tencent Hunyuan Community ✓(**EU / UK / 韓国 除外**。日本は対象内。月間 1 億 MAU 超は要契約、出力の AI 明示義務) | 公式最小 **14 GB(オフロード有効時)**。480p / 720p の T2V / I2V、蒸留版、SR(720→1080p)。ComfyUI ネイティブ                                                                                                                                     | 顔・シネマティックの評価が高い。音声なし                                                                            |
| **Kandinsky 6.0 Video Lite 3B**(追加候補) | 2026-10-06 | 3B(Pro は 29B)+ Qwen2.5-VL + CLIP、Hunyuan VAE、MMAudio | **MIT**                                                                                                          | 3B なら余裕。Pro 29B はブロックオフロード前提                                                                                                                                                                                                 | **最新のオープン動画 + 音声同期(44 kHz、リップシンク)**。ComfyUI は公式カスタムノード(`kandinsky6`)                 |
| **CogVideoX 1.5-5B**                      | 2024-11    | 5B                                                      | Apache 2.0                                                                                                       | 16 GB で動く(遅い)                                                                                                                                                                                                                            | 2024 年の基準点。I2V 1360×768・81 フレーム。比較対象としてのみ                                                      |

所見:

- **品質本命は Wan 2.2 A14B(GGUF Q5 + Lightning)と HunyuanVideo 1.5(fp8 + オフロード)**。LTX-2.5 は速さと音声が魅力だが、**22B + 12B エンコーダが 16 GB / 32 GB RAM に収まるかが最大の不確定要素**
- 「高品質な画像 → 動画化」は **I2V** の比較: Wan 2.2 I2V-A14B、HunyuanVideo 1.5 I2V(480p 蒸留 / 720p)、LTX-2.5 I2V、Kandinsky 6.0 I2V。入力画像は Phase 1 の最良モデルで作る
- 音声が要るなら LTX-2.5 か Kandinsky 6.0 の二択

## 3. 実行環境の候補

- **ComfyUI(本命)**: 画像・動画の全モデルに公式 or コミュニティのワークフローがある。Docker は `yanwk/comfyui-boot`(CUDA 12.8 slim)にホストの `models/` と `custom_nodes/` をマウント、`ComfyUI-GGUF`(Qwen-Image 2.1 は leejet fork)を入れる。ホストは Python 3.14 なので Docker が無難
- 量子化の品質比較は **同一シード・同一プロンプト**で fp8 / GGUF Q8 / Q6 / Q4 を並べる(ComfyUI は決定的)
- VRAM / 電力の採取は `nvidia-smi` 1 秒間隔(既存の `GpuSampler` を流用)

## 4. ディスク計画(概算)

| 用途                                                                                                             | 容量                        |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 画像 5〜6 モデル(GGUF / fp8 + エンコーダ)                                                                        | 約 60 GB                    |
| 動画 4〜5 モデル(Wan 14B fp8 ×4 本 57 GB は GGUF にして 40 GB 弱、Hunyuan 20 GB、LTX-2.5 35 GB、Kandinsky 10 GB) | 約 100 GB                   |
| 合計                                                                                                             | **約 160 GB > 空き 124 GB** |

対策: docker の未使用イメージ(回収可能 24 GB)と LLM の GGUF のうち実験が終わったもの(Qwen3.6 MTP UD-IQ4_XS 18 GB / UD-Q4_K_XL 21 GB、Laguna Q4_K_M 20 GB など)を消して 80 GB 以上空ける。動画モデルはフェーズごとに入れ替える。

## 5. 一次情報へのリンク

- Qwen-Image 2.1: [モデルカード](https://huggingface.co/Qwen/Qwen-Image-2.1) / [LICENSE](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE) / [Comfy ブログ](https://blog.comfy.org/p/qwen-image-21-in-comfyui-open-weight) / [unsloth ガイド](https://unsloth.ai/docs/models/qwen-image-2.1)
- FLUX.2: [公式 repo](https://github.com/black-forest-labs/flux2) / [klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) / [dev GGUF(city96)](https://huggingface.co/city96/FLUX.2-dev-gguf) / [Comfy-Org エンコーダ](https://huggingface.co/Comfy-Org/flux2-dev/tree/main/split_files/text_encoders)
- Z-Image: [Comfy docs](https://docs.comfy.org/ja/tutorials/image/z-image/z-image-turbo) / [Base 公開](https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release)
- Krea 2: [公式](https://www.krea.ai/krea-2-open-source) / [Turbo カード](https://huggingface.co/krea/Krea-2-Turbo) / [GGUF](https://huggingface.co/byteshape/Krea-2-Turbo-GGUF)
- HiDream-O1: [Comfy docs](https://docs.comfy.org/tutorials/image/hidream/hidream-o1.md)
- HunyuanVideo 1.5: [モデルカード](https://huggingface.co/tencent/HunyuanVideo-1.5) / [LICENSE](https://huggingface.co/tencent/HunyuanVideo-1.5/raw/main/LICENSE) / [Comfy ブログ](https://blog.comfy.org/p/hunyuanvideo-15-native-support) / [Comfy-Org 再パッケージ](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged)
- Wan 2.2: [Comfy-Org 再パッケージ(ファイルサイズ)](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) / [Lightning LoRA](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras) / [Wan 2.5 以降は非公開](https://howaiworks.ai/blog/alibaba-wan-open-weights-stopped-at-2-2)
- LTX: [LTX-2.5 カード](https://huggingface.co/Lightricks/LTX-2.5) / [Comfy](https://comfy.org/ltx-2.5/) / [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)
- Kandinsky 6.0: [ComfyUI wiki](https://comfyui-wiki.com/en/models/kandinsky/kandinsky-6-0)
- CogVideoX: [解説](https://stable-diffusion-art.com/cogvideox)
- ComfyUI Docker: [YanWenKun/ComfyUI-Docker](https://github.com/YanWenKun/ComfyUI-Docker)
