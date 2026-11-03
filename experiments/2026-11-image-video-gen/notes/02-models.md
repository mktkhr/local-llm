# モデル台帳(ダウンロード済み)

GPU 機の `~/comfy-data/storage-models/models/` 配下。ディスクが足りないのでフェーズごとに入れ替える。削除したら「状態」を更新する。残り(FLUX.3 Klein、FLUX.2 klein 9B、Krea 2 int8 / GGUF / エンコーダ、HiDream-O1)はダウンロード中で、完了後に追記する。

| モデル                            | ファイル                                  | 置き場              | サイズ  | 出典                               | ライセンス                    | 状態            |
| --------------------------------- | ----------------------------------------- | ------------------- | ------- | ---------------------------------- | ----------------------------- | --------------- |
| Z-Image Turbo(DiT bf16)           | `z_image_turbo_bf16.safetensors`          | `diffusion_models/` | 12.3 GB | `Comfy-Org/z_image_turbo`          | Apache 2.0(Tongyi-MAI)        | 2026-11-03 取得 |
| Qwen3-4B テキストエンコーダ(bf16) | `qwen_3_4b.safetensors`                   | `text_encoders/`    | 8.0 GB  | 同上                               | 同上                          | 同上            |
| Z-Image VAE                       | `ae.safetensors`                          | `vae/`              | 0.3 GB  | 同上                               | 同上                          | 同上            |
| Z-Image Base(DiT bf16)            | `z_image_bf16.safetensors`                | `diffusion_models/` | 12.3 GB | `Comfy-Org/z_image`                | Apache 2.0                    | 2026-11-03 取得 |
| Qwen3-4B テキストエンコーダ(fp8)  | `qwen_3_4b_fp8_mixed.safetensors`         | `text_encoders/`    | 5.6 GB  | `Comfy-Org/z_image`                | Apache 2.0                    | 同上            |
| Qwen-Image 2.1(DiT bf16)          | `qwen_image_2.1_bf16.safetensors`         | `diffusion_models/` | 14.2 GB | `Comfy-Org/Qwen-Image-2.1_ComfyUI` | Qwen Research License(非商用) | 同上            |
| Qwen-Image 2.1(DiT int8 convrot)  | `qwen_image_2.1_int8_convrot.safetensors` | `diffusion_models/` | 7.3 GB  | 同上                               | 同上                          | 同上            |
| Qwen3-VL-8B エンコーダ(int8)      | `qwen3vl_8b_int8_convrot.safetensors`     | `text_encoders/`    | 9.4 GB  | 同上                               | 同上                          | 同上            |
| Qwen-Image 2.1 VAE                | `qwen_image_2.1_vae_bf16.safetensors`     | `vae/`              | 0.7 GB  | 同上                               | 同上                          | 同上            |
| FLUX.2 [klein] 4B(bf16)           | `flux-2-klein-4b.safetensors`             | `diffusion_models/` | 7.8 GB  | `Comfy-Org/flux2-klein-4B`         | Apache 2.0                    | 同上            |
| Qwen3-8B エンコーダ(fp8 mixed)    | `qwen_3_8b_fp8mixed.safetensors`          | `text_encoders/`    | 8.7 GB  | `Comfy-Org/flux2-klein-9B`         | 同上(エンコーダ)              | 同上            |
| FLUX.2 VAE(小デコーダ)            | `full_encoder_small_decoder.safetensors`  | `vae/`              | 0.25 GB | 同上                               | 同上                          | 同上            |
| Krea 2 Turbo(DiT fp8 scaled)      | `krea2_turbo_fp8_scaled.safetensors`      | `diffusion_models/` | 13.1 GB | `Comfy-Org/Krea-2`                 | Krea 2 Community License      | 同上            |

## 環境

| 項目           | 値                                                                                                                                                                                                                                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| イメージ       | `yanwk/comfyui-boot:cu130-slim-v2`(17 GB)。Python 3.13.15、PyTorch 2.13.0+cu130                                                                                                                                                                                                                           |
| カスタムノード | `ComfyUI-GGUF`(city96、`6ea2651`)。起動ログで import 成功を確認                                                                                                                                                                                                                                           |
| 起動           | `experiments/2026-11-image-video-gen/config/compose.yml`(`docker compose up -d`)。ポート 8188 を LAN に公開                                                                                                                                                                                               |
| 注意           | コンテナは root で動くため、bind mount 配下に作られたディレクトリは root 所有になる。ホストからモデルを置くときは `docker run --rm -v ~/comfy-data:/d alpine chown -R $(id -u):$(id -g) /d` で戻すか、`docker exec comfyui mv ...` で動かす。`/tmp` は tmpfs 16 GB なのでダウンロードの一時置き場にしない |
