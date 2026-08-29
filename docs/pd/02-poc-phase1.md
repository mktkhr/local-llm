# PD分離 PoC — Phase 1: M1 単体での slot save / restore 検証

クロスバックエンド (Phase 2) の前提として、同一マシン・同一バックエンド上で slot save → restore が成立し、かつ**再 Prefill が発生しない**ことを確認した記録です。

- 実施日: 2026-08-29
- 結果: **PASS**

## 1. 環境

| 項目 | 値 |
| --- | --- |
| マシン | Apple M1 / Unified Memory 16GB / macOS 26.5.2 |
| llama.cpp | タグ `b10679` = commit `50f068ffffc3e0e4c9c2e4139281c6075224f429` |
| ビルド | `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON` |
| バックエンド | Metal (macOS 既定で ON) + BLAS (Accelerate) |
| モデル | `unsloth/Qwen3-4B-Instruct-2507-GGUF` の `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| モデルサイズ | 2,497,281,120 B |
| モデル sha256 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |

ビルドに必要で追加導入したもの: `cmake` 4.4.3 / `ninja` 1.13.2 / `iperf3` 3.21 (Homebrew)。

### 1.1 サーバ起動コマンド

```bash
./build/bin/llama-server \
  -m ~/models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 \
  -c 16384 -np 1 -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --slots --slot-save-path ~/pd-kv/ \
  --no-context-shift --no-warmup -lv 5
```

`-lv 5` は `flash_attn` の実効値を確認するために指定しています。通常の verbosity では出力されません。

## 2. 起動ログによる設定の実地確認

[01-research.md](01-research.md) §3.1 のとおり `-fa` の実効値が最重要であるため、ログで確認しました。

```
print_info: arch                  = qwen3
print_info: n_layer               = 36
print_info: n_embd_k_gqa          = 1024
print_info: n_embd_v_gqa          = 1024
llama_context: n_ctx                 = 16384
llama_context: flash_attn            = enabled
ggml_metal_library_compile_pipeline: compiling pipeline:
  base = 'kernel_flash_attn_ext_vec_q8_0_dk128_dv128'
```

- `flash_attn = enabled` → `v_trans = false`。**4080 側もこの値に一致させる必要があります**
- `n_layer` / `n_embd_k_gqa` は [01-research.md](01-research.md) §6 の想定どおりです
- Metal 側で q8_0 の Flash Attention カーネルがロードされています

## 3. 検証手順

再現スクリプトは Phase 3 で `scripts/pd/` に整理します。手順は以下のとおりです。

1. 2,026 トークンのプロンプトで `/completion` (`n_predict=1`) を実行し Prefill
2. `POST /slots/0?action=save`
3. `POST /slots/0?action=erase`
4. **対照実験**: restore せずに同じプロンプトを投げる
5. 再度 erase してから `POST /slots/0?action=restore`
6. 同じプロンプトで `/completion` (`n_predict=32`) を実行

判定は HTTP ステータスではなく `timings.cache_n` / `timings.prompt_n` で行います ([01-research.md](01-research.md) §4.2)。

## 4. 結果

| ステップ | wall | cache_n | prompt_n | prompt_ms | 備考 |
| --- | --- | --- | --- | --- | --- |
| 1. Prefill (cold) | 10,664 ms | 0 | 2,026 | 10,658 | |
| 2. slot save | 141 ms | — | — | — | `n_saved=2026` / `n_written=158,742,060` (151.4 MiB) / `save_ms=139.4` |
| 3. erase | — | — | — | — | `n_erased=2026` |
| 4. **対照 (restore なし)** | 10,208 ms | **0** | **2,026** | 10,204 | 全量が再 Prefill される |
| 5. restore | 30 ms | — | — | — | `n_restored=2026` / `n_read=158,742,060` / `restore_ms=29.0` |
| 6. **restore 後 decode** | 2,053 ms | **2,025** | **1** | 61.4 | `predicted_n=32` / 15.6 tok/s |

生成されたテキストも正常でした。

```
1. What is the main subject of the passage? 2. What is the significance of
the repeated phrases? 3. How does the repetition affect the
```

### 4.1 判定

**PASS。** restore 後は `cache_n=2025` / `prompt_n=1` であり、再 Prefill は発生していません。

`cache_n` が 2,026 ではなく 2,025 なのは、llama.cpp が最終トークンを必ず再処理する仕様によるものです。`prompt_n=1` がそれに対応します。

対照実験 (ステップ 4) で `cache_n=0` / `prompt_n=2026` となることから、この指標が再 Prefill の有無を正しく判別できることも確認できました。

## 5. 得られた基準値

| 指標 | 実測値 |
| --- | --- |
| **M1 Prefill 速度** | **190 tok/s** (2,026 tok / 10,658 ms) |
| **M1 Decode 速度** | **15.6 tok/s** |
| **KV サイズ (q8_0)** | **78,352 B/token = 76.5 KiB/token** |
| KV save 速度 | 151.4 MiB / 139.4 ms ≒ 1.09 GB/s |
| KV restore 速度 | 151.4 MiB / 29.0 ms ≒ 5.2 GB/s (ページキャッシュ命中) |

### 5.1 KV サイズは計算と一致

[01-research.md](01-research.md) §7 の計算値 76.5 KiB/token に対し、実測 76.5 KiB/token。**計算式が妥当であることが確認できました。**

差分 33,324 B はファイルヘッダ (magic / version / トークン列) と層ごとのメタデータによるものです。

### 5.2 break-even の見通しが変わりました

Phase 0 では M1 の Prefill を 300 tok/s と仮定していましたが、**実測は 190 tok/s** でした。M1 の Prefill は想定よりさらに遅く、PD分離が有利になる閾値は下がります。

2k トークンの場合の概算です。

- M1 単体の Prefill: **10.7 秒**
- PD分離のオーバーヘッド: save 0.14 秒 + 転送 (151.4 MiB) + restore 0.03 秒 + 4080 の Prefill

Wi-Fi の実効を 40 MB/s と仮定すると転送は約 3.8 秒で、合計は約 4 秒です。

**すなわち 2k の時点で、Wi-Fi 経由であっても PD分離が M1 単体より速い可能性があります。** Phase 0 時点の「初回は PD分離が遅くても問題ない」という想定より、はるかに早く優位に立つ見込みです。Phase 3 で実測します。

## 6. 次のステップ

Phase 2 として、4080 SUPER (CUDA) で保存した KV ファイルを M1 (Metal) で restore できるかを検証します。判定基準は本 Phase と同一です。
