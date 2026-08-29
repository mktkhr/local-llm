# PD分離(Prefill/Decode分離) — 事前調査

llama.cpp で「別マシンで Prefill した KV キャッシュを転送し、別マシンで Decode を継続する」ことが成立するかを、実装ソースを読んで判定した記録です。

- 調査日: 2026-08-29
- 調査対象: `ggml-org/llama.cpp` master `d7bd3bf` (2026-08-28)
- 調査方法: リポジトリを clone してソースを直接読解。あわせて GitHub Issues / Discussions を確認

## 0. 結論

**CUDA 側で保存した KV キャッシュを Metal 側で restore することは、実装上成立します。** slot 保存ファイルの形式はバックエンド非依存です。

ただし一致要件が複数あり、特に `-fa` の既定値 `auto` が最大の落とし穴です。両機で `-fa on` を明示指定する必要があります。

なお「ソース上成立する」ことと「実際に動く」ことは別です。クロスバックエンド restore の前例報告は見つかりませんでした。実測で確定させます。

## 1. なぜバックエンド非依存と言えるか

`llama_state_seq_save_file` から KV テンソルが書き出されるまでの経路をたどります。

### 1.1 書き出し経路

`src/llama-kv-cache.cpp:2277` `llama_kv_cache::state_write_data`:

```cpp
// Write key type
const int32_t k_type_i = (int32_t) k->type;
io.write(&k_type_i, sizeof(k_type_i));

// Write row size of key
const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
io.write(&k_size_row, sizeof(k_size_row));

// Read each range of cells of k_size length and write out
for (const auto & range : cr.data) {
    const size_t range_size = range.second - range.first;
    const size_t buf_size = range_size * k_size_row;
    io.write_tensor(k, range.first * k_size_row, buf_size);
}
```

### 1.2 write_tensor の実体

`src/llama-context.cpp:2688` `llama_io_write_file::write_tensor`:

```cpp
void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
    temp_buffer.resize(size);
    ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
    write(temp_buffer.data(), temp_buffer.size());
}
```

`ggml_backend_tensor_get` はデバイスからホストへのコピーです。書き出されるのは ggml の正準レイアウトに整列したホスト側データであり、CUDA / Metal のデバイスバッファをそのままダンプしたものではありません。

加えて、各層について `k_type` (ggml 型 enum) と `k_size_row` をファイルへ明示的に書き込みます。**バックエンドを識別する情報は一切書き込まれません。**

### 1.3 既存情報との相違

Discussion #20572 の要約には「slot ファイルはハードウェア/バックエンド固有の成果物」という趣旨の記述がありますが、**ソース上はこれを支持する根拠が見当たりません**。本調査ではソース読解を優先し、実測で最終判定します。

## 2. ファイル形式

`src/llama-context.cpp:3238` `llama_context::state_seq_save_file`:

| オフセット | 内容 |
| --- | --- |
| 0 | magic `GGSQ` (`LLAMA_STATE_SEQ_MAGIC`) |
| 4 | version = `3` (`LLAMA_STATE_SEQ_VERSION`, `include/llama.h:49`) |
| 8 | token 数 (u32) |
| 12 | token 列 (`llama_token` × N) |
| 以降 | KV キャッシュ本体 |

## 3. restore 側が照合する条件

`src/llama-kv-cache.cpp:2533` `llama_kv_cache::state_read_data` が拒否する条件です。

| 条件 | 失敗時のログ |
| --- | --- |
| `n_layer` 不一致 | `mismatched layer count` |
| `cell_count > cells.size()` | `not enough cells in kv cache to restore state` |
| `v_trans` 不一致 | `incompatible V transposition` |
| 層ごとの `k_type` / `v_type` 不一致 | `mismatched key type` / `mismatched value type` |
| 層ごとの `k_size_row` / `v_size_row` 不一致 | `mismatched key row size` / `mismatched value row size` |

### 3.1 v_trans は Flash Attention 設定そのもの

`src/llama-model.cpp:2239` ほか、`llama_kv_cache` のコンストラクタ引数:

```cpp
!cparams.flash_attn,   // ← v_trans
```

**Flash Attention の ON/OFF が両機で一致していないと restore は必ず失敗します。**

さらに既定値は `LLAMA_FLASH_ATTN_TYPE_AUTO` (`src/llama-context.cpp:3607`) で、AUTO は実グラフを probe してバックエンドごとに解決します (`src/llama-context.cpp:554`):

```cpp
if (cparams.auto_fa) {
    resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
    cparams.auto_fa = false;
}
```

CUDA と Metal で解決結果が異なりうるため、**両機で `-fa on` を明示指定します。これが本 PoC で最重要の設定です。**

### 3.2 照合されないもの

`state_seq_save_file` は full-state 経路 (`state_write_data`) と異なり、**arch 名すら書き込みません**。model のハッシュ・vocab・rope 設定はいずれも照合されません。

**形状さえ一致すれば別モデルでも restore が「成功」し、出力だけが壊れます。** 両機で同一 GGUF ファイル (sha256 一致) を使うことが必須です。

## 4. 計測は自前実装が不要

### 4.1 save / restore のレスポンス

`tools/server/README.md` および `tools/server/server-context.cpp:2518` 以降。必要な値がそのまま返ります。

```json
{"id_slot":0,"filename":"x.bin","n_saved":1745,"n_written":14309796,"timings":{"save_ms":49.865}}
{"id_slot":0,"filename":"x.bin","n_restored":1745,"n_read":14309796,"timings":{"restore_ms":42.937}}
```

### 4.2 再 Prefill の有無を判定する指標

`tools/server/server-common.cpp:69` `server_slot_stats::to_json`:

```json
"timings": {
  "cache_n":    "キャッシュから再利用したトークン数",
  "prompt_n":   "実際に処理したトークン数",
  "prompt_ms":  "...",
  "predicted_n": "..."
}
```

**合格判定: restore 後の decode リクエストで `cache_n` が Prefill 済みトークン数とほぼ一致し、`prompt_n` が 1〜数トークンに収まること。**

`prompt_n` がプロンプト全長であれば再 Prefill されており、PD分離は成立していません。HTTP 200 では判定しません。

OpenAI 互換エンドポイントでは `usage.prompt_tokens_details.cached_tokens` が同じ値です。

## 5. 既知の問題と回避すべき構成

| Issue | 内容 | 対応 |
| --- | --- | --- |
| [#18703](https://github.com/ggml-org/llama.cpp/issues/18703) | multi-model router モード (`--models-preset`) は slot save/restore 非対応 (400 `Invalid action`) | **既存の llama-swap 経路は使わず、素の llama-server を別ポートで立てる** |
| [#22373](https://github.com/ggml-org/llama.cpp/issues/22373) | router モードで `/slots` が利用できない | 同上 |
| [#21133](https://github.com/ggml-org/llama.cpp/issues/21133) | `--mmproj` をロードすると全 slot が `has_mtmd=true` になり save/restore がブロックされる | VL モデルを使わない |
| [#20473](https://github.com/ggml-org/llama.cpp/issues/20473) | GLM-5 で restore 時に `GGML_ASSERT(nread <= state_size)` 失敗 | 該当モデルを使わない |
| [#21140](https://github.com/ggml-org/llama.cpp/issues/21140) | recurrent state restore のマルチ GPU ROCm クラッシュ | Mamba / hybrid 系を使わない |
| [#21383](https://github.com/ggml-org/llama.cpp/issues/21383) | Qwen3.5-27B の CUDA illegal memory access (prompt cache) | 27B MoE を使わない |
| [#23210](https://github.com/ggml-org/llama.cpp/issues/23210) | Qwen3.6-27B で llama-server がクラッシュ | 同上 |
| [#22450](https://github.com/ggml-org/llama.cpp/issues/22450) | Qwen3.6-35B-A3B MoE で slot がハング | 同上 |

**Qwen3-4B (dense) に該当する既知の regression は見つかりませんでした。** 報告は MoE / hybrid / VL / SWA 系に集中しています。

## 6. Qwen3-4B を採用する構造的な理由

`src/models/qwen3.cpp` に SWA の記述がありません。HuggingFace の `config.json` も `"sliding_window": null` / `"use_sliding_window": false` です。

- `llama_kv_cache_iswa` ではなく素の `llama_kv_cache` を使う → `--swa-full` が不要
- MoE ではない → §5 の MoE 系 regression に当たらない
- recurrent ではない → #21140 に当たらない
- VL ではない → #21133 に当たらない

**既知バグの当たらない、最も単純な経路です。** PoC の第一候補として構造上妥当と判断します。

主要パラメータ (`Qwen/Qwen3-4B-Instruct-2507/config.json`):

- `num_hidden_layers`: 36
- `num_key_value_heads`: 8
- `head_dim`: 128
- → `n_embd_k_gqa` = `n_embd_v_gqa` = 8 × 128 = **1024**

## 7. KV ファイルサイズとコンテキスト長の関係

ファイルには実使用セルのみを書き出します (`cr.data` の range 単位)。**サイズは `n_ctx` ではなく実トークン数に比例します。**

1 トークンあたりのバイト数 = `36 層 × 2 (K,V) × ggml_row_size(type, 1024)`

| KV 型 | row_size | 1 token あたり |
| --- | --- | --- |
| f16 | 2,048 B | 147,456 B = **144 KiB** |
| q8_0 | 1,088 B | 78,336 B = **76.5 KiB** |
| q4_0 | 576 B | 41,472 B = **40.5 KiB** |

| tokens | f16 | q8_0 | q4_0 |
| --- | --- | --- | --- |
| 1k | 144 MiB | 76.5 MiB | 40.5 MiB |
| 2k | 288 MiB | 153 MiB | 81 MiB |
| 4k | 576 MiB | 306 MiB | 162 MiB |
| 8k | 1.13 GiB | 612 MiB | 324 MiB |
| 16k | 2.25 GiB | 1.20 GiB | 648 MiB |

**KV 量子化が転送時間を直接半減・4分の1にする最大のレバーです。** ただし型は照合されるため両機で一致させます。q8_0 の V は Flash Attention が必須であり、§3.1 の `-fa on` と整合します。

### 7.1 break-even の事前予測 (実測で置き換える)

8k / q8_0 = 612 MiB。Wi-Fi 実効 40 MB/s と仮定すると転送に約 15 秒です。M1 の 4B Q4 prefill を 300 tok/s と仮定すると 8k で約 27 秒です。

**Wi-Fi でも 8k 前後で break-even に届く可能性があります。** 10GbE (実効 1.1 GB/s) なら転送が 0.6 秒となり、break-even は 1〜2k まで下がる見込みです。

この予測を実測で検証することが本 PoC の主目的です。

## 8. ファイル転送以外の手段は現時点で存在しません

- [#21266](https://github.com/ggml-org/llama.cpp/issues/21266) `server : disaggregated prefill/decode support` は **open かつ未実装**です。紐づく PR もありません
- [#15959](https://github.com/ggml-org/llama.cpp/discussions/15959) で ggerganov 氏が「context 間の memory コピーには `llama_state_` API を使う」と回答しています。**slot save/restore はまさにその API のラッパです**
- `ggml-rpc-server` (`tools/rpc/`) は層分割による分散推論であり、PD分離ではありません。全トークンの forward が毎回ネットワークを跨ぐため目的が異なります

**現時点では slot save → ファイル転送 → slot restore が唯一かつ公式に示唆された手段です。**

## 9. 推奨構成

| 項目 | 値 | 理由 |
| --- | --- | --- |
| モデル | Qwen3-4B-Instruct-2507 Q4_K_M GGUF (両機で同一ファイル / sha256 一致) | §6。M1 16GB に余裕がある |
| commit | タグ `b10679` を両機で固定 | 調査時点の最新タグ。既存 llama-swap の b10438 とは別に立てる |
| KV 型 | `--cache-type-k q8_0 --cache-type-v q8_0` | 転送量を f16 の 53% に削減。比較軸として f16 も測る |
| Flash Attention | **`-fa on` を両機で明示** (`auto` は禁止) | §3.1。不一致で restore が確実に失敗する |
| CUDA ビルド | `cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` | |
| Metal ビルド | `cmake -B build -DCMAKE_BUILD_TYPE=Release` (Metal は macOS 既定で ON) | |
| server 共通 | `--slot-save-path <dir> --slots -np 1 --no-context-shift -ngl 99 -c <n_ctx>` | router モードと context shift を排除する |
| 4080 側 | 既存 llama-swap (:11435) は変更せず、別ポートで素の llama-server を起動 | §5 #18703 |
| 検証指標 | `/completion` の `timings.cache_n` / `timings.prompt_n` | §4.2 |

## 10. 未確定事項 (実測で解消する)

1. **クロスバックエンド restore の実動作**。ソース上は成立するが前例報告がない。PoC Phase 2 の主目的
2. M1 の実 prefill tok/s。break-even 計算の分母になる
3. Wi-Fi の実効転送速度。`iperf3` が未インストールのため導入が必要
4. `n_written` の実測値が §7 の計算と一致するか

## 11. PoC の段階

| Phase | 内容 | 完了条件 |
| --- | --- | --- |
| 0 | 調査 (本ドキュメント) | 完了 |
| 1 | M1 単体で save → restore を成立させる | `cache_n` により再 Prefill なしを確認 |
| 2 | 4080 (CUDA) → M1 (Metal) のクロスバックエンド疎通 | 同上。PD分離の成立判定 |
| 3 | スクリプト化とベンチマーク (A/B/C/D × 1k〜16k) | break-even 点の算出、10GbE 推定値 |

## 参考

- [llama.cpp `tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Discussion #20572: Persistent KV cache per session with llama-server hooks](https://github.com/ggml-org/llama.cpp/discussions/20572)
- [Discussion #15959: Prefilling Decoding Disaggregation Support?](https://github.com/ggml-org/llama.cpp/discussions/15959)
- [Issue #21266: server : disaggregated prefill/decode support](https://github.com/ggml-org/llama.cpp/issues/21266)
