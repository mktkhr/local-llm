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

| 項目                                    | 結果                                                                                                                                                                                                                                                                                                             |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| llama.cpp                               | `ghcr.io/mostlygeek/llama-swap:cuda` 同梱の **build 10920**(commit `eafe15a5e`)。`--spec-type draft-mtp / ngram-simple / ngram-mod` と `--spec-draft-n-max` が使える。deploy と同じイメージをダイジェストでピン                                                                                                  |
| MTP ヘッドの VRAM コスト                | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_S` は **14,635 MiB**。通常版 IQ3_S(13,043 MiB)より **+1,592 MiB**。128k の KV(q8_0、1,360 MiB)より大きい                                                                                                                                                                |
| C'(主力 C 相当、フル GPU)               | IQ3_S は 128k / 96k / `--n-cpu-moe 2` のいずれも KV 1,360 MiB が確保できず OOM。**UD-IQ3_XXS に落として 128k フル GPU**(ロード後 15,882 MiB、56k prefill 中ピーク 15,914 MiB、推論中の増分 32 MiB)。基準の 15.5 GB は超えるが、KV と compute buffer はロード時確保のため採用                                     |
| 受理率の取得                            | `/metrics` に `llamacpp:spec_decode_num_draft_tokens_total` / `_num_accepted_tokens_total` / `_num_drafts_total` / `_num_accepted_tokens_per_pos_total{position}` が存在。**`/v1/messages` の応答に `timings` は無い**ため、tok/s も `/metrics` 差分で取る                                                       |
| スモーク(C' / mtp2 / W1-01、56k tokens) | decode **125 tok/s**、draft 52 / accepted 36(受理率 **0.69**、draft/generated 0.81)、status 200                                                                                                                                                                                                                  |
| プロンプト                              | ccshim ログ 2,278 件から W1 × 3(224〜461 KB、13〜103 メッセージ)、W2 × 3(345〜425 KB)を抽出。`scripts/data/prompts_manifest.json` に形だけ記録(本体は commit しない)                                                                                                                                             |
| B'(対照、IQ4_XS + ncmoe)                | MTP 版 IQ4_XS は通常版(16,909 MiB)+ 約 1.6 GB。採用時の ncmoe=10 では載らず、**`--n-cpu-moe 13`** でロード後 15,216 MiB / ピーク 15,250 MiB(14 なら 14,894 MiB)。スモーク(mtp2 / W1-01)は decode 50〜55 tok/s、受理率 0.55〜0.79。thinking は無効化されていることを `block_types`(`text`, `tool_use` のみ)で確認 |

## 結果

### スモーク掃引(1 ブロック、各セル n=3、2026-10-18)

`results/smoke_spec_sweep.csv`。Phase 1 本番(3 ブロック)の前に全設定を 1 周した値。**n=3 の暫定値**で、本番で更新する。比は同じプロンプトの `none` に対する decode tok/s の比(ブロック内で対にした平均)。

| 構成                | spec                | W1 tool_use(56〜109k) | W2 コード書き換え(90〜100k) | W3 日本語要約 | W4 思考  | 受理率(W1〜W4)            |
| ------------------- | ------------------- | --------------------- | --------------------------- | ------------- | -------- | ------------------------- |
| C' IQ3_XXS フル GPU | none                | 97 tok/s              | 91                          | 150           | 150      | —                         |
|                     | **mtp1**            | **1.17**              | **1.22**                    | **1.32**      | **1.36** | 0.86 / 0.96 / 0.87 / 0.94 |
|                     | **mtp2**            | **1.14**              | **1.31**                    | **1.44**      | **1.57** | 0.74 / 0.90 / 0.77 / 0.89 |
|                     | mtp3                | 起動不可(VRAM)        |                             |               |          |                           |
|                     | ngram-simple        | 0.96                  | 1.28(sd 62)                 | 0.99          | 0.99     | 0.16 / 0.26 / — / 0.02    |
|                     | ngram-mod           | 0.97                  | 1.32(sd 42)                 | 0.99          | 0.99     | 0.15 / 0.51 / — / —       |
|                     | mtp2 + ngram-simple | 1.04                  | 1.22                        | 1.44          | 1.55     | 0.51 / 0.36 / 0.77 / 0.84 |
| B' IQ4_XS ncmoe 13  | none                | 54 tok/s              | 51                          | 64            | 64       | —                         |
|                     | mtp1                | 1.03                  | 1.07                        | 1.04          | 1.07     | 0.95 / 0.97 / 0.86 / 0.93 |
|                     | mtp2                | 0.96                  | 1.14                        | 1.02          | 1.12     | 0.81 / 0.93 / 0.76 / 0.87 |
|                     | mtp3                | 1.09                  | 1.21                        | 0.96          | 1.09     | 0.82 / 0.90 / 0.65 / 0.80 |
|                     | ngram-simple        | 0.77                  | 0.94                        | 0.99          | 0.98     |                           |
|                     | ngram-mod           | 0.91                  | 1.11                        | 0.99          | 0.99     |                           |
|                     | mtp2 + ngram-simple | 0.71                  | 0.95                        | 1.04          | 1.11     |                           |

読みどころ(暫定):

- **フル GPU(C')では MTP が全ワークロードで勝つ。** n-max=2 で要約・思考 +44〜57%、長文 tool_use でも +14%。利得は受理率に比例し、tool_use の JSON(受理 0.6〜0.7)より自然文・思考(0.9)で大きい
- **CPU オフロード(B')では利得がほぼ消える。** 同じ受理率でも +3〜14%、W1 では負けることもある。検証バッチが CPU 側 expert を通るぶん 1 ステップのコストが増え、受理で稼いだ分を食う(問い 1 の答え)
- **n-gram はコード書き換えだけ強烈に効く。** W2-03(Write で 100k 中の長いブロックを書き出す)で ngram-simple **2.10 倍**、ngram-mod 1.91 倍。一方 W2-01 では 0.69 倍。発火しない W3/W4 では無害(0.99)。MTP と併用すると MTP の draft を n-gram が邪魔して W1 で悪化
- **MTP の VRAM コスト**: 重み +1,592 MiB(IQ3_S)に加え、起動後の draft context で **+830〜890 MiB**(none 15,024 → mtp 15,854〜15,918 MiB)。合計約 2.4 GB。16 GB では IQ3_S → IQ3_XXS への格下げと mtp3 の断念を強いた
- **出力は保存されない。** mtp の生成物一致率 0〜0.67(同じプロンプト・greedy でも baseline と別のトークン列)。n-gram で draft が出なかった行だけ 1.0。tool_use が壊れるかは Phase 2 で見る
- **電力**: C' で mtp は none より平均電力が低い(146〜215 W vs 179〜217 W)のに速い。tok/s/W で見ると差はさらに広がる
- **W1/W2 の prefill は cold。** 3 本のプロンプトは別セッション由来で共通 prefix が無く、`-np 1` では直前のプロンプトの KV しか残らないため、W1 を warm で測る設計は成立しなかった(cached_n が付いたのは primer 直後に同一プロンプトが来た 1 件のみ)。warm prefix の挙動は Phase 2 の実セッションで `cache_n` を見る

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
