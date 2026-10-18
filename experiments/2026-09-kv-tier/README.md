# 階層型 KV キャッシュ: VRAM / DRAM / SSD からの KV 復元 vs 再 Prefill(2026-09-kv-tier)

llama.cpp の prompt cache(`--cache-ram` / slot save)を使い、VRAM から溢れた KV を DRAM と SATA SSD に退避して復元した場合、再 Prefill と比べてどこから速いのか(損益分岐)を RTX 4080 SUPER 1 台で実測した。[2026-08-pd-split](../2026-08-pd-split/README.md) の slot save/restore 知見の続き。

| 項目       | 値                                                                         |
| ---------- | -------------------------------------------------------------------------- |
| 期間       | 2026-09-03                                                                 |
| 機材       | RTX 4080 SUPER 16 GB / Ryzen 5 3600 / DDR4 30 GiB / Crucial BX500 SATA SSD |
| ランタイム | llama.cpp `b10679`(Docker、[config/](config/))                             |
| モデル     | Qwen3-4B-Instruct-2507 Q4_K_M(native ctx 262,144)                          |
| 関連実験   | [2026-08-pd-split](../2026-08-pd-split/README.md)                          |

## 結論

1. **200,000 tokens で、full prefill 141.6 秒が DRAM から 1.6 秒 (89.6x) / SSD から 20.2 秒 (7.0x) になりました。** 本 PoC の主目的である「長 context セッションの再開」は成立します (H3 成立)。
2. **DRAM tier は全長・全 KV 型で再 Prefill より速い (H1 成立)。** 損益分岐は存在しません。
3. **SSD tier (SATA) の損益分岐は KV 量子化で 1 桁動きます (H5 成立)。** f16 で約 10 万、q8_0 で 43,429、q4_0 で 10,633 tokens。**KV 量子化が SSD tier 成立の主レバー**であることが実測で確定しました。
4. **事前予測 ([01-research.md](01-research.md) §5) は q8_0 で誤差 6%、q4_0 で 11%** でした。1k〜16k のデータから 200k までの外挿が成立しています。
5. **階層には帯域だけでなく容量の壁があります。** DRAM に載り切らないサイズになった瞬間、実測値が SSD 速度へ落ちました (§6.1)。単純な「速度表」では設計できません。

詳細は [notes/03-results.md](notes/03-results.md)。調査は [notes/01-research.md](notes/01-research.md)、設計は [notes/02-design.md](notes/02-design.md)。

## ファイル

| パス                                                  | 内容                                                    |
| ----------------------------------------------------- | ------------------------------------------------------- |
| `config/`                                             | 計測用 llama-server の Docker 構成(`.env` は gitignore) |
| `scripts/make_context.py`                             | Coding Agent 相当のテスト context 生成                  |
| `scripts/run_cases.py`                                | Case A〜E の計測                                        |
| `scripts/sweep.sh`                                    | KV 型 × context 長の掃引                                |
| `scripts/aggregate.py`                                | CSV 出力と損益分岐の算出                                |
| `results/kvtier_runs.jsonl`                           | 生データ(17 構成 × 6 ケース)                            |
| `results/kvtier.csv` / `results/kvtier_breakeven.csv` | 集計                                                    |
| `results/raw/ctx/`                                    | 生成したテスト context(gitignore)                       |
