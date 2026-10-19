# 計画: GPU 電力リミット(150 / 220 / 320 W)と推論速度・電力効率

作成: 2026-10-19。[2026-10-speculative-decoding](../../2026-10-speculative-decoding/README.md) の計測が利用者の常用設定 **220 W 制限**(RTX 4080 SUPER の既定は 320 W)の下で行われ、C' の prefill が上限に張り付いていたことを受けた追試。

## 問い

1. 「MTP n=2 は none より速くて平均電力が低い」は 220 W 制限の産物か。制限を外す(320 W)と差は縮むか
2. prefill と decode のどちらが電力律速か。150 W に絞ったとき、どちらがどれだけ落ちるか
3. tok/s/W の最適点はどこか(150 / 220 / 320 のうち)

## 固定するもの

| 項目         | 値                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| 構成         | C' = `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_XXS`、128k、KV q8_0、フル GPU(前実験と同一の `config/model/C-iq3xxs.args`) |
| llama.cpp    | llama-swap イメージ build 10920(ダイジェストでピン、前実験と同一)                                                        |
| spec         | `none` / `draft-mtp --spec-draft-n-max 2`                                                                                |
| ワークロード | W1(前実験の private プロンプト 3 本、56〜109k トークンの cold prefill)/ W3(日本語要約 3 本)/ W4(思考 3 本、300 トークン) |
| ブロック     | 各リミットで 2 ブロック(spec の順序を入れ替え)。各セル n=6                                                               |
| 計測         | `/metrics` 差分(tok/s、受理率)+ `nvidia-smi`(memory.used / power.draw / temperature / **clocks.sm** を 1 秒間隔)         |

## 動かすもの

- 電力リミット: **220 → 150 → 320 → 220(復帰)** の順。`sudo nvidia-smi -pl N` は利用者が実行し、`scripts/run_limit.sh N` が現在値を確認してから掃引する

## 判定

- 問い 1: 320 W での mtp2 / none の比と電力差が 220 W と同じなら「制限の産物ではない」
- 問い 2: 150 W での prefill と decode の低下率を比べる。prefill(演算律速)が大きく落ち、decode(帯域律速)はあまり落ちない、という予想
- 問い 3: `decode_tok_per_w` と `prefill_tok_per_w` の最大点

## 成果物

- `scripts/run_limit.sh`(1 リミットぶんの実行)、`scripts/aggregate_power.py`(集計)、`results/power_limit.csv`、`results/raw/sweep-*-pl<W>.jsonl`(commit 対象外)
- README に結論と表
