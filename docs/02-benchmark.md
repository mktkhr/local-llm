# 02. 計測

「このモデルがこの設定で安定して動くか」を見るための計測手順と、結果のどこを読むか。

## 1. 計測の原則

| 原則                                                                      | 理由                                                                    |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **1モデルだけロードする**                                                 | 別モデルが残ったままだと VRAM を共有して値が壊れる                      |
| **並行リクエストは 1 に固定する**                                         | KVキャッシュは並行スロット数倍で確保される                              |
| **`keep_alive` を明示する**                                               | 計測中にアンロードされると値が変わる。計測後は `keep_alive: 0` で即解放 |
| **デフォルト context と 目標 context の両方で計測する**                   | デフォルト(2k〜4k)では問題なくても、目標(128k 等)で破綻するケースがある |
| **VRAM は `nvidia-smi` を、PROCESSOR(GPU/CPU比率) は `ollama ps` を読む** | 両方を見ないと「乗っているように見えて実はオフロード中」が起きる        |

## 2. 計測コマンド(汎用テンプレ)

`<MODEL>` と `<CTX>` を埋めるだけで使えるテンプレ。`for` で複数モデルを回すこともできます。

```bash
MODEL=qwen3.5:2b
CTX=131072

# 1. ロード(5分常駐)
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"hi\",
  \"stream\": false,
  \"keep_alive\": \"5m\",
  \"options\": {\"num_ctx\": $CTX}
}" | grep -oE '"done":[^,]*'

# 2. ロード後の状態
ollama ps
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# 3. 速度計測(eval_count / eval_duration から tok/s を算出)
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"日本語で1段落、自己紹介を書いてください。\",
  \"stream\": false,
  \"keep_alive\": \"5m\",
  \"options\": {\"num_ctx\": $CTX, \"num_predict\": 128}
}" | grep -oE '"(eval_count|eval_duration|prompt_eval_count|prompt_eval_duration|load_duration)":[0-9]*'

# 4. 即アンロード
curl -s http://localhost:11434/api/generate -d "{\"model\": \"$MODEL\", \"keep_alive\": 0}" > /dev/null
while ollama ps 2>/dev/null | grep -q "$MODEL"; do sleep 1; done
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

## 3. 何を読むか

### `ollama ps` の出力

```
NAME            ID    SIZE    PROCESSOR          CONTEXT    UNTIL
qwen3.5:4b      ...   12 GB   38%/62% CPU/GPU    131072     4 minutes from now
```

| 列          | 意味                                                           | 判断                                                                |
| ----------- | -------------------------------------------------------------- | ------------------------------------------------------------------- |
| `SIZE`      | モデル重み + KVキャッシュ + 推論バッファ の **想定確保サイズ** | これが実VRAM を上回ると後述のオフロードが発生する                   |
| `PROCESSOR` | レイヤー配置の GPU / CPU 比率                                  | **`100% GPU` でないと CPUオフロード発生中**。生成速度が大幅に落ちる |
| `CONTEXT`   | 実際にロードされたコンテキスト長                               | API で指定した `num_ctx` が反映されているか確認                     |

`PROCESSOR` が `100% GPU` でない時点で「GPUに収まらないモデル/設定」と判定できます。

### `nvidia-smi` の出力

```
memory.used [MiB], memory.free [MiB]
7,150 MiB,         961 MiB
```

- `memory.used` … 実測のGPUメモリ使用量。`ollama ps` の `SIZE` はあくまで「想定確保量」で、CPUオフロードが発生していると実VRAMはこれより小さい
- `memory.free` … 残りVRAM。これが極端に小さい(数百MiB以下)とコンテキスト追加時に詰む

### API レスポンスの数値

| フィールド             | 単位   | 意味                                       |
| ---------------------- | ------ | ------------------------------------------ |
| `load_duration`        | ns     | モデルロード所要時間                       |
| `prompt_eval_count`    | tokens | プロンプト(入力)をモデルに通したトークン数 |
| `prompt_eval_duration` | ns     | プロンプト評価の所要時間                   |
| `eval_count`           | tokens | 生成したトークン数                         |
| `eval_duration`        | ns     | 生成の所要時間                             |

**速度の換算**

```
生成速度 (tok/s)        = eval_count / (eval_duration / 1e9)
プロンプト評価速度       = prompt_eval_count / (prompt_eval_duration / 1e9)
```

`prompt_eval` は長文を投入したときのスループット指標。長コンテキスト運用では生成速度より先にここがボトルネックになるケースもあります。

## 4. CPUオフロードの判定

以下のいずれかが該当したらオフロード発生中。

- `ollama ps` の `PROCESSOR` 列に `% CPU` が含まれている
- `nvidia-smi` の `memory.used` が `ollama ps` の `SIZE` より大幅に小さい
- 生成速度が同モデルの GPU 100% 時の半分未満まで落ちている

→ [03-tuning.md](03-tuning.md) の判定フローへ。

## 5. 計測結果の残し方

[04-results.md](04-results.md) に1行追記。必要な情報は以下。

- ハードウェア(GPU名 / VRAM / Compute Capability)
- モデル名(`vendor:tag`)とサイズ
- 設定(`num_ctx`, KV量子化, Flash Attention)
- 観測値(VRAM使用、残VRAM、PROCESSOR、生成速度、プロンプト評価速度)
