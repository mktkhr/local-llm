# Laguna XS 2.1 vs Qwen3.6-35B-A3B: Claude Code バックエンドの置き換え候補か(2026-10-laguna-xs-2-1)

Poolside Laguna XS 2.1(33B-A3B、agentic coding 特化)を RTX 4080 SUPER 16 GB に Qwen3.6-35B-A3B と同じ条件で載せ、速度・VRAM と Claude Code の実タスク(M1 / M2)で比較する。

**完了(2026-10-19)。設計は [notes/00-plan.md](notes/00-plan.md)。**

| 項目       | 値                                                                        |
| ---------- | ------------------------------------------------------------------------- |
| 期間       | YYYY-MM-DD 〜 YYYY-MM-DD                                                  |
| 機材       | GPU / CPU / RAM / OS / Driver                                             |
| ランタイム | llama.cpp bNNNN(commit)/ Ollama vX.Y / mlx-lm X.Y                         |
| 方法論     | [docs/evaluation.md](../../docs/evaluation.md) との差分があればここに書く |
| 関連実験   | [experiments/...](../)                                                    |

## 結論

1. **置き換えない。** Claude Code の実タスク(M1 / M2 × 3 回)は Qwen3.6 が 6/6 PASS・合計 126 秒に対し、Laguna は **3/6 PASS・合計 1,114 秒**。tool_use の形式は壊れていない(JSON 破損 0)が、6 本中 3 本で「Prompt is too long」か自動コンパクション失敗に至り、1 本は要求と違う実装をした
2. **敗因の大半はモデルの賢さではなく 16 GB に載るコンテキスト長。** Laguna の KV は 1 トークン約 26 KiB(Qwen3.6 の 10.6 KiB の 2.5 倍)で、IQ3_XXS でも全層 GPU で載るのは **64k まで**(128k は KV 2 GB が確保できず OOM)。Claude Code のプロンプトは system + ツール定義で 50k 前後を使うため、64k では数ターンで溢れる。Qwen3.6 は同じ 16 GB で 128k を持てる
3. **素の速度は Laguna が速い。** 短い prompt の decode は 178〜190 tok/s(Qwen3.6 150、**+19〜26%**)、prefill も +12〜52%。長文(55k)の decode は 95 vs 108 で −12%。同じ 33B-A3B 級でも、Laguna は SWA 30 層のぶん短文で軽く、グローバル 10 層の KV が大きいぶん長文で重い
4. **Poolside 自身の表(Qwen3.6 が SWE-bench Verified 73.4 vs 70.9、Terminal-Bench 51.5 vs 37.5)と整合する。** 速さを活かすなら ctx が短くて済む用途(補完・チャット)向きで、長い system prompt を持つエージェントには 16 GB では向かない
5. 想定外の発見: **Claude Code は thinking を有効で送っている**(両モデルとも全応答に thinking ブロック)。前実験の Phase 2 も同じ条件だったので比較は崩れないが、「think=false の運用」という前提は Claude Code 経由では成り立っていない。ccshim で強制的に落とす手がある

## なぜそうなるか

- **KV の大きさが ctx の天井を決める。** Laguna は 40 層のうち 30 層が 512 トークンの SWA(KV は窓ぶんだけ)、10 層がグローバル。そのグローバル層の KV が Qwen3.6 の attention 層(こちらも 10 層)より 1 層あたり 2.5 倍大きい。重みが同じ 13 GB 級でも、128k の KV が 1.4 GB(Qwen)か 3.3 GB(Laguna)かで、16 GB では載る・載らないが分かれる
- **Claude Code は毎ターン全履歴を送る。** system(13k 字)+ ツール定義(88k 字 ≈ 25k トークン)+ 会話。64k だと会話に使える余地は 30k 強で、Read や Grep の出力が数回積まれると溢れる。自動コンパクション自体にも空きが要るため、溢れた時点で回復できない
- **短文で速いのは SWA。** 512 窓の attention は長さに依らず定数コストで、30 層ぶんの attention が軽い。長文では残り 10 層のグローバル attention と大きい KV の読み出しが効いて逆転する

## 計測条件

| 項目            | 値                                                                                                                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 機材            | RTX 4080 SUPER 16 GB(電力制限 220 W)/ Ryzen 5 3600 / DDR4 32 GB                                                                                                                                  |
| llama.cpp       | llama-swap イメージ build 10920(Laguna 対応は b10087 から)                                                                                                                                       |
| Laguna          | `bartowski/Laguna-XS-2.1-GGUF:IQ3_XXS`(14.3 GB)、**ctx 65536**、KV q8_0、フル GPU(ロード後 15,733 MiB)。128k / 96k は KV 確保で OOM                                                              |
| Qwen3.6(現行 C) | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S`(13.0 GiB)、ctx 131072、KV q8_0、フル GPU(14,939 MiB)                                                                                                     |
| 共通            | `--jinja -ngl 99 --flash-attn on -np 1 --temp 0 -b 2048 -ub 512 --cache-reuse 256`                                                                                                               |
| Phase 1         | W1 = Claude Code 実ログから 64k に収まるよう抽出し直した 3 本(220〜229 KB、55〜57k トークン、ツール 111 個)、W3 要約 3 本、W4 思考 3 本。2 ブロック、各セル n=6                                  |
| Phase 2         | M1 / M2 × 3 回、`claude -p --dangerously-skip-permissions`(2.1.283)、`CLAUDE_CODE_MAX_CONTEXT_TOKENS` はサーバの ctx に合わせる(Laguna 65536 / Qwen 131072)、判定は `go build` / `go vet` / grep |

## 結果

### Phase 1: 速度(`results/speed.csv`)

| 構成               | W1 長文(55k)decode | W3 要約 decode | W4 思考 decode | W3 prefill      | VRAM       |
| ------------------ | ------------------ | -------------- | -------------- | --------------- | ---------- |
| Qwen3.6 IQ3_S 128k | 108 tok/s          | 150            | 151            | 3,659 tok/s     | 14,939 MiB |
| Laguna IQ3_XXS 64k | 95(0.88)           | **178(1.19)**  | **190(1.26)**  | **4,029(1.12)** | 15,757 MiB |

W1 の prefill は 3 本が同じ system を共有して prefix cache に乗る(primer 後は 1〜2k トークンしか計算しない)ため、平均値は比較しない。cold で計算した 1 本(W1-02、59k トークン)は Laguna 3,091 tok/s で、Qwen3.6 の前実験の cold(56k)2,956 tok/s と同等。decode は cache に依らない。Laguna は tool_use の前に短いテキストを挟む(`['text','tool_use']`)、Qwen3.6 は tool_use だけを返す。

### Phase 2: Claude Code 実タスク(`results/claude_code.csv`)

| 条件         | M1 PASS | M1 wall(s)    | M2 PASS | M2 wall(s)      | 合計    | tool_use(破損) | 失敗の内訳                                                                                                                                                                                                                               |
| ------------ | ------- | ------------- | ------- | --------------- | ------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.6 128k | 3/3     | 82 / 118 / 64 | 3/3     | 35 / 37 / 42    | **378** | 114(0)         | —                                                                                                                                                                                                                                        |
| Laguna 64k   | 1/3     | 96 / 98 / 238 | 2/3     | 437 / 124 / 121 | 1,114   | 117(0)         | M1-2: "Prompt is too long" 後にコンパイルエラー / M1-3: コンパクション後にテキストで計画だけ返し変更 0 / M2-1: 自動コンパクション失敗(65,584 > 65,536)、変更は済んでいたので PASS / M2-3: タイムアウトでなく実行時間計測を実装(要求違い) |

- Laguna は 1 ターンの出力が多い(M2-1 で 39k トークン、Qwen は 2〜3k)。thinking が長い
- prefix cache ヒット率は Laguna 0.66〜0.87、Qwen 0.64〜0.92 で同程度
- Claude Code の 1 セッションの prefill 合計は 30〜75 万トークン(cache 込み)。64k の天井に当たるかどうかが所要時間と成否を分けた

## 落とし穴

| 罠                                             | 症状                                                               | 対処                                                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Laguna の KV は Qwen3.6 の 2.5 倍              | IQ3_XXS でも 128k は OOM(`failed to allocate buffer for kv cache`) | 32k でロードして `memory.used` の差から KV/token を出し、載る ctx を逆算する                                               |
| 64k で Claude Code を回す                      | 数ターンで "Prompt is too long"、自動コンパクションも失敗          | `CLAUDE_CODE_MAX_CONTEXT_TOKENS` を ctx に合わせても救えない。ツール定義(25k トークン)を減らすか、ctx を取れるモデルを選ぶ |
| 前実験の W1 プロンプト(85〜109k)をそのまま使う | 400 `exceed_context_size_error`                                    | `extract_prompts.py --max-bytes 230000` で抽出し直す(`PRIVATE_DIR` で切替)                                                 |
| 公式 GGUF は Q4_K_M(20.3 GB)だけ               | 16 GB ではオフロード必須                                           | bartowski の imatrix 量子化(IQ3_XXS 14.3 GB)を使う                                                                         |

## 次

- ツール定義を減らした Claude Code(MCP / plugin を切る)で Laguna を再評価する。64k でも収まれば賢さの比較になる
- 短 ctx 用途(コード補完、チャット)で Laguna の decode +20% を活かす構成
- Qwen3.6 側: Claude Code が thinking を有効で送っている件を ccshim で制御し、think off の所要時間を測る

## ファイル

| パス                                 | 内容                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| `notes/00-plan.md`                   | 設計                                                                                  |
| `config/model/*.args`                | Laguna / 対照のモデル定義(前実験の sweep.sh / cc_sweep.sh に `MODEL_ARGS_DIR` で渡す) |
| `scripts/phase1.sh` / `phase2.sh`    | 速度掃引と Claude Code 実タスクのラッパ                                               |
| `scripts/aggregate_laguna.py`        | `results/speed.csv`(基準構成比)と `results/claude_code.csv`                           |
| `scripts/data/prompts_manifest.json` | 64k 用に抽出し直したプロンプトの形(本体は commit しない)                              |
| `results/raw/`                       | 生ログ(commit 対象外)                                                                 |
