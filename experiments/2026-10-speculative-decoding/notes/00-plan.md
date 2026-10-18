# 計画: Qwen3.6-35B-A3B の投機デコード(MTP / draft / ngram)を 16 GB で計測する

作成: 2026-10-18。実装前の設計書。計測が終わったら結論は [../README.md](../README.md) に書き、本書は手順と判断理由の記録として残す。

## 1. 問い

1. **MTP は 16 GB 構成でも効くか。** Qwen3.6-35B-A3B の MTP ヘッド(`--spec-type draft-mtp`)は、RTX 3090 / 24 GB / フル GPU で decode +22.7% の先行報告がある。本機の採用構成は `--n-cpu-moe 10` で MoE 層の一部を CPU に逃がしている。投機の検証バッチ(n+1 トークン)は CPU 側 expert も通るため、prefill と同じ理由で Ryzen 5 3600 の演算能力に引っ張られて利得が消える可能性がある。**フル GPU 構成(C 系)とオフロード構成(B 系)で利得がどう違うか**を測る
2. **Claude Code の実ワークロードで受理率はいくつか。** tool_use の JSON、既存コードの書き換え(Edit / Write)、日本語要約で受理率が変わるはず。特に `ngram-simple` は「ソースコードの書き換え向き」と公式が明記しており、Claude Code のコード編集で発火するかを見る
3. **プレフィックスキャッシュと両立するか。** 採用構成は `--cache-reuse 256` でターン間ヒット率 81.7% を得ている。hybrid(GDN)モデルは context checkpoint が常に無効化される報告(llama.cpp issue #24055)があり、MTP 併用でキャッシュが効かなくなれば TTFT が悪化して正味で負ける
4. **出力は壊れないか。** 先行報告では投機デコードが出力保存的でない(A11)。tool_use の構造(JSON 破損 / 生トークン露出)が増えないことを、既存の集計観点で確認する

## 2. 勝ち条件

**主力構成は C(`ccc`、UD-IQ3_S、フル GPU、128k、KV q8_0)。** 全層 VRAM で TTFT が最速なため日常的に使っている(2026-10-18 確認)。よって比較の基準・Phase 2 の対象・deploy 反映先は **C 系**とし、B 系(ncmoe)は Phase 1 で「オフロード下で利得が残るか」を見るための対照に留める。

- **主指標**: Claude Code 実タスク(M1 / M2、[2026-08-claude-code-backend](../../2026-08-claude-code-backend/README.md) と同一)で、**M1 + M2 の合計所要時間が「現行 deploy 構成」より 15% 以上短く、両課題とも現行を下回らず、tool_use 破損 0**。これが満たせなければ `deploy/` には入れない
  - 比較対象は 2 つ置く。**(a) 現行 deploy 構成**(通常 GGUF + `llama-swap:cuda` 同梱 llama.cpp + 現行 ncmoe)が deploy 判断の基準、**(b) 同じ MTP GGUF で spec=none** が投機そのものの効果を分離する基準。MTP GGUF は 0.5〜1.6 GB 大きく ncmoe が増えうるため、(b) に勝っても (a) に負けることがありうる
  - 15% は run 間ばらつき 9.4 pp(先行事例 A16)を上回る幅として置いた閾値。n=2 で判定し、境界なら n を増やす
- **副指標**: 合成ワークロードでの decode tok/s × 受理率 × 構成の曲線。記事化の素材

## 3. 先行事例との差分

| 項目         | thc1006(RTX 3090)                                         | 本実験                                                 |
| ------------ | --------------------------------------------------------- | ------------------------------------------------------ |
| VRAM         | 24 GB、フル GPU、UD-Q4_K_XL                               | **16 GB、`--n-cpu-moe` 有り / 無しの両方**             |
| ワークロード | 固定 10 プロンプト、greedy                                | **Claude Code の実ログから抽出**した 4 系統 + 実タスク |
| prefix cache | 言及なし                                                  | **`--cache-reuse 256` 併用で cache_n を記録**          |
| 結果         | MTP n=2 +22.7%、外部 draft 0.8B は −74%、ngram 系は効かず | 同条件で再現するか、オフロードで消えるかを確認         |

先行事例から引き継ぐ教訓:

- **同一設定でも run 間で最大 9.4 pp ずれる**(A16)。baseline を各ブロックに必ず含め、対にして比較する
- **外部 draft モデルは vocab 一致が必須**。`qwen3:0.6b` は黙って失敗し、Qwen3.5-0.8B(vocab 248320)なら動く。ただし結果は −74% なので本実験では**優先度を下げ、Phase 1 の 1 設定だけ**測って打ち切る
- **思考 off の比較は出力長で交絡する**(A17)。`n_predict` を固定して打ち切る
- 結果は **llama.cpp の build に強く依存**する。build hash を全レコードに残す

## 4. 固定するもの / 動かすもの

### 固定

| 項目         | 値                                                                                                                                                 | 理由                                                                                                                                                                                                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| モデル       | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`(MTP ヘッド同梱)                                                                                                 | 通常 GGUF には MTP ヘッドが無い。baseline も同じファイルで測り、重みの差を消す                                                                                                                         |
| llama.cpp    | `ghcr.io/mostlygeek/llama-swap:cuda` の現行イメージ(llama.cpp **build 10920** / commit `eafe15a5e`、digest `sha256:199a9dd3…`)をダイジェストでピン | 2026-10-18 に GPU 機で確認済み: `--spec-type` に `draft-mtp` / `ngram-simple` / `ngram-mod` を含み、`--spec-draft-n-max` も存在する。deploy と同じ llama.cpp で測れるため Phase 4 の再確認が不要になる |
| 共通フラグ   | `-np 1 -fa on --jinja --no-mmproj -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --cache-reuse 256 --metrics`                                                 | 採用構成と同一。MTP は `-np > 1` と `--mmproj` に非対応                                                                                                                                                |
| サンプリング | `--temp 0`(greedy)、`think: false`、`n_predict` 固定(W4 のみ think on)                                                                             | 出力長・ばらつきの交絡を消す                                                                                                                                                                           |
| ctx          | 131072(Phase 3 のみ 262144)                                                                                                                        | 採用構成と同一                                                                                                                                                                                         |

### 動かすもの

| 軸           | 水準                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 構成         | **C'**: UD-IQ3_XXS、フル GPU、128k(主力 C 相当。**Phase 0 で確定**: MTP 版 GGUF は MTP ヘッド分 +1,592 MiB で、IQ3_S は 128k でも 96k でも `--n-cpu-moe 2` でも KV 1,360 MiB が確保できず OOM。IQ3_XXS はロード後 15,882 MiB / 56k prefill 時ピーク 15,914 MiB で載る。基準の 15.5 GB は超えるが、KV と compute buffer はロード時に確保済みで推論中の増分は 32 MiB だったため採用)/ **B'**: UD-IQ4_XS、ncmoe 再確定(対照) |
| spec         | `none` / `draft-mtp` n-max=1,2,3 / `ngram-simple` / `ngram-mod` / `draft-mtp,ngram-simple`(併用)/ `draft-simple`(Qwen3.5-0.8B、1 設定のみ)                                                                                                                                                                                                                                                                                |
| ワークロード | W1 tool_use 多(Claude Code 実ログの system + tools + 短指示)/ W2 コード書き換え(既存ファイルを渡して部分修正を書かせる)/ W3 日本語要約(`tools/bench/data/summary`)/ W4 思考(think on、300 トークンで打ち切り)                                                                                                                                                                                                             |

組み合わせは 2 構成 × 7 spec × 4 ワークロード × 各 3 プロンプト × 3 ブロック。1 本 10〜30 秒として**概算 3〜4 時間**。spec 設定の切替はサーバ再起動を伴うため、1 ブロック = 「spec 設定ごとにサーバを起動し、全プロンプトをシャッフルして流す」とし、ブロック間で spec 設定の順序をローテーションする。

## 5. 計測方法

| 指標           | 取得元                                                                                                                                                                                                                                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| decode tok/s   | レスポンスの `timings.predicted_per_second` / `predicted_n` / `predicted_ms`                                                                                                                                                                                                                                                                                       |
| TTFT / prefill | `timings.prompt_ms` / `prompt_n` / **`cache_n`**(prefix cache のヒット量)                                                                                                                                                                                                                                                                                          |
| 受理率         | `/metrics` の `llamacpp:spec_decode_num_draft_tokens_total` / `_num_accepted_tokens_total` / `_num_drafts_total` / `_num_accepted_tokens_per_pos_total{position}` を**リクエスト前後で差分**。サーバログの `draft acceptance rate = ...` と突き合わせる                                                                                                            |
| 発火率         | drafted / generated(投機がそもそも走った割合。ngram 系はここが小さい)                                                                                                                                                                                                                                                                                              |
| VRAM           | `nvidia-smi` ピーク([tools/bench/vram.py](../../../tools/bench/vram.py) を流用)                                                                                                                                                                                                                                                                                    |
| 電力           | `nvidia-smi --query-gpu=power.draw`(追加コストゼロ。後続の電力リミット実験の種にする)                                                                                                                                                                                                                                                                              |
| 決定性         | baseline と spec の出力トークン列の一致率(A11 の確認)                                                                                                                                                                                                                                                                                                              |
| エンドポイント | W1 / W2 は ccshim ログの `/v1/messages` 形式を**そのまま** llama-server の `/v1/messages` に投げる(実運用と同じトークン列にする)。W3 / W4 は `/v1/chat/completions`。tok/s と受理率は `/metrics` の差分(`tokens_predicted_total` / `_seconds_total` と spec カウンタ)から算出し、エンドポイントの `timings` 有無に依存しない。`timings` が取れる場合は照合用に併記 |
| build          | `/props` の `build_info` を全レコードに保存                                                                                                                                                                                                                                                                                                                        |

`timings` に draft 関連のフィールドが入るかは build 依存で、公式 README には記載がない。Phase 0 で実レスポンスを確認し、無ければ `/metrics` 差分だけを正とする。

## 6. Phase

### Phase 0: 環境確認(GPU 機、30 分)

2026-10-18 時点で確認済み(`ssh gpu`。`~/.ssh/config` の `gpu` は鍵が `ems/id_rsa` を指しているが実際に通るのは `~/.ssh/id_rsa`):

- llama-swap イメージ同梱の llama-server は build 10920。`--spec-type draft-mtp,ngram-simple,...` と `--spec-draft-n-max` が使える(上記 1 は完了)
- HF キャッシュ(docker volume `llama_llama-cache`、220 GB)に `unsloth/Qwen3.6-35B-A3B-GGUF`(50.9 GB、通常版)はあるが **MTP 版と Qwen3.5-0.8B は未取得**。上記 2 は要実施(UD-IQ3_S 15.3 GB + UD-IQ4_XS 18.2 GB)
- ccshim のログは B: 183 / C: 1,854 / D: 241 リクエスト分(2026-08-17 〜 09-18)が docker volume に残っている。W1 / W2 の抽出元として十分
- 本番スタック(llama-swap / ccshim)は 3 週間停止中。GPU は空き(136 MiB 使用)。開始時に止める手順は不要だが、終了後の再開が要るかを確認する
- **前回の M1 / M2 ハーネス(プロンプト原文・合否判定・tool_use 集計スクリプト)は残っていない**(Claude Code の scratchpad 上にあり消失)。`mktkhr/rag` の `bench/v{A,B,C}-M{0..3}` と `ccd-bench-{da,db}-m{1,2}` ブランチに**成果物の diff だけ**残っている。Phase 2 の `cc_task.sh` は [2026-08-claude-code-backend §8](../../2026-08-claude-code-backend/README.md) の課題定義とこれらの diff を参照して再構築する。過去数値との直接比較はしない

1. ピンするイメージで `llama-server --help | grep -A1 spec-type` を実行し、`draft-mtp` / `ngram-simple` / `--spec-draft-n-max` が存在することを確認
2. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` の UD-IQ3_S(15.3 GB)/ UD-IQ4_XS(18.2 GB)を取得。通常版より 0.5〜1.6 GB 大きい
3. [2026-08-claude-code-backend §5](../../2026-08-claude-code-backend/README.md) と同じ手法で VRAM 境界を再確定(1 秒間隔の `nvidia-smi` で実推論 1 本のピークを踏ませる)。**C' がフル GPU に入らない場合はここで判断**(ncmoe 最小 / IQ3_XXS / ctx 65536 のどれか)
4. `/completion` を 1 本投げ、`timings` と `/metrics` の差分が取れることを確認
5. ccshim のログ(`ccshim-log` volume の `NNN-req.json`)から W1 / W2 のプロンプトを抽出。Claude Code の system prompt とツール定義全文を含むため**抽出物は commit しない**(`scripts/data/private/`、gitignore)。抽出スクリプトと、プロンプトの形(トークン数・ツール数・ターン数)の一覧だけを残す
6. 実験中は本番スタックを止める(`docker compose -f deploy/llama-swap/compose.yml stop`)。`sweep.sh` の冒頭で停止、末尾で再開する。VRAM の判定基準は [2026-08-claude-code-backend §5](../../2026-08-claude-code-backend/README.md) と同じ **15.5 GB 以下**

### Phase 1: 合成ワークロード掃引(3〜4 時間、無人)

`scripts/sweep.sh` でブロック設計のまま回す。出力は `results/raw/*.jsonl`(gitignore)→ `scripts/aggregate.py` で `results/spec_sweep.csv` に集計。

prefix cache の扱い(Phase 0 で修正): W1 の 3 本は別セッション由来で共通 prefix が無く、`-np 1` では直前のプロンプトの KV しか残らないため「W1 を全て warm で測る」は成立しない。Phase 1 の W1 / W2 は **cold prefill** として測り、warm prefix の挙動(問い 3)は Phase 2 の実セッションの `cache_n` で判定する。primer(W1 先頭、n_predict 8)は cold TTFT の参考値として `primer-<TS>.jsonl` に残す。

比較対象 (a) 現行 deploy 構成は、本番スタックを一時再開して同じプロンプトを 1 ブロックだけ流して取る。

判定: 構成 × ワークロードごとに「baseline 比 tok/s」「受理率」「発火率」を表にし、**主指標の候補(Phase 2 に進める設定)を 1〜2 個**に絞る。

### Phase 2: Claude Code 実タスク(1〜2 時間)

**C' 系**の候補設定 vs baseline(現行 C と、同一 MTP GGUF の spec=none)で M1 / M2 を各 n=2。実験用 ccshim(ポートを変えた別インスタンス)を実験サーバに向け、`claude --dangerously-skip-permissions` で無人実行。記録するもの: 所要時間、ターン数、tool_use 総数、JSON 破損、生トークン露出、セッション合計の受理率(`/metrics` 差分)、各ターンの `cache_n`(prefix cache が維持されているか)。

### Phase 3: 256k / KV q4_0

主力は C(128k)なので任意。D(`ccd`、256k / KV q4_0)は長文時のみ使うため、Phase 2 で C' に利得が出た場合に W1 / W2 だけ追試する。

D 構成(`qwen36-35b-iq3s-256k`)に MTP を足して Phase 1 の W1 / W2 だけ再測。KV q4_0 で受理率が落ちるかを見る。

### Phase 4: 判定と反映

主指標を満たした設定だけ [deploy/llama-swap/config.yaml](../../../deploy/llama-swap/config.yaml) に反映する。その際 `llama-swap:cuda` 同梱の llama.cpp で同じフラグが通るかを再確認する(通らなければ deploy 側のイメージ更新が先)。

## 7. 成果物

| パス                         | 内容                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `config/compose.yml`         | 公式 llama.cpp イメージをピンした実験用 llama-server(ポート 18081)+ 実験用 ccshim(11439) |
| `config/*.args`              | spec 設定ごとの llama-server 引数(1 設定 1 ファイル)                                     |
| `scripts/extract_prompts.py` | ccshim ログから W1 / W2 を抽出。出力は `scripts/data/private/`(gitignore)                |
| `scripts/cc_task.sh`         | Phase 2 の M1 / M2 実行と合否判定、ccshim ログからの tool_use 集計(前回の手順を再構築)   |
| `scripts/spec_bench.py`      | 1 リクエストの計測(timings + /metrics 差分 + VRAM + 電力)→ jsonl 1 行                    |
| `scripts/sweep.sh`           | ブロック設計で構成 × spec × ワークロードを回す。サーバ起動 / 停止込み                    |
| `scripts/aggregate.py`       | `results/raw/*.jsonl` → `results/spec_sweep.csv` / `results/claude_code.csv`             |
| `results/`                   | 集計済み csv のみ commit                                                                 |
| `README.md`                  | 結論(1 枚)                                                                               |

共通 harness([tools/bench](../../../tools/bench/README.md))は Ollama 前提のため、HTTP クライアント部分は llama-server 用に書き直す。`vram.py` と `metadata.py` の採取ロジックは流用する。

## 8. リスクと未知

| リスク                                                   | 対処                                                                                                                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MTP GGUF が通常版より大きく、C' がフル GPU に収まらない  | Phase 0 で確定。収まらなければ「フル GPU vs オフロード」の比較軸を ncmoe 1〜2 vs 10 に置き換える                                                                        |
| hybrid モデルで prefix cache が効かなくなる              | `cache_n` を全リクエストで記録。Phase 2 でターン間ヒット率を baseline と比較                                                                                            |
| build によって投機の挙動・フラグ名が違う                 | 公式イメージをタグでピン。deploy 側に反映する前に再確認                                                                                                                 |
| 受理カウンタの解釈違い(A1 の「100% 受理」はカウンタの癖) | `/metrics` 差分とサーバログの両方で取り、drafted / generated も併記する                                                                                                 |
| 外部 draft(0.8B)が起動時に abort する(A6)                | 1 設定だけ試し、失敗したら記録して捨てる                                                                                                                                |
| M1 / M2 の再構築で判定基準が前回とずれる                 | 判定は機械的なもの(`go build` / `go vet` / ルート登録 / `context.WithTimeout` の有無 / 変更 0 件は FAIL)に限定し、`cc_task.sh` に固定する。前回の所要時間とは比較しない |
| 電力・温度でブロック間がずれる                           | ブロック内に baseline を含め、ブロック間は順序ローテーション。`power.draw` と `temperature.gpu` を記録                                                                  |

## 9. 見送るもの

- **DFlash / EAGLE3 / DSpark**: 専用 draft チェックポイントの変換が必要。MTP で傾向が掴めてから
- **`-np 2` 以上**: MTP 非対応。別実験(並列スループット)で扱う
- **vLLM との比較**: 別実験
