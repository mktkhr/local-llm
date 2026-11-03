# ローカル画像・動画生成: RTX 4080 SUPER 16 GB で最高品質を狙う(2026-11-image-video-gen)

16 GB + RAM 32 GB で動く画像生成(Qwen-Image 2.1 / FLUX.2 / Z-Image / Krea 2 / HiDream-O1)と動画生成(Wan 2.2 / HunyuanVideo 1.5 / LTX-2.5 / Kandinsky 6.0 / CogVideoX)を、品質最優先で比較し、常用構成を決める。

**進行中(Phase 1 画像の計測と採点まで完了。FLUX.3 Klein と FLUX.2 klein 9B は gated repo のためスキップ。次は Phase 2 動画)。調査は [notes/00-research.md](notes/00-research.md)、計画は [notes/01-plan.md](notes/01-plan.md)、取得済みモデルは [notes/02-models.md](notes/02-models.md)。**

| 項目       | 値                                                                        |
| ---------- | ------------------------------------------------------------------------- |
| 期間       | YYYY-MM-DD 〜 YYYY-MM-DD                                                  |
| 機材       | GPU / CPU / RAM / OS / Driver                                             |
| ランタイム | llama.cpp bNNNN(commit)/ Ollama vX.Y / mlx-lm X.Y                         |
| 方法論     | [docs/evaluation.md](../../docs/evaluation.md) との差分があればここに書く |
| 関連実験   | [experiments/...](../)                                                    |

## 結論

画像(Phase 1、1K、8 プロンプト × 3 seed、10 構成)の時点での常用候補は **Krea 2 Turbo int8** と **Qwen-Image 2.1 int8** の 2 本。どちらも日本語を含む文字・数と配置の指示・写実のすべてで破綻がなく、6〜9 s / 枚。Krea 2 は構図と写実の切れが一枚上で商用も条件付き可、Qwen-Image 2.1 は追従と文字が最も安定しているが非商用ライセンス。Z-Image Base は肌や光の写実では最良だが「頼んでいない文字を描き込む」「文字の 1〜2 字が化ける」が 3 seed 中 2 回あり、Z-Image Turbo は 3 seed すべてでりんごを 4 個描いた(数の指示に弱い)。HiDream-O1 は 5 s と最速で文字も正確だが HDR 調で写実感が落ちる。FLUX.2 [klein] 4B base は日本語全滅で今回のプロンプト群では最下位。量子化は Qwen bf16→int8、Krea fp8→int8→GGUF Q4、HiDream bf16→fp8 のいずれも同一 seed で見分けがつかず、**この用途では量子化を恐れなくてよい**。

| 構成(1K)               | 追従 | 構図 | 文字 | 写実・美的 | 破綻の少なさ | 合計 /25 | 1 枚   | ライセンス                              |
| ---------------------- | ---- | ---- | ---- | ---------- | ------------ | -------- | ------ | --------------------------------------- |
| Krea 2 Turbo int8      | 5    | 5    | 5    | 4.5        | 5            | **24.5** | 6.0 s  | Krea 2 Community(gated、条件付き商用可) |
| Qwen-Image 2.1 int8    | 5    | 4    | 5    | 4.5        | 5            | **23.5** | 9.0 s  | Qwen Research(非商用)                   |
| HiDream-O1 dev fp8     | 4    | 4    | 5    | 3          | 4            | 20       | 5.0 s  | MIT                                     |
| Z-Image Base bf16      | 4    | 4    | 3    | 5          | 3            | 19       | 31.1 s | Apache 2.0                              |
| Z-Image Turbo bf16     | 3    | 4    | 2    | 4          | 3            | 16       | 6.0 s  | Apache 2.0                              |
| FLUX.2 [klein] 4B base | 3    | 3    | 1    | 3          | 3            | 13       | 16.0 s | Apache 2.0                              |

採点は seed 1 の全 8 プロンプトと、文字(P1)・空間指示(P8)は seed 1〜3 の比較シートを見て付けた(人手 + VLM 補助の暫定値。量子化違いの構成は同一 seed でほぼ同じ画なので代表 1 つに集約)。根拠は「結果」の所見に書いた。

## なぜそうなるか

- **文字と空間指示はテキストエンコーダの差**。Qwen-Image 2.1(Qwen3-VL 8B)と Krea 2(Qwen3-VL 4B)、HiDream-O1(Gemma 4)は VLM 系の大きいエンコーダで、日本語の字形と「3 個・左から・前に」の関係を保持する。Z-Image は Qwen3-4B(テキストのみ)で、濁点・半濁点や字形の細部が落ちる。FLUX.2 [klein] 4B は Qwen3-4B でも日本語がほぼ出ず、学習データ側の問題が大きい
- **量子化で見分けがつかないのは、画像拡散が誤差に鈍いから**。int8 / fp8 の丸め誤差は各ステップのノイズ予測に乗るが、同じ seed・同じスケジュールなら軌道はほぼ同じで、差は高周波のテクスチャに留まる。LLM の量子化で起きる「論理の崩れ」に相当するものがない。GGUF Q4 でも同じ画になったのは、Krea 2 の 12B が 4 bit でも十分な冗長性を持つため
- **Z-Image Base の「頼んでいない文字」は CFG 4 の副作用**。Turbo(CFG 1)では出ず、Base(25 steps・CFG 4)で商品写真や棚田に説明文風の崩れた文字列が出る。学習データ(商品カタログ・観光写真)のキャプションを過剰に再現している。ネガティブプロンプトに "text, watermark" を入れれば抑えられる見込みだが、今回は公式テンプレート既定で比較した
- **HiDream-O1 が速いのは LCM 系サンプラー + 小さい DiT**。28 steps でも 1 ステップが軽く、fp8 なら VRAM 11 GB で済む。代わりに彩度とコントラストが高く、写実が「加工済み」に見える

## 計測条件

### Phase 0: 環境(2026-11-03)

- ComfyUI を Docker(`yanwk/comfyui-boot:cu130-slim-v2`)で起動。モデル・ノード・出力はホスト `~/comfy-data/` に bind mount。詳細と落とし穴は [notes/02-models.md](notes/02-models.md)
- 生成と計測は `scripts/comfy_api.py`(API 形式のグラフを `/prompt` に投げ、`/history` で完了を待ち、`nvidia-smi` を 1 秒間隔で採取)。同じプロンプト・同じシードは ComfyUI がキャッシュして 1 秒で返すので、時間を測るときはシードを変える
- GPU の電力制限は計測時に `nvidia-smi --query-gpu=power.limit` で記録する(Phase 0 のスモークは 320 W)

### スモーク: Z-Image Turbo(bf16、8 ステップ、res_multistep / simple、shift 3)

| 条件                              | 所要      | VRAM ピーク    | 平均 / 最大電力  |
| --------------------------------- | --------- | -------------- | ---------------- |
| 1024×1024、初回(モデルロード込み) | 53.8 s    | 15,589 MiB     | 63 / 290 W       |
| 1024×1024、2 回目以降             | **6.0 s** | 15,435 MiB     | 222〜247 / 288 W |
| 2048×2048                         | 30.1 s    | **15,915 MiB** | 271 / 298 W      |

- bf16 の DiT 12.3 GB + エンコーダ 8 GB は同時には載らないが、ComfyUI が順にロード / 退避して 16 GB で回る。ただし 2K は残り 460 MiB で限界。エンコーダを fp8 にするか DiT を fp8 / GGUF にすると余裕ができる
- 文字: チョークボードの「焼きたて パン」が「焼きたて **バ**ン」(半濁点が濁点に)、「OPEN 7:00-19:00」は正確。日本語の濁点・半濁点は文字生成の見どころになる
- 構図・写実は 8 ステップとして十分(作例は `results/raw/img/`、commit 対象外)

### Phase 1: 画像(2026-11-03〜)

- 実行は ComfyUI 公式テンプレート(`scripts/data/templates/`、MIT)を `scripts/tpl2api.py` で API 形式に変換し、`scripts/run_matrix.py` で構成 × 8 プロンプト × 3 seed(1, 2, 3)を回す。変換の正しさは、手書きグラフとテンプレート経由で同一 seed の出力が **ピクセル単位で一致** することで確認した(PNG の IDAT ハッシュ一致)
- 公式テンプレートの推奨設定をそのまま使う。変えるのはモデルファイル名・プロンプト・seed・解像度(1024×1024 相当)だけ。テンプレート同梱のプロンプト拡張(LLM による書き換え)は全構成で切る
- 電力制限は 320 W のまま(品質は seed 固定で決定的なので電力に依らない。時間の比較表も 320 W のまま。220 W での再計測はしないと決めた)
- 構成の一覧は `scripts/data/phase1_models.json`、生データは `results/raw/phase1/runs.jsonl`(commit 対象外)、集計は `results/phase1-summary.md`、seed 1 の比較シートは `results/phase1-sheets/`

## 結果

### Phase 1(10 構成 × 24 枚)

| 構成                        | 設定(テンプレート既定)                                                 | 1 枚あたり(2 回目以降 中央値) | VRAM ピーク | 平均電力 |
| --------------------------- | ---------------------------------------------------------------------- | ----------------------------- | ----------- | -------- |
| Z-Image Turbo bf16          | 8 steps、cfg 1、res_multistep / simple                                 | **6.0 s**                     | 15.9 GB     | 232 W    |
| Z-Image Base bf16           | 25 steps、cfg 4、res_multistep / simple                                | 31.1 s                        | 15.9 GB     | 289 W    |
| Qwen-Image 2.1 int8         | 25 steps、cfg 1、euler / simple、QwenImage21Cache                      | 9.0 s                         | 15.7 GB     | 277 W    |
| Qwen-Image 2.1 bf16         | 同上(DiT 14.2 GB をオフロードしながら)                                 | 19.0 s                        | 15.8 GB     | 280 W    |
| Krea 2 Turbo fp8            | 8 steps、cfg 1、euler / simple                                         | 9.0 s                         | 15.8 GB     | 270 W    |
| Krea 2 Turbo int8           | 同上                                                                   | **6.0 s**                     | 15.9 GB     | 258 W    |
| Krea 2 Turbo GGUF Q4_K_M    | 同上(ComfyUI-GGUF)                                                     | 16.0 s                        | **14.0 GB** | 266 W    |
| HiDream-O1 dev fp8          | 28 steps、cfg 1、LCM サンプラー、noise_scale 7.6、既定 2048 を 1024 に | **5.0 s**                     | **11.4 GB** | 263 W    |
| HiDream-O1 dev bf16         | 同上(checkpoint 16 GB をオフロード)                                    | 6.0 s                         | 15.6 GB     | 252 W    |
| FLUX.2 [klein] 4B base bf16 | 20 steps、cfg 5、euler / Flux2Scheduler                                | 16.0 s                        | 15.7 GB     | 292 W    |

初回(モデル読込込み)は 56〜186 s だが、この計測中は裏でモデルのダウンロードが走っていて読み込みが遅かった(後述の罠)。

seed 1 の比較シート([results/phase1-sheets/](results/phase1-sheets/))を見た暫定所見。人手 5 段階の本評価は全構成が揃ってから行う。

- **文字(P1 ポスター、3 seed)**: Qwen-Image 2.1 / Krea 2 / HiDream-O1 は 3 seed とも日本語見出し・英語副題・日付の 3 行が正確。Z-Image Turbo は 3 seed とも見出しが化ける(「フェ尽」「フェマ」「フ尽」)。Z-Image Base は見出しは 3 回中 2 回正確だが、副題が崩れる(seed 1)・指示していない文字列を足す(seed 3)で完全正解は 1 回。FLUX.2 4B は 3 seed とも見出しが読めない。Phase 0 の「バン」もそうだが、Z-Image 系は日本語の 1〜2 文字が化ける
- **複雑構図(P2 ラーメン屋 3 人)**: 4 構成とも 3 人と役割(料理人・新聞を読む男性・食べる女性)を再現。Z-Image Base がフィルム調の自然な画で最も良く、Qwen は暗めでシネマ寄り
- **風景(P4 棚田)**: Qwen が良い。Z-Image は田んぼの形が整いすぎていて現実離れしている(ユーザー所見)。「きれいに整った幾何」はこの用途では減点
- **写実ポートレート(P3)**: いずれも破綻なし。Z-Image Base の肌の質感が最も細かく「無加工」の指示に合う。Qwen / Krea はクリーンで整った写り。HiDream は頼んでいないニット帽を被せ笑顔にする(追従の減点)、FLUX.2 4B は桟橋の文脈が消えてスタジオ写真になる
- **商品(P5)・イラスト(P6)・建築(P7)**: 商品は Krea が「オーク板・豆・グレー背景」をすべて満たし最もカタログらしい。FLUX.2 4B は板が別物になり、HiDream は頼んでいない湯気とコーヒーを足す。イラストは全構成とも成立し、指定パレット(ネイビー・クリーム・コーラル)への忠実さは Krea > Qwen > 他。建築は Z-Image Turbo が片持ち屋根と二点透視を最も忠実に描き、Krea も良い。Qwen / Z-Image Base は建物の一部しか写らない
- **商品(P5 マグカップ)**: Qwen が最も商品写真らしい。Z-Image Base は指示していない文字(商品説明風の崩れた文字列)を上部に描き込んだ
- **空間指示(P8 りんご 3 個とタグ「No.2」)**: Z-Image Turbo は **3 seed すべてでりんごを 4 個** 描いた。Z-Image Base / Qwen / Krea / HiDream は 3 seed とも 3 個でタグも正確。FLUX.2 4B は seed 1 で赤りんごの色が混ざりタグが「No´ 2」(3 seed 中 2 回成功)
- **Krea 2 Turbo**: 文字は 3 行とも正確(見出しは 2 行に折り返し)。ラーメン屋・棚田・りんごのいずれも写実で、棚田は霧と不揃いな畦が自然。fp8 / int8 / GGUF Q4_K_M の 3 つは同一 seed で **ほぼ見分けがつかない**。int8 が最速(6 s)、GGUF は 2 GB 軽いが 2.5 倍遅い
- **HiDream-O1 dev**: 1K なら 5 s と最速で文字も正確。ただし写実系は HDR 調で彩度が高く、ラーメン屋では看板の漢字が崩れ、顔がやや作り物っぽい。棚田も劇画調で現実感は Qwen / Krea に劣る。fp8 と bf16 は同一 seed でほぼ同一
- **FLUX.2 [klein] 4B base**: 今回の 10 構成で最も弱い。日本語見出しは「豪の鶴車フェア」と全滅、英語も「Auturm Bookook」。棚田は畦が整いすぎて緑一色、りんごは赤の色が混ざりタグも「No´ 2」。ラーメン屋の構図だけは良い
- **量子化(Qwen-Image 2.1 bf16 vs int8、同一 seed)**: 8 プロンプトすべてで構図・文字・色がほぼ同一。差は細部のテクスチャ程度で、シートの縮小では見分けがつかない。int8 は bf16 の **半分の時間**(9 s vs 19 s)なので、Qwen-Image 2.1 は int8 で使ってよい

### Phase 1: 2K(2048×2048、上位 4 構成 × P1〜P4、seed 1)

| 構成(2K)            | 1 枚(2 回目以降) | VRAM ピーク | 所見                                                                                       |
| ------------------- | ---------------- | ----------- | ------------------------------------------------------------------------------------------ |
| Krea 2 Turbo int8   | **33 s**         | 15.9 GB     | 文字正確。ラーメン屋・棚田とも 1K より描き込みが増え、霧の棚田は 4 構成で最も雰囲気がある  |
| Qwen-Image 2.1 int8 | 60 s             | 15.8 GB     | 文字正確(明朝系の書体になる)。等倍で見ると肌の毛穴・産毛まで自然で、写実のディテールは最良 |
| HiDream-O1 dev fp8  | 19 s             | **9.4 GB**  | 文字正確だが看板の漢字は崩れる。彩度の高い HDR 調は 2K でも同じ                            |
| Z-Image Base bf16   | 169 s            | 15.9 GB     | 2K では英語副題が「Book                                                                    | Fair」と崩れ余計な記号も入る。棚田は 2K でも畦が整いすぎ。16 GB では時間も見合わない |

1K → 2K で時間は 3.7〜5.5 倍(画素 4 倍)。VRAM はどれも 16 GB に収まる(HiDream は余裕あり)。生データは `results/raw/phase1/runs-2k.jsonl`、集計は `results/phase1-2k-summary.md`。

## 落とし穴

| 罠                                     | 症状                                                                                                                                             | 対処                                                                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RAM 32 GB にモデルが溜まる             | ComfyUI は退避したモデルを CPU RAM に残す。構成を切り替えると前のモデルが残ったまま次を読み、スワップして数分固まる                              | 構成の切り替え時に `POST /free {"unload_models": true, "free_memory": true}` を呼ぶ(`run_matrix.py` が自動で呼ぶ)。RAM 25 GB → 4 GB に戻る                  |
| ダウンロードと同時に計測しない         | `hf download` の書き込み(〜380 MB/s)でモデル読み込みが 10 MB/s まで落ち、初回が 3 分以上かかる                                                   | 計測中はダウンロードを `kill -STOP`、終わったら `kill -CONT`。`pkill -f` のパターンが自分の ssh シェルにも当たるので `bash -s` のヒアドキュメントで実行する |
| ComfyUI の出力ファイル名               | `filename_prefix` に `_00001_` の連番が付く。同じ prefix で再実行すると番号がずれる                                                              | 取得時に `prompt_seed.png` へ改名して保存する(`run_matrix.py`)                                                                                              |
| サブグラフ入りテンプレートの widget 値 | サブグラフノードの `widgets_values` はノード側の inputs 順ではなく **サブグラフ定義の inputs 順**。ずれると steps に 4096 が入る等の無言の誤動作 | `tpl2api.py` は定義順で割り当て、テストで公式テンプレートの既定値(steps 25 / cfg 1 など)を固定している                                                      |

## 次

- FLUX.3 Klein(distilled / base)と FLUX.2 klein 9B fp8 は HF の gated repo(`black-forest-labs/flux-3-image-klein-comfy`、`black-forest-labs/FLUX.2-klein-base-9b-fp8`)。ライセンス承認済みの HF トークンを GPU 機に置いてから取得し、同じ行列で回す。FLUX.2 klein 4B base は取得済みで実行待ち
- Z-Image Base にネガティブプロンプト(text, watermark)を入れて「頼んでいない文字」が消えるか 1 回だけ試す
- Krea 2 のプロンプト拡張(同梱の Qwen3-VL 4B で書き換え)を ON にしたときの変化を最良モデルで 1 回見る
- Phase 2(動画)の T2V / I2V、Phase 3(画像 → 動画)

## ファイル

| パス       | 内容                                                              |
| ---------- | ----------------------------------------------------------------- |
| `notes/`   | 調査メモ・手順書・途中経過(複数可、番号 prefix で順序づけ)        |
| `scripts/` | この実験専用の計測スクリプト。共通 harness は `tools/` を使う     |
| `config/`  | この実験専用の llama-swap entry / compose / systemd override など |
| `results/` | 集計済みの csv / jsonl。生ログは `results/raw/`(gitignore)        |
