# 06. 実用性評価(品質含む)

docs/02-benchmark.md は単純な速度計測を扱う。本書はそこから一歩進めて、**コーディング / 会議要約 / 長コンテキスト** の 3 用途で **品質まで含めた実用性** を比較評価するための方法論。

ハードウェア非依存。実機固有のモデル選定・実測値は [docs/04-results.md](04-results.md) の該当 GPU セクションへ。

## 1. 目的

本評価で答えたい問い:

1. CPU offload なしで GPU 上に乗る **最大モデル / 最大 ctx**
2. 重み量子化・KV 量子化が **品質と速度** に与える影響
3. **コーディング / 要約 / 長コンテキスト** での用途適性
4. 「最も実用的な構成」(用途別)

## 2. 前提

[01-setup.md](01-setup.md) / [02-benchmark.md](02-benchmark.md) / [03-tuning.md](03-tuning.md) を前提として読む(Mac は [mac/01-setup.md](mac/01-setup.md) / [mac/02-benchmark.md](mac/02-benchmark.md))。固定する設定の **理由** は [README の「評価の前提」](../README.md#評価の前提なぜこの既定値なのか) を参照。

**Linux + Ollama 環境:**

- 評価対象 GPU 1 台、CPU offload 禁止
- Ollama 最新安定版(計測時点)
- `OLLAMA_NUM_PARALLEL=1`
- Flash Attention は GPU が対応していれば ON

**Mac + mlx-lm 環境:**

- 評価対象は Apple Silicon 1 機、ユニファイドメモリ
- mlx / mlx-lm 最新安定版(計測時点)
- `max_kv_size` をリクエストごとに明示
- 「CPU offload」概念はなく、代わりに `peak_memory_mib > effective_gpu_limit - safety_margin` で NG 判定
- 計測中は LM Studio / Ollama Metal などの常駐 GPU プロセスを停止

## 3. バージョン記録(必須)

バージョン差で挙動が変わるため、評価結果には以下を必ず併記する。

**Linux + Ollama 環境:**

| 記録項目       | 取得コマンド                                        |
| -------------- | --------------------------------------------------- |
| Ollama version | `ollama --version`                                  |
| Driver / CUDA  | `nvidia-smi` ヘッダー                               |
| GPU 名         | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| Kernel         | `uname -r`                                          |
| OS             | `lsb_release -ds`                                   |
| 計測スクリプト | `git rev-parse HEAD`(scripts/bench 配下)            |
| Model digest   | `ollama show <tag>` または `ollama list` の ID 列   |

これらをまとめた `metadata.json` を計測実行ごとに `scripts/bench/results/<timestamp>/metadata.json` として保存する。

評価開始前に Ollama 最新版へ更新する:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

**Mac + mlx-lm 環境:**

| 記録項目         | 取得コマンド / 採取元                          |
| ---------------- | ---------------------------------------------- |
| mlx version      | `python -c "import mlx; ..."` / `pip show mlx` |
| mlx-lm version   | `pip show mlx-lm`                              |
| Chip / Model     | `sysctl -n machdep.cpu.brand_string hw.model`  |
| Kernel           | `uname -r`                                     |
| macOS            | `sw_vers`                                      |
| GPU 上限         | `mlx.core.device_info()`                       |
| 計測スクリプト   | `git rev-parse HEAD`(scripts/bench-mlx 配下)   |
| Model HF repo id | `mlx-community/<repo>` フル ID                 |

これらは `scripts/bench-mlx/metadata_mlx.py` が自動採取し、`results/<timestamp>/metadata.json` に保存する。

## 4. 評価用 systemd 設定 / MLX 設定

### Linux + Ollama

[01-setup.md](01-setup.md) の override.conf に対して、本評価では:

- `OLLAMA_KV_CACHE_TYPE` は記載しない(評価軸として動的に切り替える)
- `OLLAMA_CONTEXT_LENGTH` も記載しない(API リクエストで明示する)

KV タイプを変える時は計測スクリプトが override.conf を書き換えて `systemctl restart`。`sudo` が要るので評価開始時に `sudo -v` でキャッシュしておく。

### Mac + mlx-lm

systemd / 環境変数による固定は **不要**。すべてリクエスト時の引数で制御する:

- `max_kv_size`: コンテキスト上限。リクエストごとに渡す
- `kv_bits`: KV キャッシュ量子化(`4` / `8` / 未指定=非量子化)。リクエストごとに渡す
- `prefill_step_size`: prefill チャンク。既定 2048 で固定

Ollama のような「サーバ再起動」フローが無いため、KV 量子化を振った計測も同一プロセス内で完結する(別プロセスにするのは clean な peak_memory のため)。

## 5. 評価軸

| 軸              | 値                                                         | 探索方針                              |
| --------------- | ---------------------------------------------------------- | ------------------------------------- |
| モデル系列      | 対象 GPU の VRAM に乗るものを docs/04 で列挙               | 全て計測                              |
| 重み量子化      | Q4_K_M(起点)→ Q8_0(品質)→ 強量子化(大型を押し込む)→ Q5_K_M | 優先順                                |
| KV 量子化       | q8_0(主力)→ f16 / q4_0(救済)                               | 最大 ctx が目標に届かない場合のみ振る |
| Context Length  | 4k 〜 モデル既定上限                                       | **二分探索で最大値を求める**          |
| Flash Attention | ON 固定(GPU 対応時)/ OFF(非対応 GPU)                       | 固定                                  |
| Batch Size      | 1                                                          | 固定                                  |

優先順: モデルサイズ → 重み量子化 → ctx → KV 量子化 → Batch。

### 5.1 ctx 二分探索

例: 既定上限 256K、起点 64k

- 64k OK → 128k 試行
- 128k OK → 192k 試行
- 192k NG → 160k 試行
- ...

停止条件: 残 VRAM ≥ 500 MiB を確保しつつ `PROCESSOR=100% GPU`。

## 6. 探索フロー

```
モデル選択 → Q4_K_M / KV=q8_0 でロード
              ↓
         100% GPU で乗るか?
            ├─ No  → KV を q8_0 → q4_0
            │        ↓
            │        ダメなら ctx を半分
            │        ↓
            │        ダメなら重みを強量子化(Q3_K_M / IQ4_XS / INT4 等、実在タグ次第)
            │        ↓
            │        それでもダメなら対象外
            └─ Yes → ctx を二分探索で最大化
                       ↓
                  速度計測(02-benchmark.md のテンプレ)
                       ↓
                  品質評価(Needle → Summary → Coding)
                       ↓
                  ログ記録
```

## 7. 品質評価(自動採点)

**LLM-as-a-judge は使わない**。再現性を最優先に **自動採点** で統一する。データを採点可能な形に設計する。

ディレクトリ構成:

```
scripts/bench/
  ├─ data/
  │   ├─ coding/      # pytest 可能な Python タスク
  │   ├─ summary/     # 期待キーポイント・期待 TODO を仕込んだ文字起こし
  │   └─ needle/      # 長文 + needle + 質問
  ├─ run_speed.py
  ├─ run_needle.py
  ├─ run_coding.py
  ├─ run_summary.py
  └─ scorer.py
```

### 7.1 Coding(pytest 実行ベース)

タスクは「単体テスト可能な Python 関数」に限定する。

データフォーマット:

```json
{
  "task_id": "refactor_001",
  "task_type": "refactor|bugfix|test_gen|impl",
  "prompt": "...",
  "scaffold": "def target_function(...): ...",
  "tests": [
    { "input": [1, 2], "expected": 3 },
    { "input": [-1, 5], "expected": 4 }
  ]
}
```

採点フロー:

1. LLM 出力から ` ```python ` ブロックを抽出
2. 一時ファイルに書き出し `exec`
3. tests を回して pass / fail を取る
4. スコア = passing / total

抽出失敗 / syntax error は 0 点。タスク種別(refactor / bugfix / test_gen / impl)ごとに合格率を集計。

### 7.2 Summary(被覆率 + ROUGE-L)

入力: 2,000〜10,000 字程度の文字起こし。

データフォーマット:

```json
{
  "task_id": "meeting_001",
  "transcript": "...",
  "expected_keypoints": ["コスト超過", "..."],
  "expected_decisions": ["..."],
  "expected_todos": ["田中さんが X を調査", "..."]
}
```

採点:

- 被覆率: 出力に対して各 `expected_*` 要素の **部分文字列マッチ(N-gram 包含)** で被覆判定 → カテゴリ別被覆率
- ROUGE-L: 参考要約との重複(オプション、F1 を 0-1 で記録)
- 合計スコア = 各カテゴリ被覆率の平均

### 7.3 Needle In A Haystack(文字列マッチ)

採点: 出力に needle 文字列(または埋め込み ID)が含まれれば成功、それ以外は失敗。主観性ゼロ。

評価点:

- 各モデルの最大 ctx 達成値の **25% / 50% / 75% / 100%** × needle 配置 **10% / 50% / 90%** = 12 試行
- 文書本体は青空文庫 / Wikipedia ダンプ等の自然文から組み立てる

### 7.4 LLM-as-a-judge を使わない理由

- 採点者モデル選定・コスト・揺らぎが評価軸を増やす
- 自動採点で十分な解像度が出る(被覆率・pytest 合格率は数値比較可能)
- 後段で実データに差し替えた段階で主観的妥当性が要件になったら、`scorer.py` にオプションとして追加可能な構造にしておく

## 8. 合否判定

各 (モデル × 重み量子化 × KV 量子化 × ctx) の組合せに対して:

### Linux + Ollama

**OK 条件**(全て満たす):

- `ollama ps` で `100% GPU`(CPU オフロードなし)
- 残 VRAM ≥ 500 MiB
- 推論が完走(タイムアウト・OOM なし)
- Needle テスト @ ctx 50% × 位置 50% で成功
- decode tok/s ≥ 5

**NG 条件**(いずれか該当):

- OOM、または `PROCESSOR` に CPU% が混入
- 応答停止(タイムアウト)
- Needle テスト失敗
- decode tok/s < 5

### Mac + mlx-lm

**OK 条件**(全て満たす):

- `peak_memory_mib ≤ effective_gpu_limit_mib - safety_margin`(既定 safety_margin = 1024 MiB、長時間運用なら 2048 MiB 推奨)
- 推論が完走(`generate_with_metrics` が例外を投げない)
- `vm_stat` で計測中に大量 swap が発生していない
- Needle テスト @ ctx 50% × 位置 50% で成功
- generation_tps ≥ 5

**NG 条件**(いずれか該当):

- `peak_memory` が `effective_gpu_limit` を超過(macOS が swap に逃がす)
- 計測中に `vm_stat` の `pages swapped out` が顕著に増加
- mlx-lm が例外(`RuntimeError` / `MemoryError` 等)
- Needle テスト失敗
- generation_tps < 5

## 9. 出力フォーマット

実測結果は [docs/04-results.md](04-results.md) の該当 GPU セクションに追記する。

### 9.1 測定環境ヘッダー(必須)

詳細表の冒頭に。Linux 版:

```markdown
## 測定環境(YYYY-MM-DD 計測)

- GPU: <名前> (<VRAM>)
- Driver: <ver> / CUDA <ver>
- OS: <distro>, Linux <kernel>
- Ollama: v<X.Y.Z>
- 計測スクリプト: scripts/bench/ @ <git short hash>
- 計測者: <username>
```

Mac 版:

```markdown
## 測定環境(YYYY-MM-DD 計測)

- Chip: Apple M<X> (<Pro/Max/Ultra>) / <Unified Memory>GB
- Effective GPU limit: <max_recommended_working_set>GB
- OS: macOS <ver> (<build>), Darwin <kernel>
- mlx: <ver> / mlx-lm: <ver>
- 計測スクリプト: scripts/bench-mlx/ @ <git short hash>
- 計測者: <username>
```

バージョンを変えて再計測した場合は **別ブロックとして追記** し、同じテーブルには混ぜない。

### 9.2 詳細表

```
| Model (tag) | Digest | Weight | KV | Ctx | FA | VRAM Peak | Prompt tok/s | Decode tok/s | TTFT | Needle@50% | Coding | Summary | 判定 |
```

Coding は pytest 合格率(0-1)、Summary は被覆率(0-1)、Needle は成功率(0-1)。

### 9.3 採用推奨表

採用候補のみを抜粋し、用途別に推奨構成を 1〜数行で示す。docs/04 既存セクションの「採用推奨」表と同フォーマット。

### 9.4 最終分析

該当 GPU セクション末尾に文章で:

1. 最大搭載可能モデル
2. 最大搭載可能 ctx
3. 最速構成
4. 最高品質構成
5. コーディング用途最適構成
6. 会議要約用途最適構成
7. 長コンテキスト用途最適構成
8. 推奨運用構成
9. 推奨しない構成
10. トレードオフ分析(品質 vs 速度 vs VRAM vs 長文性能)

## 10. 実行手順テンプレ

任意の GPU 環境で本評価を実施する場合の標準手順:

1. Ollama を最新安定版へ更新
2. metadata 記録(§3)
3. systemd `override.conf` を §4 の状態に
4. `scripts/bench/` を最新化、合成データを生成
5. 評価対象モデルを Ollama Library で確認 → `ollama pull`
   - 対象 / 対象外と根拠を docs/04-results.md の該当 GPU セクションに記載
6. **Pass 1**: Q4_K_M / KV=q8_0 で最大 ctx を二分探索
7. **Pass 2**: 採用候補で品質評価(Needle / Coding / Summary)
8. **Pass 3**(必要なら): 強量子化や別 KV 設定で長 ctx 救済
9. 結果を docs/04-results.md に追記
10. 最終分析を執筆

## 11. リスク・既知の制約(汎用)

| 項目                      | 内容                                                | 対処                                                                                 |
| ------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Ollama バージョン依存     | バージョンで FA・KV 挙動が変わる                    | バージョンを必ず併記。バージョン跨ぎ比較は別ブロック                                 |
| thinking モデルの応答遅延 | TTFT・decode が大幅悪化                             | `think:false` / `think:true` の両方を計測。代表値は `think:false`、`true` は付記     |
| 量子化タグの実在性        | モデルによって用意される量子化が異なる              | 評価前に ollama.com/library で実在確認。対象モデルでサポートのない量子化はスコープ外 |
| 既定 ctx の上書き可否     | タグごとに既定 ctx が極端に小さいケースあり(例: 4K) | リクエストで `num_ctx` を明示、ロード後の `ollama ps` CONTEXT 列で反映確認           |
| 合成データの代表性        | 実タスクとの乖離                                    | 「合成データの初版」と明記、実データへ差し替え可能な構造にしておく                   |
| VRAM サンプリング粒度     | `nvidia-smi` 1Hz は瞬間ピークを取り逃す可能性       | 初版は 1Hz、必要なら `pynvml` で精緻化                                               |
| GPU 世代による FA 効果差  | 旧世代は FA の速度メリット限定                      | [docs/03-tuning.md](03-tuning.md) §5 参照                                            |

---

本書は方法論。実機固有の評価対象選定・実測値は [docs/04-results.md](04-results.md) の該当 GPU セクションへ。
