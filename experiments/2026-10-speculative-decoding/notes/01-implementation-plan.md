# 投機デコード計測ツール 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** [00-plan.md](00-plan.md) の Phase 1(合成ワークロード掃引)と Phase 2(Claude Code 実タスク)を無人で回し、`results/*.csv` に集計できる一式(config / scripts / tests)を作る。

**Architecture:** 実験用 llama-server(llama-swap と同じイメージをダイジェストでピン)と実験用 ccshim を `config/compose.yml` で立て、`scripts/spec_bench.py` が 1 リクエストごとに `/metrics` の差分・`timings`・nvidia-smi を 1 行の JSONL に落とす。`scripts/sweep.sh` がサーバの起動 / 停止込みでブロック設計を回し、`scripts/aggregate.py` が JSONL をブロック内対比較で CSV にする。Phase 2 は `scripts/cc_task.sh` が `claude -p` を無人実行し、`scripts/cc_metrics.py` が ccshim ログから tool_use の健全性とセッション合計の受理率を出す。

**Tech Stack:** Python 3.12+ 標準ライブラリのみ(GPU 機は Python 3.14、uv なし。`urllib` / `json` / `unittest`)、bash(GPU 機は GNU coreutils)、docker compose、nvidia-smi。テストは `python3 -m unittest`。

**Spec:** [00-plan.md](00-plan.md)

## Global Constraints

- 実行場所は **GPU 機**(`ssh gpu`、リポジトリは `~/ghq/github.com/mktkhr/local-llm`)。Phase 2 も GPU 機で回す(`~/.local/bin/claude`、`~/.asdf/shims` の go / node が必要。非対話シェルでは PATH に含まれないので `cc_task.sh` が自前で足す)
- llama-server イメージは `ghcr.io/mostlygeek/llama-swap@sha256:199a9dd3e30b77fda1b9861eafe61722b665918f6d9348dabd8a30c3bc4ebb33`(llama.cpp build 10920)。バイナリは `/app/llama-server`
- 共通フラグ: `--host 0.0.0.0 --port 18081 --jinja -ngl 99 --flash-attn on -np 1 --metrics --temp 0`
- モデル: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`(`UD-IQ3_S` = C'、`UD-IQ4_XS` = B')。比較対象 (a) は `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S`
- HF キャッシュは docker volume `llama_llama-cache`(external)を `/root/.cache` にマウント
- 受理率・tok/s は `/metrics` の差分から算出する。使うカウンタ名(接頭辞 `llamacpp:`): `prompt_tokens_total` / `prompt_tokens_cached_total` / `prompt_seconds_total` / `tokens_predicted_total` / `tokens_predicted_seconds_total` / `n_decode_total` / `spec_decode_num_draft_tokens_total` / `spec_decode_num_accepted_tokens_total` / `spec_decode_num_drafts_total` / `spec_decode_num_accepted_tokens_per_pos_total{position="N"}`
- `n_predict` 既定: W1 = 256 / W2 = 768 / W3 = 512 / W4 = 300(think on)
- 抽出したプロンプト(`scripts/data/private/`)と生ログ(`results/raw/`)は commit しない。集計 csv と `scripts/data/prompts_manifest.json`(形だけ)は commit する
- 外部ネットワーク: GPU 機から HF への GGUF 取得のみ(約 34 GB)

## Review Focus

1. `/metrics` に `spec_decode_*` が**存在しない**(spec=none のサーバ)→ `metrics_delta` は 0 扱いで落ちないこと(Task 2 のテストで固定)
2. リクエスト失敗(HTTP 4xx/5xx、タイムアウト、接続拒否)→ JSONL に `status` と `error` を残して次へ進むこと(Task 3 のテストで固定)
3. `/v1/messages` 応答に `timings` が無い build でも計測が成立すること(Task 2: `extract_timings` は `None` を返し、aggregate は metrics 由来の列だけで動く)
4. SSE の `input_json_delta` が複数チャンクに割れる → 連結してから JSON 判定すること。部分断片を破損と誤判定しない(Task 7 のテストで固定)
5. サーバ起動待ちのタイムアウト(初回 GGUF 取得で 30 分超)→ `sweep.sh` は `docker logs` の末尾を出して非 0 終了し、以降のブロックを続けない(Task 5 のドライランと手順で確認)

---

### Task 1: 実験用 compose と引数ファイル

**Files:**

- Create: `experiments/2026-10-speculative-decoding/config/compose.yml`
- Create: `experiments/2026-10-speculative-decoding/config/.env.example`
- Create: `experiments/2026-10-speculative-decoding/config/common.args`
- Create: `experiments/2026-10-speculative-decoding/config/model/{C,C-ctx96k,C-iq3xxs,C-ncmoe2,B,D,current-C}.args`
- Create: `experiments/2026-10-speculative-decoding/config/spec/{none,mtp1,mtp2,mtp3,ngram-simple,ngram-mod,mtp2-ngram-simple,draft08}.args`
- Modify: `.gitignore`(`experiments/2026-10-speculative-decoding/config/.env`、`experiments/2026-10-speculative-decoding/scripts/data/private/`)

**Interfaces:**

- Produces: `LLAMA_ARGS` 環境変数で llama-server の全引数を受け取る service `llama-spec`(ホスト `:18081`)と、`ccshim-spec`(ホスト `:11439` → `llama-spec:18081`、ログは `../results/raw/ccshim-log`)。`*.args` は 1 行 1 引数ではなく**空白区切りの 1 行**(`sweep.sh` が `cat` で連結する)

- [ ] **Step 1: `compose.yml` を書く**

`llama-spec`: `image: ${LLAMA_IMAGE}`、`entrypoint: ["/app/llama-server"]`、`command: ${LLAMA_ARGS}`、`gpus: all`、`ports: ["${SPEC_SERVER_PORT:-18081}:18081"]`、volumes `llama_llama-cache:/root/.cache`(`volumes:` で `external: true`、`name: llama_llama-cache`)、`environment: HF_TOKEN`、`restart: "no"`。`ccshim-spec`: `image: python:3.13-alpine`、`command: ["python3","/app/ccshim.py","11439","http://llama-spec:18081","/log"]`、volumes `../../../deploy/llama-swap:/app:ro` と `../results/raw/ccshim-log:/log`、`ports: ["${SPEC_SHIM_PORT:-11439}:11439"]`、`depends_on: [llama-spec]`。

- [ ] **Step 2: `.env.example` と引数ファイルを書く**

`.env.example`: `LLAMA_IMAGE=ghcr.io/mostlygeek/llama-swap@sha256:199a9dd3e30b77fda1b9861eafe61722b665918f6d9348dabd8a30c3bc4ebb33`、`SPEC_SERVER_PORT=18081`、`SPEC_SHIM_PORT=11439`、`HF_TOKEN=`。

`common.args`: Global Constraints の共通フラグ。`model/C.args`: `-hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_S --no-mmproj --ctx-size 131072 --kv-unified -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --cache-reuse 256`。`C-ctx96k` は `--ctx-size 98304`、`C-iq3xxs` は `:UD-IQ3_XXS`、`C-ncmoe2` は `--n-cpu-moe 2` を足す。`B.args`: `:UD-IQ4_XS --n-cpu-moe 10`(他は C と同じ)。`current-C.args`: `-hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S`(他は C と同じ)。`D.args`: C に対して `--ctx-size 262144 -ctk q4_0 -ctv q4_0`(Phase 3 用)。

`spec/none.args` は空ファイル。`mtp1`/`mtp2`/`mtp3`: `--spec-type draft-mtp --spec-draft-n-max N`。`ngram-simple`: `--spec-type ngram-simple`。`ngram-mod`: `--spec-type ngram-mod`。`mtp2-ngram-simple`: `--spec-type draft-mtp,ngram-simple --spec-draft-n-max 2`。`draft08`: `--spec-type draft-simple -hfd unsloth/Qwen3.5-0.8B-GGUF:Q8_0 -ngld 99 --spec-draft-n-max 8`。

- [ ] **Step 3: compose の構文を検証する**

Run: `cd experiments/2026-10-speculative-decoding/config && cp -n .env.example .env && LLAMA_ARGS="$(cat common.args model/C.args spec/mtp2.args | tr '\n' ' ')" docker compose config | grep -A2 "command:"`
Expected: `command:` の配列に `--spec-type` と `draft-mtp` と `-hf` が含まれる。エラーなし。

- [ ] **Step 4: Commit**

```bash
git add experiments/2026-10-speculative-decoding/config .gitignore
git commit -m "add: 投機デコード計測用の llama-server / ccshim compose と引数ファイル (#no-issue)"
```

---

### Task 2: `spec_bench.py` の純関数(metrics 解析・差分・timings 抽出)

**Files:**

- Create: `experiments/2026-10-speculative-decoding/scripts/spec_bench.py`
- Test: `experiments/2026-10-speculative-decoding/scripts/tests/test_spec_bench.py`
- Create: `experiments/2026-10-speculative-decoding/scripts/tests/__init__.py`(空)

**Interfaces:**

- Produces:
  - `parse_metrics(text: str) -> dict[str, float]` … Prometheus テキストを `{"llamacpp:tokens_predicted_total": 123.0, 'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}': 5.0}` のように**ラベル込みの名前をキー**にして返す。`#` 行は無視
  - `metrics_delta(before: dict, after: dict) -> dict` … 以下のキーを持つ dict。`predicted_n` `predicted_s` `decode_tok_s` `prompt_n` `prompt_s` `prefill_tok_s` `cached_n` `n_decode` `draft_n` `accepted_n` `drafts_n` `acceptance`(draft_n が 0 なら `None`)`draft_per_generated`(predicted_n が 0 なら `None`)`accepted_per_pos`(position 昇順の list[int])。存在しないカウンタは 0
  - `extract_timings(resp: dict) -> dict | None` … `/v1/messages` / `/v1/chat/completions` どちらの応答でもトップレベル `timings` があれば `{"prompt_n","prompt_ms","cache_n","predicted_n","predicted_ms","predicted_per_second"}` を返し、無ければ `None`
  - `extract_text(resp: dict) -> str` … messages 形式なら `content[]` の `type=="text"` を連結、chat 形式なら `choices[0].message.content`。thinking / tool_use ブロックは含めない

- [ ] **Step 1: 失敗するテストを書く**

```python
METRICS = """# HELP llamacpp:tokens_predicted_total x
llamacpp:tokens_predicted_total 100
llamacpp:tokens_predicted_seconds_total 2.5
llamacpp:prompt_tokens_total 1000
llamacpp:prompt_tokens_cached_total 900
llamacpp:prompt_seconds_total 0.5
llamacpp:spec_decode_num_draft_tokens_total 60
llamacpp:spec_decode_num_accepted_tokens_total 45
llamacpp:spec_decode_num_drafts_total 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 15
"""

class ParseMetrics(unittest.TestCase):
    def test_parses_labels(self):
        m = parse_metrics(METRICS)
        self.assertEqual(m["llamacpp:tokens_predicted_total"], 100.0)
        self.assertEqual(m['llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"}'], 15.0)

class MetricsDelta(unittest.TestCase):
    def test_delta_and_derived(self):
        before = parse_metrics(METRICS)
        after = dict(before); after["llamacpp:tokens_predicted_total"] = 300; after["llamacpp:tokens_predicted_seconds_total"] = 4.5
        after["llamacpp:spec_decode_num_draft_tokens_total"] = 160; after["llamacpp:spec_decode_num_accepted_tokens_total"] = 125
        after['llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}'] = 80
        d = metrics_delta(before, after)
        self.assertEqual(d["predicted_n"], 200)
        self.assertAlmostEqual(d["decode_tok_s"], 100.0)
        self.assertEqual(d["draft_n"], 100); self.assertEqual(d["accepted_n"], 80)
        self.assertAlmostEqual(d["acceptance"], 0.8)
        self.assertAlmostEqual(d["draft_per_generated"], 0.5)
        self.assertEqual(d["accepted_per_pos"], [50, 0])
    def test_no_spec_counters(self):
        d = metrics_delta({}, {"llamacpp:tokens_predicted_total": 10.0, "llamacpp:tokens_predicted_seconds_total": 1.0})
        self.assertEqual(d["draft_n"], 0); self.assertIsNone(d["acceptance"]); self.assertEqual(d["accepted_per_pos"], [])

class Timings(unittest.TestCase):
    def test_absent(self): self.assertIsNone(extract_timings({"choices": []}))
    def test_present(self):
        t = extract_timings({"timings": {"prompt_n": 5, "prompt_ms": 1.0, "cache_n": 100, "predicted_n": 7, "predicted_ms": 70.0, "predicted_per_second": 100.0}})
        self.assertEqual(t["cache_n"], 100)

class Text(unittest.TestCase):
    def test_messages(self):
        self.assertEqual(extract_text({"content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "a"}, {"type": "tool_use"}, {"type": "text", "text": "b"}]}), "ab")
    def test_chat(self):
        self.assertEqual(extract_text({"choices": [{"message": {"content": "hi"}}]}), "hi")
```

テストファイル先頭で `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))` してから `from spec_bench import ...`。

- [ ] **Step 2: 失敗を確認する**

Run: `cd experiments/2026-10-speculative-decoding && python3 -m unittest scripts.tests.test_spec_bench -v`
Expected: `ImportError`(spec_bench が無い)

- [ ] **Step 3: 4 関数を `spec_bench.py` に実装する**

`parse_metrics` は各行を最後の空白で `name value` に分ける(ラベルの中に空白は無い)。`metrics_delta` の `accepted_per_pos` は `position="N"` を正規表現で拾い、N の最大値まで 0 埋めする。

- [ ] **Step 4: テストが通ることを確認する**

Run: 同上
Expected: `OK`(7 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts
git commit -m "feat: 投機デコード計測の metrics 差分・timings 抽出を実装 (#no-issue)"
```

---

### Task 3: `spec_bench.py` の CLI(HTTP 実行・GPU サンプリング・JSONL 出力)

**Files:**

- Modify: `experiments/2026-10-speculative-decoding/scripts/spec_bench.py`
- Test: `experiments/2026-10-speculative-decoding/scripts/tests/test_spec_bench_cli.py`

**Interfaces:**

- Consumes: Task 2 の 4 関数
- Produces:
  - CLI: `python3 spec_bench.py --url http://127.0.0.1:18081 --endpoint {messages,chat} --prompt-file F --n-predict N --out results/raw/x.jsonl --config C --spec mtp2 --workload W1 --prompt-id W1-02 --block 1 [--think] [--nvidia-smi 1]`
  - `run_once(url: str, endpoint: str, body: dict, timeout: float = 1800) -> tuple[int, dict | None, float, str]` … `(status, resp_json, wall_s, error)`。例外時は `status=0`、`error` にメッセージ
  - `build_body(endpoint: str, prompt: dict, n_predict: int, think: bool) -> dict` … `messages` なら prompt(保存済みの `/v1/messages` 本体)に `max_tokens=n_predict` `temperature=0` `stream=False` を上書き、`thinking` を `{"type":"disabled"}`(think=False)に固定。`chat` なら `{"messages": prompt["messages"], "max_tokens": n_predict, "temperature": 0, "timings_per_token": True, "chat_template_kwargs": {"enable_thinking": think}}`
  - `GpuSampler(interval_s=1.0)` … コンテキストマネージャ。`nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits` を別スレッドで叩き、`summary()` が `{"vram_peak_mib","power_mean_w","power_max_w","temp_max_c","samples"}` を返す。nvidia-smi が無ければ全て `None`
  - **JSONL 1 行のスキーマ**(aggregate / cc_metrics が依存): `ts, block, config, spec, workload, prompt_id, endpoint, n_predict, think, url, status, error, wall_s, metrics(= metrics_delta の dict), timings(= extract_timings か null), text_sha256, text_len, text_head(先頭 200 字), gpu(= GpuSampler.summary())`

- [ ] **Step 1: 失敗するテストを書く**

`http.server` を別スレッドで立て、`/metrics` は呼び出し回数で値が増えるテキストを返し、`POST /v1/chat/completions` は `{"choices":[{"message":{"content":"hello"}}],"timings":{...}}` を返す fake を作る。

```python
class CliRun(unittest.TestCase):
    def test_writes_one_jsonl_row(self):
        # fake server 起動 → main([...--url fake --endpoint chat --prompt-file p.json --out out.jsonl --nvidia-smi 0 ...])
        row = json.loads(pathlib.Path(out).read_text().splitlines()[0])
        self.assertEqual(row["status"], 200)
        self.assertEqual(row["metrics"]["predicted_n"], 20)      # fake の 2 回目 - 1 回目
        self.assertEqual(row["text_len"], 5)
        self.assertEqual(row["timings"]["cache_n"], 100)
        self.assertEqual(row["workload"], "W3")
    def test_connection_refused_is_recorded(self):
        # 閉じたポートに向ける → status 0、error 非空、行は書かれる
    def test_build_body_messages(self):
        b = build_body("messages", {"model": "x", "max_tokens": 8192, "stream": True, "messages": [], "system": "s"}, 256, False)
        self.assertEqual(b["max_tokens"], 256); self.assertFalse(b["stream"]); self.assertEqual(b["temperature"], 0)
        self.assertEqual(b["thinking"], {"type": "disabled"})
```

- [ ] **Step 2: 失敗を確認する**

Run: `python3 -m unittest scripts.tests.test_spec_bench_cli -v`
Expected: `AttributeError`(`main` / `build_body` が無い)

- [ ] **Step 3: `main(argv) -> int` と上記 3 つを実装する**

順序: `GET /metrics` → `GpuSampler` 開始 → `run_once` → 停止 → `GET /metrics` → 1 行追記。`--nvidia-smi 0` でサンプリングを省略。`/metrics` 取得失敗も `error` に残して行は書く。

- [ ] **Step 4: テストが通ることを確認する**

Run: `python3 -m unittest discover -s scripts/tests -v`(実験ディレクトリで)
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts
git commit -m "feat: spec_bench.py に 1 リクエスト計測 CLI(metrics 差分 / GPU サンプリング / JSONL)を追加 (#no-issue)"
```

---

### Task 4: `extract_prompts.py`(ccshim ログから W1 / W2、既存データから W3 / W4)

**Files:**

- Create: `experiments/2026-10-speculative-decoding/scripts/extract_prompts.py`
- Create: `experiments/2026-10-speculative-decoding/scripts/data/w4_prompts.json`
- Test: `experiments/2026-10-speculative-decoding/scripts/tests/test_extract_prompts.py`

**Interfaces:**

- Consumes: ccshim ログディレクトリ(`NNN-req.json` + `NNN-resp.bin` + `trace.jsonl`。`docker run --rm -v llama_ccshim-log-c:/l -v $PWD:/out alpine cp -r /l /out/ccshim-c` で取り出す)
- Produces:
  - `tool_uses_in_resp(raw: bytes) -> list[dict]` … 非 SSE JSON と SSE(`event: content_block_start` の `tool_use` と `input_json_delta` の連結)の両方から `[{"name": str, "input_json": str}]` を返す
  - `classify(req: dict, tool_uses: list[dict]) -> str | None` … `req` に `tools` が無ければ `None`。tool_uses に `Write` か `Edit` があり、その `input_json` 長が 1500 以上なら `"W2"`。それ以外で tool_uses が 1 件以上なら `"W1"`。tool_uses が 0 なら `None`
  - `select(cands: dict[str, list[dict]], k: int = 3, seed: int = 42) -> dict[str, list[dict]]` … 各 W について req 本体サイズ(bytes)の 10 / 50 / 90 パーセンタイルに最も近い 1 件ずつ(k=3)を選ぶ
  - `build_w3(meetings: list[dict]) -> list[dict]` … `tools/bench/data/summary/meetings.json` の各 transcript と、全 transcript 連結版の計 3 本。各要素は `{"id": "W3-01", "messages": [{"role": "user", "content": "次の会議の文字起こしを日本語で要約してください。決定事項と TODO を箇条書きで。\n\n<transcript>"}]}`
  - 出力: `scripts/data/private/{W1,W2}-0N.json`(req 本体そのまま。`stream` キーは削除)、`scripts/data/private/W3-0N.json`、`scripts/data/w4_prompts.json`(手書き 3 本、commit 対象)、`scripts/data/prompts_manifest.json`(各 prompt の `id, workload, bytes, n_messages, n_tools, system_chars` のみ。commit 対象)
  - CLI: `python3 extract_prompts.py --ccshim-dir DIR [--ccshim-dir DIR2 ...] --meetings ../../../tools/bench/data/summary/meetings.json --out-private scripts/data/private --manifest scripts/data/prompts_manifest.json`

- [ ] **Step 1: 失敗するテストを書く**

```python
SSE = (b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"Edit","input":{}}}\n\n'
       b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\":\\"a.go\\",\\"new_string\\":\\""}}\n\n'
       b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"' + b'x'*1600 + b'\\"}"}}\n\n'
       b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n')

class ToolUses(unittest.TestCase):
    def test_sse_concatenates_partial_json(self):
        tus = tool_uses_in_resp(SSE)
        self.assertEqual(tus[0]["name"], "Edit"); self.assertGreater(len(tus[0]["input_json"]), 1500)
        json.loads(tus[0]["input_json"])
    def test_plain_json(self):
        raw = json.dumps({"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "x"}}]}).encode()
        self.assertEqual(tool_uses_in_resp(raw)[0]["name"], "Read")

class Classify(unittest.TestCase):
    def test_w2_requires_tools_and_big_edit(self):
        self.assertEqual(classify({"tools": [1]}, [{"name": "Edit", "input_json": "x"*1600}]), "W2")
        self.assertEqual(classify({"tools": [1]}, [{"name": "Read", "input_json": "{}"}]), "W1")
        self.assertIsNone(classify({}, [{"name": "Read", "input_json": "{}"}]))
        self.assertIsNone(classify({"tools": [1]}, []))

class Select(unittest.TestCase):
    def test_picks_percentiles(self):
        c = {"W1": [{"bytes": b, "req": {}} for b in range(100, 1100, 100)]}
        picked = select(c, k=3)
        self.assertEqual([p["bytes"] for p in picked["W1"]], [200, 600, 1000])

class W3(unittest.TestCase):
    def test_three_prompts(self):
        ps = build_w3([{"transcript": "a"}, {"transcript": "b"}])
        self.assertEqual(len(ps), 3); self.assertIn("a", ps[2]["messages"][0]["content"]); self.assertIn("b", ps[2]["messages"][0]["content"])
```

- [ ] **Step 2: 失敗を確認する**

Run: `python3 -m unittest scripts.tests.test_extract_prompts -v`
Expected: `ImportError`

- [ ] **Step 3: 実装する**

`w4_prompts.json` は 3 本手書き(例: 「1 から 100 の素数の和を手順を示して求めよ」「与えられた Go の関数のバグを説明せよ」+ 短い関数、「日本の祝日を列挙し、移動祝日の規則を説明せよ」)。形式は W3 と同じ `{"id","messages"}`。

- [ ] **Step 4: テストが通ることを確認する**

Run: `python3 -m unittest discover -s scripts/tests -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts
git commit -m "feat: ccshim ログと既存データから投機デコード計測用プロンプトを抽出するスクリプトを追加 (#no-issue)"
```

---

### Task 5: `sweep.sh`(ブロック設計でサーバ起動 / 計測 / 停止)

**Files:**

- Create: `experiments/2026-10-speculative-decoding/scripts/sweep.sh`

**Interfaces:**

- Consumes: Task 1 の compose と `*.args`、Task 3 の CLI、Task 4 の `scripts/data/private/*.json` と `w4_prompts.json`
- Produces: `results/raw/sweep-<YYYYmmdd-HHMMSS>.jsonl`(1 掃引 1 ファイル)、同名 `.log`、`results/raw/props-<config>-<spec>-<block>.json`(`/props` の保存)
- 環境変数: `BLOCKS`(既定 3)`CONFIGS`(既定 `C B`)`SPECS`(既定 `none mtp1 mtp2 mtp3 ngram-simple ngram-mod mtp2-ngram-simple`)`WORKLOADS`(既定 `W1 W2 W3 W4`)`DRY_RUN`(1 ならコマンドを `echo` するだけ)`HEALTH_TIMEOUT`(既定 2400 秒)

- [ ] **Step 1: 書く**

骨子:

1. 実験ディレクトリに `cd`、`.env` の存在を確認、`results/raw` を作成
2. `for b in 1..BLOCKS`: `SPECS` を `b-1` 個回転させた順で `for config; for spec`:
   - `LLAMA_ARGS="$(cat config/common.args config/model/$config.args config/spec/$spec.args | tr '\n' ' ')" docker compose -f config/compose.yml up -d --force-recreate llama-spec`
   - `/health` を 5 秒間隔で `HEALTH_TIMEOUT` まで待つ。超えたら `docker logs --tail 50 llama-spec` を出して `exit 1`
   - `curl -s :18081/props > results/raw/props-$config-$spec-$b.json`
   - 捨てリクエスト: W1 の先頭プロンプトを `--n-predict 8 --out /dev/null` で 1 本
   - `WORKLOADS` の全 prompt ファイルを `shuf --random-source=<(yes $b)` で並べ替え、各 1 本 `spec_bench.py`(`--endpoint` は W1/W2 → `messages`、W3/W4 → `chat`。W4 は `--think`。`n_predict` は Global Constraints の既定)
   - `docker compose -f config/compose.yml down llama-spec`
3. 比較対象 (a) は `CONFIGS=current-C SPECS=none BLOCKS=1 bash scripts/sweep.sh` で別掃引として取る(同じ JSONL スキーマなので aggregate がそのまま読む)
4. `DRY_RUN=1` のときは `docker` / `curl` / `python3` を `echo` に置き換える関数経由で呼ぶ

- [ ] **Step 2: 構文とドライランを確認する**

Run: `bash -n scripts/sweep.sh && cd experiments/2026-10-speculative-decoding && DRY_RUN=1 BLOCKS=1 CONFIGS=C SPECS="none mtp2" WORKLOADS=W4 bash scripts/sweep.sh | head -20`
Expected: `docker compose ... up` → `curl .../health` → `python3 scripts/spec_bench.py ... --spec none` ×(W4 の 3 本) → `down` の順に echo され、続いて `--spec mtp2` の同じ並び。終了コード 0

- [ ] **Step 3: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts/sweep.sh
git commit -m "feat: 投機デコード設定をブロック設計で掃引する sweep.sh を追加 (#no-issue)"
```

---

### Task 6: `aggregate.py`(ブロック内対比較で CSV 化)

**Files:**

- Create: `experiments/2026-10-speculative-decoding/scripts/aggregate.py`
- Test: `experiments/2026-10-speculative-decoding/scripts/tests/test_aggregate.py`

**Interfaces:**

- Consumes: Task 3 の JSONL スキーマ(`results/raw/sweep-*.jsonl` 全部を読む)
- Produces:
  - `summarize(rows: list[dict]) -> list[dict]` … `(config, spec, workload)` ごとに 1 行。列: `config, spec, workload, n, decode_tok_s_mean, decode_tok_s_sd, ratio_vs_none_mean`(同じ `block, config, workload, prompt_id` の `spec=none` 行で割った比の平均。none 自身は 1.0、対が無ければ空)`acceptance_mean, draft_per_generated_mean, prefill_tok_s_mean, cached_n_mean, vram_peak_mib_max, power_mean_w_mean, text_match_rate`(同じ `block, config, workload, prompt_id` の none 行と `text_sha256` が一致した割合)`errors`(status != 200 の件数)
  - `per_pos(rows) -> list[dict]` … `(config, spec)` ごとに position 別の受理数合計
  - CLI: `python3 aggregate.py [--raw results/raw] [--out results/spec_sweep.csv] [--per-pos results/spec_accept_per_pos.csv]`。`status != 200` の行は集計から除き `errors` に数える

- [ ] **Step 1: 失敗するテストを書く**

```python
def row(block, spec, pid, tok_s, sha, acc=None, draft=0, status=200):
    return {"block": block, "config": "C", "spec": spec, "workload": "W1", "prompt_id": pid, "status": status,
            "metrics": {"decode_tok_s": tok_s, "acceptance": acc, "draft_per_generated": draft, "prefill_tok_s": 1000, "cached_n": 5, "accepted_per_pos": [3, 1]},
            "text_sha256": sha, "gpu": {"vram_peak_mib": 15000, "power_mean_w": 200}}

class Summarize(unittest.TestCase):
    def test_paired_ratio_and_match(self):
        rows = [row(1, "none", "p1", 100, "a"), row(1, "mtp2", "p1", 120, "a", 0.8, 0.7),
                row(2, "none", "p1", 110, "b"), row(2, "mtp2", "p1", 110, "c", 0.6, 0.7)]
        s = {(r["spec"]): r for r in summarize(rows)}
        self.assertAlmostEqual(s["mtp2"]["ratio_vs_none_mean"], (1.2 + 1.0) / 2)
        self.assertAlmostEqual(s["mtp2"]["text_match_rate"], 0.5)
        self.assertEqual(s["none"]["ratio_vs_none_mean"], 1.0)
        self.assertEqual(s["mtp2"]["n"], 2)
    def test_errors_excluded(self):
        rows = [row(1, "none", "p1", 100, "a"), row(1, "none", "p2", 0, "", status=500)]
        s = summarize(rows)[0]
        self.assertEqual(s["n"], 1); self.assertEqual(s["errors"], 1)
    def test_per_pos(self):
        self.assertEqual(per_pos([row(1, "mtp2", "p1", 1, "a"), row(1, "mtp2", "p2", 1, "a")])[0]["pos_0"], 6)
```

- [ ] **Step 2: 失敗を確認する** … Run: `python3 -m unittest scripts.tests.test_aggregate -v` / Expected: `ImportError`

- [ ] **Step 3: 実装する**(標準ライブラリ `csv` / `statistics`)

- [ ] **Step 4: テストが通ることを確認する** … Run: `python3 -m unittest discover -s scripts/tests -v` / Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts
git commit -m "feat: 投機デコード掃引の JSONL をブロック内対比較で集計する aggregate.py を追加 (#no-issue)"
```

---

### Task 7: `cc_metrics.py`(Claude Code セッションの tool_use 健全性と受理率)

**Files:**

- Create: `experiments/2026-10-speculative-decoding/scripts/cc_metrics.py`
- Test: `experiments/2026-10-speculative-decoding/scripts/tests/test_cc_metrics.py`

**Interfaces:**

- Consumes: Task 4 の `tool_uses_in_resp`(import して使う)、ccshim ログディレクトリ(`results/raw/ccshim-log`)、Task 2 の `parse_metrics` / `metrics_delta`
- Produces:
  - `session_stats(logdir: pathlib.Path, idx_from: int, idx_to: int) -> dict` … `trace.jsonl` の `idx` が範囲内で `path` が `/v1/messages` の各リクエストについて: `requests`(件数)`tool_use_total`、`json_broken`(`input_json` が `json.loads` できない件数)、`raw_token_leak`(text ブロックに `<tool_call>` または `<|im_` を含む件数)、`cache_n_list`(各応答の `timings.cache_n`。無ければ `None`)、`wall_s_sum`(trace の `dur_s` 合計)
  - `text_blocks_in_resp(raw: bytes) -> list[str]` … SSE / JSON 双方から text ブロック本文
  - CLI: `python3 cc_metrics.py --logdir results/raw/ccshim-log --from N --to M --metrics-before F --metrics-after G --out results/raw/cc-<label>.json` … `session_stats` と `metrics_delta`(`/metrics` を before/after で保存したテキストから)を 1 つの JSON にまとめる

- [ ] **Step 1: 失敗するテストを書く**

```python
class SessionStats(unittest.TestCase):
    def setUp(self):
        # tmpdir に trace.jsonl(idx 1..3、1 と 3 が /v1/messages)、001-resp.bin = Task 4 の SSE(正常)、
        # 003-resp.bin = input_json が壊れた tool_use + text "<tool_call>" を含む JSON を書く
    def test_counts(self):
        s = session_stats(self.dir, 1, 3)
        self.assertEqual(s["requests"], 2)
        self.assertEqual(s["tool_use_total"], 2)
        self.assertEqual(s["json_broken"], 1)
        self.assertEqual(s["raw_token_leak"], 1)
    def test_range_filter(self):
        self.assertEqual(session_stats(self.dir, 2, 2)["requests"], 0)
```

- [ ] **Step 2: 失敗を確認する** … Run: `python3 -m unittest scripts.tests.test_cc_metrics -v` / Expected: `ImportError`

- [ ] **Step 3: 実装する**

- [ ] **Step 4: テストが通ることを確認する** … Run: `python3 -m unittest discover -s scripts/tests -v` / Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts
git commit -m "feat: Claude Code セッションの tool_use 健全性と受理率を ccshim ログから集計する cc_metrics.py を追加 (#no-issue)"
```

---

### Task 8: `cc_task.sh`(M1 / M2 の無人実行と判定)

**Files:**

- Create: `experiments/2026-10-speculative-decoding/scripts/cc_task.sh`
- Create: `experiments/2026-10-speculative-decoding/scripts/data/cc_tasks.json`

**Interfaces:**

- Consumes: Task 1 の `ccshim-spec`(`:11439`)、Task 7 の CLI、GPU 機の `~/ghq/github.com/mktkhr/rag`(`c2181d0`)、`~/.local/bin/claude`、`~/.asdf/shims/go`
- Produces:
  - `cc_tasks.json`: `[{"id":"M1","prompt":"backend に GET /healthz を追加してください。DB(dbClient)と Ollama(ollamaClient)の疎通を確認し、どちらかが失敗したら 503 で、両方成功なら 200 で JSON を返します。既存の gin ルーティング(backend/cmd/api/main.go)に登録してください。go build と go vet が通る状態にしてください。","judge":"cd backend && go build ./... && go vet ./... && grep -q '\"/healthz\"' cmd/api/main.go"},{"id":"M2","prompt":"backend/internal/db/postgres.go の SearchSimilar に 30 秒のコンテキストタイムアウトを付け、エラーは %w で wrap して返してください。go build と go vet が通る状態にしてください。","judge":"cd backend && go build ./... && go vet ./... && grep -q 'context.WithTimeout' internal/db/postgres.go && grep -q '%w' internal/db/postgres.go"}]`
  - CLI: `bash scripts/cc_task.sh <M1|M2> <label>`。手順: `results/raw/cc/<label>-<task>/` を作る → `git -C ~/ghq/github.com/mktkhr/rag worktree add <dir>/wt c2181d0 --detach` → `/metrics` を `before.txt` に保存、`trace.jsonl` の行数を記録 → `cd wt && timeout 900 env ANTHROPIC_BASE_URL=http://127.0.0.1:11439 ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN=dummy ANTHROPIC_MODEL=spec-bench API_TIMEOUT_MS=1800000 CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 claude -p "<prompt>" --dangerously-skip-permissions --output-format json > claude.json` → 終了コードと壁時計を記録 → `/metrics` を `after.txt` に保存 → `git diff --stat` を保存、**変更 0 件なら FAIL**、`timeout`(124)も FAIL → judge を `bash -c` で実行して PASS / FAIL → `cc_metrics.py --from <開始 idx> --to <終了 idx> --metrics-before before.txt --metrics-after after.txt` → `summary.json`(`task, label, pass, wall_s, exit_code, num_turns(claude.json の num_turns), changed_files, cc_metrics`)→ `git worktree remove --force`
  - PATH: 先頭で `export PATH="$HOME/.local/bin:$HOME/.asdf/shims:$PATH"`

- [ ] **Step 1: 書く**(`DRY_RUN=1` で `claude` と `git worktree` を echo に差し替える)

- [ ] **Step 2: 構文とドライランを確認する**

Run: `bash -n scripts/cc_task.sh && DRY_RUN=1 bash scripts/cc_task.sh M1 dry | head`
Expected: `git worktree add` → `claude -p` → judge → `cc_metrics.py` の順に echo。終了コード 0。`python3 -c 'import json; json.load(open("scripts/data/cc_tasks.json"))'` が通る

- [ ] **Step 3: Commit**

```bash
git add experiments/2026-10-speculative-decoding/scripts
git commit -m "feat: Claude Code 実タスク(M1/M2)を無人実行して判定する cc_task.sh を追加 (#no-issue)"
```

---

### Task 9: Phase 0 環境確認(GPU 機で実行)

**Files:**

- Modify: `experiments/2026-10-speculative-decoding/README.md`(Phase 0 の結果を「計測条件」に追記)
- Modify: `experiments/2026-10-speculative-decoding/notes/00-plan.md`(C' の最終形を §4 に記録)

**Interfaces:**

- Consumes: Task 1〜5

- [ ] **Step 1: リポジトリを GPU 機で最新化し `.env` を置く**

Run(GPU 機): `cd ~/ghq/github.com/mktkhr/local-llm && git pull -q && cd experiments/2026-10-speculative-decoding && cp -n config/.env.example config/.env && mkdir -p results/raw scripts/data/private`

- [ ] **Step 2: MTP GGUF と draft を取得しつつ C' の VRAM を測る**

Run: `LLAMA_ARGS="$(cat config/common.args config/model/C.args config/spec/mtp2.args | tr '\n' ' ')" docker compose -f config/compose.yml up -d llama-spec && until curl -sf :18081/health; do sleep 10; done; curl -s :18081/props | python3 -m json.tool | grep -E 'build|n_ctx'; nvidia-smi --query-gpu=memory.used --format=csv,noheader`
Expected: health 200(初回は DL で 20〜40 分)。`memory.used` が **15,500 MiB 以下**なら C' 確定。超過または OOM なら `C-ctx96k` → `C-iq3xxs` → `C-ncmoe2` の順に同じコマンドで試し、通った最初のものを `CONFIGS` の C として採用(`00-plan.md` §4 に記録)

- [ ] **Step 3: `/metrics` と `/v1/messages` を 1 本確認する**

Run: `curl -s :18081/metrics | grep -E 'spec_decode|tokens_predicted_total|prompt_tokens_cached'` と、`docker run --rm -v llama_ccshim-log-c:/l -v $PWD/results/raw:/out alpine cp -r /l /out/ccshim-c && python3 scripts/extract_prompts.py --ccshim-dir results/raw/ccshim-c --meetings ../../tools/bench/data/summary/meetings.json --out-private scripts/data/private --manifest scripts/data/prompts_manifest.json && python3 scripts/spec_bench.py --url http://127.0.0.1:18081 --endpoint messages --prompt-file scripts/data/private/W1-01.json --n-predict 64 --out results/raw/smoke.jsonl --config C --spec mtp2 --workload W1 --prompt-id W1-01 --block 0 && tail -1 results/raw/smoke.jsonl | python3 -m json.tool | head -40`
Expected: metrics に `spec_decode_num_accepted_tokens_total` が存在。smoke 行の `status` 200、`metrics.draft_n > 0`、`metrics.acceptance` が 0〜1、`gpu.vram_peak_mib` 非 null。`timings` の有無を README に記録

- [ ] **Step 4: B' の ncmoe を確定する**

Run: `B.args` の `--n-cpu-moe` を 10 → 11 → 12 と増やしながら Step 2 と同じ手順で `memory.used` ≤ 15,500 MiB になる最小値を探し、`B.args` を更新

- [ ] **Step 5: 1 ブロックだけ本番と同じ掃引を流す**

Run: `BLOCKS=1 nohup bash scripts/sweep.sh > results/raw/sweep-smoke.log 2>&1 &` → 終了後 `python3 scripts/aggregate.py && column -s, -t results/spec_sweep.csv | head -30`
Expected: 7 spec × 2 config × 12 prompt = 168 行、`errors` 0。`ratio_vs_none_mean` が全 spec で埋まる

- [ ] **Step 6: Commit**

```bash
git add experiments/2026-10-speculative-decoding
git commit -m "docs: 投機デコード計測の Phase 0(環境確認・VRAM 境界・スモーク)の結果を記録 (#no-issue)"
```

Phase 1 本番(`BLOCKS=3`)と Phase 2(`cc_task.sh` × 候補設定 × baseline × n=2)はこの計画の後、[00-plan.md](00-plan.md) §6 に従って実行する。
