#!/usr/bin/env python3
"""プロンプト集合を greedy で生成し、各トークンの上位 logprob を JSONL に残す(GPU 機で実行)。

/apply-template で chat を整形(thinking off)→ /completion を temperature -1(greedy、確率は生の softmax)で呼ぶ。
プロンプトキャッシュは使わない(プロンプト側の計算を全実行で同じにする)。
  python3 gen_probe.py --url http://127.0.0.1:18091 --prompts data/prompts.json --n 400 --tag build=b12464,spec=none,rep=1 --out runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post(url: str, path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metric(url: str, key: str) -> float:
    text = urllib.request.urlopen(url + "/metrics", timeout=60).read().decode()
    for line in text.splitlines():
        if line.startswith(f"llamacpp:{key} "):
            return float(line.split()[1])
    return 0.0


def compact(cp: list[dict]) -> list[dict]:
    """completion_probabilities → [{id, lp, top: [[id, lp], ...]}]。"""
    return [{"id": e["id"], "lp": e["logprob"], "top": [[t["id"], t["logprob"]] for t in e["top_logprobs"]]} for e in cp]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    url = args.url.rstrip("/")
    tags = dict(kv.split("=", 1) for kv in filter(None, args.tag.split(",")))
    for p in json.load(open(args.prompts)):
        prompt = post(url, "/apply-template", {"messages": p["messages"], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        d0, a0 = metric(url, "spec_decode_num_draft_tokens_total"), metric(url, "spec_decode_num_accepted_tokens_total")
        r = post(url, "/completion", {"prompt": prompt, "n_predict": args.n, "temperature": -1, "n_probs": 5, "cache_prompt": False})
        d1, a1 = metric(url, "spec_decode_num_draft_tokens_total"), metric(url, "spec_decode_num_accepted_tokens_total")
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **tags, "prompt_id": p["id"], "kind": p["kind"],
               "prompt_n": r["timings"]["prompt_n"], "predicted_n": r["timings"]["predicted_n"],
               "decode_tok_s": round(r["timings"]["predicted_per_second"], 2),
               "drafts": int(d1 - d0), "accepted": int(a1 - a0), "tokens": compact(r["completion_probabilities"])}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{p['id']} {tags} n={row['predicted_n']} drafts={row['drafts']} acc={row['accepted']} {row['decode_tok_s']} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
