#!/usr/bin/env python3
"""プロンプトを順に greedy で生成し、1 リクエスト 1 行の JSON Lines を残す(トークン列と timings)。
  python3 decode_run.py --url http://127.0.0.1:18093 --prompts prompts.json --n 400 --tag cache_mib=4096 --out runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request


def post(url: str, path: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


def row_from(resp: dict, tags: dict, prompt_id: str) -> dict:
    t = resp.get("timings", {})
    return {**tags, "prompt_id": prompt_id, "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
            "ms_per_tok": t.get("predicted_per_token_ms"), "tokens": resp.get("tokens", [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompts", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    tags = dict(kv.split("=", 1) for kv in args.tag.split(","))
    for p in json.loads(args.prompts.read_text()):
        prompt = post(args.url, "/apply-template", {"messages": p["messages"],
                                                     "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        resp = post(args.url, "/completion", {"prompt": prompt, "n_predict": args.n, "temperature": 0,
                                              "cache_prompt": False, "return_tokens": True})
        row = row_from(resp, tags, p["id"])
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{p['id']} {tags} n={row['predicted_n']} {row['ms_per_tok']:.2f} ms/tok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
