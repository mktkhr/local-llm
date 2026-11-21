#!/usr/bin/env python3
"""同じトークン列の prefill を繰り返し、timings の prompt_ms を残す。
トークン列が無ければ、Claude Code のリクエスト(--src)をサーバで整形・トークナイズして先頭 --n トークンを保存する。
  python3 prefill_probe.py --url http://127.0.0.1:18094 --tokens results/raw/prompt-tokens.json \
      [--src W1-03.json --turn-scripts ~/turn/scripts --n 32768] --reps 3 --tag build=b12464,ub=512 --out runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request


def post(url: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


def build_tokens(url: str, src: pathlib.Path, turn_scripts: pathlib.Path, n: int) -> list[int]:
    sys.path.insert(0, str(turn_scripts))
    from turn_probe import anthropic_to_oai, variant
    body = anthropic_to_oai(variant(json.loads(src.read_text()), n_tools=None, n_msgs=None, max_tokens=1))
    prompt = post(url, "/apply-template", body)["prompt"]
    toks = post(url, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
    if len(toks) < n:
        raise SystemExit(f"prompt has only {len(toks)} tokens")
    return toks[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokens", type=pathlib.Path, required=True)
    ap.add_argument("--src", type=pathlib.Path)
    ap.add_argument("--turn-scripts", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    if not args.tokens.exists():
        args.tokens.write_text(json.dumps(build_tokens(args.url, args.src, args.turn_scripts, args.n)))
    toks = json.loads(args.tokens.read_text())
    tags = dict(kv.split("=", 1) for kv in args.tag.split(","))
    for i in range(args.reps):
        t = post(args.url, "/completion", {"prompt": toks, "n_predict": 1, "temperature": 0, "cache_prompt": False})["timings"]
        row = {**tags, "rep": i, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
               "tok_s": t["prompt_n"] / t["prompt_ms"] * 1000}
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{tags} rep={i} n={row['prompt_n']} {row['prompt_ms']:.0f} ms {row['tok_s']:.0f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
