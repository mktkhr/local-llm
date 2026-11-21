#!/usr/bin/env python3
"""llama-server に同じ短いプロンプトで n トークン生成させ、timings から decode の 1 トークン時間を測る。

  python3 decode_probe.py --url http://127.0.0.1:18090 --reps 5 --n 256 --tag k=8,graphs=on --out runs.jsonl
1 回目は捨て(warmup)、残りの predicted_per_token_ms の中央値と最小値を 1 行にまとめて追記する。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def summarize(per_token_ms: list[float]) -> dict:
    xs = per_token_ms[1:] if len(per_token_ms) > 1 else per_token_ms  # warmup を捨てる
    med = statistics.median(xs)
    return {"n_reps": len(xs), "ms_median": round(med, 4), "ms_min": round(min(xs), 4),
            "ms_max": round(max(xs), 4), "tok_s_median": round(1000 / med, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prompt", default="Write a long, detailed story about a lighthouse keeper.")
    ap.add_argument("--tag", default="", help="key=value,... を行に足す")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    body = {"prompt": args.prompt, "n_predict": args.n, "temperature": 0, "ignore_eos": True, "cache_prompt": False}
    per = []
    for _ in range(args.reps + 1):
        r = post(args.url.rstrip("/") + "/completion", body)
        t = r["timings"]
        per.append(t["predicted_per_token_ms"])
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **summarize(per), "n_predict": args.n, "raw_ms": [round(x, 4) for x in per]}
    for kv in filter(None, args.tag.split(",")):
        k, _, v = kv.partition("=")
        row[k] = v
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in row if k != "raw_ms"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
