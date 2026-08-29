#!/usr/bin/env python3
"""単一マシンで Prefill と Decode を完結させたときの基準値を測る。

PD分離の比較対象(構成 A: M1 単体 / 構成 B: 4080 単体)に使う。
slot save/restore は一切行わない。
"""
import argparse
import json
import sys
import time
import urllib.request

from pd_run import FILLER, build_prompt, completion, post  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", required=True)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", help="JSON Lines の追記先")
    args = ap.parse_args()

    rows = []
    for target in args.tokens:
        prompt, ntok = build_prompt(args.url, target)

        # 前の試行の prompt cache を必ず捨てる。残っていると Prefill が計測できない。
        post(args.url, "/slots/0?action=erase", {})

        t0 = time.perf_counter()
        body, t, wall = completion(args.url, prompt, args.n_predict)
        total_ms = (time.perf_counter() - t0) * 1000.0

        if t["cache_n"] != 0:
            print(f"WARN: cache が残っている cache_n={t['cache_n']}", file=sys.stderr)

        row = {
            "label": args.label,
            "target_tokens": target,
            "prompt_tokens": ntok,
            "prompt_n": t["prompt_n"],
            "cache_n": t["cache_n"],
            "prefill_ms": t["prompt_ms"],
            "prefill_tok_per_s": t["prompt_per_second"],
            "predicted_n": t["predicted_n"],
            "decode_ms": t["predicted_ms"],
            "decode_tok_per_s": t["predicted_per_second"],
            # Prefill 完了 = 最初のトークンが出る時点とみなす
            "ttft_ms": t["prompt_ms"],
            "total_ms": total_ms,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
