#!/usr/bin/env python3
"""llama-server に対する Needle In A Haystack ランナー。

KV キャッシュ量子化が長文からの情報抽出能力を損なわないかを測る。
データ生成は scripts/bench/data/needle/generate.py をそのまま使う
(ランタイム非依存のため)。

採点は needle_id が応答に含まれるかの 2 値。主観が入らない。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def post(base, path, payload, timeout=3600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def n_tokens(base, text):
    return len(post(base, "/tokenize", {"content": text})["tokens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="llama-server の URL")
    ap.add_argument("--needle", type=pathlib.Path, required=True, help="generate.py が出力した JSON")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv-type", required=True)
    ap.add_argument("--backend", required=True, help="prefill と decode を行った実体 (例 cuda / metal / cuda->metal)")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--out", help="JSON Lines の追記先")
    # PD分離の実経路で品質が保たれるかを確認するためのモード。
    # 指定すると Prefill を別ホストで行い、KV を転送してから Decode する。
    ap.add_argument("--prefill-url", help="PD分離モード。Prefill 側 llama-server")
    ap.add_argument("--kv-url", help="PD分離モード。KV ファイル配信元")
    ap.add_argument("--decode-kv-dir", help="PD分離モード。Decode 側 --slot-save-path")
    ap.add_argument("--conns", type=int, default=8)
    args = ap.parse_args()

    d = json.loads(args.needle.read_text())
    prompt = d["prompt"]
    needle_id = d["needle_id"]

    ntok = n_tokens(args.url, prompt)

    pd_mode = bool(args.prefill_url)
    if pd_mode:
        import os
        from pd_run import fetch_kv

        fname = f"needle_{args.needle.stem}.bin"
        post(args.prefill_url, "/slots/0?action=erase", {})
        post(args.prefill_url, "/completion",
             {"prompt": prompt, "n_predict": 1, "id_slot": 0, "cache_prompt": True,
              "temperature": 0.0, "seed": 1234})
        post(args.prefill_url, "/slots/0?action=save", {"filename": fname})
        fetch_kv(f"{args.kv_url}/{fname}", os.path.join(args.decode_kv_dir, fname), conns=args.conns)
        post(args.url, "/slots/0?action=erase", {})
        post(args.url, "/slots/0?action=restore", {"filename": fname})
    else:
        # 前の試行の prompt cache を必ず捨てる。残っていると別条件の KV を
        # 再利用してしまい、KV 量子化の比較が壊れる。
        post(args.url, "/slots/0?action=erase", {})

    t0 = time.perf_counter()
    body = post(
        args.url,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": args.n_predict,
            "id_slot": 0,
            "cache_prompt": True,
            # 採点の再現性のため貪欲デコードに固定する。
            "temperature": 0.0,
            "seed": 1234,
        },
    )
    wall = (time.perf_counter() - t0) * 1000.0

    content = body["content"]
    t = body["timings"]
    ok = needle_id in content

    row = {
        "model": args.model,
        "kv_type": args.kv_type,
        "backend": args.backend,
        "needle_file": args.needle.name,
        "position_pct": d["position_pct"],
        "prompt_tokens": ntok,
        "prompt_chars": d["actual_chars"],
        "needle_id": needle_id,
        "found": ok,
        "prompt_n": t["prompt_n"],
        "cache_n": t["cache_n"],
        "pd_mode": pd_mode,
        "prefill_ms": t["prompt_ms"],
        "decode_tok_per_s": t["predicted_per_second"],
        "wall_ms": wall,
        "response": content.strip()[:200],
    }

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mark = "OK  " if ok else "FAIL"
    print(
        f"  {mark} kv={args.kv_type:<5} pos={d['position_pct']:<4} "
        f"tokens={ntok:>6} prefill={t['prompt_ms']:>8.0f}ms  resp={content.strip()[:40]!r}",
        file=sys.stderr,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
