#!/usr/bin/env python3
"""none 実行の各位置の (1 位, 2 位) トークンについて、lm head の行の差のノルム ||w_a - w_b|| を出す。

δ = (w_a - w_b)·Δh と見ると、隠れ状態の揺れ Δh が同じでも、ペアの行が近いほど δ は小さくなる。
GGUF は gguf-py(numpy)で読むため、GPU 機の python コンテナで実行する。
  python3 pair_norms.py --runs runs.jsonl --build b12464 --gguf model.gguf --out norms-b12464.json
"""

from __future__ import annotations

import argparse
import json
import math


def collect_pairs(rows: list[dict], build: str) -> set[tuple[int, int]]:
    out = set()
    for r in rows:
        if r["build"] != build or r["spec"] != "none":
            continue
        for t in r["tokens"]:
            top = sorted(t["top"], key=lambda x: -x[1])
            if len(top) > 1:
                out.add((top[0][0], top[1][0]))
    return out


def pair_norms(pairs: set[tuple[int, int]], fetch) -> dict[tuple[int, int], float]:
    """fetch(ids) は ids の順に行ベクトルを返す関数。"""
    ids = sorted({i for p in pairs for i in p})
    vec = dict(zip(ids, fetch(ids)))
    return {(a, b): math.sqrt(sum((x - y) ** 2 for x, y in zip(vec[a], vec[b]))) for a, b in pairs}


def gguf_rows(path: str):
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(path)
    by = {t.name: t for t in reader.tensors}
    t = by.get("output.weight") or by["token_embd.weight"]

    def fetch(ids):
        raw = np.asarray(t.data)[ids]
        return dequantize(raw, t.tensor_type).astype(np.float64).reshape(len(ids), -1).tolist()

    return t.name, fetch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.runs) if l.strip()]
    pairs = collect_pairs(rows, args.build)
    name, fetch = gguf_rows(args.gguf)
    norms = pair_norms(pairs, fetch)
    json.dump({"tensor": name, "norms": {f"{a},{b}": round(v, 5) for (a, b), v in norms.items()}}, open(args.out, "w"))
    print(f"{args.build}: {len(norms)} pairs from {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
