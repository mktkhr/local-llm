#!/usr/bin/env python3
"""2 つの imatrix(GGUF 形式)の重み付けがどれだけ似ているかを、テンソルごとの cos 類似度で比べる。

量子化は Σ_j a_j (Δw_j)^2(a_j = 入力 j の二乗平均 = in_sum2 / counts)を小さくするように丸め方を選ぶ。
較正データの a と、実際に使う領域の a の向きが同じ(cos ≈ 1)なら、領域専用の imatrix にしても丸め方はほとんど変わらない。
gguf-py と numpy が要るので python コンテナで実行する。
  python3 imatrix_cos.py --a cc.gguf --b unsloth.gguf --out results/imatrix-cos.json
  python3 imatrix_cos.py --a cc.gguf --b uniform --out results/imatrix-cos-uniform.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else float("nan")


def kind_of(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "", name).removesuffix(".weight")


def summarize(rows: list[dict]) -> list[dict]:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(kind_of(r["name"]), []).append(r["cos"])
    return [{"kind": k, "n": len(v), "median": round(statistics.median(v), 4), "min": round(min(v), 4)}
            for k, v in sorted(by.items())]


def read_means(path: str) -> dict:
    import numpy as np
    from gguf import GGUFReader
    r = GGUFReader(path)
    t = {x.name: np.asarray(x.data, dtype=np.float64) for x in r.tensors}
    out = {}
    for name in t:
        if name.endswith(".in_sum2"):
            base = name.removesuffix(".in_sum2")
            s2, c = t[name], t.get(base + ".counts")
            if c is None:
                continue
            s2 = s2.reshape(len(c.reshape(-1)), -1) if c.size > 1 else s2.reshape(1, -1)
            out[base] = (s2 / np.maximum(c.reshape(-1, 1), 1)).reshape(-1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import numpy as np
    A = read_means(args.a)
    # --b uniform は imatrix なし(全入力を同じ重みで扱う)との比較
    B = {k: np.ones_like(v) for k, v in A.items()} if args.b == "uniform" else read_means(args.b)
    rows = []
    for name in sorted(set(A) & set(B)):
        a, b = A[name], B[name]
        if a.shape != b.shape:
            continue
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        rows.append({"name": name, "cos": round(cos, 5)})
    summ = summarize(rows)
    json.dump({"a": args.a, "b": args.b, "only_a": sorted(set(A) - set(B))[:10], "only_b": sorted(set(B) - set(A))[:10],
               "n_common": len(rows), "median_cos": round(statistics.median(r["cos"] for r in rows), 4),
               "by_kind": summ, "tensors": rows}, open(args.out, "w"), indent=1)
    for s in summ:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
