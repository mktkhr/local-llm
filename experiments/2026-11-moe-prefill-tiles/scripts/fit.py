#!/usr/bin/env python3
"""prefill の 1 トークン時間を、ubatch ごとの読み出し量で説明できるかを最小二乗で比べる。

  M0  t = a + c/ub                    ubatch ごとの固定費だけ
  M1  t = a + (B_dense/ub + X_tile)/BW  タイルごとに expert の重みを読み直す
  M2  t = a + (B_dense/ub + X_once)/BW  ubatch 内で使う expert を 1 回ずつ読む(L2 で読み直しが消える場合)
B_dense は毎 ubatch 1 回読む密部分(2026-11-moe-decode-anatomy の GGUF 集計で 1.674 GB)。
  python3 fit.py --runs results/raw/prefill.jsonl --tiles results/tiles.json --build b12464
"""

from __future__ import annotations

import argparse
import json
import statistics

B_DENSE_MB = 1674.0


def lstsq(X: list[list[float]], y: list[float]) -> tuple[list[float], float]:
    """正規方程式で解く。(係数, R²)。"""
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(n)]
    for c in range(n):  # ガウスの消去法(部分ピボット)
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(n):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [x - f * z for x, z in zip(A[r], A[c])]
                b[r] -= f * b[c]
    coef = [b[i] / A[i][i] for i in range(n)]
    pred = [sum(c * x for c, x in zip(coef, r)) for r in X]
    my = statistics.fmean(y)
    ss = sum((v - my) ** 2 for v in y)
    r2 = 1 - sum((v - p) ** 2 for v, p in zip(y, pred)) / ss if ss else 1.0
    return coef, r2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--build", default="b12464")
    ap.add_argument("--out", default="results/fit.json")
    args = ap.parse_args(argv)
    runs = [json.loads(l) for l in open(args.runs) if l.strip()]
    tiles = {d["ub"]: d for d in json.load(open(args.tiles))["by_ub"]}
    pts = []
    for ub in sorted({int(r["ub"]) for r in runs if r["build"] == args.build}):
        sel = [r for r in runs if r["build"] == args.build and int(r["ub"]) == ub]
        us = statistics.median(1e6 / r["tok_s"] for r in sel)
        d = tiles[ub]
        tok = d["chunks"] * ub
        x_tile = d["bytes_read"] / tok / 1e6
        x_once = d["active_experts"] * d["layers"] * (d["bytes_read"] / max(d["tiles"], 1)) / ub / 1e6
        pts.append({"ub": ub, "us_per_tok": us, "x_tile_mb": x_tile, "x_once_mb": x_once})
    y = [p["us_per_tok"] for p in pts]
    out = {"build": args.build, "points": pts, "models": {}}
    for name, row in (("M0", lambda p: [1, 1 / p["ub"]]),
                      ("M1", lambda p: [1, B_DENSE_MB / p["ub"] + p["x_tile_mb"]]),
                      ("M2", lambda p: [1, B_DENSE_MB / p["ub"] + p["x_once_mb"]])):
        X = [row(p) for p in pts]
        coef, r2 = lstsq(X, y)
        m = {"a_us": round(coef[0], 2), "r2": round(r2, 4),
             "pred": [round(sum(c * x for c, x in zip(coef, r)), 1) for r in X]}
        if name == "M0":
            m["c_us_per_ubatch"] = round(coef[1], 1)
        else:
            m["bw_gbs"] = round(1000 / coef[1], 1)  # µs/MB → GB/s
        out["models"][name] = m
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
