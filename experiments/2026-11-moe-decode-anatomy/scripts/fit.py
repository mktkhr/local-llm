#!/usr/bin/env python3
"""k 掃引と密モデル対照から decode 1 トークン時間を分解する。

モデル: t = T_fix + B_dense/BW_dense + k·b_exp/BW_exp
- 密モデル 2 量子化(同じグラフ、B だけ違う)の差から BW_dense と T_fix(密)を出す
- MoE の t(k) の傾きから BW_exp(expert の散らばった読み出しの実効帯域)を出す
- MoE の切片 a から、密部分の帯域時間を引いた残りが固定費。ノード数で割ると 1 ノードあたり
  python3 fit.py --runs results/raw/k-sweep.jsonl --nodes results/raw/nodes.jsonl --bytes results/bytes-*.json
"""

from __future__ import annotations

import argparse
import json
import pathlib


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """最小二乗の (切片, 傾き, R²)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 1.0


def two_point_bw(b1: float, t1_ms: float, b2: float, t2_ms: float) -> tuple[float, float]:
    """同じグラフで B だけ違う 2 点から (BW GB/s, 固定費 ms)。t = T + B/BW。"""
    bw = (b1 - b2) / 1e9 / ((t1_ms - t2_ms) / 1000)
    fix = t1_ms - b1 / 1e9 / bw * 1000
    return bw, fix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--bytes-dir", default="results")
    ap.add_argument("--build", required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.runs) if l.strip()]
    rows = [r for r in rows if r.get("build") == args.build]
    nodes = {(r["model"], r["build"]): int(r["graph_nodes"]) for r in map(json.loads, open(args.nodes)) if r.get("graph_nodes")}
    by = lambda name: json.load(open(pathlib.Path(args.bytes_dir) / f"bytes-{name}.json"))
    moe, d8, d4 = by("qwen36-iq3s"), by("qwen35-9b-q8"), by("qwen35-9b-q4km")
    out = {"build": args.build}
    for g in ("on", "off"):
        dense = {r["model"]: r["ms_median"] for r in rows if r["k"] == "-" and r["graphs"] == g}
        if "qwen35-9b-Q8_0" in dense and "qwen35-9b-Q4_K_M" in dense:
            bw_d, fix_d = two_point_bw(d8["dense_bytes"], dense["qwen35-9b-Q8_0"], d4["dense_bytes"], dense["qwen35-9b-Q4_K_M"])
        else:
            bw_d, fix_d = None, None
        ks = sorted((int(r["k"]), r["ms_median"]) for r in rows if r["model"] == "qwen36-iq3s" and r["k"] != "-" and r["graphs"] == g)
        if len(ks) < 2:
            continue
        a, b, r2 = linfit([k for k, _ in ks], [t for _, t in ks])
        bw_e = moe["expert_bytes_per_k"] / 1e9 / (b / 1000)
        dense_ms = moe["dense_bytes"] / 1e9 / bw_d * 1000 if bw_d else None
        fix_moe = a - dense_ms if dense_ms is not None else None
        n_moe, n_d = nodes.get(("qwen36-iq3s", args.build)), nodes.get(("qwen35-9b-Q8_0", args.build))
        out[f"graphs_{g}"] = {
            "moe_t_k": ks, "moe_intercept_ms": round(a, 3), "moe_slope_ms_per_k": round(b, 4), "moe_r2": round(r2, 4),
            "bw_expert_GBps": round(bw_e, 1), "bw_dense_GBps": round(bw_d, 1) if bw_d else None,
            "dense_fix_ms": round(fix_d, 3) if fix_d is not None else None,
            "moe_dense_bw_ms": round(dense_ms, 3) if dense_ms else None, "moe_fix_ms": round(fix_moe, 3) if fix_moe is not None else None,
            "nodes_moe": n_moe, "nodes_dense": n_d,
            "fix_us_per_node_moe": round(fix_moe / n_moe * 1000, 3) if fix_moe and n_moe else None,
            "fix_us_per_node_dense": round(fix_d / n_d * 1000, 3) if fix_d and n_d else None,
            "moe_t8_share": {"fix": round(fix_moe / (a + 8 * b), 3) if fix_moe else None,
                             "dense_bw": round(dense_ms / (a + 8 * b), 3) if dense_ms else None,
                             "expert_bw": round(8 * b / (a + 8 * b), 3)},
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
