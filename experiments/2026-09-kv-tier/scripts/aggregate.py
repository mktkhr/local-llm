#!/usr/bin/env python3
"""kvtier_runs.jsonl を CSV へ集計し、SSD tier の損益分岐点を算出する。

損益分岐点の求め方:
  full prefill    T_A(N) = N * (a + b*N)      アテンションが二次で効くため N の2次
  SSD からの復元  T_D(N) = s*N + c            転送量は N に線形
  この2式が交わる N が、SSD から読み戻すほうが速くなる境界。
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乗で y = slope*x + intercept を返す。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    return slope, my - slope * mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "results" / "kvtier_runs.jsonl"))
    ap.add_argument("--out", default=str(REPO / "results" / "kvtier.csv"))
    ap.add_argument("--breakeven", default=str(REPO / "results" / "kvtier_breakeven.csv"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.runs, encoding="utf-8")]
    by: dict[tuple[int, str], dict[str, dict]] = {}
    for r in rows:
        by.setdefault((r["target_tokens"], r["kv_type"], r["n_ctx"]), {})[r["case"]] = r

    fields = [
        "tokens", "kv_type", "n_ctx", "kv_mib",
        "A_full_prefill_ms", "B_vram_hit_ms",
        "C_dram_promptcache_ms", "Dw_dram_pagecache_ms", "D_ssd_ms",
        "C_speedup", "Dw_speedup", "D_speedup",
        "ssd_mib_s", "dram_pagecache_mib_s", "dram_promptcache_mib_s",
        "decode_tok_s", "E_reprefill_tokens", "E_prefill_ms",
    ]
    out = []
    for (tok, kv, nctx), c in sorted(by.items()):
        a, b, cc, dw, d = (c["A_full_prefill"], c["B_vram_hit"], c["C_dram_restore"],
                           c["Dw_pagecache_restore"], c["D_ssd_restore"])
        mib = d["n_read"] / 2**20
        # C は restore 時間が API から取れないので wall の差分で求める
        c_ms = round(cc["wall_ms"] - b["wall_ms"], 1)
        out.append({
            "tokens": tok, "kv_type": kv, "n_ctx": nctx, "kv_mib": round(mib, 1),
            "A_full_prefill_ms": a["prompt_ms"], "B_vram_hit_ms": b["prompt_ms"],
            "C_dram_promptcache_ms": c_ms,
            "Dw_dram_pagecache_ms": dw["restore_ms"], "D_ssd_ms": d["restore_ms"],
            "C_speedup": round(a["prompt_ms"] / c_ms, 2),
            "Dw_speedup": round(a["prompt_ms"] / dw["restore_ms"], 2),
            "D_speedup": round(a["prompt_ms"] / d["restore_ms"], 2),
            "ssd_mib_s": round(mib / (d["restore_ms"] / 1000)),
            "dram_pagecache_mib_s": round(mib / (dw["restore_ms"] / 1000)),
            "dram_promptcache_mib_s": round(mib / (c_ms / 1000)),
            "decode_tok_s": a["predicted_per_second"],
            "E_reprefill_tokens": c["E_partial_prefix"]["prompt_n"],
            "E_prefill_ms": c["E_partial_prefix"]["prompt_ms"],
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    # 損益分岐は KV 型ごとに算出する
    be = []
    for kv in sorted({r["kv_type"] for r in out}):
        sub = [r for r in out if r["kv_type"] == kv]
        if len(sub) < 3:
            continue
        be.extend(breakeven(kv, sub))
    with open(args.breakeven, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(be[0].keys()))
        w.writeheader()
        w.writerows(be)

    for r in out:
        print(f"{r['tokens']:>7} tok {r['kv_type']:>5} | KV {r['kv_mib']:>8.1f} MiB | "
              f"A {r['A_full_prefill_ms']:>9.1f} | C {r['C_dram_promptcache_ms']:>8.1f} ({r['C_speedup']:>6.2f}x) | "
              f"Dw {r['Dw_dram_pagecache_ms']:>8.1f} ({r['Dw_speedup']:>6.2f}x) | "
              f"D {r['D_ssd_ms']:>9.1f} ({r['D_speedup']:>5.2f}x)")
    print()
    for r in be:
        print(f"{r['kv_type']:>5} {r['tier']:<18} 復元 {r['restore_ms_per_tok']:.5f} ms/tok  損益分岐 = {r['breakeven_tokens']}")


def breakeven(kv: str, out: list[dict]) -> list[dict]:
    xs = [float(r["tokens"]) for r in out]
    b_slope, b_int = linfit(xs, [r["A_full_prefill_ms"] / r["tokens"] for r in out])
    s_slope, s_int = linfit(xs, [float(r["D_ssd_ms"]) for r in out])
    n = 40000.0
    for _ in range(200):  # 不動点反復
        n = (s_slope - b_int - s_int / n) / b_slope
    be = []
    be.append({"kv_type": kv, "tier": "ssd", "prefill_a_ms_per_tok": round(b_int, 6),
               "prefill_b_ms_per_tok2": f"{b_slope:.4e}",
               "restore_ms_per_tok": round(s_slope, 5),
               "restore_intercept_ms": round(s_int, 1),
               "breakeven_tokens": round(n)})
    for tier, key in (("dram_pagecache", "Dw_dram_pagecache_ms"),
                      ("dram_promptcache", "C_dram_promptcache_ms")):
        t_slope, t_int = linfit(xs, [float(r[key]) for r in out])
        m = 40000.0
        try:
            for _ in range(200):
                m = (t_slope - b_int - t_int / m) / b_slope
            m = round(m) if m > 0 else "なし (全長で復元が有利)"
        except ZeroDivisionError:
            m = "なし"
        be.append({"kv_type": kv, "tier": tier, "prefill_a_ms_per_tok": round(b_int, 6),
                   "prefill_b_ms_per_tok2": f"{b_slope:.4e}",
                   "restore_ms_per_tok": round(t_slope, 5),
                   "restore_intercept_ms": round(t_int, 1),
                   "breakeven_tokens": m})

    return be


if __name__ == "__main__":
    main()
