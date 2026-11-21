#!/usr/bin/env python3
"""トレース 1 本からヒット率曲線(LRU / Che / Belady / llama.cpp 実装)と局所性の指標を出し、
キャッシュサイズごとの実測速度を「t = t0 + ミスのバイト数 / BW」で当てはめる。

  python3 analyze.py --trace results/raw/trace-c11100.trace --runs results/raw/runs.jsonl \
      --tune ~/ghq/github.com/ggml-org/llama.cpp/scripts/moe-cache-tune.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict

from locality import MAX_BATCH, belady_hits, che_hit_rate, decode_refs, popularity, read_trace, stack_distances


def _tokens(recs, layers):
    """層ごとに、キャッシュを通るルックアップのトークン単位の expert 集合を時刻順に。"""
    by = defaultdict(list)
    ls = set(layers)
    for il, nu, ids in recs:
        if il not in ls or len(ids) // nu > MAX_BATCH:
            continue
        for t in range(len(ids) // nu):
            by[il].append(frozenset(ids[t * nu:(t + 1) * nu]))
    return by


def temporal_overlap(recs, layers, n_expert) -> dict:
    """連続するトークンが共有する expert の数の平均と、独立に選んだ場合の期待値 Σ q_e^2。"""
    obs_sum = ind_sum = 0.0
    n_pairs = 0
    per_layer = {}
    for il, seq in sorted(_tokens(recs, layers).items()):
        if len(seq) < 2:
            continue
        c = Counter(e for s in seq for e in s)
        q2 = sum((v / len(seq)) ** 2 for v in c.values())
        pairs = len(seq) - 1
        obs = sum(len(a & b) for a, b in zip(seq, seq[1:]))
        per_layer[il] = {"observed": round(obs / pairs, 4), "independent": round(q2, 4)}
        obs_sum += obs
        ind_sum += q2 * pairs
        n_pairs += pairs
    return {"observed": obs_sum / n_pairs, "independent": ind_sum / n_pairs, "pairs": n_pairs, "per_layer": per_layer}


def fit_time(pts) -> tuple[float, float, float]:
    """[(ミスのバイト数 / トークン, ms / トークン)] から t = t0 + bytes / BW。(t0 ms, BW B/s, R²)。"""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / ss if ss else 1.0
    return a, 1000.0 / b, r2


def bw_fixed_t0(miss_bytes: float, ms: float, t0: float) -> float:
    """t0 を既知として 1 点から BW(B/s)を出す。"""
    return miss_bytes / ((ms - t0) / 1000)


def load_tune(path: str):
    spec = importlib.util.spec_from_file_location("moe_cache_tune", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_misses(tune, ttr, mib: float) -> dict[int, int]:
    """llama.cpp の実装(moe-cache-tune.py の ClassLRU)を共有 0 で回し、グループごとのミス数を返す。"""
    n_layer, n_used, groups, records = ttr
    caches, group_of = [], {}
    for ig, g in enumerate(groups):
        n = tune.n_slots_of(g, mib, 0, n_used)
        caches.append(tune.ClassLRU(g.layers, g.n_expert, n) if n > 0 else None)
        group_of.update((il, ig) for il in g.layers)
    miss = defaultdict(int)
    for il, nu, ids in records:
        ig = group_of[il]
        c = caches[ig]
        n_tokens = len(ids) // nu
        if n_tokens > MAX_BATCH:
            continue
        if c is None or min(n_tokens * nu, groups[ig].n_expert) > c.n_slots:
            miss[ig] += len(set(ids))
            continue
        n_fill, _ = c.plan(il, ids, nu, 0, lambda n: 0)
        miss[ig] += n_fill
    return miss


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tune", required=True, help="llama.cpp の scripts/moe-cache-tune.py")
    ap.add_argument("--label", default="default", help="速度の当てはめに使う runs の label")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)

    n_layer, n_used, groups, recs = read_trace(args.trace)
    tune = load_tune(args.tune)
    ttr = tune.read_trace(args.trace)
    refs = [decode_refs(recs, g) for g in groups]
    dists = [stack_distances(r) for r in refs]
    pops = [popularity(r) for r in refs]
    n_tok = sum(len(ids) // nu for il, nu, ids in recs if il == min(groups[0]["layers"]) and len(ids) // nu <= MAX_BATCH)
    total_refs = sum(len(r) for r in refs)

    runs = [json.loads(l) for l in open(args.runs) if l.strip()]
    measured = sorted({int(r["cache_mib"]) for r in runs})
    grid = sorted(set(measured) | {256, 512, 768} | set(range(1024, 11200, 512)))

    rows = []
    for mib in grid:
        slots = [tune.n_slots_of(tg, mib, 0, n_used) for tg in ttr[2]]
        lru = sum(sum(1 for d in dist if d < s) for dist, s in zip(dists, slots))
        che = sum(che_hit_rate(p, s) * len(r) for p, s, r in zip(pops, slots, refs))
        row = {"mib": mib, "frac": round(sum(slots) / sum(len(g["layers"]) * g["n_expert"] for g in groups), 4),
               "lru": lru / total_refs, "che": che / total_refs}
        if mib in measured or mib % 2048 == 0:
            row["belady"] = sum(belady_hits(r, s) for r, s in zip(refs, slots)) / total_refs
            m = replay_misses(tune, ttr, mib)
            row["llama"] = 1 - sum(m.values()) / total_refs
            row["miss_bytes_per_tok"] = sum(m[ig] * g["expert_bytes"] for ig, g in enumerate(groups)) / n_tok
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    with (out / "curves.csv").open("w", newline="") as f:
        cols = ["mib", "frac", "lru", "che", "belady", "llama", "miss_bytes_per_tok"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 5) if isinstance(r.get(k), float) else r.get(k)) for k in cols})

    # 実測速度(プロンプトごとの ms/トークンを生成トークン数で重み付け)
    def speeds(label):
        out = {}
        for mib in measured:
            sel = [r for r in runs if int(r["cache_mib"]) == mib and r.get("label", "default") == label]
            if sel:
                out[mib] = sum(r["predicted_ms"] for r in sel) / sum(r["predicted_n"] for r in sel)
        return out

    speed = speeds(args.label)
    by_mib = {r["mib"]: r for r in rows}
    pts = [(by_mib[m]["miss_bytes_per_tok"], speed[m]) for m in sorted(speed)]
    t0, bw, r2 = fit_time(pts)
    locality = {
        "decode_tokens": n_tok, "refs": total_refs,
        "groups": [{"layers": len(g["layers"]), "expert_bytes": g["expert_bytes"], "refs": len(r),
                    "distinct": len(set(r))} for g, r in zip(groups, refs)],
        "overlap": temporal_overlap(recs, [il for g in groups for il in g["layers"]], groups[0]["n_expert"]),
        "time_fit": {"t0_ms": round(t0, 3), "bw_gbs": round(bw / 1e9, 3), "r2": round(r2, 4),
                     "points": [{"mib": m, "measured_ms": round(speed[m], 3),
                                 "miss_mb_per_tok": round(by_mib[m]["miss_bytes_per_tok"] / 1e6, 2),
                                 "pred_ms": round(t0 + by_mib[m]["miss_bytes_per_tok"] / bw * 1000, 3)} for m in sorted(speed)]},
    }
    others = sorted({r.get("label", "default") for r in runs} - {args.label})
    locality["other_labels"] = {
        lab: [{"mib": m, "measured_ms": round(v, 3),
               "bw_gbs_fixed_t0": round(bw_fixed_t0(by_mib[m]["miss_bytes_per_tok"], v, t0) / 1e9, 2)}
              for m, v in sorted(speeds(lab).items())] for lab in others}
    (out / "locality.json").write_text(json.dumps(locality, indent=1) + "\n")
    print(json.dumps(locality, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
