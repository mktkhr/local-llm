#!/usr/bin/env python3
"""投機なし(none)と投機あり(spec)の greedy 生成を突き合わせ、分岐をロジット余裕のハザードで説明する。

- 余裕 m_i = none の 1 位と 2 位の logprob 差(= logit 差)
- 揺れ δ_i = spec での同じ 2 トークンの logprob 差 − none での差(分岐位置 D まで。文脈は同一)
- 予測: 位置 i で反転する確率 h_i = P(δ < −m_i)。δ の経験分布で計算する(δ が 0 の位置 = まだ揺れていない位置は除外)
  python3 hazard.py --runs results/raw/runs.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict


def margins(tokens: list[dict]) -> list[float]:
    out = []
    for t in tokens:
        top = sorted(t["top"], key=lambda x: -x[1])
        out.append(top[0][1] - top[1][1] if len(top) > 1 else math.inf)
    return out


def top_pairs(tokens: list[dict]) -> list[tuple[int, int]]:
    """各位置の (1 位, 2 位) トークン ID。"""
    out = []
    for t in tokens:
        top = sorted(t["top"], key=lambda x: -x[1])
        out.append((top[0][0], top[1][0]) if len(top) > 1 else (top[0][0], -1))
    return out


def first_divergence(a: list[dict], b: list[dict]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x["id"] != y["id"]:
            return i
    return None


def deltas(none: list[dict], spec: list[dict], upto: int) -> list[float | None]:
    """位置 0..upto(含む)の δ。none の 1 位・2 位が spec の top に無ければ None。"""
    out = []
    for i in range(min(upto + 1, len(none), len(spec))):
        top_n = sorted(none[i]["top"], key=lambda x: -x[1])
        a, b = top_n[0][0], top_n[1][0]
        lp_s = {tid: lp for tid, lp in spec[i]["top"]}
        if a in lp_s and b in lp_s:
            out.append((lp_s[a] - lp_s[b]) - (top_n[0][1] - top_n[1][1]))
        else:
            out.append(None)
    return out


class Ecdf:
    def __init__(self, xs: list[float]):
        self.xs = sorted(xs)

    def below(self, x: float) -> float:
        """P(δ < x)。"""
        return bisect.bisect_left(self.xs, x) / len(self.xs) if self.xs else 0.0


BUCKETS = [0, 0.01, 0.03, 0.1, 0.3, 1.0, math.inf]


def bucket(m: float) -> int:
    return max(i for i, lo in enumerate(BUCKETS[:-1]) if m >= lo)


def _spec_pairs(rows: list[dict], build: str):
    by = {(r["spec"], r["prompt_id"]): r for r in rows if r["build"] == build}
    for (s, pid), r in sorted(by.items()):
        if s not in ("none", "none2") and ("none", pid) in by:
            yield s, pid, by[("none", pid)], r


def reference_deltas(rows: list[dict], build: str, max_margin: float) -> list[float]:
    """build 内の全投機方式から、余裕 < max_margin の位置の δ(揺れ始め以降、分岐位置まで)を集める。"""
    out = []
    for _, _, n, s in _spec_pairs(rows, build):
        D = first_divergence(n["tokens"], s["tokens"])
        upto = D if D is not None else min(len(n["tokens"]), len(s["tokens"])) - 1
        m = margins(n["tokens"])
        out += [d for i, d in enumerate(deltas(n["tokens"], s["tokens"], upto))
                if d is not None and abs(d) > 1e-7 and m[i] < max_margin]
    return out


def delta_by_margin(rows: list[dict], build: str, edges=(0, 0.1, 0.3, 1, 2, 4, 8, math.inf)) -> list[dict]:
    """揺れ始め以降の |δ| を none の余裕で区切って集計する(build 内の全投機方式をまとめる)。"""
    pts = []
    for _, _, n, s in _spec_pairs(rows, build):
        D = first_divergence(n["tokens"], s["tokens"])
        upto = D if D is not None else min(len(n["tokens"]), len(s["tokens"])) - 1
        m = margins(n["tokens"])
        pts += [(m[i], abs(d)) for i, d in enumerate(deltas(n["tokens"], s["tokens"], upto))
                if d is not None and abs(d) > 1e-7 and m[i] != math.inf]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = sorted(a for mm, a in pts if lo <= mm < hi)
        if sel:
            out.append({"margin": f"[{lo}, {hi})", "n": len(sel), "abs_p50": round(sel[len(sel) // 2], 4),
                        "abs_p90": round(sel[min(int(len(sel) * 0.9), len(sel) - 1)], 4)})
    return out


def analyze(rows: list[dict], build: str, spec: str, scale: dict | None = None, ref: list[float] | None = None) -> dict:
    """δ の経験分布でハザードを計算する。

    scale = {(a, b): ||w_a - w_b||} を渡すと、δ と余裕をペアのノルムで割った空間で計算する。
    ref を渡すと、δ の分布にその spec の全位置ではなく ref を使う(例: reference_deltas で余裕の小さい位置だけ)。
    """
    sc = (lambda pair: scale.get(pair, 1.0)) if scale is not None else (lambda pair: 1.0)
    by = {(r["spec"], r["prompt_id"]): r for r in rows if r["build"] == build}
    pairs = [(pid, by[("none", pid)], by[(spec, pid)]) for (s, pid) in by if s == spec and ("none", pid) in by]
    # 1) δ の経験分布(揺れていない位置 = |δ| < 1e-7 を除く)
    per_run = []
    all_d = []
    for pid, n, s in pairs:
        D = first_divergence(n["tokens"], s["tokens"])
        L = min(len(n["tokens"]), len(s["tokens"]))
        upto = D if D is not None else L - 1
        ds = deltas(n["tokens"], s["tokens"], upto)
        pairs_n = top_pairs(n["tokens"])
        nz = [d / sc(pairs_n[i]) for i, d in enumerate(ds) if d is not None and abs(d) > 1e-7]
        all_d += nz
        p0 = next((i for i, d in enumerate(ds) if d is not None and abs(d) > 1e-7), None)
        per_run.append({"prompt_id": pid, "kind": n["kind"], "D": D, "L": L, "p0": p0, "m": margins(n["tokens"]),
                        "s": [sc(pr) for pr in pairs_n],
                        "drafts": s["drafts"], "accepted": s["accepted"]})
    F = Ecdf(all_d if ref is None else ref)
    # 2) 予測と観測
    exp_div, obs_div, brier = 0.0, 0, 0.0
    at_risk = defaultdict(int)
    events = defaultdict(int)
    pred_h = defaultdict(float)
    for r in per_run:
        end = r["D"] if r["D"] is not None else r["L"] - 1
        surv = 1.0
        if r["p0"] is not None:
            for i in range(r["p0"], end + 1):
                h = F.below(-r["m"][i] / r["s"][i])
                b = bucket(r["m"][i])
                at_risk[b] += 1
                pred_h[b] += h
                surv *= (1 - h)
            if r["D"] is not None:
                events[bucket(r["m"][r["D"]])] += 1
        # 分岐しなかった実行は L まで生き残った、分岐した実行は D で起きた。P(D 以前に分岐) で評価
        p_div = 1 - surv if r["p0"] is not None else 0.0
        r["p_div_pred"] = round(p_div, 3)
        exp_div += p_div
        obs = 1 if r["D"] is not None else 0
        obs_div += obs
        brier += (p_div - obs) ** 2
    calib = []
    for b in sorted(at_risk):
        lo, hi = BUCKETS[b], BUCKETS[b + 1]
        calib.append({"margin": f"[{lo}, {hi})", "at_risk": at_risk[b], "events": events.get(b, 0),
                      "obs_hazard": round(events.get(b, 0) / at_risk[b], 5), "pred_hazard": round(pred_h[b] / at_risk[b], 5)})
    absd = sorted(abs(x) for x in all_d)
    q = lambda p: round(absd[min(int(p * len(absd)), len(absd) - 1)], 5) if absd else None
    return {"build": build, "spec": spec, "n_runs": len(per_run), "n_delta": len(all_d),
            "n_ref": len(all_d) if ref is None else len(ref),
            "abs_delta_q": {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": absd[-1] if absd else None},
            "observed_divergences": obs_div, "expected_divergences": round(exp_div, 2), "brier": round(brier / max(len(per_run), 1), 4),
            "margin_at_D": sorted(round(r["m"][r["D"]], 4) for r in per_run if r["D"] is not None),
            "calibration": calib,
            "runs": [{k: r[k] for k in ("prompt_id", "kind", "D", "L", "p0", "drafts", "accepted", "p_div_pred")} for r in per_run]}


def determinism(rows: list[dict], build: str) -> dict:
    by = {(r["spec"], r["prompt_id"]): r for r in rows if r["build"] == build}
    same_ids = same_lp = n = 0
    for (s, pid), r in by.items():
        if s != "none" or ("none2", pid) not in by:
            continue
        n += 1
        a, b = r["tokens"], by[("none2", pid)]["tokens"]
        same_ids += [t["id"] for t in a] == [t["id"] for t in b]
        same_lp += [t["lp"] for t in a] == [t["lp"] for t in b]
    return {"pairs": n, "identical_tokens": same_ids, "identical_logprobs": same_lp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ref-margin", type=float, default=None,
                    help="指定すると、δ の分布を build 内の余裕 < この値の位置だけから作る")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.runs) if l.strip()]
    out = []
    for build in sorted({r["build"] for r in rows}):
        out.append({"build": build, "determinism": determinism(rows, build), "delta_by_margin": delta_by_margin(rows, build)})
        ref = reference_deltas(rows, build, args.ref_margin) if args.ref_margin is not None else None
        for spec in sorted({r["spec"] for r in rows if r["build"] == build} - {"none", "none2"}):
            out.append(analyze(rows, build, spec, ref=ref))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
