#!/usr/bin/env python3
"""電力リミット別の掃引結果(results/raw/sweep-*-pl<W>.jsonl)を集計して results/power_limit.csv にする。

行の形式は experiments/2026-10-speculative-decoding/scripts/spec_bench.py の JSONL と同じ。
電力リミットはファイル名の pl<W> から取る(sweep.sh の RESUME_TS に "pl150" のような接尾辞を付けて回す)。
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLUMNS = ["limit_w", "spec", "workload", "n", "decode_tok_s_mean", "decode_tok_s_sd", "decode_vs_220w", "ratio_vs_none",
           "prefill_tok_s_mean", "prefill_vs_220w", "acceptance_mean", "power_mean_w", "power_max_w", "decode_tok_per_w",
           "prefill_tok_per_w", "temp_max_c", "clock_sm_mean_mhz"]
_PL = re.compile(r"-pl(\d+)\.jsonl$")


def limit_from_name(name: str) -> int | None:
    m = _PL.search(name)
    return int(m.group(1)) if m else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else ""


def _sd(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.stdev(xs), 3) if len(xs) >= 2 else ""


def summarize(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r.get("status") == 200 and isinstance(r.get("metrics"), dict)]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        groups[(r["limit_w"], r["spec"], r["workload"])].append(r)
    base = {(r["sweep"], r["block"], r["workload"], r["prompt_id"]): r for r in ok if r["spec"] == "none"}
    ref220 = {(k[1], k[2]): statistics.fmean(x["metrics"]["decode_tok_s"] for x in v)
              for k, v in groups.items() if k[0] == 220 and k[1] == "none"}
    ref220_pre = {(k[1], k[2]): statistics.fmean(x["metrics"]["prefill_tok_s"] for x in v if x["metrics"].get("prefill_tok_s"))
                  for k, v in groups.items() if k[0] == 220 and k[1] == "none"
                  and any(x["metrics"].get("prefill_tok_s") for x in v)}
    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        limit, spec, wl = key
        dec = [r["metrics"]["decode_tok_s"] for r in g]
        pre = [r["metrics"].get("prefill_tok_s") for r in g]
        pw = [(r.get("gpu") or {}).get("power_mean_w") for r in g]
        ratios = []
        for r in g:
            b = base.get((r["sweep"], r["block"], r["workload"], r["prompt_id"]))
            if b and b is not r and b["metrics"].get("decode_tok_s"):
                ratios.append(r["metrics"]["decode_tok_s"] / b["metrics"]["decode_tok_s"])
        dec_mean = statistics.fmean(dec)
        pw_mean = statistics.fmean([x for x in pw if x is not None]) if any(x is not None for x in pw) else None
        pre_mean = statistics.fmean([x for x in pre if x]) if any(pre) else None
        r220 = ref220.get(("none", wl))
        r220p = ref220_pre.get(("none", wl))
        out.append({
            "limit_w": limit, "spec": spec, "workload": wl, "n": len(g),
            "decode_tok_s_mean": round(dec_mean, 3), "decode_tok_s_sd": _sd(dec),
            "decode_vs_220w": round(dec_mean / r220, 3) if r220 else "",
            "ratio_vs_none": _mean(ratios) if spec != "none" else 1.0,
            "prefill_tok_s_mean": round(pre_mean, 1) if pre_mean else "",
            "prefill_vs_220w": round(pre_mean / r220p, 3) if pre_mean and r220p else "",
            "acceptance_mean": _mean([r["metrics"].get("acceptance") for r in g]),
            "power_mean_w": round(pw_mean, 1) if pw_mean else "",
            "power_max_w": max([(r.get("gpu") or {}).get("power_max_w") or 0 for r in g], default=""),
            "decode_tok_per_w": round(dec_mean / pw_mean, 3) if pw_mean else "",
            "prefill_tok_per_w": round(pre_mean / pw_mean, 2) if pre_mean and pw_mean else "",
            "temp_max_c": max([(r.get("gpu") or {}).get("temp_max_c") or 0 for r in g], default=""),
            "clock_sm_mean_mhz": _mean([(r.get("gpu") or {}).get("clock_sm_mean_mhz") for r in g]),
        })
    return out


def load_rows(raw_dir: pathlib.Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.glob("sweep-*-pl*.jsonl")):
        limit = limit_from_name(p.name)
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["limit_w"] = limit
                r.setdefault("sweep", p.stem)
                rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=pathlib.Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "power_limit.csv")
    args = ap.parse_args(argv)
    rows = load_rows(args.raw)
    if not rows:
        print(f"{args.raw} に sweep-*-pl*.jsonl がありません")
        return 1
    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"wrote {args.out} ({len(summary)} rows, {len(rows)} raw)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
