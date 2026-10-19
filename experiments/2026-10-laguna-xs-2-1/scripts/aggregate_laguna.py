#!/usr/bin/env python3
"""Laguna vs Qwen3.6 の集計。

speed: results/raw/sweep-*.jsonl(2026-10-speculative-decoding の spec_bench 形式)を config × workload で集計し、
       同ブロック・同プロンプトの基準構成(既定 current-C)に対する decode / prefill 比を出す → results/speed.csv
cc:    results/raw/cc/*/summary.json(cc_task.sh の出力)を 2026-10-speculative-decoding/scripts/aggregate.py の
       summarize_cc で 1 行 1 試行にする → results/claude_code.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = ROOT.parent / "2026-10-speculative-decoding" / "scripts"
sys.path.insert(0, str(SPEC))
from aggregate import CC_COLUMNS, load_cc_summaries, summarize_cc  # noqa: E402

COLUMNS = ["config", "workload", "n", "decode_tok_s_mean", "decode_tok_s_sd", "decode_vs_ref", "prefill_tok_s_mean",
           "prefill_vs_ref", "prompt_n_mean", "vram_peak_mib_max", "power_mean_w_mean", "errors"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else ""


def summarize_speed(rows: list[dict], reference: str = "current-C") -> list[dict]:
    ok = [r for r in rows if r.get("status") == 200 and isinstance(r.get("metrics"), dict)]
    errors: dict[tuple, int] = defaultdict(int)
    for r in rows:
        if not (r.get("status") == 200 and isinstance(r.get("metrics"), dict)):
            errors[(r.get("config"), r.get("workload"))] += 1
    ref = {(r.get("sweep"), r["block"], r["workload"], r["prompt_id"]): r for r in ok if r["config"] == reference}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        groups[(r["config"], r["workload"])].append(r)
    out = []
    for key in sorted(set(groups) | set(errors)):
        g = groups.get(key, [])
        dec, pre = [], []
        for r in g:
            b = ref.get((r.get("sweep"), r["block"], r["workload"], r["prompt_id"]))
            if b:
                if b["metrics"].get("decode_tok_s") and r["metrics"].get("decode_tok_s"):
                    dec.append(r["metrics"]["decode_tok_s"] / b["metrics"]["decode_tok_s"])
                if b["metrics"].get("prefill_tok_s") and r["metrics"].get("prefill_tok_s"):
                    pre.append(r["metrics"]["prefill_tok_s"] / b["metrics"]["prefill_tok_s"])
        out.append({
            "config": key[0], "workload": key[1], "n": len(g),
            "decode_tok_s_mean": _mean([r["metrics"].get("decode_tok_s") for r in g]),
            "decode_tok_s_sd": round(statistics.stdev([r["metrics"]["decode_tok_s"] for r in g]), 3) if len(g) >= 2 else "",
            "decode_vs_ref": _mean(dec), "prefill_tok_s_mean": _mean([r["metrics"].get("prefill_tok_s") for r in g]),
            "prefill_vs_ref": _mean(pre), "prompt_n_mean": _mean([r["metrics"].get("prompt_n") for r in g]),
            "vram_peak_mib_max": max([(r.get("gpu") or {}).get("vram_peak_mib") or 0 for r in g], default=""),
            "power_mean_w_mean": _mean([(r.get("gpu") or {}).get("power_mean_w") for r in g]),
            "errors": errors.get(key, 0),
        })
    return out


def _write(path: pathlib.Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=pathlib.Path, default=ROOT / "results" / "raw")
    ap.add_argument("--reference", default="current-C")
    args = ap.parse_args(argv)
    rows = []
    for p in sorted(args.raw.glob("sweep-*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r.setdefault("sweep", p.stem)
                rows.append(r)
    wrote = []
    if rows:
        _write(ROOT / "results" / "speed.csv", summarize_speed(rows, args.reference), COLUMNS)
        wrote.append("results/speed.csv")
    cc_dir = args.raw / "cc"
    if cc_dir.exists():
        cc = load_cc_summaries(cc_dir)
        if cc:
            _write(ROOT / "results" / "claude_code.csv", summarize_cc(cc), CC_COLUMNS)
            wrote.append("results/claude_code.csv")
    print("wrote " + ", ".join(wrote) if wrote else "入力がありません")
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
