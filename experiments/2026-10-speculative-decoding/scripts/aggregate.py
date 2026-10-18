#!/usr/bin/env python3
"""sweep.sh が書いた results/raw/sweep-*.jsonl を集計して CSV にする。

比較は「同じ block / config / workload / prompt_id の spec=none 行」と対にして行う
(run 間のばらつきが大きいため、ブロックをまたいだ平均同士の比較はしない)。

出力:
  results/spec_sweep.csv           (config, spec, workload) ごとの 1 行
  results/spec_accept_per_pos.csv  (config, spec) ごとの draft 位置別の受理数
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

COLUMNS = ["config", "spec", "workload", "n", "decode_tok_s_mean", "decode_tok_s_sd", "ratio_vs_none_mean",
           "acceptance_mean", "draft_per_generated_mean", "prefill_tok_s_mean", "cached_n_mean",
           "vram_peak_mib_max", "power_mean_w_mean", "text_match_rate", "errors"]


def _mean(xs: list) -> float | str:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else ""


def _sd(xs: list) -> float | str:
    xs = [x for x in xs if x is not None]
    return round(statistics.stdev(xs), 3) if len(xs) >= 2 else ""


def _pair_key(r: dict) -> tuple:
    return (r.get("block"), r.get("config"), r.get("workload"), r.get("prompt_id"))


def summarize(rows: list[dict]) -> list[dict]:
    """(config, spec, workload) ごとの要約。status != 200 は errors に数えて除外する。"""
    ok = [r for r in rows if r.get("status") == 200]
    errors: dict[tuple, int] = defaultdict(int)
    for r in rows:
        if r.get("status") != 200:
            errors[(r.get("config"), r.get("spec"), r.get("workload"))] += 1

    baseline = {_pair_key(r): r for r in ok if r.get("spec") == "none"}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        groups[(r.get("config"), r.get("spec"), r.get("workload"))].append(r)

    out = []
    for key in sorted(set(groups) | set(errors)):
        g = groups.get(key, [])
        m = [r.get("metrics") or {} for r in g]
        ratios, matches = [], []
        for r in g:
            b = baseline.get(_pair_key(r))
            if b is None:
                continue
            bt = (b.get("metrics") or {}).get("decode_tok_s")
            rt = (r.get("metrics") or {}).get("decode_tok_s")
            if bt and rt:
                ratios.append(rt / bt)
            matches.append(1.0 if r.get("text_sha256") == b.get("text_sha256") else 0.0)
        out.append({
            "config": key[0], "spec": key[1], "workload": key[2], "n": len(g),
            "decode_tok_s_mean": _mean([x.get("decode_tok_s") for x in m]),
            "decode_tok_s_sd": _sd([x.get("decode_tok_s") for x in m]),
            "ratio_vs_none_mean": _mean(ratios),
            "acceptance_mean": _mean([x.get("acceptance") for x in m]),
            "draft_per_generated_mean": _mean([x.get("draft_per_generated") for x in m]),
            "prefill_tok_s_mean": _mean([x.get("prefill_tok_s") for x in m]),
            "cached_n_mean": _mean([x.get("cached_n") for x in m]),
            "vram_peak_mib_max": max([(r.get("gpu") or {}).get("vram_peak_mib") or 0 for r in g], default=""),
            "power_mean_w_mean": _mean([(r.get("gpu") or {}).get("power_mean_w") for r in g]),
            "text_match_rate": _mean(matches),
            "errors": errors.get(key, 0),
        })
    return out


def per_pos(rows: list[dict]) -> list[dict]:
    """(config, spec) ごとに draft 位置別の受理トークン数を合計する。"""
    acc: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("status") != 200:
            continue
        pp = (r.get("metrics") or {}).get("accepted_per_pos") or []
        key = (r.get("config"), r.get("spec"))
        cur = acc[key]
        if len(cur) < len(pp):
            cur.extend([0] * (len(pp) - len(cur)))
        for i, v in enumerate(pp):
            cur[i] += int(v)
    out = []
    for (config, spec), vals in sorted(acc.items()):
        row = {"config": config, "spec": spec}
        row.update({f"pos_{i}": v for i, v in enumerate(vals)})
        out.append(row)
    return out


def load_rows(raw_dir: pathlib.Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.glob("sweep-*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        return
    cols = columns or sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "spec"), k))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=pathlib.Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "spec_sweep.csv")
    ap.add_argument("--per-pos", type=pathlib.Path, default=ROOT / "results" / "spec_accept_per_pos.csv")
    args = ap.parse_args(argv)

    rows = load_rows(args.raw)
    if not rows:
        print(f"{args.raw} に sweep-*.jsonl がありません")
        return 1
    summary = summarize(rows)
    _write_csv(args.out, summary, COLUMNS)
    _write_csv(args.per_pos, per_pos(rows))
    print(f"wrote {args.out} ({len(summary)} rows) / {args.per_pos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
