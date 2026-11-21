#!/usr/bin/env python3
"""perplexity --kl-divergence のログから要約を抜き、表(CSV)にする。
  python3 parse_kld.py results/raw/kld-*.log --out results/kld.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re

_PATTERNS = {
    "ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "ln_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*(-?[\d.]+)",
    "kld": r"Mean\s+KLD:\s*([\d.]+)",
    "kld_unc": r"Mean\s+KLD:\s*[\d.]+\s*±\s*([\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([\d.]+)",
    "kld_median": r"Median\s+KLD:\s*([\d.]+)",
    "rms_dp": r"RMS Δp\s*:\s*([\d.]+)",
    "same_top": r"Same top p:\s*([\d.]+)",
}


def parse(text: str) -> dict:
    out = {}
    for k, pat in _PATTERNS.items():
        m = re.search(pat, text)
        out[k] = float(m.group(1)) if m else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(args.logs):
        name = pathlib.Path(f).stem.removeprefix("kld-")  # <eval>-<quant>-<imatrix>
        ev, q = name.split("-", 1)
        rows.append({"eval": ev, "quant": q, **parse(pathlib.Path(f).read_text(errors="replace"))})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
