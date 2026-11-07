#!/usr/bin/env python3
"""bench.sh の JSONL を構成ごとに集計して Markdown 表にする。

初見(prompt_n >= 10)と接頭辞キャッシュ済み(prompt_n < 10、同じ文の再送)を分けて中央値を出す。
  python3 aggregate.py results/raw/bench-*.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def summarize(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["config"], {"n_ok": 0, "n_err": 0, "vram_peak_mib": None, "_fresh": [], "_cached": []})
        if r.get("status") != 200 or not r.get("metrics"):
            s["n_err"] += 1
            continue
        s["n_ok"] += 1
        v = (r.get("gpu") or {}).get("vram_peak_mib")
        if v is not None:
            s["vram_peak_mib"] = max(s["vram_peak_mib"] or 0, v)
        m = r["metrics"]
        per100 = (r["sda_read_mib"] / m["predicted_n"] * 100) if r.get("sda_read_mib") is not None and m.get("predicted_n") else None
        rec = {"prefill": m.get("prefill_tok_s"), "decode": m.get("decode_tok_s"), "per100": per100, "wall": r.get("wall_s")}
        (s["_cached"] if (m.get("prompt_n") or 0) < 10 else s["_fresh"]).append(rec)
    for s in out.values():
        for key in ("fresh", "cached"):
            recs = s.pop("_" + key)
            s[key] = {"n": len(recs), "prefill_med": _med([x["prefill"] for x in recs]), "decode_med": _med([x["decode"] for x in recs]),
                      "sda_mib_per_100tok_med": _med([x["per100"] for x in recs]), "wall_med": _med([x["wall"] for x in recs])}
    return out


def to_markdown(summary: dict[str, dict]) -> str:
    lines = ["| 構成 | 成功/失敗 | VRAM ピーク | 初見 n | 初見 prefill tok/s | 初見 decode tok/s | 初見 SSD 読み出し MiB/100tok | 再送 n | 再送 decode tok/s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, s in summary.items():
        f, k = s["fresh"], s["cached"]
        lines.append(f"| {c} | {s['n_ok']}/{s['n_err']} | {s['vram_peak_mib']} MiB | {f['n']} | {f['prefill_med']} | {f['decode_med']} | {f['sda_mib_per_100tok_med']} | {k['n']} | {k['decode_med']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    rows = []
    for p in (argv or sys.argv[1:]):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    print(to_markdown(summarize(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
