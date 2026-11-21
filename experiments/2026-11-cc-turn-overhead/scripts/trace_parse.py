#!/usr/bin/env python3
"""-lv 4 のサーバログから、同一本文の再送(checkpoint 復元あり)ごとにスロット側の区間を ms で切り出す。

  select_to_launch   スロット選択からタスク開始まで(grammar のパースとサンプラ構築)
  restore            GDN 状態の checkpoint を VRAM に戻す
  sampler_init       プロンプト全トークンをサンプラに流す
  checkpoint_create  末尾 4 トークン手前の checkpoint を作り直す(VRAM → ホスト)
  eval_and_sample    末尾 4 トークンの評価と 1 トークンのサンプリング
  finish             print_timing から release まで

  python3 trace_parse.py results/raw/trace-b12464.log
"""

from __future__ import annotations

import re
import statistics
import sys

_TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
_TASK = re.compile(r"\| task (-?\d+) \|")


def _ms(line: str) -> float | None:
    m = _TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return (mi * 60 + s) * 1000 + ms + us / 1000


def phases(lines: list[str]) -> list[dict]:
    out = []
    cur: dict = {}
    for line in lines:
        t = _ms(line)
        if t is None:
            continue
        if "selected slot by" in line:
            cur = {"select": t}
        elif "processing task" in line and "select" in cur:
            cur["launch"] = t
            cur["task"] = int(_TASK.search(line).group(1))
        elif "restored context checkpoint" in line and "launch" in cur:
            cur["restored"] = t
        elif "init sampler" in line and "restored" in cur:
            cur["sampler"] = t
            cur["n_tokens"] = int(re.search(r"total = (\d+)", line).group(1))
        elif "created context checkpoint" in line and "sampler" in cur:
            cur["created"] = t
        elif "prompt eval time" in line and "created" in cur:
            cur["timing"] = t
        elif "stop processing" in line and "timing" in cur:
            out.append({
                "task": cur["task"], "n_tokens": cur["n_tokens"],
                "select_to_launch": round(cur["launch"] - cur["select"], 3),
                "restore": round(cur["restored"] - cur["launch"], 3),
                "sampler_init": round(cur["sampler"] - cur["restored"], 3),
                "checkpoint_create": round(cur["created"] - cur["sampler"], 3),
                "eval_and_sample": round(cur["timing"] - cur["created"], 3),
                "finish": round(t - cur["timing"], 3),
            })
            cur = {}
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ps = phases(open(argv[0]).read().splitlines())
    keys = ("select_to_launch", "restore", "sampler_init", "checkpoint_create", "eval_and_sample", "finish")
    for n in sorted({p["n_tokens"] for p in ps}):
        sel = [p for p in ps if p["n_tokens"] == n]
        med = {k: round(statistics.median(p[k] for p in sel), 2) for k in keys}
        print(f"n_tokens={n} reps={len(sel)} " + " ".join(f"{k}={v}" for k, v in med.items())
              + f" total={round(sum(med.values()), 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
