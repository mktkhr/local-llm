#!/usr/bin/env python3
"""Claude Code セッション(ccshim ログ)の tool_use 健全性と、セッション合計の投機受理率をまとめる。

ccshim は 1 リクエストごとに NNN-req.json / NNN-resp.bin を書き、trace.jsonl に 1 行追記する。
cc_task.sh が課題の開始前後で trace.jsonl の行数と /metrics を控えておき、本スクリプトがその範囲を集計する。

  python3 cc_metrics.py --logdir results/raw/ccshim-log --from 12 --to 40 \\
      --metrics-before before.txt --metrics-after after.txt --out results/raw/cc/x/cc_metrics.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extract_prompts import tool_uses_in_resp  # noqa: E402
from spec_bench import metrics_delta, parse_metrics  # noqa: E402

LEAK_MARKERS = ("<tool_call>", "<|im_")


def _sse_events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:].strip()))
            except ValueError:
                continue
    return out


def text_blocks_in_resp(raw: bytes) -> list[str]:
    """応答(JSON / SSE)の text ブロック本文を順に返す。SSE は text_delta を index ごとに連結する。"""
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            body = json.loads(stripped)
        except ValueError:
            return []
        return [b.get("text", "") for b in body.get("content") or []
                if isinstance(b, dict) and b.get("type") == "text"]
    blocks: dict[int, list[str]] = {}
    for ev in _sse_events(text):
        t = ev.get("type")
        if t == "content_block_start" and (ev.get("content_block") or {}).get("type") == "text":
            blocks.setdefault(ev["index"], []).append(ev["content_block"].get("text", ""))
        elif t == "content_block_delta" and (ev.get("delta") or {}).get("type") == "text_delta":
            blocks.setdefault(ev.get("index", 0), []).append(ev["delta"].get("text", ""))
    return ["".join(parts) for _, parts in sorted(blocks.items())]


def cache_n_in_resp(raw: bytes) -> int | None:
    """応答のどこかにある timings.cache_n。JSON ならトップレベル、SSE ならどれかのイベント。"""
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            t = json.loads(stripped).get("timings")
        except ValueError:
            return None
        return t.get("cache_n") if isinstance(t, dict) else None
    for ev in _sse_events(text):
        t = ev.get("timings")
        if isinstance(t, dict) and "cache_n" in t:
            return t["cache_n"]
    return None


def session_stats(logdir: pathlib.Path, idx_from: int | None = None, idx_to: int | None = None,
                  line_from: int | None = None, line_to: int | None = None) -> dict:
    """trace.jsonl の範囲内の /v1/messages リクエストを集計する。

    範囲は idx([idx_from, idx_to])か、trace.jsonl の 1 始まりの行番号([line_from, line_to])で指定する。
    ccshim は再起動すると idx を 1 から振り直す(NNN-resp.bin も上書きされる)ため、
    セッションをまたぐ集計には行番号を使い、resp.bin は実行直後に退避しておくこと(cc_task.sh が行う)。
    """
    stats = {"requests": 0, "tool_use_total": 0, "json_broken": 0, "raw_token_leak": 0,
             "cache_n_list": [], "wall_s_sum": 0.0}
    trace_path = logdir / "trace.jsonl"
    if not trace_path.exists():
        return stats
    for lineno, line in enumerate(trace_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if line_from is not None and line_to is not None and not (line_from <= lineno <= line_to):
            continue
        rec = json.loads(line)
        idx = rec.get("idx")
        if idx is None:
            continue
        if idx_from is not None and idx_to is not None and not (idx_from <= idx <= idx_to):
            continue
        if not str(rec.get("path", "")).startswith("/v1/messages"):
            continue
        stats["requests"] += 1
        stats["wall_s_sum"] += float(rec.get("dur_s") or 0.0)
        resp_path = logdir / f"{idx:03d}-resp.bin"
        if not resp_path.exists():
            stats["cache_n_list"].append(None)
            continue
        raw = resp_path.read_bytes()
        for tu in tool_uses_in_resp(raw):
            stats["tool_use_total"] += 1
            try:
                json.loads(tu["input_json"])
            except ValueError:
                stats["json_broken"] += 1
        if any(m in t for t in text_blocks_in_resp(raw) for m in LEAK_MARKERS):
            stats["raw_token_leak"] += 1
        stats["cache_n_list"].append(cache_n_in_resp(raw))
    stats["wall_s_sum"] = round(stats["wall_s_sum"], 2)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", type=pathlib.Path, required=True)
    ap.add_argument("--from", dest="idx_from", type=int, help="idx の下限(ccshim を再起動していないとき)")
    ap.add_argument("--to", dest="idx_to", type=int)
    ap.add_argument("--line-from", type=int, help="trace.jsonl の行番号の下限(1 始まり)。再起動をまたぐときはこちら")
    ap.add_argument("--line-to", type=int)
    ap.add_argument("--metrics-before", type=pathlib.Path)
    ap.add_argument("--metrics-after", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(argv)

    if (args.idx_from is None or args.idx_to is None) and (args.line_from is None or args.line_to is None):
        ap.error("--from/--to か --line-from/--line-to のどちらかを指定してください")
    result = {"session": session_stats(args.logdir, args.idx_from, args.idx_to, args.line_from, args.line_to),
              "metrics": None}
    if args.metrics_before and args.metrics_after and args.metrics_before.exists() and args.metrics_after.exists():
        result["metrics"] = metrics_delta(parse_metrics(args.metrics_before.read_text()),
                                          parse_metrics(args.metrics_after.read_text()))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if result["session"]["requests"] == 0:
        print("警告: 範囲内に /v1/messages のリクエストがありません(範囲の指定ミスか ccshim の再起動)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
