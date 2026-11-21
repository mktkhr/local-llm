#!/usr/bin/env python3
"""turns.jsonl を条件ごとに中央値で畳み、1 リクエストの固定費を部品に分ける。

  base     GET /health の往復(以降の値はすべてこれを引く)
  apply    JSON パース + jinja + grammar 文字列生成
  tokenize レンダ済みプロンプトのトークナイズ
  count    Anthropic 変換 + apply + tokenize。conv = count - apply - tokenize
  resid    再送 1 回の wall から base・count・prefill・decode を引いた残り
           (grammar のパースとサンプラ初期化、スロット割り当て、キャッシュ照合、応答生成)
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

_KEY = ("build", "axis", "n_tools", "n_msgs")


def _med(rs: list[dict], field: str) -> float | None:
    vals = [r[field] for r in rs if r.get("status") == 200 and r.get(field) is not None]
    return statistics.median(vals) if vals else None


def _ms(x: float | None) -> float | None:
    return None if x is None else x * 1000


def _sub(a, *bs):
    if a is None or any(b is None for b in bs):
        return None
    return a - sum(bs)


def _per_tok(rs: list[dict]) -> float | None:
    ok = [r for r in rs if r.get("status") == 200 and r.get("predicted_n")]
    if not ok:
        return None
    return sum(r["predicted_s"] for r in ok) / sum(r["predicted_n"] for r in ok) * 1000


def decompose(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r.get(f) for f in _KEY)
        groups[k][r["kind"]].append(r)
        meta.setdefault(k, {f: r.get(f) for f in (*_KEY, "chars_tools", "chars_msgs", "chars_system")})
    out = []
    for k, by in groups.items():
        base = _med(by["health"], "wall_s")
        apply, tok, count = (_sub(_med(by[x], "wall_s"), base) for x in ("apply", "tokenize", "count"))
        warm = _med(by["msg_warm"], "wall_s")
        p, d = _med(by["msg_warm"], "prompt_s"), _med(by["msg_warm"], "predicted_s")
        anyw = _med(by["msg_any_warm"], "wall_s")
        pa, da = _med(by["msg_any_warm"], "prompt_s"), _med(by["msg_any_warm"], "predicted_s")
        n_tok, n_in = _med(by["tokenize"], "n_tokens"), _med(by["count"], "input_tokens")
        apply_none = _sub(_med(by["apply_none"], "wall_s"), base)

        def chat_resid(kind, pre):
            return _sub(_med(by[kind], "wall_s"), base, pre, tok,
                        _med(by[kind], "prompt_s"), _med(by[kind], "predicted_s"))

        cr, cnr = chat_resid("chat_warm", apply), chat_resid("chat_none_warm", apply_none)
        da, dn = _per_tok(by["dec_chat_auto"]), _per_tok(by["dec_chat_none"])
        out.append({
            **meta[k],
            "prompt_chars": _med(by["apply"], "prompt_chars"),
            "n_tokens": n_tok,
            "tokens_match": n_tok is not None and n_tok == n_in,
            "base_ms": _ms(base),
            "apply_ms": _ms(apply), "tokenize_ms": _ms(tok), "count_ms": _ms(count),
            "conv_ms": _ms(_sub(count, apply, tok)),
            "cold_ms": _ms(_med(by["msg_cold"], "wall_s")),
            "cold_prompt_n": _med(by["msg_cold"], "prompt_n"),
            "cold_prompt_ms": _ms(_med(by["msg_cold"], "prompt_s")),
            "warm_ms": _ms(warm), "warm_prompt_n": _med(by["msg_warm"], "prompt_n"),
            "warm_prompt_ms": _ms(p), "warm_decode_ms": _ms(d),
            "resid_ms": _ms(_sub(warm, base, count, p, d)),
            "chat_warm_ms": _ms(_med(by["chat_warm"], "wall_s")),
            "any_warm_ms": _ms(anyw),
            "any_resid_ms": _ms(_sub(anyw, base, count, pa, da)),
            "dec_auto_ms": _per_tok(by["dec_auto"]), "dec_any_ms": _per_tok(by["dec_any"]),
            "apply_none_ms": _ms(apply_none),
            "same_prompt_none": bool(by["apply_none"]) and _med(by["apply_none"], "prompt_chars") == _med(by["apply"], "prompt_chars"),
            "grammar_build_ms": _ms(_sub(apply, apply_none)),
            "chat_resid_ms": _ms(cr), "chat_none_resid_ms": _ms(cnr),
            "grammar_init_ms": _ms(_sub(cr, cnr)),
            "dec_chat_auto_ms": da, "dec_chat_none_ms": dn,
            "grammar_tok_ms": _sub(da, dn),
        })
    return sorted(out, key=lambda g: tuple(str(g[f]) for f in ("build", "axis")) + (g["n_tools"] or 0, g["n_msgs"] or 0))


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """y = a + b x の最小二乗。(a, b, R²)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / ss if ss else 1.0
    return a, b, r2


def fits(groups: list[dict]) -> list[dict]:
    out = []
    for build in sorted({g["build"] for g in groups}):
        for axis, xf in (("tools", "chars_tools"), ("msgs", "chars_msgs")):
            gs = [g for g in groups if g["build"] == build and g["axis"] == axis]
            for yf in ("apply_ms", "tokenize_ms", "count_ms", "resid_ms", "grammar_build_ms", "grammar_init_ms"):
                pts = [(g[xf] / 1000, g[yf]) for g in gs if g[yf] is not None]
                if len(pts) >= 3:
                    a, b, r2 = linfit(*zip(*pts))
                    out.append({"build": build, "axis": axis, "y": yf, "x": f"{xf}/1k",
                                "a_ms": round(a, 3), "b_ms_per_kchar": round(b, 4), "r2": round(r2, 4), "n": len(pts)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("results/raw/turns.jsonl"))
    ap.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("results/decompose.csv"))
    ap.add_argument("--fits", type=pathlib.Path, default=pathlib.Path("results/fits.json"))
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in args.runs.read_text().splitlines() if line.strip()]
    groups = decompose(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(groups[0]))
        w.writeheader()
        for g in groups:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in g.items()})
    fs = fits(groups)
    args.fits.write_text(json.dumps(fs, indent=1) + "\n")
    cols = ("build", "axis", "n_tools", "n_msgs", "n_tokens", "tokens_match", "apply_ms", "tokenize_ms", "conv_ms",
            "count_ms", "warm_prompt_n", "warm_prompt_ms", "resid_ms", "warm_ms", "grammar_build_ms", "grammar_init_ms",
            "dec_chat_auto_ms", "dec_chat_none_ms", "grammar_tok_ms")
    print("\t".join(cols))
    for g in groups:
        print("\t".join(f"{g[c]:.1f}" if isinstance(g[c], float) else str(g[c]) for c in cols))
    for f_ in fs:
        print(f_, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
