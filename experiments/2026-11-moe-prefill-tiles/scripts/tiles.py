#!/usr/bin/env python3
"""MoE の prefill で、ubatch あたり各 expert に何行(トークン)届き、MMQ のタイルがどれだけ埋まるかを、
実プロンプトのルーティング(-ub 32 で取った LLAMA_MOE_CACHE_TRACE)から計算する。

llama.cpp の MMQ(MUL_MAT_ID)はタイル幅 J を「expert あたりの平均行数 ceil(ub·k/E)」を覆う最小の 8 の倍数(上限 128)にする
(ggml-cuda/mmq.cu の ggml_cuda_mmq_get_J_best)。expert e に n_e 行届くと ceil(n_e/J) 枚のタイルを処理するので、
詰め物込みの行数は Σ J·ceil(n_e/J)、埋まり率は Σ n_e / Σ J·ceil(n_e/J)。
  python3 tiles.py --trace results/raw/trace-prefill.trace --ubs 64,128,256,512,1024,2048,4096
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "2026-11-moe-expert-locality" / "scripts"))
from locality import read_trace  # noqa: E402


def tile_width(ub: int, k: int, n_expert: int) -> int:
    m = math.ceil(ub * k / n_expert)
    return min(128, 8 * math.ceil(m / 8))


def token_experts(recs) -> dict[int, list[tuple]]:
    """層ごとに、トレース順(= プロンプト順)のトークン単位の expert 列。"""
    out = defaultdict(list)
    for il, nu, ids in recs:
        for t in range(len(ids) // nu):
            out[il].append(tuple(ids[t * nu:(t + 1) * nu]))
    return out


def prompt_only(recs, full: int = 32):
    """サーバ起動時のウォームアップ(数トークン)を除き、最初の満杯の ubatch から後だけを残す。"""
    for i, (il, nu, ids) in enumerate(recs):
        if len(ids) // nu >= full:
            return recs[i:]
    return []


def chunk_stats(toks: dict[int, list[tuple]], ub: int, k: int, n_expert: int, expert_bytes: dict | None = None) -> dict:
    """ub トークンずつに区切り、行数・詰め物込みの行数・タイル数・タイルごとに読み直す expert の重みのバイト数を数える。

    MMQ は (expert, J 列のタイル) ごとに expert の重み全体を読むので、読み出し量は Σ ceil(n_e/J) × W_e。
    最終層のように prefill ではほとんど計算されない層(満杯の chunk が無い層)は数えない。
    """
    J = tile_width(ub, k, n_expert)
    rows = padded = tiles = active = n_chunks = 0
    bytes_read = 0
    layers_used = 0
    for il, seq in toks.items():
        nc = len(seq) // ub
        if nc == 0:
            continue
        layers_used += 1
        for c in range(nc):
            n = Counter(e for t in seq[c * ub:(c + 1) * ub] for e in t)
            nt = sum(math.ceil(v / J) for v in n.values())
            rows += sum(n.values())
            padded += J * nt
            tiles += nt
            bytes_read += nt * (expert_bytes or {}).get(il, 0)
            active += len(n)
            n_chunks += 1
    return {"ub": ub, "J": J, "chunks": n_chunks // max(layers_used, 1), "layers": layers_used, "rows": rows,
            "padded": padded, "tiles": tiles, "bytes_read": bytes_read,
            "util": rows / padded if padded else None, "active_experts": active / n_chunks if n_chunks else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ubs", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--out", default="results/tiles.json")
    args = ap.parse_args(argv)
    n_layer, k, groups, recs = read_trace(args.trace)
    toks = token_experts(prompt_only(recs))
    E = groups[0]["n_expert"]
    eb = {il: g["expert_bytes"] for g in groups for il in g["layers"]}
    res = [chunk_stats(toks, int(u), k, E, eb) for u in args.ubs.split(",")]
    n_tok = min(len(v) for v in toks.values())
    pathlib.Path(args.out).write_text(json.dumps({"tokens": n_tok, "k": k, "n_expert": E, "by_ub": res}, indent=1) + "\n")
    for r in res:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
