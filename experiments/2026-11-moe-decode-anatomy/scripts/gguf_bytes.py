#!/usr/bin/env python3
"""GGUF のテンソル表から、decode 1 トークンで GPU が読むバイト数 B(k) を出す。

B(k) = dense(毎トークン全部読むテンソル)+ k × expert 1 本あたりのバイト(全層合計)
- expert テンソル(*_exps)は E 本ぶんを 1 テンソルに持つので n_bytes / E が 1 本分
- token_embd は 1 行しか引かない(get_rows)ので 1 行ぶんだけ数える
- それ以外(attention、GDN、shared expert、norm、output)は毎トークン全部読む
  python3 gguf_bytes.py model.gguf [--k 1 2 4 8 16]
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def classify(name: str) -> str:
    if name.startswith("token_embd"):
        return "embd"
    if re.search(r"_exps(\.|$)", name):
        return "expert"
    return "dense"


def summarize(tensors: list[tuple[str, int, list[int]]], n_expert: int) -> dict:
    """tensors: [(name, n_bytes, shape)]。shape は GGUF の ne 順(ne[0] が行の長さ)。"""
    out = {"dense_bytes": 0, "expert_bytes_all": 0, "embd_row_bytes": 0, "n_expert": n_expert, "by_kind": {}}
    for name, nb, shape in tensors:
        kind = classify(name)
        out["by_kind"].setdefault(kind, 0)
        out["by_kind"][kind] += nb
        if kind == "dense":
            out["dense_bytes"] += nb
        elif kind == "expert":
            out["expert_bytes_all"] += nb
        elif kind == "embd":
            rows = shape[1] if len(shape) > 1 else 1
            out["embd_row_bytes"] += nb // max(rows, 1)
    out["expert_bytes_per_k"] = out["expert_bytes_all"] / n_expert if n_expert else 0
    return out


def bytes_per_token(s: dict, k: int) -> float:
    return s["dense_bytes"] + s["embd_row_bytes"] + k * s["expert_bytes_per_k"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--k", type=int, nargs="*", default=[1, 2, 4, 6, 8, 10, 12, 16])
    args = ap.parse_args(argv)
    from gguf import GGUFReader  # pip install gguf
    r = GGUFReader(args.gguf)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    def kv(key):
        f = r.fields.get(f"{arch}.{key}")
        return int(f.parts[-1][0]) if f is not None else None
    n_expert, n_used = kv("expert_count") or 0, kv("expert_used_count")
    tensors = [(t.name, int(t.n_bytes), [int(x) for x in t.shape]) for t in r.tensors]
    s = summarize(tensors, n_expert)
    s.update({"arch": arch, "expert_used_count_default": n_used, "n_layer": kv("block_count"),
              "file_tensor_bytes": sum(nb for _, nb, _ in tensors),
              "B_k": {k: bytes_per_token(s, k) for k in args.k}})
    json.dump(s, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
