#!/usr/bin/env python3
"""MoE cache のルーティングトレース(LLAMA_MOE_CACHE_TRACE)から、expert 参照の局所性とヒット率曲線を出す。

  LRU      スタック距離(前回の参照から今回までに参照された別の expert の数)の分布 F で H(C) = F(C)
  Che      参照が独立で人気度 p_k だけで決まると仮定した LRU の近似(時間的な偏りを無視した場合)
  Belady   次に使うのが最も遠いものを捨てる最適方式(上限)
  llama    llama.cpp の実装(scripts/moe-cache-tune.py の replay)をそのまま回した値

キャッシュは同じレイアウトの層の expert を 1 つのプールで共有するので(llama-moe-cache.cpp の group)、
参照列はグループごとに層をまたいで時刻順に並べる。32 トークンを超えるバッチ(prefill)はキャッシュを通らないので除く。
"""

from __future__ import annotations

import heapq
import math
import struct
from collections import Counter

MAX_BATCH = 32  # llama_moe_cache::impl::max_batch


def read_trace(path: str):
    data = open(path, "rb").read()
    if data[:4] != b"MCT1":
        raise ValueError(f"{path}: not a MoE cache trace")
    pos = 4

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    n_layer, n_used, n_group = take("3I")
    groups = []
    for _ in range(n_group):
        gid, n_expert, n_layers = take("3I")
        layers = list(take(f"{n_layers}I"))
        split, split_sum = take("2d")
        host_bytes, dev_host_bytes, alignment = take("3Q")
        (n_t,) = take("I")
        nb2 = list(take(f"{n_t}Q"))
        groups.append({"id": gid, "n_expert": n_expert, "layers": layers, "split": split, "split_sum": split_sum,
                       "host_bytes": host_bytes, "dev_host_bytes": dev_host_bytes, "alignment": alignment,
                       "nb2": nb2, "expert_bytes": sum(nb2)})
    recs = []
    while pos < len(data):
        il, nu, n_ids = struct.unpack_from("<3i", data, pos)
        pos += 12
        recs.append((il, nu, struct.unpack_from(f"<{n_ids}i", data, pos)))
        pos += 4 * n_ids
    return n_layer, n_used, groups, recs


def decode_refs(recs, group) -> list[int]:
    """グループの層の、キャッシュを通る参照(1 ルックアップ内の重複は 1 回)を時刻順に。キーは il * n_expert + e。"""
    layers, ne = set(group["layers"]), group["n_expert"]
    out = []
    for il, nu, ids in recs:
        if il not in layers or len(ids) // nu > MAX_BATCH:
            continue
        seen = set()
        for e in ids:
            if e not in seen:
                seen.add(e)
                out.append(il * ne + e)
    return out


class _Bit:
    def __init__(self, n):
        self.n, self.t = n, [0] * (n + 1)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def prefix(self, i):  # sum of [0, i)
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def stack_distances(keys) -> list[float]:
    """Mattson のスタック距離。初回参照は inf。"""
    bit, last, out = _Bit(len(keys)), {}, []
    for t, k in enumerate(keys):
        p = last.get(k)
        if p is None:
            out.append(math.inf)
        else:
            out.append(bit.prefix(t) - bit.prefix(p + 1))
            bit.add(p, -1)
        bit.add(t, 1)
        last[k] = t
    return out


def hit_curve(dist, caps) -> list[float]:
    """容量 C(スロット数)の LRU のヒット率 = P(距離 < C)。"""
    n = len(dist)
    fin = sorted(d for d in dist if d != math.inf)
    out = []
    import bisect
    for c in caps:
        out.append(bisect.bisect_left(fin, c) / n)
    return out


def belady_hits(keys, cap: int) -> int:
    nxt, last = [0] * len(keys), {}
    for t in range(len(keys) - 1, -1, -1):
        nxt[t] = last.get(keys[t], math.inf)
        last[keys[t]] = t
    cache, heap, hits = {}, [], 0  # key -> 次の使用時刻、heap は (-次の使用時刻, key)
    for t, k in enumerate(keys):
        if k in cache:
            hits += 1
        elif len(cache) >= cap:
            while True:
                nu, v = heapq.heappop(heap)
                if v in cache and cache[v] == -nu:
                    del cache[v]
                    break
        cache[k] = nxt[t]
        heapq.heappush(heap, (-nxt[t], k))
    return hits


def che_hit_rate(p, cap: float) -> float:
    """Che 近似: Σ (1 - e^{-p_i T}) = C を T について解き、H = Σ p_i (1 - e^{-p_i T})。"""
    p = [x for x in p if x > 0]
    if cap >= len(p):
        return 1.0
    occ = lambda T: sum(1 - math.exp(-x * T) for x in p)
    lo, hi = 0.0, 1.0
    while occ(hi) < cap:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if occ(mid) < cap else (lo, mid)
    T = (lo + hi) / 2
    return sum(x * (1 - math.exp(-x * T)) for x in p)


def popularity(keys) -> list[float]:
    c = Counter(keys)
    n = len(keys)
    return [v / n for v in c.values()]
