#!/usr/bin/env python3
"""階層型 KV キャッシュの Case A〜E を計測する。

Case 定義 (notes/02-design.md §2):

    A  cache なし        未見の prompt を投入 -> full prefill
    B  VRAM hit          同一 prompt を連続投入 -> prefill skip
    C  DRAM 退避 -> 再開  別 session に slot を奪わせ、prompt cache から復元
    D  SSD 退避 -> 再開   slot save -> page cache を落とす -> restore
    E  partial prefix    prefix は共有し tail だけ差し替える

再 Prefill が起きたかは HTTP 200 では判定しない。/completion の
timings.cache_n (再利用トークン数) と prompt_n (実処理トークン数) で判定する。
docs/pd/01-research.md §4.2 と同じ基準。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]


def post(base_url: str, path: str, payload: dict | None = None, timeout: int = 3600) -> dict:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


def get(base_url: str, path: str, timeout: int = 60):
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as res:
        return json.loads(res.read())


def completion(base_url: str, prompt: str, n_predict: int = 16, cache_prompt: bool = True) -> dict:
    t0 = time.perf_counter()
    r = post(
        base_url,
        "/completion",
        {"prompt": prompt, "n_predict": n_predict, "cache_prompt": cache_prompt, "temperature": 0},
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    t = r["timings"]
    return {
        "wall_ms": round(wall_ms, 1),
        "prompt_ms": round(t["prompt_ms"], 1),
        "prompt_n": t["prompt_n"],
        "cache_n": t["cache_n"],
        "prompt_per_second": round(t.get("prompt_per_second") or 0.0, 1),
        "predicted_ms": round(t["predicted_ms"], 1),
        "predicted_n": t["predicted_n"],
        "predicted_per_second": round(t.get("predicted_per_second") or 0.0, 1),
        "content_head": r.get("content", "")[:48],
    }


def drop_file_page_cache(path: pathlib.Path) -> bool:
    """特定ファイルの page cache だけを落とす。root 権限は不要。

    これをやらないと Case D は Case C になる。書いた直後の KV ファイルは
    ほぼ確実に page cache に残っているため、SSD から読んだことにならない。
    POSIX_FADV_DONTNEED は clean page にしか効かないので先に fsync する。
    """
    if not path.exists():
        return False
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def cached_bytes(path: pathlib.Path) -> int:
    """mincore 相当。page cache に載っているバイト数を返す。"""
    try:
        out = subprocess.run(
            ["python3", "-c", CACHED_SNIPPET, str(path)], capture_output=True, text=True, timeout=60
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1


CACHED_SNIPPET = """
import ctypes, os, sys
p = sys.argv[1]
fd = os.open(p, os.O_RDONLY)
size = os.fstat(fd).st_size
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_long]
PROT_READ, MAP_SHARED = 0x1, 0x1
addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
pages = (size + 4095) // 4096
vec = (ctypes.c_ubyte * pages)()
libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
os.close(fd)
print(sum(v & 1 for v in vec) * 4096)
"""


def gpu_mem_mib() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(r.stdout.strip().splitlines()[0])


def pcie_gen() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return r.stdout.strip().splitlines()[0].replace(", ", "x")


def host_mem_mib() -> dict:
    info = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) // 1024
    return {"mem_available_mib": info["MemAvailable"], "cached_mib": info["Cached"]}


def slot_save(base_url: str, filename: str) -> dict:
    r = post(base_url, f"/slots/0?action=save", {"filename": filename})
    return {"save_ms": round(r["timings"]["save_ms"], 1), "n_written": r["n_written"],
            "n_saved": r["n_saved"]}


def slot_restore(base_url: str, filename: str) -> dict:
    r = post(base_url, f"/slots/0?action=restore", {"filename": filename})
    return {"restore_ms": round(r["timings"]["restore_ms"], 1), "n_read": r["n_read"],
            "n_restored": r["n_restored"]}


def slot_erase(base_url: str) -> None:
    post(base_url, "/slots/0?action=erase", {})


def run(base_url: str, kv_dir: pathlib.Path, ctx_dir: pathlib.Path, target: int,
        kv_type: str, n_ctx: int) -> list[dict]:
    prefix = (ctx_dir / f"ctx_{target}_prefix.txt").read_text(encoding="utf-8")
    tail = (ctx_dir / f"ctx_{target}_tail.txt").read_text(encoding="utf-8")
    prompt = prefix + tail
    # slot を奪うための、共通接頭辞を持たない別 session
    other = "OTHER SESSION. " + ("独立したセッションの本文です。" * 400)

    base = {"target_tokens": target, "kv_type": kv_type, "n_ctx": n_ctx,
            "pcie": pcie_gen(), **host_mem_mib()}
    rows: list[dict] = []
    fname = f"kvtier_{target}_{kv_type}.bin"
    fpath = kv_dir / fname

    # --- Case A: cache なし (full prefill) ---
    slot_erase(base_url)
    post(base_url, "/completion", {"prompt": other, "n_predict": 1, "temperature": 0})
    slot_erase(base_url)
    r = completion(base_url, prompt)
    rows.append({**base, "case": "A_full_prefill", **r, "vram_mib": gpu_mem_mib()})

    # --- Case B: VRAM hit ---
    r = completion(base_url, prompt)
    rows.append({**base, "case": "B_vram_hit", **r, "vram_mib": gpu_mem_mib()})

    # --- Case D: SSD 退避 -> 再開 ---
    # 先に保存しておく (この時点の slot は目的の KV を保持している)
    sv = slot_save(base_url, fname)
    # slot を別 session に奪わせ、さらに erase して VRAM/DRAM 両方から消す
    post(base_url, "/completion", {"prompt": other, "n_predict": 1, "temperature": 0})
    slot_erase(base_url)
    before = cached_bytes(fpath)
    drop_file_page_cache(fpath)
    after = cached_bytes(fpath)
    rs = slot_restore(base_url, fname)
    r = completion(base_url, prompt)
    rows.append({**base, "case": "D_ssd_restore", **sv, **rs, **r,
                 "page_cached_before_mib": round(before / 2**20, 1),
                 "page_cached_after_mib": round(after / 2**20, 1),
                 "vram_mib": gpu_mem_mib()})

    # --- Case D-warm: page cache を落とさない対照 ---
    post(base_url, "/completion", {"prompt": other, "n_predict": 1, "temperature": 0})
    slot_erase(base_url)
    rs = slot_restore(base_url, fname)
    r = completion(base_url, prompt)
    rows.append({**base, "case": "Dw_pagecache_restore", **sv, **rs, **r,
                 "page_cached_before_mib": round(cached_bytes(fpath) / 2**20, 1),
                 "vram_mib": gpu_mem_mib()})

    # --- Case C: DRAM 退避 -> 再開 ---
    # prompt を一度処理して slot に載せ、別 session を投げて prompt cache へ追い出す。
    # erase はしない (erase すると prompt cache からも消える)。
    r0 = completion(base_url, prompt)
    post(base_url, "/completion", {"prompt": other, "n_predict": 1, "temperature": 0})
    r = completion(base_url, prompt)
    rows.append({**base, "case": "C_dram_restore", **r, "vram_mib": gpu_mem_mib()})

    # --- Case E: partial prefix reuse ---
    tail2 = tail.replace("列挙し", "列挙したうえで優先度をつけ")
    r = completion(base_url, prefix + tail2)
    rows.append({**base, "case": "E_partial_prefix", **r, "vram_mib": gpu_mem_mib()})

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18090")
    ap.add_argument("--kv-dir", default=str(REPO / "config" / "kv"))
    ap.add_argument("--ctx-dir", default=str(REPO / "results" / "raw" / "ctx"))
    ap.add_argument("--tokens", type=int, nargs="+", required=True)
    ap.add_argument("--kv-type", default="q8_0")
    ap.add_argument("--n-ctx", type=int, default=49152)
    ap.add_argument("--out", default=str(REPO / "results" / "kvtier_runs.jsonl"))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for target in args.tokens:
            for row in run(args.base_url, pathlib.Path(args.kv_dir), pathlib.Path(args.ctx_dir),
                           target, args.kv_type, args.n_ctx):
                row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{row['case']:>22} target={target:>6} "
                      f"prompt_n={row['prompt_n']:>6} cache_n={row['cache_n']:>6} "
                      f"prompt_ms={row['prompt_ms']:>9} wall_ms={row['wall_ms']:>9} "
                      f"restore_ms={row.get('restore_ms','-')}")


if __name__ == "__main__":
    main()
