"""ユニファイドメモリの使用量計測 (Apple Silicon / MLX 用)。

役割は bench/vram.py に相当。
- Metal の wired_limit / max_recommended_working_set_size から「実用上の GPU 上限」を取る
- mx.get_peak_memory() でモデル+KV のピークを取る
- system memory pressure を補足情報として吐く
"""

from __future__ import annotations

import subprocess
from typing import Any

import mlx.core as mx


def device_info() -> dict[str, Any]:
    """Metal デバイスの情報を取得。GPU の実効容量はここから決める。

    - memory_size: ユニファイドメモリ総量(byte)
    - max_recommended_working_set_size: GPU が同時確保すべきでない上限(byte、≒VRAM 相当)
    - max_buffer_length: 単一バッファ上限(モデル重みの単一テンソル割当で効く)
    """
    info = mx.device_info()
    return {
        "device_name": info.get("device_name"),
        "architecture": info.get("architecture"),
        "memory_size_bytes": int(info.get("memory_size", 0)),
        "max_recommended_working_set_bytes": int(info.get("max_recommended_working_set_size", 0)),
        "max_buffer_length_bytes": int(info.get("max_buffer_length", 0)),
        "resource_limit": info.get("resource_limit"),
    }


def get_wired_limit_mb() -> int:
    """sysctl iogpu.wired_limit_mb を取得。0 なら macOS デフォルト(動的)。"""
    try:
        out = subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip()
        return int(out)
    except Exception:
        return -1


def effective_gpu_limit_mib() -> int:
    """実用上の GPU 上限を MiB で返す。

    - iogpu.wired_limit_mb > 0 が設定されていればそれを採用
    - そうでなければ max_recommended_working_set_size を採用
    """
    wl = get_wired_limit_mb()
    if wl and wl > 0:
        return wl
    info = device_info()
    return int(info["max_recommended_working_set_bytes"] / (1024 * 1024))


def vm_stat_summary() -> dict[str, int]:
    """vm_stat を解析して page free / active / inactive / wired を返す(MiB)。"""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return {}
    lines = out.splitlines()
    page_size = 16384  # M4 / arm64 Mac は 16K ページ
    header = lines[0] if lines else ""
    if "page size of" in header:
        try:
            page_size = int(header.split("page size of")[1].split("bytes")[0].strip())
        except Exception:
            pass
    fields: dict[str, int] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        val = val.strip().rstrip(".")
        try:
            pages = int(val)
        except ValueError:
            continue
        fields[key] = pages * page_size // (1024 * 1024)
    return fields


def get_peak_memory_mib() -> float:
    """これまでのピーク使用量を MiB で返す。"""
    return mx.get_peak_memory() / (1024 * 1024)


def get_active_memory_mib() -> float:
    """現在の active 使用量を MiB で返す。"""
    return mx.get_active_memory() / (1024 * 1024)


def reset_peak() -> None:
    mx.reset_peak_memory()


def clear_cache() -> None:
    """MLX 内部メモリプールをクリア。モデルアンロード後に呼ぶ。"""
    mx.clear_cache()


def set_wired_limit(mib: int) -> int:
    """Metal の wired limit を引き上げる(byte 単位で渡す)。

    成功時は実際に設定された値(byte)を返す。失敗時は 0。
    過度な値を渡すと OS が拒否する点に注意。
    """
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "set_wired_limit"):
            return int(mx.metal.set_wired_limit(mib * 1024 * 1024))
    except Exception:
        pass
    return 0
