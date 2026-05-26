"""測定時バージョン・環境情報の取得 (Apple Silicon / MLX)。

docs/06-evaluation.md §3 のバージョン記録要件に対応した MLX 版。
bench/metadata.py に対応するが、採取項目は Mac/MLX 用に差し替えている。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlx.core as mx

from memory_mlx import device_info, effective_gpu_limit_mib, get_wired_limit_mb


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return ""


def _mac_info() -> dict[str, Any]:
    """system_profiler / sw_vers / sysctl から Mac の情報を採取。"""
    sw = _run(["sw_vers"])
    os_ver = ""
    build = ""
    for line in sw.splitlines():
        if line.startswith("ProductVersion:"):
            os_ver = line.split(":", 1)[1].strip()
        elif line.startswith("BuildVersion:"):
            build = line.split(":", 1)[1].strip()

    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    ncpu = _run(["sysctl", "-n", "hw.ncpu"])
    ncpu_p = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
    ncpu_e = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
    memsize = _run(["sysctl", "-n", "hw.memsize"])
    model_id = _run(["sysctl", "-n", "hw.model"])
    kernel = _run(["uname", "-r"])

    try:
        mem_bytes = int(memsize) if memsize else 0
    except ValueError:
        mem_bytes = 0

    return {
        "chip": chip,
        "model_identifier": model_id,
        "ncpu_total": ncpu,
        "ncpu_performance": ncpu_p,
        "ncpu_efficiency": ncpu_e,
        "ram_total_gib": round(mem_bytes / (1024 ** 3), 2) if mem_bytes else None,
        "kernel": kernel,
        "macos_version": os_ver,
        "macos_build": build,
    }


def _mlx_info() -> dict[str, Any]:
    dev = device_info()
    wired = get_wired_limit_mb()
    return {
        "mlx_version": _pkg_version("mlx"),
        "mlx_lm_version": _pkg_version("mlx-lm"),
        "default_device": str(mx.default_device()),
        "device_name": dev["device_name"],
        "architecture": dev["architecture"],
        "memory_size_gib": round(dev["memory_size_bytes"] / (1024 ** 3), 2) if dev["memory_size_bytes"] else None,
        "max_recommended_working_set_gib": round(dev["max_recommended_working_set_bytes"] / (1024 ** 3), 2) if dev["max_recommended_working_set_bytes"] else None,
        "max_buffer_length_gib": round(dev["max_buffer_length_bytes"] / (1024 ** 3), 2) if dev["max_buffer_length_bytes"] else None,
        "iogpu_wired_limit_mb": wired,
        "effective_gpu_limit_mib": effective_gpu_limit_mib(),
    }


def _git_hash(repo_dir: Path) -> str:
    return _run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"])


@dataclass
class Metadata:
    timestamp: str
    runtime: str
    runtime_versions: dict[str, str]
    gpu: dict[str, Any]
    mac: dict[str, Any]
    git_hash: str
    user: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect(repo_dir: Path | None = None, extra: dict[str, Any] | None = None) -> Metadata:
    """現環境のメタデータを採取する。"""
    repo_dir = repo_dir or Path(__file__).resolve().parent.parent.parent
    mlx_info = _mlx_info()
    return Metadata(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        runtime="mlx-lm",
        runtime_versions={
            "mlx": mlx_info["mlx_version"],
            "mlx_lm": mlx_info["mlx_lm_version"],
        },
        gpu=mlx_info,
        mac=_mac_info(),
        git_hash=_git_hash(repo_dir),
        user=os.environ.get("USER", ""),
        extra=extra or {},
    )


def write(meta: Metadata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))
