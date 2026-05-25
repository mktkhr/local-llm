"""測定時バージョン・環境情報の取得。

docs/06-evaluation.md §3 のバージョン記録要件に対応。各計測実行ごとに
results/<timestamp>/metadata.json として保存される情報を組み立てる。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _ollama_version() -> str:
    out = _run(["ollama", "--version"])
    m = re.search(r"version is (\S+)", out)
    return m.group(1) if m else out


def _nvidia_info() -> dict[str, str]:
    fields = ["name", "driver_version", "memory.total", "compute_cap"]
    out = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader",
        ]
    )
    parts = [p.strip() for p in out.split(",")] if out else [""] * len(fields)
    info = dict(zip(fields, parts, strict=False))

    cuda_out = _run(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*(\S+)", cuda_out)
    info["cuda_version"] = cuda_match.group(1) if cuda_match else ""

    return info


def _os_info() -> dict[str, str]:
    return {
        "kernel": _run(["uname", "-r"]),
        "distribution": _run(["lsb_release", "-ds"]).strip('"'),
    }


def _git_hash(repo_dir: Path) -> str:
    return _run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"])


@dataclass
class Metadata:
    timestamp: str
    ollama_version: str
    gpu: dict[str, str]
    os: dict[str, str]
    git_hash: str
    user: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect(repo_dir: Path | None = None, extra: dict[str, Any] | None = None) -> Metadata:
    """現環境のメタデータを採取する。"""
    repo_dir = repo_dir or Path(__file__).resolve().parent.parent.parent
    return Metadata(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        ollama_version=_ollama_version(),
        gpu=_nvidia_info(),
        os=_os_info(),
        git_hash=_git_hash(repo_dir),
        user=os.environ.get("USER", ""),
        extra=extra or {},
    )


def write(meta: Metadata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))
