"""nvidia-smi をポーリングして VRAM 使用量のピークを取る。

GPU 利用率も同時にサンプリング。docs/06-evaluation.md §6.3 に対応。
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class VramSample:
    timestamp: float
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int


@dataclass
class VramStats:
    samples: list[VramSample]

    @property
    def peak_used_mib(self) -> int:
        return max((s.memory_used_mib for s in self.samples), default=0)

    @property
    def min_free_mib(self) -> int:
        return min((s.memory_free_mib for s in self.samples), default=0)

    @property
    def mean_utilization_pct(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.utilization_pct for s in self.samples) / len(self.samples)


def _query() -> VramSample | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    parts = [p.strip() for p in out.strip().split(",")]
    if len(parts) < 3:
        return None
    try:
        return VramSample(
            timestamp=time.monotonic(),
            memory_used_mib=int(parts[0]),
            memory_free_mib=int(parts[1]),
            utilization_pct=int(parts[2]),
        )
    except ValueError:
        return None


class VramMonitor:
    """nvidia-smi を別スレッドで定期サンプリングするコンテキストマネージャ。"""

    def __init__(self, interval_sec: float = 1.0):
        self._interval = interval_sec
        self._samples: list[VramSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> VramMonitor:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_a: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = _query()
            if sample is not None:
                self._samples.append(sample)
            self._stop.wait(self._interval)

    def stats(self) -> VramStats:
        return VramStats(samples=list(self._samples))
