#!/usr/bin/env python3
"""モデルの最大 num_ctx を二分探索で求める。

各プローブで Ollama に短いプロンプトを投げて load → /api/ps と nvidia-smi で
状態確認 → unload を繰り返す。OK 条件は以下:

- /api/ps の size == size_vram(= 100% GPU、CPU オフロードなし)
- nvidia-smi の free memory ≥ min-free-mib

OK ならその ctx は採用可、NG ならより小さい ctx を試す、を二分探索する。

使い方:
    python ctx_search.py --model qwen3.5:9b
    python ctx_search.py --model qwen3.5:9b --low 4096 --high 262144 --tolerance 4096
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from client import OllamaClient
from metadata import collect, write


@dataclass
class ProbeResult:
    ctx: int
    effective_ctx: int  # ollama ps の context_length(モデル上限でキャップされた場合は要求より小さい)
    success: bool
    reason: str  # "ok" | "ok_capped" | "oom" | "cpu_offload" | "low_vram" | "error"
    size_bytes: int = 0
    size_vram_bytes: int = 0
    vram_free_mib: int = 0
    error: str | None = None


@dataclass
class SearchResult:
    model: str
    max_ctx: int  # 採用可能だった最大値。0 なら下限すら通らなかった
    low: int
    high: int
    tolerance: int
    min_free_mib: int
    history: list[ProbeResult] = field(default_factory=list)


def _nvidia_free_mib() -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split("\n")[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0


def probe(
    client: OllamaClient,
    model: str,
    ctx: int,
    min_free_mib: int = 500,
) -> ProbeResult:
    """指定 ctx でロードして OK/NG を判定し、必ず unload して返す。

    Ollama がモデル上限で要求 ctx をキャップした場合は effective_ctx に実値を入れ、
    reason を "ok_capped" にする。
    """
    try:
        client.generate(
            model,
            "hi",
            num_ctx=ctx,
            num_predict=1,
            think=False,
            keep_alive="2m",
        )
    except Exception as e:
        return ProbeResult(ctx=ctx, effective_ctx=0, success=False, reason="error", error=str(e))

    info = client.model_info_in_ps(model)
    if info is None:
        client.unload(model)
        client.wait_unloaded(model)
        return ProbeResult(
            ctx=ctx, effective_ctx=0, success=False, reason="oom",
            error="not in ps after load",
        )

    size = int(info.get("size", 0))
    size_vram = int(info.get("size_vram", 0))
    effective = int(info.get("context_length", ctx))
    free_mib = _nvidia_free_mib()

    client.unload(model)
    client.wait_unloaded(model)

    if size > 0 and size_vram < size:
        return ProbeResult(
            ctx=ctx, effective_ctx=effective,
            success=False, reason="cpu_offload",
            size_bytes=size, size_vram_bytes=size_vram, vram_free_mib=free_mib,
        )

    if free_mib < min_free_mib:
        return ProbeResult(
            ctx=ctx, effective_ctx=effective,
            success=False, reason="low_vram",
            size_bytes=size, size_vram_bytes=size_vram, vram_free_mib=free_mib,
        )

    return ProbeResult(
        ctx=ctx,
        effective_ctx=effective,
        success=True,
        reason="ok_capped" if effective < ctx else "ok",
        size_bytes=size, size_vram_bytes=size_vram, vram_free_mib=free_mib,
    )


def _align(value: int, step: int) -> int:
    return max(step, (value // step) * step)


def binary_search_max_ctx(
    client: OllamaClient,
    model: str,
    *,
    low: int = 4096,
    high: int = 262144,
    tolerance: int = 4096,
    min_free_mib: int = 500,
    align_step: int = 1024,
) -> SearchResult:
    """[low, high] 範囲で最大 ctx を二分探索。"""
    history: list[ProbeResult] = []
    result = SearchResult(
        model=model,
        max_ctx=0,
        low=low,
        high=high,
        tolerance=tolerance,
        min_free_mib=min_free_mib,
        history=history,
    )

    print(f"[probe] low={low}")
    res_low = probe(client, model, low, min_free_mib)
    history.append(res_low)
    print(f"  -> {'ok' if res_low.success else 'NG'} ({res_low.reason})")
    if not res_low.success:
        return result  # 下限も通らない
    result.max_ctx = res_low.effective_ctx if res_low.reason == "ok_capped" else low

    print(f"[probe] high={high}")
    res_high = probe(client, model, high, min_free_mib)
    history.append(res_high)
    print(f"  -> {'ok' if res_high.success else 'NG'} ({res_high.reason})")
    if res_high.success:
        # capped の場合は実効値を max_ctx として返し、これ以上探索しない
        result.max_ctx = res_high.effective_ctx if res_high.reason == "ok_capped" else high
        return result

    lo, hi = low, high
    while hi - lo > tolerance:
        mid = _align((lo + hi) // 2, align_step)
        if mid <= lo or mid >= hi:
            break
        print(f"[probe] mid={mid} (range {lo}..{hi})")
        res = probe(client, model, mid, min_free_mib)
        history.append(res)
        print(f"  -> {'ok' if res.success else 'NG'} ({res.reason})")
        if res.success:
            if res.reason == "ok_capped":
                # mid 要求が capped された = モデル上限が effective_ctx
                result.max_ctx = res.effective_ctx
                return result
            lo = mid
            result.max_ctx = mid
        else:
            hi = mid

    return result


def _result_to_dict(res: SearchResult) -> dict[str, Any]:
    return {
        "model": res.model,
        "max_ctx": res.max_ctx,
        "search_range_low": res.low,
        "search_range_high": res.high,
        "tolerance": res.tolerance,
        "min_free_mib": res.min_free_mib,
        "history": [asdict(p) for p in res.history],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--low", type=int, default=4096)
    parser.add_argument("--high", type=int, default=262144)
    parser.add_argument("--tolerance", type=int, default=4096)
    parser.add_argument("--min-free-mib", type=int, default=500)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect(
        extra={
            "runner": "ctx_search",
            "model": args.model,
            "low": args.low,
            "high": args.high,
            "tolerance": args.tolerance,
            "min_free_mib": args.min_free_mib,
        }
    )
    write(meta, out_dir / "metadata.json")

    with OllamaClient() as client:
        result = binary_search_max_ctx(
            client,
            args.model,
            low=args.low,
            high=args.high,
            tolerance=args.tolerance,
            min_free_mib=args.min_free_mib,
        )

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"ctx_search_{sanitized}.json"
    out_file.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))

    print()
    print(f"[result] max_ctx = {result.max_ctx} ({len(result.history)} probes)")
    print(f"[done] {out_file}")


if __name__ == "__main__":
    main()
