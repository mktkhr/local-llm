#!/usr/bin/env python3
"""Needle In A Haystack ランナー。

事前生成された needle データ(data/needle/*.json)を 1 件読み、対象モデルに
問い合わせて needle_id が応答に含まれるかを判定する。

使い方:
    python run_needle.py --model qwen3.5:9b --ctx 32768 \\
        --needle data/needle/32k_p50.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from client import OllamaClient
from metadata import collect, write
from vram import VramMonitor


def score(response: str, needle_id: str) -> bool:
    return needle_id in response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--needle", type=Path, required=True, help="data/needle/*.json")
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--think", choices=["true", "false"], default="false")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    think = args.think == "true"
    needle_data = json.loads(args.needle.read_text())

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect(
        extra={
            "runner": "run_needle",
            "model": args.model,
            "num_ctx": args.ctx,
            "needle_file": str(args.needle),
            "needle_id": needle_data["needle_id"],
            "position_pct": needle_data["position_pct"],
            "think": think,
        }
    )
    write(meta, out_dir / "metadata.json")

    with OllamaClient() as client:
        print(
            f"[run] model={args.model} ctx={args.ctx} "
            f"needle_id={needle_data['needle_id']} pos={needle_data['position_pct']}"
        )
        with VramMonitor(interval_sec=0.5) as vm:
            result = client.generate(
                args.model,
                needle_data["prompt"],
                num_ctx=args.ctx,
                num_predict=args.num_predict,
                think=think,
                keep_alive="5m",
            )
        vram_stats = vm.stats()
        ps_after = client.ps()
        client.unload(args.model)

    success = score(result.response_text, needle_data["needle_id"])

    out = {
        "model": args.model,
        "num_ctx": args.ctx,
        "needle_id": needle_data["needle_id"],
        "position_pct": needle_data["position_pct"],
        "approximate_chars": needle_data.get("approximate_chars"),
        "think": think,
        "success": success,
        "response_excerpt": result.response_text[:500],
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "decode_tokens_per_sec": round(result.decode_tokens_per_sec, 2),
        "vram_peak_used_mib": vram_stats.peak_used_mib,
        "vram_min_free_mib": vram_stats.min_free_mib,
        "ollama_ps_after": ps_after,
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = (
        out_dir
        / f"needle_{sanitized}_ctx{args.ctx}_pos{int(needle_data['position_pct'] * 100)}.json"
    )
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"[done] success={success}")
    print(f"  {out_file}")


if __name__ == "__main__":
    main()
