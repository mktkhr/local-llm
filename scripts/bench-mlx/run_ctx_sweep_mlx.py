#!/usr/bin/env python3
"""ctx 依存の速度スイープ (Apple Silicon / MLX 版)。

同一モデルで ctx を変えながら decode / prefill / TTFT を測り、
「decode は帯域律速で ctx に対し緩やかに低下、prefill(TTFT)は compute 律速で
ctx に対し急激に悪化」を 1 モデル内で分離して示すためのデータを取る。

各 ctx 点で「ctx 相当のフィラーを prefill → num_predict トークン生成」を実行し、
prompt_tps(prefill 速度)/ ttft_sec / generation_tps(その ctx での decode 速度)/
peak メモリ / swap(vm_stat 差分)を記録する。

使い方:
    uv run python run_ctx_sweep_mlx.py --model mlx-community/Qwen3.5-4B-MLX-4bit \\
        --kv-bits 8 --ctx-points 4096,16384,65536 --max-ctx 262144
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from client_mlx import generate_with_metrics, load_model, warmup
from ctx_search_mlx import _build_filler_prompt
from memory_mlx import clear_cache, get_active_memory_mib, reset_peak, vm_stat_summary
from metadata_mlx import collect, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument(
        "--ctx-points",
        default="4096,16384,65536",
        help="カンマ区切りの ctx 点。--max-ctx を渡すとそれも末尾に加える",
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=None,
        help="ctx_search で求めた最大 ctx。指定すると末尾の点として追加",
    )
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    ctx_points = [int(x) for x in args.ctx_points.split(",") if x.strip()]
    if args.max_ctx and args.max_ctx not in ctx_points:
        ctx_points.append(args.max_ctx)
    ctx_points = sorted(set(ctx_points))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect(
        extra={
            "runner": "run_ctx_sweep_mlx",
            "model": args.model,
            "kv_bits": args.kv_bits,
            "ctx_points": ctx_points,
            "num_predict": args.num_predict,
        }
    )
    write(meta, out_dir / "metadata.json")

    print(f"[load] {args.model}")
    model, tokenizer = load_model(args.model)
    print(f"  active={get_active_memory_mib():.0f} MiB")

    print(f"[warmup] 1 token")
    warmup(model, tokenizer, kv_bits=args.kv_bits)

    points: list[dict[str, Any]] = []
    for ctx in ctx_points:
        target_tokens = max(64, ctx - args.num_predict - 32)
        prompt_text, actual_prompt_tokens = _build_filler_prompt(tokenizer, target_tokens)

        swap_before = vm_stat_summary().get("swapouts", 0)
        reset_peak()
        print(f"[sweep] ctx≈{ctx} (prompt {actual_prompt_tokens} tok)")
        result = generate_with_metrics(
            model,
            tokenizer,
            prompt_text,
            max_tokens=args.num_predict,
            kv_bits=args.kv_bits,
        )
        swap_after = vm_stat_summary().get("swapouts", 0)

        point = {
            "ctx": ctx,
            "prompt_tokens": result.prompt_tokens,
            "prefill_tps": round(result.prompt_tps, 2),
            "ttft_sec": round(result.ttft_sec, 3),
            "decode_tps": round(result.generation_tps, 2),
            "generation_tokens": result.generation_tokens,
            "peak_memory_mib": round(result.peak_memory_mib_mx, 1),
            "swap_delta_mib": swap_after - swap_before,
            "finish_reason": result.finish_reason,
        }
        points.append(point)
        print(
            f"  prefill={point['prefill_tps']} tok/s  TTFT={point['ttft_sec']}s  "
            f"decode={point['decode_tps']} tok/s  peak={point['peak_memory_mib']} MiB"
        )

    out = {
        "model": args.model,
        "kv_bits": args.kv_bits,
        "num_predict": args.num_predict,
        "points": points,
    }
    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"ctxsweep_{sanitized}.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"[done] {out_file}")

    del model, tokenizer
    gc.collect()
    clear_cache()


if __name__ == "__main__":
    main()
