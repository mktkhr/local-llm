#!/usr/bin/env python3
"""Needle In A Haystack ランナー (Apple Silicon / MLX 版)。

事前生成された needle データ(data/needle/*.json)を 1 件読み、対象モデルに
問い合わせて needle_id が応答に含まれるかを判定する。

使い方:
    uv run python run_needle_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --ctx 32768 --needle data/needle/32k_p50.json
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

from client_mlx import apply_chat_template, generate_with_metrics, load_model, warmup
from memory_mlx import clear_cache, get_active_memory_mib, reset_peak
from metadata_mlx import collect, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, required=True, help="max_kv_size")
    parser.add_argument("--needle", type=Path, required=True)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--think", choices=["true", "false"], default="false")
    parser.add_argument("--kv-bits", type=int, choices=[4, 8], default=None)
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
            "runner": "run_needle_mlx",
            "model": args.model,
            "max_kv_size": args.ctx,
            "needle_file": str(args.needle),
            "needle_id": needle_data["needle_id"],
            "position_pct": needle_data["position_pct"],
            "think": think,
            "kv_bits": args.kv_bits,
        }
    )
    write(meta, out_dir / "metadata.json")

    print(f"[load] {args.model}")
    model, tokenizer = load_model(args.model)
    print(f"  active={get_active_memory_mib():.0f} MiB")

    print(f"[warmup] 1 token")
    warmup(model, tokenizer, kv_bits=args.kv_bits)
    reset_peak()

    text = apply_chat_template(tokenizer, needle_data["prompt"], enable_thinking=think)

    print(
        f"[run] ctx={args.ctx} needle_id={needle_data['needle_id']} "
        f"pos={needle_data['position_pct']}"
    )
    result = generate_with_metrics(
        model,
        tokenizer,
        text,
        max_tokens=args.num_predict,
        kv_bits=args.kv_bits,
        max_kv_size=args.ctx,
    )

    success = needle_data["needle_id"] in result.response_text

    out = {
        "model": args.model,
        "max_kv_size": args.ctx,
        "needle_id": needle_data["needle_id"],
        "position_pct": needle_data["position_pct"],
        "approximate_chars": needle_data.get("approximate_chars"),
        "think": think,
        "kv_bits": args.kv_bits,
        "success": success,
        "response_excerpt": result.response_text[:500],
        "prompt_tokens": result.prompt_tokens,
        "prompt_tps": round(result.prompt_tps, 2),
        "generation_tokens": result.generation_tokens,
        "generation_tps": round(result.generation_tps, 2),
        "ttft_sec": round(result.ttft_sec, 3),
        "peak_memory_mib": round(result.peak_memory_mib_mx, 1),
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = (
        out_dir
        / f"needle_{sanitized}_ctx{args.ctx}_pos{int(needle_data['position_pct'] * 100)}.json"
    )
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"[done] success={success}")
    print(f"  {out_file}")

    del model, tokenizer
    gc.collect()
    clear_cache()


if __name__ == "__main__":
    main()
