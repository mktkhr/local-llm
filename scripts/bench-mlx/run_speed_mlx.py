#!/usr/bin/env python3
"""速度計測ランナー (Apple Silicon / MLX 用)。

ロード → ストリーミング生成 (TTFT + 速度) → アンロード、を 1 モデル × 1 構成で実行する。

mlx-lm の stream_generate は prompt_tps / generation_tps / peak_memory を内部で計測して
返してくれるため、bench/ 版のように非ストリーミング+ストリーミングの 2 回回しは不要。

使い方:
    uv run python run_speed_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --ctx 32768

結果は results/<timestamp>/ 配下に speed_*.json と metadata.json として保存。
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from client_mlx import apply_chat_template, generate_with_metrics, load_model, warmup
from memory_mlx import clear_cache, get_active_memory_mib, get_peak_memory_mib, reset_peak
from metadata_mlx import collect, write

DEFAULT_PROMPT = (
    "日本語で1段落、自身の1日の予定について簡潔に書いてください。"
    "出力は3〜5文程度にしてください。"
)


def _sanitize(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace repo id またはローカルパス")
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="max_kv_size (KV キャッシュ最大長)。未指定なら無制限(モデル既定)",
    )
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--think",
        choices=["true", "false"],
        default="false",
        help="thinking モデル向け (Qwen3 系の enable_thinking)。代表値は false",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=[4, 8],
        default=None,
        help="KV キャッシュ量子化 bit 数。未指定なら非量子化",
    )
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--prefill-step", type=int, default=2048, help="prefill_step_size")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    think = args.think == "true"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect(
        extra={
            "runner": "run_speed_mlx",
            "model": args.model,
            "max_kv_size": args.ctx,
            "num_predict": args.num_predict,
            "think": think,
            "kv_bits": args.kv_bits,
            "kv_group_size": args.kv_group_size,
            "prefill_step_size": args.prefill_step,
        }
    )
    write(meta, out_dir / "metadata.json")

    reset_peak()
    print(f"[load] {args.model}")
    model, tokenizer = load_model(args.model)
    mem_after_load_mib = get_active_memory_mib()
    peak_after_load_mib = get_peak_memory_mib()
    print(f"  active={mem_after_load_mib:.1f} MiB, peak={peak_after_load_mib:.1f} MiB")

    # Metal shader 初回 JIT 分を計測から外す
    print(f"[warmup] 1 token")
    warmup(model, tokenizer, kv_bits=args.kv_bits)
    reset_peak()

    text = apply_chat_template(
        tokenizer,
        args.prompt,
        enable_thinking=(think if think else False),
    )

    print(f"[measure] generate ctx_max={args.ctx} kv_bits={args.kv_bits}")
    result = generate_with_metrics(
        model,
        tokenizer,
        text,
        max_tokens=args.num_predict,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        max_kv_size=args.ctx,
        prefill_step_size=args.prefill_step,
    )

    out = {
        "model": args.model,
        "max_kv_size": args.ctx,
        "num_predict": args.num_predict,
        "think": think,
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size,
        "prefill_step_size": args.prefill_step,
        "prompt_tokens": result.prompt_tokens,
        "prompt_tps": round(result.prompt_tps, 2),
        "generation_tokens": result.generation_tokens,
        "generation_tps": round(result.generation_tps, 2),
        "ttft_sec": round(result.ttft_sec, 3),
        "total_elapsed_sec": round(result.total_elapsed_sec, 3),
        "peak_memory_mib": round(result.peak_memory_mib_mx, 1),
        "peak_memory_gib_from_mlx_lm": round(result.peak_memory_gib, 3),
        "active_memory_mib_after_load": round(mem_after_load_mib, 1),
        "finish_reason": result.finish_reason,
        "response_excerpt": result.response_text[:300],
    }

    sanitized = _sanitize(args.model)
    ctx_label = args.ctx if args.ctx else "default"
    out_file = out_dir / f"speed_{sanitized}_ctx{ctx_label}.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"[done] {out_file}")
    print(f"  decode  : {out['generation_tps']} tok/s ({out['generation_tokens']} tok)")
    print(f"  prefill : {out['prompt_tps']} tok/s ({out['prompt_tokens']} tok)")
    print(f"  TTFT    : {out['ttft_sec']}s")
    print(f"  peak mem: {out['peak_memory_mib']} MiB")
    print(f"  finish  : {out['finish_reason']}")

    del model, tokenizer
    gc.collect()
    clear_cache()


if __name__ == "__main__":
    main()
