#!/usr/bin/env python3
"""速度計測ランナー。

ロード → ollama ps 取得 → 非ストリーミング計測(eval_* メトリクス)→
ストリーミング計測(TTFT)→ アンロード、を 1 モデル × 1 構成で実行する。

使い方:
    python run_speed.py --model qwen3.5:9b --ctx 32768

結果は results/<timestamp>/ 配下に speed_*.json と metadata.json として保存。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from client import OllamaClient
from metadata import collect, write
from vram import VramMonitor

DEFAULT_PROMPT = (
    "日本語で1段落、自身の1日の予定について簡潔に書いてください。"
    "出力は3〜5文程度にしてください。"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, default=None, help="num_ctx を明示。未指定なら Ollama 既定値")
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--think",
        choices=["true", "false"],
        default="false",
        help="thinking モデル向け。代表値は false",
    )
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
            "runner": "run_speed",
            "model": args.model,
            "num_ctx": args.ctx,
            "num_predict": args.num_predict,
            "think": think,
        }
    )
    write(meta, out_dir / "metadata.json")

    with OllamaClient() as client:
        print(f"[load] {args.model} ctx={args.ctx} think={think}")
        client.generate(
            args.model,
            "hi",
            num_ctx=args.ctx,
            num_predict=1,
            think=think,
            keep_alive="5m",
        )

        ps_after_load = client.ps()

        with VramMonitor(interval_sec=0.5) as vm:
            print("[measure] non-streaming for eval_* metrics ...")
            non_stream = client.generate(
                args.model,
                args.prompt,
                num_ctx=args.ctx,
                num_predict=args.num_predict,
                think=think,
                keep_alive="5m",
            )
            print("[measure] streaming for TTFT ...")
            stream = client.generate_stream(
                args.model,
                args.prompt,
                num_ctx=args.ctx,
                num_predict=args.num_predict,
                think=think,
                keep_alive="5m",
            )
        vram_stats = vm.stats()

        print("[unload]")
        client.unload(args.model)

    result = {
        "model": args.model,
        "num_ctx": args.ctx,
        "num_predict": args.num_predict,
        "think": think,
        "prompt_eval_count": non_stream.prompt_eval_count,
        "prompt_eval_duration_ns": non_stream.prompt_eval_duration_ns,
        "eval_count": non_stream.eval_count,
        "eval_duration_ns": non_stream.eval_duration_ns,
        "load_duration_ns": non_stream.load_duration_ns,
        "total_duration_ns": non_stream.total_duration_ns,
        "prefill_tokens_per_sec": round(non_stream.prefill_tokens_per_sec, 2),
        "decode_tokens_per_sec": round(non_stream.decode_tokens_per_sec, 2),
        "ttft_sec": round(stream.ttft_sec, 3),
        "stream_total_elapsed_sec": round(stream.total_elapsed_sec, 3),
        "vram_peak_used_mib": vram_stats.peak_used_mib,
        "vram_min_free_mib": vram_stats.min_free_mib,
        "gpu_mean_utilization_pct": round(vram_stats.mean_utilization_pct, 1),
        "ollama_ps_after_load": ps_after_load,
        "response_excerpt": non_stream.response_text[:300],
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"speed_{sanitized}_ctx{args.ctx}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"[done] {out_file}")
    print(f"  decode  : {result['decode_tokens_per_sec']} tok/s")
    print(f"  prefill : {result['prefill_tokens_per_sec']} tok/s")
    print(f"  TTFT    : {result['ttft_sec']}s")
    print(f"  VRAM    : peak={result['vram_peak_used_mib']} MiB, free={result['vram_min_free_mib']} MiB")
    print(f"  GPU util: {result['gpu_mean_utilization_pct']}%")


if __name__ == "__main__":
    main()
