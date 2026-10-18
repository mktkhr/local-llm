#!/usr/bin/env python3
"""Coding 評価ランナー (Apple Silicon / MLX 版)。

タスクファイル(JSON 配列)を読み、各タスクを 1 件ずつモデルに投げて応答を採点する。
モデルは 1 回だけロードして全タスクを連続実行する(bench/ 版より無駄が少ない)。

使い方:
    uv run python run_coding_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --ctx 16384 --tasks data/coding/tasks.json
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from client_mlx import apply_chat_template, generate_with_metrics, load_model, warmup
from memory_mlx import clear_cache, get_active_memory_mib, reset_peak
from metadata_mlx import collect, write
from scorer import score_coding_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, required=True, help="max_kv_size")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--think", choices=["true", "false"], default="false")
    parser.add_argument("--kv-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--exec-timeout-sec", type=int, default=5)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    think = args.think == "true"
    tasks = json.loads(args.tasks.read_text())
    if not isinstance(tasks, list):
        raise ValueError("tasks file must be a JSON array")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect(
        extra={
            "runner": "run_coding_mlx",
            "model": args.model,
            "max_kv_size": args.ctx,
            "tasks_file": str(args.tasks),
            "task_count": len(tasks),
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

    per_task: list[dict[str, Any]] = []
    peak_overall = 0.0
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['task_id']} ({task.get('task_type', '?')})")
        text = apply_chat_template(tokenizer, task["prompt"], enable_thinking=think)
        reset_peak()
        result = generate_with_metrics(
            model,
            tokenizer,
            text,
            max_tokens=args.num_predict,
            kv_bits=args.kv_bits,
            max_kv_size=args.ctx,
        )
        peak_overall = max(peak_overall, result.peak_memory_mib_mx)
        scored = score_coding_task(task, result.response_text, exec_timeout_sec=args.exec_timeout_sec)
        scored_dict = scored.to_dict()
        scored_dict["response_excerpt"] = result.response_text[:600]
        scored_dict["generation_tps"] = round(result.generation_tps, 2)
        scored_dict["generation_tokens"] = result.generation_tokens
        scored_dict["finish_reason"] = result.finish_reason
        scored_dict["truncated"] = result.finish_reason == "length"
        per_task.append(scored_dict)
        trunc = " [TRUNCATED]" if result.finish_reason == "length" else ""
        print(f"   passed={scored.passed}/{scored.total} (rate={scored.pass_rate:.2f}){trunc}")

    by_type: dict[str, list[float]] = {}
    for r in per_task:
        by_type.setdefault(r["task_type"], []).append(r["pass_rate"])
    type_summary = {t: round(sum(rs) / len(rs), 3) for t, rs in by_type.items() if rs}

    truncated_count = sum(1 for r in per_task if r.get("truncated"))
    summary = {
        "model": args.model,
        "max_kv_size": args.ctx,
        "think": think,
        "kv_bits": args.kv_bits,
        "num_predict": args.num_predict,
        "task_count": len(tasks),
        "truncated_count": truncated_count,
        "overall_pass_rate": round(
            sum(r["pass_rate"] for r in per_task) / len(per_task), 3
        ) if per_task else 0.0,
        "by_task_type": type_summary,
        "peak_memory_mib": round(peak_overall, 1),
        "tasks": per_task,
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"coding_{sanitized}_ctx{args.ctx}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print()
    print(f"[done] overall_pass_rate = {summary['overall_pass_rate']}")
    print(f"  by type: {summary['by_task_type']}")
    print(f"  {out_file}")

    del model, tokenizer
    gc.collect()
    clear_cache()


if __name__ == "__main__":
    main()
