#!/usr/bin/env python3
"""Coding 評価ランナー。

タスクファイル(JSON 配列)を読み、各タスクを 1 件ずつモデルに投げて応答を採点する。

使い方:
    python run_coding.py --model qwen3.5:9b-q4_K_M --ctx 8192 \\
        --tasks data/coding/tasks.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from client import OllamaClient
from metadata import collect, write
from scorer import score_coding_task
from vram import VramMonitor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="JSON 配列のタスクファイル")
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--think", choices=["true", "false"], default="false")
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
            "runner": "run_coding",
            "model": args.model,
            "num_ctx": args.ctx,
            "tasks_file": str(args.tasks),
            "task_count": len(tasks),
            "think": think,
        }
    )
    write(meta, out_dir / "metadata.json")

    per_task: list[dict[str, Any]] = []
    with OllamaClient() as client:
        with VramMonitor(interval_sec=1.0) as vm:
            for i, task in enumerate(tasks, 1):
                print(f"[{i}/{len(tasks)}] {task['task_id']} ({task.get('task_type', '?')})")
                gen = client.generate(
                    args.model,
                    task["prompt"],
                    num_ctx=args.ctx,
                    num_predict=args.num_predict,
                    think=think,
                    keep_alive="5m",
                )
                scored = score_coding_task(task, gen.response_text, exec_timeout_sec=args.exec_timeout_sec)
                scored_dict = scored.to_dict()
                scored_dict["response_excerpt"] = gen.response_text[:600]
                per_task.append(scored_dict)
                print(f"   passed={scored.passed}/{scored.total} (rate={scored.pass_rate:.2f})")
        vram_stats = vm.stats()
        client.unload(args.model)

    by_type: dict[str, list[float]] = {}
    for r in per_task:
        by_type.setdefault(r["task_type"], []).append(r["pass_rate"])
    type_summary = {t: round(sum(rs) / len(rs), 3) for t, rs in by_type.items() if rs}

    summary = {
        "model": args.model,
        "num_ctx": args.ctx,
        "think": think,
        "task_count": len(tasks),
        "overall_pass_rate": round(
            sum(r["pass_rate"] for r in per_task) / len(per_task), 3
        ) if per_task else 0.0,
        "by_task_type": type_summary,
        "vram_peak_used_mib": vram_stats.peak_used_mib,
        "vram_min_free_mib": vram_stats.min_free_mib,
        "tasks": per_task,
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"coding_{sanitized}_ctx{args.ctx}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print()
    print(f"[done] overall_pass_rate = {summary['overall_pass_rate']}")
    print(f"  by type: {summary['by_task_type']}")
    print(f"  {out_file}")


if __name__ == "__main__":
    main()
