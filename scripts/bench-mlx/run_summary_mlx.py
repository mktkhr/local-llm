#!/usr/bin/env python3
"""会議要約評価ランナー (Apple Silicon / MLX 版)。

会議文字起こし JSON(transcript + 期待要点/決定/TODO)を読み、固定の
要約プロンプトに渡して、応答を期待リストとの n-gram 被覆率で採点する。

使い方:
    uv run python run_summary_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --ctx 16384 --meetings data/summary/meetings.json
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
from scorer import score_summary_task

SUMMARY_PROMPT_TEMPLATE = """以下は会議の文字起こしです。会議の内容を読み、以下の3カテゴリで簡潔に箇条書きしてください。
カテゴリに該当する内容が無ければ「(なし)」と書いてください。

【文字起こし】
{transcript}

【出力フォーマット】
要点:
- ...

決定事項:
- ...

TODO:
- ...
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, required=True, help="max_kv_size")
    parser.add_argument("--meetings", type=Path, required=True)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--think", choices=["true", "false"], default="false")
    parser.add_argument("--kv-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    think = args.think == "true"
    meetings = json.loads(args.meetings.read_text())
    if not isinstance(meetings, list):
        raise ValueError("meetings file must be a JSON array")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect(
        extra={
            "runner": "run_summary_mlx",
            "model": args.model,
            "max_kv_size": args.ctx,
            "meetings_file": str(args.meetings),
            "meeting_count": len(meetings),
            "ngram": args.ngram,
            "threshold": args.threshold,
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

    per_meeting: list[dict[str, Any]] = []
    peak_overall = 0.0
    for i, m in enumerate(meetings, 1):
        print(f"[{i}/{len(meetings)}] {m['task_id']}")
        prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=m["transcript"])
        text = apply_chat_template(tokenizer, prompt, enable_thinking=think)
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
        scored = score_summary_task(
            m, result.response_text, ngram=args.ngram, threshold=args.threshold
        )
        d = scored.to_dict()
        d["response_excerpt"] = result.response_text[:800]
        d["generation_tps"] = round(result.generation_tps, 2)
        d["generation_tokens"] = result.generation_tokens
        d["finish_reason"] = result.finish_reason
        d["truncated"] = result.finish_reason == "length"
        per_meeting.append(d)
        print(f"   overall={scored.overall:.2f}")
        for c in scored.categories:
            print(f"   {c.category}: {c.matched_count}/{c.total}")

    overall = (
        sum(p["overall_match_rate"] for p in per_meeting) / len(per_meeting)
        if per_meeting
        else 0.0
    )
    truncated_count = sum(1 for p in per_meeting if p.get("truncated"))
    summary = {
        "model": args.model,
        "max_kv_size": args.ctx,
        "think": think,
        "kv_bits": args.kv_bits,
        "num_predict": args.num_predict,
        "ngram": args.ngram,
        "threshold": args.threshold,
        "meeting_count": len(meetings),
        "truncated_count": truncated_count,
        "overall_match_rate": round(overall, 3),
        "peak_memory_mib": round(peak_overall, 1),
        "meetings": per_meeting,
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"summary_{sanitized}_ctx{args.ctx}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print()
    print(f"[done] overall_match_rate = {summary['overall_match_rate']}")
    print(f"  {out_file}")

    del model, tokenizer
    gc.collect()
    clear_cache()


if __name__ == "__main__":
    main()
