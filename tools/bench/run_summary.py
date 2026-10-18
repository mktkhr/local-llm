#!/usr/bin/env python3
"""会議要約評価ランナー。

会議文字起こし JSON(transcript + 期待要点/決定/TODO)を読み、固定の
要約プロンプトに渡して、応答を期待リストとの n-gram 被覆率で採点する。

使い方:
    python run_summary.py --model qwen3.5:9b-q4_K_M --ctx 8192 \\
        --meetings data/summary/meetings.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from client import OllamaClient
from metadata import collect, write
from scorer import score_summary_task
from vram import VramMonitor

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
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--meetings", type=Path, required=True, help="JSON 配列の会議データ")
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--think", choices=["true", "false"], default="false")
    parser.add_argument("--ngram", type=int, default=3, help="被覆率判定の n-gram サイズ")
    parser.add_argument("--threshold", type=float, default=0.5, help="matched 判定の被覆率閾値")
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
            "runner": "run_summary",
            "model": args.model,
            "num_ctx": args.ctx,
            "meetings_file": str(args.meetings),
            "meeting_count": len(meetings),
            "ngram": args.ngram,
            "threshold": args.threshold,
            "think": think,
        }
    )
    write(meta, out_dir / "metadata.json")

    per_meeting: list[dict[str, Any]] = []
    with OllamaClient() as client:
        with VramMonitor(interval_sec=1.0) as vm:
            for i, m in enumerate(meetings, 1):
                print(f"[{i}/{len(meetings)}] {m['task_id']}")
                prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=m["transcript"])
                gen = client.generate(
                    args.model,
                    prompt,
                    num_ctx=args.ctx,
                    num_predict=args.num_predict,
                    think=think,
                    keep_alive="5m",
                )
                scored = score_summary_task(
                    m, gen.response_text, ngram=args.ngram, threshold=args.threshold
                )
                d = scored.to_dict()
                d["response_excerpt"] = gen.response_text[:800]
                per_meeting.append(d)
                print(f"   overall={scored.overall:.2f}")
                for c in scored.categories:
                    print(f"   {c.category}: {c.matched_count}/{c.total}")
        vram_stats = vm.stats()
        client.unload(args.model)

    overall = (
        sum(p["overall_match_rate"] for p in per_meeting) / len(per_meeting) if per_meeting else 0.0
    )
    summary = {
        "model": args.model,
        "num_ctx": args.ctx,
        "think": think,
        "ngram": args.ngram,
        "threshold": args.threshold,
        "meeting_count": len(meetings),
        "overall_match_rate": round(overall, 3),
        "vram_peak_used_mib": vram_stats.peak_used_mib,
        "vram_min_free_mib": vram_stats.min_free_mib,
        "meetings": per_meeting,
    }

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"summary_{sanitized}_ctx{args.ctx}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print()
    print(f"[done] overall_match_rate = {summary['overall_match_rate']}")
    print(f"  {out_file}")


if __name__ == "__main__":
    main()
