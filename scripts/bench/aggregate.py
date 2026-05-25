#!/usr/bin/env python3
"""results/ 配下の最新計測結果を集計して docs/04-results.md 用の Markdown 表を出力する。

各モデルについて、最新の ctx_search / speed / needle / coding / summary を拾って 1 行にまとめる。

使い方:
    uv run python aggregate.py                     # 既定 13 モデルを集計
    uv run python aggregate.py foo:tag bar:tag     # 指定モデルだけ集計
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MODELS = [
    "qwen3.5:4b-q4_K_M",
    "qwen3.5:4b-q8_0",
    "qwen3.5:9b-q4_K_M",
    "qwen3.5:9b-q8_0",
    "gemma4:e2b-it-q4_K_M",
    "gemma4:e4b-it-q4_K_M",
    "gemma4:e4b-it-q8_0",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "deepseek-r1:7b-qwen-distill-q8_0",
    "deepseek-r1:8b-0528-qwen3-q4_K_M",
    "deepseek-r1:14b-qwen-distill-q4_K_M",
    "deepseek-coder-v2:16b-lite-instruct-q4_0",
    "deepseek-coder-v2:16b-lite-instruct-q4_K_M",
]


def sanitize(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def find_latest(results_dir: Path, pattern: str) -> Path | None:
    matches = list(results_dir.rglob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def collect_for_model(results_dir: Path, model: str) -> dict[str, Any]:
    s = sanitize(model)
    row: dict[str, Any] = {"model": model}

    if (d := load_json(find_latest(results_dir, f"ctx_search_{s}.json"))) is not None:
        row["max_ctx"] = d.get("max_ctx")
        row["ctx_search_probes"] = len(d.get("history", []))

    if (d := load_json(find_latest(results_dir, f"speed_{s}_ctx*.json"))) is not None:
        row["decode_tps"] = d.get("decode_tokens_per_sec")
        row["prefill_tps"] = d.get("prefill_tokens_per_sec")
        row["ttft_sec"] = d.get("ttft_sec")
        row["vram_peak_mib"] = d.get("vram_peak_used_mib")
        row["vram_free_mib"] = d.get("vram_min_free_mib")
        row["gpu_util_pct"] = d.get("gpu_mean_utilization_pct")
        row["requested_ctx"] = d.get("num_ctx")
        # ollama ps の context_length が「実効 ctx」。Ollama がモデル上限でキャップ
        # した場合は要求値より小さくなる(例: gemma4 で 262144 要求 → 131072)。
        ps = d.get("ollama_ps_after_load") or []
        for entry in ps:
            if entry.get("name") == model or entry.get("model") == model:
                row["effective_ctx"] = entry.get("context_length")
                break

    if (d := load_json(find_latest(results_dir, f"needle_{s}_ctx*.json"))) is not None:
        row["needle_success"] = d.get("success")
        row["needle_position_pct"] = d.get("position_pct")

    if (d := load_json(find_latest(results_dir, f"coding_{s}_ctx*.json"))) is not None:
        row["coding_pass_rate"] = d.get("overall_pass_rate")
        row["coding_by_type"] = d.get("by_task_type", {})

    if (d := load_json(find_latest(results_dir, f"summary_{s}_ctx*.json"))) is not None:
        row["summary_match_rate"] = d.get("overall_match_rate")

    return row


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _effective_max_ctx(row: dict[str, Any]) -> int | None:
    """ctx_search.max_ctx と speed の ollama_ps.context_length の小さい方。"""
    cs = row.get("max_ctx")
    eff = row.get("effective_ctx")
    if cs is None and eff is None:
        return None
    if cs is None:
        return eff
    if eff is None:
        return cs
    return min(cs, eff)


def format_main_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Max ctx",
        "Decode tok/s",
        "Prefill tok/s",
        "TTFT (s)",
        "VRAM peak (MiB)",
        "VRAM free (MiB)",
        "Needle",
        "Coding",
        "Summary",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        max_ctx = _effective_max_ctx(r)
        # ctx_search 値と effective(ollama 側でキャップ後)が食い違う場合はマーク
        capped = (
            r.get("max_ctx") is not None
            and r.get("effective_ctx") is not None
            and r.get("effective_ctx") < r.get("max_ctx")
        )
        max_ctx_str = f"{_fmt(max_ctx)}{' *' if capped else ''}"
        out.append(
            "| "
            + " | ".join(
                [
                    r.get("model", ""),
                    max_ctx_str,
                    _fmt(r.get("decode_tps")),
                    _fmt(r.get("prefill_tps")),
                    _fmt(r.get("ttft_sec")),
                    _fmt(r.get("vram_peak_mib")),
                    _fmt(r.get("vram_free_mib")),
                    _fmt(r.get("needle_success")),
                    _fmt(r.get("coding_pass_rate")),
                    _fmt(r.get("summary_match_rate")),
                ]
            )
            + " |"
        )
    out.append("")
    out.append("`*` = Ollama がモデル上限で要求 ctx をキャップしている(ctx_search が VRAM ではなくモデル上限に到達した)。")
    return "\n".join(out)


def format_coding_breakdown(rows: list[dict[str, Any]]) -> str:
    headers = ["Model", "Overall", "impl", "bugfix", "refactor"]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        by_type = r.get("coding_by_type") or {}
        out.append(
            "| "
            + " | ".join(
                [
                    r.get("model", ""),
                    _fmt(r.get("coding_pass_rate")),
                    _fmt(by_type.get("impl")),
                    _fmt(by_type.get("bugfix")),
                    _fmt(by_type.get("refactor")),
                ]
            )
            + " |"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    rows = [collect_for_model(args.results_dir, m) for m in models]

    print("## Main table")
    print()
    print(format_main_table(rows))
    print()
    print("## Coding breakdown by task type")
    print()
    print(format_coding_breakdown(rows))


if __name__ == "__main__":
    main()
