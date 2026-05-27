#!/usr/bin/env python3
"""results/ 配下の最新計測結果を集計して docs/04-results.md 用の Markdown 表を出力する (MLX 版)。

各モデルについて、最新の ctx_search / speed / needle / coding / summary を拾って 1 行にまとめる。
KV 量子化はリクエスト時引数なので、metadata.json の extra.kv_bits を見て KV を判別する。

使い方:
    uv run python aggregate_mlx.py
    uv run python aggregate_mlx.py mlx-community/Qwen3.5-9B-MLX-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MODELS = [
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
    "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    "mlx-community/Qwen3.5-27B-4bit",
    "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
]


def sanitize(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _result_kv_bits(result_file: Path) -> Any:
    """同じディレクトリの metadata.json から extra.kv_bits を取得。無ければ None。"""
    meta = load_json(result_file.parent / "metadata.json")
    if meta is None:
        return None
    return meta.get("extra", {}).get("kv_bits")


def find_latest(results_dir: Path, pattern: str, kv_bits: Any = "ANY") -> Path | None:
    """pattern にマッチする最新ファイルを返す。

    kv_bits が "ANY" 以外なら、同ディレクトリ metadata.json の extra.kv_bits が
    一致するものだけに絞る(q4/q8/f16 の混成行を防ぐ)。
    """
    matches = list(results_dir.rglob(pattern))
    if kv_bits != "ANY":
        matches = [m for m in matches if _result_kv_bits(m) == kv_bits]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def collect_for_model(results_dir: Path, model: str, kv_bits: Any = "ANY") -> dict[str, Any]:
    s = sanitize(model)
    row: dict[str, Any] = {"model": model, "filter_kv_bits": kv_bits}

    def latest(pat: str) -> Path | None:
        return find_latest(results_dir, pat, kv_bits)

    if (d := load_json(latest(f"ctx_search_{s}.json"))) is not None:
        row["max_ctx"] = d.get("max_ctx")
        row["ctx_search_probes"] = len(d.get("history", []))
        row["model_max_position"] = d.get("model_max_position")

    if (d := load_json(latest(f"speed_{s}_ctx*.json"))) is not None:
        row["generation_tps"] = d.get("generation_tps")
        row["prompt_tps"] = d.get("prompt_tps")
        row["ttft_sec"] = d.get("ttft_sec")
        row["peak_memory_mib"] = d.get("peak_memory_mib")
        row["requested_ctx"] = d.get("max_kv_size")
        row["kv_bits"] = d.get("kv_bits")

    if (d := load_json(latest(f"needle_{s}_ctx*.json"))) is not None:
        row["needle_success"] = d.get("success")
        row["needle_position_pct"] = d.get("position_pct")
        row["needle_prefill_tps"] = d.get("prompt_tps")
        row["needle_ttft_sec"] = d.get("ttft_sec")
        row["needle_decode_tps"] = d.get("generation_tps")
        row["needle_truncated"] = d.get("truncated")

    if (d := load_json(latest(f"coding_{s}_ctx*.json"))) is not None:
        row["coding_pass_rate"] = d.get("overall_pass_rate")
        row["coding_by_type"] = d.get("by_task_type", {})
        row["coding_truncated"] = d.get("truncated_count")

    if (d := load_json(latest(f"summary_{s}_ctx*.json"))) is not None:
        row["summary_match_rate"] = d.get("overall_match_rate")
        row["summary_truncated"] = d.get("truncated_count")

    return row


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def format_main_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Max ctx",
        "Decode tok/s",
        "Prefill tok/s",
        "TTFT (s)",
        "Peak mem (MiB)",
        "Needle",
        "Coding",
        "Summary",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        max_ctx = r.get("max_ctx")
        capped = (
            max_ctx is not None
            and r.get("model_max_position")
            and max_ctx == r.get("model_max_position")
        )
        max_ctx_str = f"{_fmt(max_ctx)}{' `*`' if capped else ''}"
        out.append(
            "| "
            + " | ".join(
                [
                    r.get("model", "").replace("mlx-community/", ""),
                    max_ctx_str,
                    _fmt(r.get("generation_tps")),
                    _fmt(r.get("prompt_tps")),
                    _fmt(r.get("ttft_sec")),
                    _fmt(r.get("peak_memory_mib")),
                    _fmt(r.get("needle_success")),
                    _fmt(r.get("coding_pass_rate")),
                    _fmt(r.get("summary_match_rate")),
                ]
            )
            + " |"
        )
    out.append("")
    out.append("`*` = モデル既定 ctx 上限でキャップ(これ以上は探索しない)")
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
                    r.get("model", "").replace("mlx-community/", ""),
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
    parser.add_argument(
        "--kv-bits",
        default="8",
        help="集計対象を絞る KV bits(metadata の extra.kv_bits)。'any' で全部混ぜる(非推奨)。既定 8",
    )
    args = parser.parse_args()

    kv_filter: Any = "ANY" if args.kv_bits.lower() == "any" else int(args.kv_bits)
    models = args.models or DEFAULT_MODELS
    rows = [collect_for_model(args.results_dir, m, kv_filter) for m in models]

    print(f"## Main table (KV filter = {kv_filter})")
    print()
    print(format_main_table(rows))
    print()
    print("## Coding breakdown by task type")
    print()
    print(format_coding_breakdown(rows))


if __name__ == "__main__":
    main()
