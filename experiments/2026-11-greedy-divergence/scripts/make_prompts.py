#!/usr/bin/env python3
"""このリポジトリのコード片から「小さな編集をして全体を書き直す」プロンプトを作る(n-gram の投機が当たる型)。
対照として、写しの無い推論プロンプト(10 月実験の W4)も入れる。行範囲で切り出すので再現できる。
  python3 make_prompts.py --root ../../.. --out data/prompts.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

EDITS = [
    ("experiments/2026-11-image-video-gen/scripts/tpl2api.py", 1, 60, "Add Python type hints to every function signature that lacks them, keeping everything else identical."),
    ("experiments/2026-11-image-video-gen/scripts/run_matrix.py", 25, 75, "Rename the variable `api` to `graph` everywhere, keeping everything else identical."),
    ("experiments/2026-11-image-video-gen/scripts/aggregate_phase1.py", 1, 60, "Translate all Japanese comments and docstrings to English, keeping the code identical."),
    ("experiments/2026-11-image-video-gen/scripts/comfy_api.py", 50, 110, "Replace every `urllib.request` call with a call through a helper named `_open(req, timeout)` that you define at the top, keeping everything else identical."),
    ("experiments/2026-11-moe-decode-anatomy/scripts/fit.py", 1, 70, "Rename `linfit` to `least_squares` everywhere and keep everything else identical."),
    ("experiments/2026-11-moe-decode-anatomy/scripts/gguf_bytes.py", 1, 70, "Add a docstring to every function that lacks one, keeping everything else identical."),
    ("experiments/2026-11-moe-decode-anatomy/scripts/sweep_k.sh", 1, 40, "Change the default port from 18090 to 18190 everywhere, keeping everything else identical."),
    ("experiments/2026-10-speculative-decoding/scripts/spec_bench.py", 20, 80, "Rename the function `metrics_delta` to `diff_metrics` everywhere, keeping everything else identical."),
    ("experiments/2026-10-speculative-decoding/scripts/aggregate.py", 1, 60, "Convert all f-strings to str.format calls, keeping everything else identical."),
    ("deploy/llama-swap/config.yaml", 1, 60, "Rewrite this YAML exactly, but change every `-ctk q8_0 -ctv q8_0` to `-ctk f8_e4m3 -ctv f8_e4m3`."),
    ("deploy/llama-swap/compose.yml", 1, 50, "Rewrite this compose file exactly, but change the published port 11435 to 11535."),
    ("experiments/2026-11-flash-next-moe-cache/scripts/aggregate.py", 1, 60, "Rename `summarize` to `build_summary` everywhere, keeping everything else identical."),
    ("tools/bench/run_speed.py", 1, 60, "Add a comment above every function describing what it returns, keeping everything else identical."),
    ("README.md", 1, 30, "Rewrite this Markdown exactly, but replace every occurrence of `Claude Code` with `the coding agent`."),
]

W4 = "../../2026-10-speculative-decoding/scripts/data/w4_prompts.json"


def snippet(root: pathlib.Path, path: str, a: int, b: int) -> str:
    lines = (root / path).read_text().splitlines()
    return "\n".join(lines[a - 1:b])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    out = []
    for i, (path, a, b, task) in enumerate(EDITS, 1):
        lang = {"py": "python", "sh": "bash", "yaml": "yaml", "yml": "yaml", "md": "markdown"}[path.rsplit(".", 1)[1]]
        body = f"{task} Output only the full result in one code block.\n\n```{lang}\n{snippet(args.root, path, a, b)}\n```"
        out.append({"id": f"E{i:02d}", "kind": "edit", "source": f"{path}:{a}-{b}", "messages": [{"role": "user", "content": body}]})
    w4 = json.loads((pathlib.Path(__file__).resolve().parent / W4).read_text()) if (pathlib.Path(__file__).resolve().parent / W4).exists() else []
    for p in w4:
        out.append({"id": p["id"], "kind": "reason", "source": "W4", "messages": p["messages"]})
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(len(out), "prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
