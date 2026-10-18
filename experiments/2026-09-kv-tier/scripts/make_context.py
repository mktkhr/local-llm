#!/usr/bin/env python3
"""Coding Agent 相当のテスト context を生成する。

synthetic な繰り返し文字列では prefix 共有と圧縮率が実運用と乖離するため、
本リポジトリ自身のソースを素材にして以下の構造を組み立てる。

    [system prompt]        固定・全 context 共通(最も再利用されるべき prefix)
    [repository context]   実ソースの連結
    [tool output]          ls / grep / cat 相当の構造化テキスト
    [conversation history] user / assistant の往復
    [tail]                 Case E で差し替える末尾

トークン数は llama-server の /tokenize で確定させる。文字数からの推定はしない。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
# 素材にするリポジトリ本体(experiments/<実験>/scripts/ から 3 つ上)。
# 2026-09 の計測時は旧構成(docs/ scripts/ deploy/)を素材にしており、再生成すると内容は変わる。
SRC_ROOT = REPO.parents[1]

SYSTEM_PROMPT = """You are a coding agent working inside a local repository.

Rules:
- Always read a file before editing it.
- Prefer the smallest change that satisfies the request.
- Never introduce a new dependency without stating why.
- Report test failures verbatim; do not summarise them away.
- When you are uncertain about an API, read the source rather than guessing.

Available tools: read_file, write_file, list_dir, grep, run_tests, git_diff.
Respond with a short plan, then the tool calls, then a summary of what changed.
"""


def tokenize(base_url: str, text: str) -> int:
    req = urllib.request.Request(
        f"{base_url}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as res:
        return len(json.loads(res.read())["tokens"])


def source_blocks() -> list[str]:
    """リポジトリの実ファイルを read_file のツール出力風に整形する。"""
    out = []
    patterns = ["docs/**/*.md", "experiments/**/*.md", "tools/**/*.py", "tools/**/*.sh", "deploy/**/*.yml", "README.md"]
    seen = set()
    for pat in patterns:
        for p in sorted(SRC_ROOT.glob(pat)):
            if ".venv" in p.parts or p in seen:
                continue
            seen.add(p)
            try:
                body = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = p.relative_to(SRC_ROOT)
            out.append(f"<tool_result tool=read_file path={rel}>\n{body}\n</tool_result>\n")
    return out


def tool_output_blocks() -> list[str]:
    """実際に実行したコマンドの出力を素材にする。"""
    cmds = [
        ["git", "log", "--oneline", "-40"],
        ["git", "diff", "--stat", "HEAD~3", "HEAD"],
        ["ls", "-laR", "scripts"],
        ["ls", "-laR", "docs"],
    ]
    out = []
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, cwd=SRC_ROOT, capture_output=True, text=True, timeout=30)
            out.append(f"<tool_result tool=run cmd={' '.join(cmd)!r}>\n{r.stdout}\n</tool_result>\n")
        except (OSError, subprocess.SubprocessError):
            continue
    return out


def conversation_blocks(n: int) -> list[str]:
    out = []
    for i in range(n):
        out.append(
            f"<turn index={i}>\n"
            f"user: 計測スクリプトの {i} 番目の指標について、どこから値を取っているか教えてください。\n"
            f"assistant: その値は llama-server の /completion レスポンスの timings から取得しています。"
            f"具体的には prompt_ms が prefill 時間、cache_n が再利用されたトークン数です。"
            f"再 Prefill が起きたかどうかは prompt_n を見て判定します。\n"
            f"</turn>\n"
        )
    return out


def build(base_url: str, target_tokens: int, tail_tokens: int) -> tuple[str, str]:
    """(prefix, tail) を返す。prefix + tail のトークン数が target_tokens になる。"""
    tail = (
        "<turn index=final>\n"
        "user: ここまでの内容を踏まえて、階層型 KV キャッシュの計測スクリプトに追加すべき"
        "メトリクスを列挙し、それぞれどのエンドポイントから取得できるかを述べてください。"
        + " なお前提として本機の SSD は SATA であり NVMe ではありません。" * 12
        + "\n</turn>\nassistant:"
    )

    pool: list[str] = []
    blocks = source_blocks()
    tools = tool_output_blocks()
    convs = conversation_blocks(200)
    # 素材が尽きたら周回させる。周回ごとに見出しを変えて完全一致を避ける。
    for cycle in range(50):
        pool.append(f"\n<context_window cycle={cycle}>\n")
        pool.extend(blocks)
        pool.extend(tools)
        pool.extend(convs[cycle * 4 : cycle * 4 + 4])

    head = SYSTEM_PROMPT + "\n<repository_context>\n"
    budget = target_tokens - tokenize(base_url, tail)

    # 粗く積む
    acc: list[str] = []
    cur = tokenize(base_url, head)
    for b in pool:
        t = tokenize(base_url, b)
        if cur + t > budget:
            break
        acc.append(b)
        cur += t
    prefix = head + "".join(acc)

    # 文字数の二分探索で目標に寄せる
    remainder = "".join(pool[len(acc) :])
    lo, hi = 0, min(len(remainder), (budget - cur) * 8 + 4096)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tokenize(base_url, prefix + remainder[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    prefix += remainder[:lo]
    return prefix, tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18090")
    ap.add_argument("--tokens", type=int, nargs="+", required=True)
    ap.add_argument("--tail-tokens", type=int, default=256)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "raw" / "ctx"))
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for target in args.tokens:
        prefix, tail = build(args.base_url, target, args.tail_tokens)
        total = tokenize(args.base_url, prefix + tail)
        (out / f"ctx_{target}_prefix.txt").write_text(prefix, encoding="utf-8")
        (out / f"ctx_{target}_tail.txt").write_text(tail, encoding="utf-8")
        print(f"target={target:>7} actual={total:>7} tokens  prefix={len(prefix)} chars")


if __name__ == "__main__":
    main()
