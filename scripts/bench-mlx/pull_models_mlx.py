#!/usr/bin/env python3
"""mlx-community / 任意のリポジトリから MLX モデルを一括取得する。

bench/pull_models.sh の MLX 版。HuggingFace Hub 経由で snapshot_download を呼ぶ。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

# RTX 4080 SUPER 評価対象の MLX 等価タグ + 48GB Mac 用追加大型モデル。
# 実在性は 2026-05-26 時点で huggingface.co/mlx-community 配下を確認したもの。
#
# 命名:
# - `*-MLX-4bit` / `*-MLX-8bit`: 旧 Qwen3.5 系の MLX 公式変換タグ
# - `*-4bit` / `*-8bit`: それ以外(Qwen3.6 / Gemma 4 / DeepSeek 系)
# - `*-OptiQ-4bit`: OptiQ 量子化(GPTQ ベース、品質寄り)
# 小さい順に並べる。pull の途中で打ち切られても評価対象数が最大化されるように。
# 全モデルを同時にディスクに置く必要は無く、`eval_cycle_mlx.sh` は 1 モデル毎に
# pull → 評価 → 削除 のサイクルで回す。本リストはサイクルの評価順を兼ねる。
DEFAULT_MODELS: list[str] = [
    # ---- 小型 (< 5GB)、起動と動作確認が早い ----
    "mlx-community/Qwen3.5-4B-MLX-4bit",                # 2.85 GB
    "mlx-community/gemma-4-e2b-it-4bit",                # 3.37 GB
    "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",   # 4.00 GB
    "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",     # 4.30 GB
    "mlx-community/Qwen3.5-4B-MLX-8bit",                # 4.81 GB
    "mlx-community/gemma-4-e4b-it-4bit",                # 4.89 GB
    # ---- 中型 (5〜10GB) ----
    "mlx-community/Qwen3.5-9B-MLX-4bit",                # 5.57 GB
    "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit",   # 7.55 GB
    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",  # 7.75 GB
    "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",  # 8.24 GB (MoE)
    "mlx-community/gemma-4-e4b-it-8bit",                # 8.38 GB
    "mlx-community/Qwen3.5-9B-MLX-8bit",                # 9.74 GB
    # ---- 大型 (10〜20GB) ----
    "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit",  # 15.55 GB (MoE)
    "mlx-community/gemma-4-26b-a4b-it-4bit",            # 14.57 GB (MoE)
    "mlx-community/Qwen3.5-27B-4bit",                   # 14.98 GB
    "mlx-community/Qwen3.6-27B-4bit",                   # 14.98 GB
    "mlx-community/gemma-4-31b-it-4bit",                # 17.18 GB
    "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",  # 17.18 GB
    # ---- 極大 (~30GB)、KV 余地は薄いが計測対象 ----
    "mlx-community/Qwen3.5-27B-8bit",                   # 27.50 GB
    "mlx-community/Qwen3.6-27B-8bit",                   # 27.50 GB
]


def pull_one(repo_id: str, log_file: Path) -> bool:
    """1 モデルを snapshot_download。成功時 True、失敗時 False(理由は log_file へ)。"""
    print(f"[pull] {repo_id}")
    t0 = time.monotonic()
    try:
        path = snapshot_download(repo_id=repo_id)
    except Exception as e:  # noqa: BLE001
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(f"{repo_id}\tFAIL\t{type(e).__name__}: {e}\n")
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False
    dt = time.monotonic() - t0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as f:
        f.write(f"{repo_id}\tOK\t{dt:.1f}s\t{path}\n")
    print(f"  OK ({dt:.1f}s) -> {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        help="HuggingFace の repo_id を任意個指定。省略時は DEFAULT_MODELS を使う",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).parent / "results" / f"pull-{int(time.time())}.log",
    )
    args = parser.parse_args()

    models = args.models if args.models else DEFAULT_MODELS
    failed: list[str] = []
    for m in models:
        if not pull_one(m, args.log):
            failed.append(m)

    print()
    print(f"[done] {len(models) - len(failed)}/{len(models)} succeeded")
    print(f"  log: {args.log}")
    if failed:
        print("  failed:")
        for m in failed:
            print(f"    - {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
