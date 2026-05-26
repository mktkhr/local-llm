#!/usr/bin/env python3
"""モデルの最大 ctx を二分探索で求める (Apple Silicon / MLX 版)。

Ollama のように KV キャッシュをロード時に予約しないため、MLX 版では:

1. 「ctx 相当のプロンプト長」を実際に流して prefill させる
2. mx.get_peak_memory() でピーク使用量を取得
3. effective_gpu_limit_mib (max_recommended_working_set or iogpu.wired_limit) と
   比較して、安全マージン内に収まるかで OK/NG を判定

する。OOM が起きると mlx 側が例外を投げるので、それも NG として扱う。

モデル自身の max_position_embeddings を超える ctx 要求は capped とみなし、
max_position をそのまま採用する(これ以上探索しない)。

使い方:
    uv run python ctx_search_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    uv run python ctx_search_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --low 4096 --high 262144 --tolerance 4096
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

from client_mlx import generate_with_metrics, load_model
from memory_mlx import (
    clear_cache,
    effective_gpu_limit_mib,
    get_peak_memory_mib,
    reset_peak,
)
from metadata_mlx import collect, write

# 日本語ベースのフィラー。tokenizer によって若干トークン数は変動するため、
# 実測してから繰り返し回数を決める。
FILLER_BASE = (
    "本評価は Apple Silicon / MLX 上での長コンテキスト動作確認用のフィラー文です。"
    "繰り返し挿入することで KV キャッシュにロードされるトークン数を増やし、"
    "実際にユニファイドメモリへ確保される量を計測することを目的としています。"
)


@dataclass
class ProbeResult:
    requested_ctx: int
    effective_ctx: int  # モデルの max_position_embeddings でキャップされた場合は実値
    prompt_tokens_loaded: int  # 実際に prefill した token 数
    success: bool
    reason: str  # "ok" | "ok_capped" | "oom" | "low_vram" | "error"
    peak_memory_mib: float = 0.0
    prompt_tps: float = 0.0
    error: str | None = None


@dataclass
class SearchResult:
    model: str
    max_ctx: int  # 採用可能だった最大値。0 なら下限すら通らなかった
    low: int
    high: int
    tolerance: int
    gpu_limit_mib: int
    safety_margin_mib: int
    model_max_position: int
    history: list[ProbeResult] = field(default_factory=list)


def _align(value: int, step: int) -> int:
    return max(step, (value // step) * step)


def _build_filler_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    """target_tokens 程度のトークン数になる日本語フィラー文字列を作る。

    繰り返し→encode→必要に応じて decode で切り詰める。
    返り値は (prompt_text, actual_token_count)。
    """
    if target_tokens <= 0:
        return "", 0
    base_ids = tokenizer.encode(FILLER_BASE)
    per_repeat = max(1, len(base_ids))
    repeats = max(1, target_tokens // per_repeat + 1)
    text = FILLER_BASE * repeats
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        text = tokenizer.decode(ids)
    return text, len(ids)


def probe(
    model: Any,
    tokenizer: Any,
    ctx: int,
    *,
    kv_bits: int | None,
    gpu_limit_mib: int,
    safety_margin_mib: int,
    model_max_position: int,
    num_predict: int = 4,
) -> ProbeResult:
    """指定 ctx で実際に prefill を流し、peak メモリで OK/NG を判定。

    - ctx > model_max_position の場合は ok_capped を即返す(これ以上は意味がない)
    """
    if model_max_position > 0 and ctx > model_max_position:
        # 実際は max_position でキャップされるので、そちらを試す
        capped_ctx = model_max_position
        res = probe(
            model,
            tokenizer,
            capped_ctx,
            kv_bits=kv_bits,
            gpu_limit_mib=gpu_limit_mib,
            safety_margin_mib=safety_margin_mib,
            model_max_position=model_max_position,
            num_predict=num_predict,
        )
        # 要求 ctx を覚えつつ、結果は capped 扱いに振り直す
        res.requested_ctx = ctx
        if res.success:
            res.reason = "ok_capped"
            res.effective_ctx = capped_ctx
        return res

    # prompt 長は ctx - num_predict - 数十 token 程度に収める
    target_tokens = max(64, ctx - num_predict - 32)
    prompt_text, actual_prompt_tokens = _build_filler_prompt(tokenizer, target_tokens)

    reset_peak()
    try:
        result = generate_with_metrics(
            model,
            tokenizer,
            prompt_text,
            max_tokens=num_predict,
            kv_bits=kv_bits,
            max_kv_size=ctx,
        )
    except (RuntimeError, ValueError, MemoryError) as e:
        gc.collect()
        clear_cache()
        return ProbeResult(
            requested_ctx=ctx,
            effective_ctx=0,
            prompt_tokens_loaded=actual_prompt_tokens,
            success=False,
            reason="oom",
            error=str(e),
            peak_memory_mib=get_peak_memory_mib(),
        )

    peak_mib = result.peak_memory_mib_mx
    limit_with_margin = gpu_limit_mib - safety_margin_mib

    if peak_mib > limit_with_margin:
        return ProbeResult(
            requested_ctx=ctx,
            effective_ctx=ctx,
            prompt_tokens_loaded=result.prompt_tokens,
            success=False,
            reason="low_vram",
            peak_memory_mib=peak_mib,
            prompt_tps=result.prompt_tps,
        )

    return ProbeResult(
        requested_ctx=ctx,
        effective_ctx=ctx,
        prompt_tokens_loaded=result.prompt_tokens,
        success=True,
        reason="ok",
        peak_memory_mib=peak_mib,
        prompt_tps=result.prompt_tps,
    )


def binary_search_max_ctx(
    model: Any,
    tokenizer: Any,
    model_id: str,
    *,
    model_max_position: int,
    low: int,
    high: int,
    tolerance: int,
    gpu_limit_mib: int,
    safety_margin_mib: int,
    kv_bits: int | None = None,
    align_step: int = 1024,
) -> SearchResult:
    history: list[ProbeResult] = []
    result = SearchResult(
        model=model_id,
        max_ctx=0,
        low=low,
        high=high,
        tolerance=tolerance,
        gpu_limit_mib=gpu_limit_mib,
        safety_margin_mib=safety_margin_mib,
        model_max_position=model_max_position,
        history=history,
    )

    def _probe(ctx: int) -> ProbeResult:
        return probe(
            model,
            tokenizer,
            ctx,
            kv_bits=kv_bits,
            gpu_limit_mib=gpu_limit_mib,
            safety_margin_mib=safety_margin_mib,
            model_max_position=model_max_position,
        )

    print(f"[probe] low={low}")
    res_low = _probe(low)
    history.append(res_low)
    print(f"  -> {'ok' if res_low.success else 'NG'} ({res_low.reason}, peak={res_low.peak_memory_mib:.0f} MiB)")
    if not res_low.success:
        return result
    result.max_ctx = res_low.effective_ctx

    # 上限プローブ
    upper = high
    if model_max_position > 0 and upper > model_max_position:
        upper = model_max_position
    print(f"[probe] high={upper}")
    res_high = _probe(upper)
    history.append(res_high)
    print(f"  -> {'ok' if res_high.success else 'NG'} ({res_high.reason}, peak={res_high.peak_memory_mib:.0f} MiB)")
    if res_high.success:
        result.max_ctx = res_high.effective_ctx
        return result

    lo, hi = low, upper
    while hi - lo > tolerance:
        mid = _align((lo + hi) // 2, align_step)
        if mid <= lo or mid >= hi:
            break
        print(f"[probe] mid={mid} (range {lo}..{hi})")
        res = _probe(mid)
        history.append(res)
        print(f"  -> {'ok' if res.success else 'NG'} ({res.reason}, peak={res.peak_memory_mib:.0f} MiB)")
        if res.success:
            lo = mid
            result.max_ctx = res.effective_ctx
            if res.reason == "ok_capped":
                return result
        else:
            hi = mid

    return result


def _result_to_dict(res: SearchResult) -> dict[str, Any]:
    return {
        "model": res.model,
        "max_ctx": res.max_ctx,
        "search_range_low": res.low,
        "search_range_high": res.high,
        "tolerance": res.tolerance,
        "gpu_limit_mib": res.gpu_limit_mib,
        "safety_margin_mib": res.safety_margin_mib,
        "model_max_position": res.model_max_position,
        "history": [asdict(p) for p in res.history],
    }


def _model_max_position(model: Any) -> int:
    if hasattr(model, "args") and hasattr(model.args, "max_position_embeddings"):
        try:
            return int(model.args.max_position_embeddings)
        except Exception:
            return 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--low", type=int, default=4096)
    parser.add_argument("--high", type=int, default=262144)
    parser.add_argument("--tolerance", type=int, default=4096)
    parser.add_argument(
        "--safety-margin-mib",
        type=int,
        default=1024,
        help="effective_gpu_limit から差し引くマージン(MiB)。これ以下に peak が収まれば OK",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=[4, 8],
        default=None,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_limit = effective_gpu_limit_mib()
    print(f"[env] effective_gpu_limit = {gpu_limit} MiB, safety_margin = {args.safety_margin_mib} MiB")

    meta = collect(
        extra={
            "runner": "ctx_search_mlx",
            "model": args.model,
            "low": args.low,
            "high": args.high,
            "tolerance": args.tolerance,
            "safety_margin_mib": args.safety_margin_mib,
            "kv_bits": args.kv_bits,
        }
    )
    write(meta, out_dir / "metadata.json")

    print(f"[load] {args.model}")
    model, tokenizer = load_model(args.model)
    model_max_pos = _model_max_position(model)
    print(f"  model.max_position_embeddings = {model_max_pos}")
    print(f"  active mem after load = {mx.get_active_memory() / (1024 * 1024):.0f} MiB")

    result = binary_search_max_ctx(
        model,
        tokenizer,
        args.model,
        model_max_position=model_max_pos,
        low=args.low,
        high=args.high,
        tolerance=args.tolerance,
        gpu_limit_mib=gpu_limit,
        safety_margin_mib=args.safety_margin_mib,
        kv_bits=args.kv_bits,
    )

    sanitized = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"ctx_search_{sanitized}.json"
    out_file.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))

    print()
    print(f"[result] max_ctx = {result.max_ctx} ({len(result.history)} probes)")
    print(f"[done] {out_file}")

    del model, tokenizer
    gc.collect()
    clear_cache()


if __name__ == "__main__":
    main()
