"""mlx-lm 推論クライアント。

責務:
- mlx_lm.load でモデルとトークナイザを取得
- ストリーミング生成で TTFT を計測
- prompt_tps / generation_tps / peak_memory を採取
- chat template の自動適用
- KV キャッシュ量子化 (kv_bits) を引数で切替
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

import mlx.core as mx
from mlx_lm import load, stream_generate


@dataclass
class GenerateResult:
    """ストリーミング生成の結果と各種メトリクス。

    速度系は mlx-lm の GenerationResponse から取得(モデル側の計測値)。
    TTFT のみ Python 側で wallclock を取る。
    """

    response_text: str
    ttft_sec: float
    total_elapsed_sec: float
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory_gib: float  # mlx-lm が返す GB(1e9 で割った値)
    peak_memory_mib_mx: float  # mx.get_peak_memory() から MiB 換算
    finish_reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model(model_id: str, *, trust_remote_code: bool = False) -> tuple[Any, Any]:
    """HuggingFace の repo_id またはローカルパスを渡す。

    `mlx-community/Qwen2.5-7B-Instruct-4bit` のような mlx-community 配下の
    pre-quantized モデルを想定。
    """
    model, tokenizer = load(
        model_id,
        tokenizer_config={"trust_remote_code": trust_remote_code} if trust_remote_code else None,
    )
    return model, tokenizer


def apply_chat_template(tokenizer: Any, prompt: str, *, system: Optional[str] = None, enable_thinking: Optional[bool] = None) -> str:
    """ユーザ入力にチャットテンプレートを適用して文字列を返す。

    enable_thinking は Qwen3 系の thinking 切替に使う(対応モデルのみ)。
    template が拒んだら無視する。
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def reset_peak_memory() -> None:
    """生成前にピーク値をリセット。計測の独立性を保つ。"""
    mx.reset_peak_memory()


def warmup(model: Any, tokenizer: Any, *, kv_bits: int | None = None) -> None:
    """Metal shader の初回 JIT コンパイル分を本計測から外すため、ダミー 1 トークン生成を回す。

    Linux + Ollama 版の run_speed.py が "hi" + num_predict=1 で同じ役割を果たしている。
    呼び出し後に reset_peak_memory() を呼ぶことで warmup のメモリピークも除外される。
    """
    text = apply_chat_template(tokenizer, "hi")
    # stream_generate を 1 token だけ回す。結果は捨てる
    from mlx_lm import stream_generate

    kwargs: dict[str, Any] = {"max_tokens": 1}
    if kv_bits is not None:
        kwargs["kv_bits"] = kv_bits
    for _ in stream_generate(model, tokenizer, text, **kwargs):
        break


def generate_with_metrics(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 128,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    max_kv_size: Optional[int] = None,
    prefill_step_size: int = 2048,
) -> GenerateResult:
    """ストリーミング generate でメトリクス採取。

    - kv_bits: None=非量子化, 4 or 8
    - max_kv_size: KV キャッシュ最大長(プロンプト+生成が超えると古い分が捨てられる仕様)
    """
    reset_peak_memory()
    start = time.monotonic()
    ttft: Optional[float] = None
    chunks: list[str] = []
    last_resp = None

    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "kv_group_size": kv_group_size,
    }
    if kv_bits is not None:
        kwargs["kv_bits"] = kv_bits
    # max_kv_size を渡すと RotatingKVCache が使われるが、それは KV 量子化と
    # 排他(`RotatingKVCache Quantization NYI`)。本評価では rotation 不要のため、
    # kv_bits が指定された時は max_kv_size を無視する。
    if max_kv_size is not None and kv_bits is None:
        kwargs["max_kv_size"] = max_kv_size

    for resp in stream_generate(model, tokenizer, prompt, **kwargs):
        if ttft is None and resp.text:
            ttft = time.monotonic() - start
        chunks.append(resp.text)
        last_resp = resp

    total = time.monotonic() - start

    if last_resp is None:
        return GenerateResult(
            response_text="",
            ttft_sec=total,
            total_elapsed_sec=total,
            prompt_tokens=0,
            prompt_tps=0.0,
            generation_tokens=0,
            generation_tps=0.0,
            peak_memory_gib=0.0,
            peak_memory_mib_mx=mx.get_peak_memory() / (1024 * 1024),
            finish_reason=None,
        )

    return GenerateResult(
        response_text="".join(chunks),
        ttft_sec=ttft if ttft is not None else total,
        total_elapsed_sec=total,
        prompt_tokens=last_resp.prompt_tokens,
        prompt_tps=last_resp.prompt_tps,
        generation_tokens=last_resp.generation_tokens,
        generation_tps=last_resp.generation_tps,
        peak_memory_gib=last_resp.peak_memory,
        peak_memory_mib_mx=mx.get_peak_memory() / (1024 * 1024),
        finish_reason=last_resp.finish_reason,
    )
