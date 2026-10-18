#!/usr/bin/env python3
"""llama-server に 1 リクエスト投げ、/metrics の差分・timings・GPU 状態を JSONL 1 行に残す。

受理率と tok/s は /metrics(Prometheus テキスト)のリクエスト前後の差分から算出する。
レスポンスの timings はエンドポイントや build によって無いことがあるため、あれば照合用に併記する。
"""

from __future__ import annotations

import re

_PER_POS = re.compile(r'^llamacpp:spec_decode_num_accepted_tokens_per_pos_total\{position="(\d+)"\}$')


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus テキストを {名前(ラベル込み): 値} にする。コメント行と空行は無視。"""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def _d(before: dict[str, float], after: dict[str, float], key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)


def metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    """リクエスト前後の /metrics 差分と派生指標。存在しないカウンタは 0 として扱う。"""
    p = "llamacpp:"
    predicted_n = _d(before, after, p + "tokens_predicted_total")
    predicted_s = _d(before, after, p + "tokens_predicted_seconds_total")
    prompt_n = _d(before, after, p + "prompt_tokens_total")
    prompt_s = _d(before, after, p + "prompt_seconds_total")
    draft_n = _d(before, after, p + "spec_decode_num_draft_tokens_total")
    accepted_n = _d(before, after, p + "spec_decode_num_accepted_tokens_total")

    per_pos: dict[int, float] = {}
    for key in set(before) | set(after):
        m = _PER_POS.match(key)
        if m:
            per_pos[int(m.group(1))] = _d(before, after, key)
    accepted_per_pos = [int(per_pos.get(i, 0)) for i in range(max(per_pos) + 1)] if per_pos else []

    return {
        "predicted_n": int(predicted_n),
        "predicted_s": predicted_s,
        "decode_tok_s": predicted_n / predicted_s if predicted_s > 0 else None,
        "prompt_n": int(prompt_n),
        "prompt_s": prompt_s,
        "prefill_tok_s": prompt_n / prompt_s if prompt_s > 0 else None,
        "cached_n": int(_d(before, after, p + "prompt_tokens_cached_total")),
        "n_decode": int(_d(before, after, p + "n_decode_total")),
        "draft_n": int(draft_n),
        "accepted_n": int(accepted_n),
        "drafts_n": int(_d(before, after, p + "spec_decode_num_drafts_total")),
        "acceptance": accepted_n / draft_n if draft_n > 0 else None,
        "draft_per_generated": draft_n / predicted_n if predicted_n > 0 else None,
        "accepted_per_pos": accepted_per_pos,
    }


_TIMING_KEYS = ("prompt_n", "prompt_ms", "cache_n", "predicted_n", "predicted_ms", "predicted_per_second")


def extract_timings(resp: dict) -> dict | None:
    """トップレベル timings があれば主要キーだけ返す。無ければ None。"""
    t = resp.get("timings")
    if not isinstance(t, dict):
        return None
    return {k: t.get(k) for k in _TIMING_KEYS}


def extract_text(resp: dict) -> str:
    """生成本文。messages 形式は text ブロックのみ連結、chat 形式は choices[0].message.content。"""
    content = resp.get("content")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""
    return ""
