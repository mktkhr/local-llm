#!/usr/bin/env python3
"""llama-server に 1 リクエスト投げ、/metrics の差分・timings・GPU 状態を JSONL 1 行に残す。

受理率と tok/s は /metrics(Prometheus テキスト)のリクエスト前後の差分から算出する。
レスポンスの timings はエンドポイントや build によって無いことがあるため、あれば照合用に併記する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

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


def _strip_ids(obj):
    if isinstance(obj, dict):
        return {k: _strip_ids(v) for k, v in obj.items() if k != "id"}
    if isinstance(obj, list):
        return [_strip_ids(x) for x in obj]
    return obj


def generated_payload(resp: dict) -> str:
    """決定性比較用の生成物。text だけでなく tool_use / thinking / tool_calls まで含め、id は除く。"""
    if isinstance(resp.get("content"), list):
        return json.dumps(_strip_ids(resp["content"]), ensure_ascii=False, sort_keys=True)
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        return json.dumps({"reasoning_content": msg.get("reasoning_content"), "content": msg.get("content"),
                           "tool_calls": _strip_ids(msg.get("tool_calls"))}, ensure_ascii=False, sort_keys=True)
    return ""


def block_types(resp: dict) -> list[str]:
    """応答に含まれるブロック種別。thinking が切れているか、tool_use が出ているかを後から確認するため。"""
    if isinstance(resp.get("content"), list):
        return [b.get("type", "?") for b in resp["content"] if isinstance(b, dict)]
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        out = []
        if msg.get("reasoning_content"):
            out.append("reasoning")
        if msg.get("content"):
            out.append("text")
        if msg.get("tool_calls"):
            out.append("tool_calls")
        return out
    return []


def stop_reason(resp: dict) -> str | None:
    if "stop_reason" in resp:
        return resp.get("stop_reason")
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        return choices[0].get("finish_reason")
    return None


# --------------------------------------------------------------------------- HTTP

def build_body(endpoint: str, prompt: dict, n_predict: int, think: bool) -> dict:
    """保存済みプロンプトから送信本体を作る。

    messages: ccshim ログの /v1/messages 本体をそのまま使い、長さ・温度・stream・thinking だけ固定する。
    chat:     {"messages": [...]} から /v1/chat/completions 本体を組み立てる。
    """
    if endpoint == "messages":
        body = dict(prompt)
        body["max_tokens"] = n_predict
        body["temperature"] = 0
        body["stream"] = False
        body["thinking"] = {"type": "enabled", "budget_tokens": n_predict} if think else {"type": "disabled"}
        # Anthropic 形式の thinking 指定が無視される build に備え、テンプレート側のスイッチも併記する
        body["chat_template_kwargs"] = {"enable_thinking": think}
        return body
    return {
        "messages": prompt["messages"],
        "max_tokens": n_predict,
        "temperature": 0,
        "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": think},
    }


_PATHS = {"messages": "/v1/messages", "chat": "/v1/chat/completions"}


def run_once(url: str, endpoint: str, body: dict, timeout: float = 1800) -> tuple[int, dict | None, float, str]:
    """1 リクエスト。戻りは (status, resp_json, wall_s, error)。例外は status=0 で返す。"""
    req = urllib.request.Request(
        url.rstrip("/") + _PATHS[endpoint],
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
            status = res.status
    except urllib.error.HTTPError as e:
        return e.code, None, time.monotonic() - t0, e.read().decode(errors="replace")[:500]
    except Exception as e:  # 接続拒否・タイムアウト
        return 0, None, time.monotonic() - t0, repr(e)
    wall = time.monotonic() - t0
    try:
        return status, json.loads(raw), wall, ""
    except ValueError:
        return status, None, wall, "non-JSON response"


def fetch_metrics(url: str, timeout: float = 30) -> tuple[dict[str, float], str]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as res:
            return parse_metrics(res.read().decode()), ""
    except Exception as e:
        return {}, repr(e)


def fetch_build(url: str, timeout: float = 30) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/props", timeout=timeout) as res:
            return json.loads(res.read()).get("build_info")
    except Exception:
        return None


# --------------------------------------------------------------------------- GPU

class GpuSampler:
    """nvidia-smi を別スレッドで定期的に叩き、VRAM ピーク・電力・温度を集める。"""

    _QUERY = ["nvidia-smi", "--query-gpu=memory.used,power.draw,temperature.gpu",
              "--format=csv,noheader,nounits"]

    def __init__(self, interval_s: float = 1.0, enabled: bool = True):
        self._interval = interval_s
        self._enabled = enabled
        self._samples: list[tuple[int, float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        if self._enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_a: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(self._QUERY, stderr=subprocess.DEVNULL, timeout=5).decode()
                mem, power, temp = [x.strip() for x in out.strip().splitlines()[0].split(",")]
                self._samples.append((int(float(mem)), float(power), int(float(temp))))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def summary(self) -> dict:
        if not self._samples:
            return {"vram_peak_mib": None, "power_mean_w": None, "power_max_w": None,
                    "temp_max_c": None, "samples": 0}
        mems, powers, temps = zip(*self._samples)
        return {"vram_peak_mib": max(mems), "power_mean_w": round(sum(powers) / len(powers), 1),
                "power_max_w": max(powers), "temp_max_c": max(temps), "samples": len(self._samples)}


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--endpoint", choices=sorted(_PATHS), required=True)
    ap.add_argument("--prompt-file", type=pathlib.Path, required=True)
    ap.add_argument("--n-predict", type=int, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True, help="JSON Lines の追記先")
    ap.add_argument("--config", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--nvidia-smi", type=int, default=1, help="1 で GPU をサンプリング、0 で省略")
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args(argv)

    prompt = json.loads(args.prompt_file.read_text())
    body = build_body(args.endpoint, prompt, args.n_predict, args.think)

    build = fetch_build(args.url)
    before, err_before = fetch_metrics(args.url)
    with GpuSampler(enabled=bool(args.nvidia_smi)) as gpu:
        status, resp, wall, error = run_once(args.url, args.endpoint, body, timeout=args.timeout)
    after, err_after = fetch_metrics(args.url)

    text = extract_text(resp) if resp else ""
    payload = generated_payload(resp) if resp else ""
    errors = [e for e in (error, err_before and f"metrics before: {err_before}",
                          err_after and f"metrics after: {err_after}") if e]
    # /metrics が片方でも取れなければ差分は信用できない。None にして集計側で errors に数えさせる
    metrics_ok = not err_before and not err_after
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "block": args.block, "config": args.config, "spec": args.spec, "workload": args.workload,
        "prompt_id": args.prompt_id, "endpoint": args.endpoint, "n_predict": args.n_predict,
        "think": args.think, "url": args.url, "build": build,
        "status": status, "error": "; ".join(errors), "wall_s": round(wall, 3),
        "metrics": metrics_delta(before, after) if metrics_ok else None,
        "timings": extract_timings(resp) if resp else None,
        "stop_reason": stop_reason(resp) if resp else None,
        "block_types": block_types(resp) if resp else [],
        "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_len": len(text), "text_head": text[:200],
        "gpu": gpu.summary(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    m = row["metrics"] or {}
    print(f"[{args.config}/{args.spec}/{args.prompt_id}] status={status} wall={wall:.1f}s "
          f"decode={m.get('decode_tok_s')} acc={m.get('acceptance')} stop={row['stop_reason']} "
          f"blocks={row['block_types']}", flush=True)
    return 0 if status == 200 and metrics_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
