#!/usr/bin/env python3
"""Claude Code の 1 リクエストを、ツール数・メッセージ数を変えて llama-server に投げ、
前処理(変換・テンプレート・grammar 文字列・トークナイズ)と残りの固定費を分けて測る。

各条件で次を順に叩き、1 呼び出し 1 行の JSON Lines を残す。
  health        GET /health                       HTTP の往復だけ
  apply         POST /apply-template (OpenAI 形式)  JSON パース + jinja + grammar 文字列生成
  tokenize      POST /tokenize                    レンダ済みプロンプトのトークナイズ
  count         POST /v1/messages/count_tokens    Anthropic 変換 + apply + tokenize(本番と同じ前処理)
  msg_cold/warm POST /v1/messages max_tokens=1    1 回目は prefill 込み、以降は同一本文の再送
  chat_warm     POST /v1/chat/completions         変換なしの比較用
  apply_none / chat_none_warm / dec_chat_none
                tool_choice=none(ツール定義は残し grammar だけ外す)。auto との差が grammar のコスト
  msg_any_warm  tool_choice=any(非 lazy grammar。llama.cpp では呼び出し前の本文も許す)
  dec_auto/any  max_tokens=N の decode(grammar あり/なしの 1 トークン時間)

プロンプト本文はリポジトリに残さない。行に入るのはサイズ・トークン数・時間だけ。
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time
import urllib.error
import urllib.request

_STRIP = ("metadata", "context_management", "output_config")


def _dump(obj) -> str:
    """nlohmann::ordered_json の dump() 相当(区切りに空白なし、UTF-8 のまま)。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def anthropic_to_oai(body: dict) -> dict:
    """llama-server の server_chat_convert_anthropic_to_oai を写したもの(画像は扱わない)。"""
    msgs: list[dict] = []
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            text = system
        else:
            text = "".join(b.get("text", "") for b in system if b.get("type") == "text")
        msgs.append({"role": "system", "content": text})

    for msg in body["messages"]:
        role = msg.get("role", "")
        content = msg.get("content")
        if not isinstance(content, list):
            if content is None and role == "assistant":
                continue
            msgs.append(msg)
            continue
        converted, tool_calls, tool_results, reasoning = [], [], [], ""
        for b in content:
            t = b.get("type")
            if t == "text":
                converted.append(b)
            elif t == "thinking":
                reasoning += b.get("thinking", "")
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id", ""), "type": "function",
                                   "function": {"name": b.get("name", ""),
                                                "arguments": _dump(b.get("input", {}))}})
            elif t == "tool_result":
                rc = b.get("content")
                if isinstance(rc, str):
                    text = rc
                elif isinstance(rc, list):
                    text = "".join(c.get("text", "") for c in rc if c.get("type") == "text")
                else:
                    text = ""
                tool_results.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": text})
        if converted or tool_calls or reasoning:
            m: dict = {"role": role}
            m["content"] = converted if converted else ""
            if tool_calls:
                m["tool_calls"] = tool_calls
            if reasoning:
                m["reasoning_content"] = reasoning
            msgs.append(m)
        msgs.extend(tool_results)

    oai: dict = {"messages": msgs}
    if isinstance(body.get("tools"), list):
        oai["tools"] = [{"type": "function", "function": {
            "name": t.get("name", ""), "description": t.get("description", ""),
            "parameters": t.get("input_schema", {})}} for t in body["tools"]]
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        if tc.get("type") == "auto":
            oai["tool_choice"] = "auto"
        elif tc.get("type") in ("any", "tool"):
            oai["tool_choice"] = "required"
    oai["max_tokens"] = body.get("max_tokens", 4096)
    for k in ("temperature", "top_p", "top_k", "stream", "chat_template_kwargs", "id_slot"):
        if k in body:
            oai[k] = body[k]
    return oai


def variant(body: dict, n_tools: int | None, n_msgs: int | None, max_tokens: int,
            tool_choice: str | None = None) -> dict:
    """ツール先頭 n_tools 個・メッセージ先頭 n_msgs 件に切り、サンプリングと build 依存の欄を固定する。"""
    v = copy.deepcopy(body)
    for k in _STRIP:
        v.pop(k, None)
    if n_tools is not None:
        v["tools"] = v.get("tools", [])[:n_tools]
    if not v.get("tools"):
        v.pop("tools", None)
    if n_msgs is not None:
        v["messages"] = v["messages"][:n_msgs]
    v["max_tokens"] = max_tokens
    v["temperature"] = 0
    v["stream"] = False
    v["thinking"] = {"type": "disabled"}
    v["chat_template_kwargs"] = {"enable_thinking": False}
    if tool_choice:
        v["tool_choice"] = {"type": tool_choice}
    return v


def sizes(body: dict) -> dict:
    return {
        "n_tools": len(body.get("tools", [])),
        "n_msgs": len(body["messages"]),
        "chars_system": len(_dump(body.get("system", ""))),
        "chars_tools": len(_dump(body.get("tools", []))),
        "chars_msgs": len(_dump(body["messages"])),
    }


# --------------------------------------------------------------------------- HTTP

def call(url: str, path: str, body: dict | None = None, timeout: float = 1800) -> tuple[int, dict | None, float, str]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
            status = res.status
    except urllib.error.HTTPError as e:
        return e.code, None, time.perf_counter() - t0, e.read().decode(errors="replace")[:300]
    except Exception as e:
        return 0, None, time.perf_counter() - t0, repr(e)
    wall = time.perf_counter() - t0
    try:
        return status, json.loads(raw), wall, ""
    except ValueError:
        return status, None, wall, "non-JSON"


def metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30) as res:
        out = {}
        for line in res.read().decode().splitlines():
            if line and not line.startswith("#"):
                name, _, val = line.rpartition(" ")
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
        return out


_KEYS = {"prompt_n": "prompt_tokens_total", "prompt_s": "prompt_seconds_total",
         "predicted_n": "tokens_predicted_total", "predicted_s": "tokens_predicted_seconds_total"}


def delta(before: dict, after: dict) -> dict:
    return {k: after.get("llamacpp:" + v, 0.0) - before.get("llamacpp:" + v, 0.0) for k, v in _KEYS.items()}


def gen(url: str, path: str, body: dict) -> dict:
    before = metrics(url)
    status, resp, wall, err = call(url, path, body)
    d = delta(before, metrics(url))
    row = {"status": status, "wall_s": wall, "error": err, **d}
    if resp:
        t = resp.get("timings") or {}
        row["cache_n"] = t.get("cache_n")
        row["stop"] = resp.get("stop_reason") or (resp.get("choices") or [{}])[0].get("finish_reason")
        blocks = resp.get("content")
        if isinstance(blocks, list):
            row["blocks"] = [b.get("type") for b in blocks]
    return row


def probe(url: str, body: dict, reps: int, decode_n: int):
    """1 条件ぶん。(kind, rep, row) を順に返す。"""
    oai = anthropic_to_oai(body)
    for i in range(reps):
        st, _, wall, err = call(url, "/health")
        yield "health", i, {"status": st, "wall_s": wall, "error": err}
    prompt = None
    for i in range(reps):
        st, resp, wall, err = call(url, "/apply-template", oai)
        prompt = (resp or {}).get("prompt", prompt)
        yield "apply", i, {"status": st, "wall_s": wall, "error": err, "prompt_chars": len(prompt or "")}
    for i in range(reps):
        st, resp, wall, err = call(url, "/tokenize", {"content": prompt or "", "add_special": True, "parse_special": True})
        yield "tokenize", i, {"status": st, "wall_s": wall, "error": err, "n_tokens": len((resp or {}).get("tokens", []))}
    for i in range(reps):
        st, resp, wall, err = call(url, "/v1/messages/count_tokens", body)
        yield "count", i, {"status": st, "wall_s": wall, "error": err, "input_tokens": (resp or {}).get("input_tokens")}
    yield "msg_cold", 0, gen(url, "/v1/messages", body)
    for i in range(reps):
        yield "msg_warm", i, gen(url, "/v1/messages", body)
    for i in range(reps):
        yield "chat_warm", i, gen(url, "/v1/chat/completions", oai)
    if body.get("tools"):
        # tool_choice=none はツール定義をプロンプトに残したまま grammar だけ外す(同一プロンプトで grammar の有無を比べる)
        none = {**oai, "tool_choice": "none"}
        for i in range(reps):
            st, resp, wall, err = call(url, "/apply-template", none)
            yield "apply_none", i, {"status": st, "wall_s": wall, "error": err,
                                    "prompt_chars": len((resp or {}).get("prompt", ""))}
        for i in range(reps):
            yield "chat_none_warm", i, gen(url, "/v1/chat/completions", none)
        yield "dec_chat_auto", 0, gen(url, "/v1/chat/completions", {**oai, "max_tokens": decode_n})
        yield "dec_chat_none", 0, gen(url, "/v1/chat/completions", {**none, "max_tokens": decode_n})
        anyb = {**body, "tool_choice": {"type": "any"}}
        for i in range(reps):
            yield "msg_any_warm", i, gen(url, "/v1/messages", anyb)
        yield "dec_any", 0, gen(url, "/v1/messages", {**anyb, "max_tokens": decode_n})
    yield "dec_auto", 0, gen(url, "/v1/messages", {**body, "max_tokens": decode_n})


def plan(src: dict, axis: str) -> list[tuple[int | None, int | None]]:
    """axis=tools はツール数、axis=msgs はユーザー発話で終わるメッセージ数を振る。"""
    if axis == "tools":
        n = len(src.get("tools", []))
        return [(k, None) for k in (0, 8, 16, 33, 64, n) if k <= n]
    ms = src["messages"]
    users = [i + 1 for i, m in enumerate(ms) if m["role"] == "user"]
    want = (1, 9, 25, 49, len(ms))
    return [(None, min(users, key=lambda u: abs(u - w))) for w in want]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--src", type=pathlib.Path, required=True, help="ccshim から抜いた /v1/messages 本文")
    ap.add_argument("--axis", choices=("tools", "msgs"), required=True)
    ap.add_argument("--tag", required=True, help="build などを key=value,... で")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--decode-n", type=int, default=96)
    args = ap.parse_args(argv)

    src = json.loads(args.src.read_text())
    tags = dict(kv.split("=", 1) for kv in args.tag.split(","))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n_tools, n_msgs in plan(src, args.axis):
        body = variant(src, n_tools=n_tools, n_msgs=n_msgs, max_tokens=1)
        base = {**tags, "axis": args.axis, "src": args.src.stem, **sizes(body)}
        for kind, rep, row in probe(args.url, body, args.reps, args.decode_n):
            with args.out.open("a") as f:
                f.write(json.dumps({**base, "kind": kind, "rep": rep, **row}) + "\n")
            if kind in ("count", "msg_cold") or kind.startswith("dec"):
                print(f"[{base['src']} tools={base['n_tools']} msgs={base['n_msgs']}] {kind} "
                      f"wall={row['wall_s']:.3f}s prompt_n={row.get('prompt_n')} "
                      f"tok={row.get('input_tokens')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
