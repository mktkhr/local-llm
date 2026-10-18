#!/usr/bin/env python3
"""計測用プロンプトを用意する。

W1 / W2: ccshim のログ(NNN-req.json / NNN-resp.bin)から、応答に tool_use を含んだリクエストを拾う。
         W2 は Write / Edit で 1,500 文字以上を書いたもの(コード書き換え)、W1 はそれ以外の tool_use。
         各 W につき本体サイズの 10 / 50 / 90 パーセンタイルに近い 3 本を選ぶ。
W3:      tools/bench/data/summary/meetings.json の文字起こし 2 本 + 連結 1 本の日本語要約。
W4:      scripts/data/w4_prompts.json(手書き)。本スクリプトは触らない。

W1 / W2 は Claude Code の system prompt とツール定義全文を含むので scripts/data/private/ に書き、commit しない。
形(サイズ・メッセージ数など)だけ prompts_manifest.json に残す。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

W2_MIN_CHARS = 1500
W2_TOOLS = {"Write", "Edit", "MultiEdit"}
W3_INSTRUCTION = "次の会議の文字起こしを日本語で要約してください。決定事項と TODO を箇条書きで。"

_IDX = re.compile(r"(\d{3,})-req\.json$")


def tool_uses_in_resp(raw: bytes) -> list[dict]:
    """応答本体(非 SSE JSON または SSE)から tool_use を [{"name", "input_json"}] で返す。"""
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            body = json.loads(stripped)
        except ValueError:
            return []
        out = []
        for block in body.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append({"name": block.get("name", ""),
                            "input_json": json.dumps(block.get("input", {}), ensure_ascii=False)})
        return out

    # SSE: content_block_start(tool_use) → input_json_delta の partial_json を index ごとに連結
    blocks: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            continue
        t = ev.get("type")
        if t == "content_block_start" and (ev.get("content_block") or {}).get("type") == "tool_use":
            blocks[ev.get("index", 0)] = {"name": ev["content_block"].get("name", ""), "parts": []}
        elif t == "content_block_delta" and (ev.get("delta") or {}).get("type") == "input_json_delta":
            if ev.get("index") in blocks:
                blocks[ev["index"]]["parts"].append(ev["delta"].get("partial_json", ""))
    return [{"name": b["name"], "input_json": "".join(b["parts"])} for _, b in sorted(blocks.items())]


def classify(req: dict, tool_uses: list[dict]) -> str | None:
    """tools 付きリクエストのうち、大きな Write/Edit を含むものを W2、それ以外の tool_use を W1 とする。"""
    if not req.get("tools") or not tool_uses:
        return None
    for tu in tool_uses:
        if tu["name"] in W2_TOOLS and len(tu["input_json"]) >= W2_MIN_CHARS:
            return "W2"
    return "W1"


def select(cands: dict[str, list[dict]], k: int = 3, seed: int = 42,
           max_bytes: int | None = None) -> dict[str, list[dict]]:
    """各 W で bytes の 10/50/90 パーセンタイル(k=3)に最も近い候補を 1 つずつ選ぶ。重複は飛ばす。

    max_bytes を与えると、それを超える候補を先に除く(計測側の ctx に収まらないプロンプトを避ける)。
    """
    out: dict[str, list[dict]] = {}
    for w, items in cands.items():
        items = sorted((x for x in items if max_bytes is None or x["bytes"] <= max_bytes), key=lambda x: x["bytes"])
        if not items:
            out[w] = []
            continue
        n = len(items)
        picks: list[dict] = []
        quantiles = [0.1, 0.5, 0.9] if k == 3 else [(i + 1) / (k + 1) for i in range(k)]
        for q in quantiles:
            idx = min(n - 1, max(0, round(q * n)))
            cand = items[idx]
            if cand in picks:
                continue
            picks.append(cand)
        out[w] = picks
    return out


def build_w3(meetings: list[dict]) -> list[dict]:
    """各 transcript と全連結の計 len+1 本。chat 形式の {"id", "messages"}。"""
    texts = [m["transcript"] for m in meetings]
    texts.append("\n\n---\n\n".join(texts))
    return [{"id": f"W3-{i + 1:02d}",
             "messages": [{"role": "user", "content": f"{W3_INSTRUCTION}\n\n{t}"}]}
            for i, t in enumerate(texts)]


def _scan_ccshim(dirs: list[pathlib.Path]) -> dict[str, list[dict]]:
    cands: dict[str, list[dict]] = {"W1": [], "W2": []}
    for d in dirs:
        for req_path in sorted(d.glob("*-req.json")):
            m = _IDX.search(req_path.name)
            if not m:
                continue
            resp_path = req_path.with_name(f"{m.group(1)}-resp.bin")
            if not resp_path.exists() or req_path.stat().st_size < 1000:
                continue
            try:
                req = json.loads(req_path.read_text())
            except ValueError:
                continue
            if not isinstance(req, dict) or "messages" not in req:
                continue
            w = classify(req, tool_uses_in_resp(resp_path.read_bytes()))
            if w:
                cands[w].append({"bytes": req_path.stat().st_size, "req": req,
                                 "source": f"{d.name}/{req_path.name}"})
    return cands


def _shape(req: dict) -> dict:
    sysf = req.get("system")
    sys_chars = sum(len(b.get("text", "")) for b in sysf) if isinstance(sysf, list) else len(sysf or "")
    return {"n_messages": len(req.get("messages") or []), "n_tools": len(req.get("tools") or []),
            "system_chars": sys_chars}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ccshim-dir", type=pathlib.Path, action="append", default=[],
                    help="ccshim ログのディレクトリ(複数可)")
    ap.add_argument("--meetings", type=pathlib.Path, required=True)
    ap.add_argument("--out-private", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="これを超える req 本体は候補から外す(目安: ctx 96k なら 300000、128k なら 420000。約 3.3 bytes/token)")
    args = ap.parse_args(argv)

    args.out_private.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    picked = select(_scan_ccshim(args.ccshim_dir), k=args.k, max_bytes=args.max_bytes)
    for w, items in picked.items():
        for i, it in enumerate(items):
            pid = f"{w}-{i + 1:02d}"
            req = dict(it["req"])
            req.pop("stream", None)
            (args.out_private / f"{pid}.json").write_text(json.dumps(req, ensure_ascii=False))
            manifest.append({"id": pid, "workload": w, "bytes": it["bytes"], "source": it["source"], **_shape(req)})
        print(f"{w}: {len(items)} 本", file=sys.stderr)

    for p in build_w3(json.loads(args.meetings.read_text())):
        (args.out_private / f"{p['id']}.json").write_text(json.dumps(p, ensure_ascii=False))
        manifest.append({"id": p["id"], "workload": "W3", "bytes": len(json.dumps(p, ensure_ascii=False)),
                         "source": args.meetings.name, "n_messages": 1, "n_tools": 0, "system_chars": 0})

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"manifest: {args.manifest} ({len(manifest)} 本)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
