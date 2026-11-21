#!/usr/bin/env python3
"""ccshim のログ(Claude Code の /v1/messages リクエスト)から、imatrix 用と評価用の素のテキストを作る。

imatrix と perplexity は特殊トークンを解釈せずにトークナイズするので、チャットテンプレートは使わず、
system・ツール定義・会話(本文、thinking、tool_use の入力、tool_result)を順に並べたテキストにする。
system とツール定義は全セッションでほぼ同じなので、較正側に 1 回だけ入れ、評価側には入れない(リーク防止)。
セッションは最初のユーザー発話で識別し、各セッションで最も長いリクエストだけを使う(履歴は毎回送り直されるため)。
  python3 make_corpus.py --logs /l1 /l2 /l3 --out-dir results/raw
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib


def _text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", ""))
        elif t == "thinking":
            out.append(b.get("thinking", ""))
        elif t == "tool_use":
            out.append(f"{b.get('name', '')} {json.dumps(b.get('input', {}), ensure_ascii=False)}")
        elif t == "tool_result":
            out.append(_text(b.get("content")))
    return "\n".join(x for x in out if x)


def render_preamble(body: dict) -> str:
    sys_ = body.get("system")
    sys_txt = sys_ if isinstance(sys_, str) else "\n".join(b.get("text", "") for b in sys_ or [] if isinstance(b, dict))
    tools = "\n".join(json.dumps(t, ensure_ascii=False) for t in body.get("tools") or [])
    return sys_txt + "\n\n" + tools


def render_messages(body: dict) -> str:
    return "\n\n".join(f"{m.get('role', '')}: {_text(m.get('content'))}" for m in body.get("messages") or [])


def session_key(body: dict) -> str:
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            return hashlib.sha1(_text(m.get("content"))[:300].encode()).hexdigest()[:12]
    return ""


def latest_per_session(bodies: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for b in bodies:
        k = session_key(b)
        if k and (k not in best or len(b["messages"]) > len(best[k]["messages"])):
            best[k] = b
    return list(best.values())


def split(bodies: list[dict], eval_mod: int = 3) -> tuple[list[dict], list[dict]]:
    """セッションキーのハッシュで較正用と評価用に分ける(決定的)。"""
    cal, ev = [], []
    for b in sorted(bodies, key=session_key):
        (ev if int(session_key(b), 16) % eval_mod == 0 else cal).append(b)
    return cal, ev


def shuffled_pieces(text: str, size: int = 2000, seed: int | None = 0) -> list[str]:
    """行の切れ目で size 文字以下の断片に分け、seed で並べ替える(seed=None なら元の順)。

    imatrix は先頭から決まった数のチャンクしか読まないので、並べ替えてコーパス全体から均等に拾わせる。
    """
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:size])
            line = line[size:]
        if cur and len(cur) + 1 + len(line) > size:
            out.append(cur)
            cur = line
        else:
            cur = line if not cur else cur + "\n" + line
    if cur:
        out.append(cur)
    if seed is not None:
        import random
        random.Random(seed).shuffle(out)
    return out


def load(dirs: list[str]) -> list[dict]:
    out = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*-req.json")):
            try:
                b = json.load(open(f))
            except ValueError:
                continue
            if isinstance(b, dict) and b.get("messages"):
                out.append(b)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--eval-mod", type=int, default=3)
    args = ap.parse_args(argv)
    sessions = latest_per_session(load(args.logs))
    cal, ev = split(sessions, args.eval_mod)
    pre = render_preamble(max(cal, key=lambda b: len(render_preamble(b))))
    cal_txt = "\n".join(shuffled_pieces(pre + "\n\n" + "\n\n".join(render_messages(b) for b in cal)))
    ev_txt = "\n\n".join(render_messages(b) for b in ev)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cc-calib.txt").write_text(cal_txt)
    (args.out_dir / "cc-eval.txt").write_text(ev_txt)
    meta = {"sessions": len(sessions), "calib_sessions": len(cal), "eval_sessions": len(ev),
            "calib_chars": len(cal_txt), "eval_chars": len(ev_txt), "preamble_chars": len(pre)}
    (args.out_dir / "corpus-meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
