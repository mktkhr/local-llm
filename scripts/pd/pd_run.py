#!/usr/bin/env python3
"""PD分離(Prefill/Decode分離)の1試行を実行し、各段階の実測値を JSON で返す。

判定は HTTP ステータスではなく timings.cache_n / timings.prompt_n で行う。
restore が 200 を返しても、Decode 側で prompt 全体が再 Prefill されていれば
PD分離は成立していない。
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

# 再現性のため固定文。トークナイザが同一なら両機で同じ列になる。
FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    "Meanwhile, seventeen astronomers catalogued forty-three variable stars. "
)


def post(base, path, payload, timeout=1800):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, (time.perf_counter() - t0) * 1000.0


def n_tokens(base, text):
    body, _ = post(base, "/tokenize", {"content": text})
    return len(body["tokens"])


def build_prompt(base, target):
    """target トークン以上になる最小の繰り返し回数を二分探索で求める。"""
    lo, hi = 1, 8
    while n_tokens(base, FILLER * hi) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if n_tokens(base, FILLER * mid) < target:
            lo = mid + 1
        else:
            hi = mid
    text = FILLER * lo
    return text, n_tokens(base, text)


def completion(base, prompt, n_predict, slot=0):
    body, wall = post(
        base,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "id_slot": slot,
            "cache_prompt": True,
            "temperature": 0.0,
            "seed": 1234,
        },
    )
    return body, body["timings"], wall


def _fetch_range(url, start, end, path):
    """[start, end] を Range で取得し、path の該当オフセットへ書く。"""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=3600) as r, open(path, "r+b") as f:
        f.seek(start)
        shutil.copyfileobj(r, f, length=1 << 20)


def fetch_kv(url, dest, conns=1):
    """KV ファイルを取得し、転送時間と実効速度を返す。

    一時ファイルへ書いてから置換する。同一マシンで検証するとき、
    取得元と保存先が同じパスになり、書き込みが読み出し中の実体を
    切り詰めてしまうため。

    conns > 1 のときは Range で分割して並列に取得する。Wi-Fi のように
    遅延と再送があるリンクでは、単一 TCP フローは輻輳ウィンドウが繰り返し
    落とされてリンク容量の半分以下しか出ない。並列にすると各フローが独立に
    回復するためパイプが埋まる。配信側が Range に対応している必要がある
    (nginx は対応、Python の http.server は非対応)。
    """
    tmp = dest + ".part"

    if conns <= 1:
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=3600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        ms = (time.perf_counter() - t0) * 1000.0
        size = os.path.getsize(tmp)
        os.replace(tmp, dest)
        return ms, size

    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as r:
        size = int(r.headers["Content-Length"])
        ranges_ok = r.headers.get("Accept-Ranges", "").lower() == "bytes"
    if not ranges_ok:
        print("WARN: 配信側が Range 非対応。単一接続へ退避します", file=sys.stderr)
        return fetch_kv(url, dest, conns=1)

    # 疎ファイルを先に確保してから各区間を並列に書き込む。
    with open(tmp, "wb") as f:
        f.truncate(size)

    step = size // conns
    parts = [(i * step, size - 1 if i == conns - 1 else (i + 1) * step - 1) for i in range(conns)]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(conns) as ex:
        list(ex.map(lambda p: _fetch_range(url, p[0], p[1], tmp), parts))
    ms = (time.perf_counter() - t0) * 1000.0

    got = os.path.getsize(tmp)
    if got != size:
        raise RuntimeError(f"並列取得のサイズ不一致 {got} != {size}")
    os.replace(tmp, dest)
    return ms, size


def push_kv(src, dest_spec):
    """KV ファイルを scp で送り出す。

    Decode 側が着信を受けられない場合(macOS のファイアウォール等)に使う。
    ssh の暗号化ぶんだけ HTTP より不利になるため、HTTP pull と直接比較しない。
    """
    t0 = time.perf_counter()
    subprocess.run(["scp", "-q", src, dest_spec], check=True)
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, os.path.getsize(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-url", required=True, help="Prefill 側 llama-server (例 http://PREFILL_HOST:18080)")
    ap.add_argument("--decode-url", required=True, help="Decode 側 llama-server (例 http://127.0.0.1:18080)")
    ap.add_argument("--transfer-mode", choices=["http-pull", "scp-push"], default="http-pull",
                    help="http-pull: Decode 側が取りに行く / scp-push: Prefill 側が送り出す")
    ap.add_argument("--kv-url", help="http-pull 用。KV ファイル配信元 (例 http://PREFILL_HOST:18081)")
    ap.add_argument("--decode-kv-dir", help="http-pull 用。Decode 側 --slot-save-path の実パス")
    ap.add_argument("--prefill-kv-dir", help="scp-push 用。Prefill 側 --slot-save-path の実パス")
    ap.add_argument("--push-dest", help="scp-push 用。scp の宛先 (例 USER@DECODE_HOST:/path/to/kv)")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--conns", type=int, default=8,
                    help="http-pull 時の並列接続数。1 で従来どおり単一接続")
    ap.add_argument("--label", default="")
    ap.add_argument("--model", required=True,
                    help="モデルの識別子。集計時に構成同士を突き合わせる鍵になる")
    ap.add_argument("--out", help="JSON Lines の追記先")
    args = ap.parse_args()

    if args.transfer_mode == "http-pull":
        missing = [n for n in ("kv_url", "decode_kv_dir") if not getattr(args, n)]
    else:
        missing = [n for n in ("prefill_kv_dir", "push_dest") if not getattr(args, n)]
    if missing:
        ap.error(f"--transfer-mode {args.transfer_mode} には {missing} が必要です")

    # 構成ごとにファイル名を分ける。同名だと、別構成のコンテナが root 権限で
    # 作ったファイルを上書きできず Permission denied になる。
    tag = re.sub(r"[^A-Za-z0-9]+", "-", args.label).strip("-") or "run"
    filename = f"pd_{tag}_{args.tokens}.bin"
    out = {"label": args.label, "model": args.model, "target_tokens": args.tokens, "n_predict": args.n_predict}

    # プロンプトは Decode 側で生成する。両機のトークナイザが一致していることは
    # 同一 GGUF (sha256 一致) が前提。
    prompt, ntok = build_prompt(args.decode_url, args.tokens)
    ntok_prefill = n_tokens(args.prefill_url, prompt)
    if ntok != ntok_prefill:
        print(f"FATAL: トークン数が両機で不一致 decode={ntok} prefill={ntok_prefill}", file=sys.stderr)
        print("  同一の GGUF を使っているか確認してください", file=sys.stderr)
        return 2
    out["prompt_tokens"] = ntok

    # --- Prefill 側 ---
    post(args.prefill_url, "/slots/0?action=erase", {})
    _, t_pre, wall_pre = completion(args.prefill_url, prompt, 1)
    out["prefill"] = {
        "prompt_n": t_pre["prompt_n"],
        "cache_n": t_pre["cache_n"],
        "prompt_ms": t_pre["prompt_ms"],
        "prompt_tok_per_s": t_pre["prompt_per_second"],
        "wall_ms": wall_pre,
    }
    if t_pre["prompt_n"] < ntok - 8:
        print(f"WARN: Prefill 側でキャッシュが効いている prompt_n={t_pre['prompt_n']}", file=sys.stderr)

    # --- KV save ---
    b, wall = post(args.prefill_url, "/slots/0?action=save", {"filename": filename})
    out["save"] = {
        "n_saved": b["n_saved"],
        "n_written": b["n_written"],
        "mib": b["n_written"] / 1048576.0,
        "bytes_per_token": b["n_written"] / b["n_saved"],
        "save_ms": b["timings"]["save_ms"],
        "wall_ms": wall,
    }

    # --- 転送 ---
    if args.transfer_mode == "http-pull":
        dest = os.path.join(args.decode_kv_dir, filename)
        xfer_ms, size = fetch_kv(f"{args.kv_url}/{filename}", dest, conns=args.conns)
    else:
        src = os.path.join(args.prefill_kv_dir, filename)
        xfer_ms, size = push_kv(src, f"{args.push_dest}/{filename}")
    if size != b["n_written"]:
        print(f"FATAL: 転送サイズ不一致 {size} != {b['n_written']}", file=sys.stderr)
        return 2
    out["transfer"] = {
        "bytes": size,
        "ms": xfer_ms,
        "mib_per_s": size / 1048576.0 / (xfer_ms / 1000.0),
        "mbps": size * 8 / 1e6 / (xfer_ms / 1000.0),
        "mode": args.transfer_mode,
        "conns": args.conns if args.transfer_mode == "http-pull" else 1,
    }

    # --- Decode 側 restore ---
    post(args.decode_url, "/slots/0?action=erase", {})
    b, wall = post(args.decode_url, "/slots/0?action=restore", {"filename": filename})
    out["restore"] = {
        "n_restored": b["n_restored"],
        "n_read": b["n_read"],
        "restore_ms": b["timings"]["restore_ms"],
        "wall_ms": wall,
    }

    # --- Decode ---
    body, t_dec, wall_dec = completion(args.decode_url, prompt, args.n_predict)
    out["decode"] = {
        "cache_n": t_dec["cache_n"],
        "prompt_n": t_dec["prompt_n"],
        "prompt_ms": t_dec["prompt_ms"],
        "predicted_n": t_dec["predicted_n"],
        "predicted_ms": t_dec["predicted_ms"],
        "decode_tok_per_s": t_dec["predicted_per_second"],
        "wall_ms": wall_dec,
    }
    out["output_head"] = body["content"][:120]

    # --- 判定 ---
    # 最終トークンは必ず再処理されるため prompt_n は 0 にならない。
    residual = t_dec["prompt_n"]
    out["residual_prefill_tokens"] = residual
    out["pd_ok"] = bool(t_dec["cache_n"] >= ntok - 8 and residual <= 8)

    out["pd_overhead_ms"] = (
        out["save"]["save_ms"] + out["transfer"]["ms"] + out["restore"]["restore_ms"] + t_dec["prompt_ms"]
    )
    out["e2e_prefill_path_ms"] = out["prefill"]["prompt_ms"] + out["pd_overhead_ms"]

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 掃引時に読みやすいよう、1 行の要約を stderr に出す。
    print(
        f"  tokens={out['prompt_tokens']} pd_ok={out['pd_ok']} "
        f"prefill={out['prefill']['prompt_ms']:.0f}ms "
        f"save={out['save']['save_ms']:.0f}ms "
        f"xfer={out['transfer']['ms']:.0f}ms({out['transfer']['mib_per_s']:.1f}MiB/s) "
        f"restore={out['restore']['restore_ms']:.0f}ms "
        f"residual={out['residual_prefill_tokens']}tok",
        file=sys.stderr,
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["pd_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
