"""Claude Code -> llama.cpp (llama-server / llama-swap) の変換シム。

Claude Code は Anthropic Messages API に対し、トップレベル system とは別に
messages 配列の中へ role:"system" のメッセージを注入してくる。
Qwen 系の Jinja チャットテンプレートはこれを
  raise_exception('System message must be at the beginning')
で拒否するため、llama-server は 500 を返す(実測・最小再現済み)。

本シムは messages 内の role:"system" を抜き出し、トップレベル system の
末尾ブロックとして畳み込んでから上流へ渡す。内容は失われない。

併せて、V-2(プレフィックスキャッシュのヒット率)計測のため、
ターンごとのプロンプト構成を JSONL に記録する。

  usage: ccshim.py <listen_port> <upstream_base> <logdir> [force_model]

force_model を指定すると、全リクエストの model をその名前に書き換える。
Claude Code はサブエージェント(Explore 等)に claude-sonnet-5 / claude-opus-5 を
使うため、指定しないと llama-swap がメインとサブでモデルを往復ロードし、
そのたびにプレフィックスキャッシュが破棄されて再プリフィルが走る(実測 2秒 -> 21秒)。
"""

import http.server, json, os, socketserver, sys, threading, time
import urllib.request, urllib.error

PORT = int(sys.argv[1])
UPSTREAM = sys.argv[2].rstrip("/")
LOGDIR = sys.argv[3]
FORCE_MODEL = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(LOGDIR, exist_ok=True)
TRACE = os.path.join(LOGDIR, "trace.jsonl")

_lock = threading.Lock()
_n = [0]
_t0 = [None]


def _text_of(content):
    """Anthropic の content(str | ブロック配列)を素のテキストにする。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def fold_system(payload):
    """messages 内の role:"system" をトップレベル system に畳み込む。

    返り値: (変換後 payload, 畳み込んだ件数)
    """
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return payload, 0

    stray = [m for m in msgs if m.get("role") == "system"]
    if not stray:
        return payload, 0

    kept = [m for m in msgs if m.get("role") != "system"]

    # トップレベル system をブロック配列に正規化してから追記する
    sys_field = payload.get("system")
    if sys_field is None:
        blocks = []
    elif isinstance(sys_field, str):
        blocks = [{"type": "text", "text": sys_field}]
    elif isinstance(sys_field, list):
        blocks = list(sys_field)
    else:
        blocks = []

    for m in stray:
        t = _text_of(m.get("content"))
        if t:
            blocks.append({"type": "text", "text": t})

    payload["system"] = blocks
    payload["messages"] = kept
    return payload, len(stray)


def prompt_shape(payload):
    """V-2 用: このリクエストのプロンプト構成を要約する。"""
    sysf = payload.get("system")
    sys_chars = sum(len(b.get("text", "")) for b in sysf) if isinstance(sysf, list) \
        else len(sysf or "")
    msgs = payload.get("messages") or []
    per_msg = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            kinds = [b.get("type") for b in c]
            chars = sum(len(json.dumps(b, ensure_ascii=False)) for b in c)
        else:
            kinds = ["str"]
            chars = len(c or "")
        per_msg.append({"role": m.get("role"), "blocks": kinds, "chars": chars})
    return {
        "system_chars": sys_chars,
        "system_blocks": len(sysf) if isinstance(sysf, list) else 1,
        "tools": len(payload.get("tools") or []),
        "n_messages": len(msgs),
        "messages": per_msg,
        "total_chars": sys_chars + sum(m["chars"] for m in per_msg),
    }


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        with _lock:
            _n[0] += 1
            idx = _n[0]
            if _t0[0] is None:
                _t0[0] = time.time()
            elapsed = time.time() - _t0[0]

        folded = 0
        shape = None
        forced = False
        body = raw
        if raw and self.path.startswith("/v1/messages"):
            try:
                payload = json.loads(raw)
                if FORCE_MODEL and payload.get("model") != FORCE_MODEL:
                    payload["model"] = FORCE_MODEL
                    forced = True
                payload, folded = fold_system(payload)
                shape = prompt_shape(payload)
                body = json.dumps(payload, ensure_ascii=False).encode()
            except Exception as e:
                print(f"[shim] payload 変換に失敗: {e}", flush=True)

        with open(os.path.join(LOGDIR, f"{idx:03d}-req.json"), "wb") as f:
            f.write(body or b"{}")

        req = urllib.request.Request(UPSTREAM + self.path, data=body or None, method=method)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "connection", "accept-encoding"):
                continue
            req.add_header(k, v)

        t_start = time.time()
        stream = False
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
            status, resp_headers = resp.status, list(resp.getheaders())
            ctype = resp.headers.get("Content-Type", "")
            stream = "text/event-stream" in ctype
        except urllib.error.HTTPError as e:
            resp = None
            status, resp_headers, data = e.code, list(e.headers.items()), e.read()
        except Exception as e:
            resp = None
            status = 599
            resp_headers = [("Content-Type", "application/json")]
            data = json.dumps({"shim_error": str(e)}).encode()

        respf = open(os.path.join(LOGDIR, f"{idx:03d}-resp.bin"), "wb")

        if resp is not None and stream:
            # SSE は逐次転送する。ここをバッファすると、thinking が長いモデルで
            # クライアントから「無応答の接続」に見え、アイドルタイムアウトによる
            # リトライを誘発する(実測: thinking 26k tok で 282 秒ごとに再送)。
            self.send_response(status)
            for k, v in resp_headers:
                if k.lower() in ("transfer-encoding", "content-length", "connection"):
                    continue
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            nbytes = 0
            try:
                while True:
                    chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                    if not chunk:
                        break
                    nbytes += len(chunk)
                    respf.write(chunk)
                    self.wfile.write(b"%x\r\n" % len(chunk))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception as e:
                print(f"[shim] #{idx} ストリーム転送中に切断: {e}", flush=True)
            finally:
                resp.close()
            data_len = nbytes
        else:
            if resp is not None:
                data = resp.read()
                resp.close()
            respf.write(data)
            data_len = len(data)
            self.send_response(status)
            for k, v in resp_headers:
                if k.lower() in ("transfer-encoding", "content-length", "connection"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        respf.close()
        dur = time.time() - t_start
        rec = {"idx": idx, "t": round(elapsed, 2), "path": self.path, "status": status,
               "folded_system": folded, "forced_model": forced, "dur_s": round(dur, 2),
               "stream": stream, "bytes": data_len, "shape": shape}
        with open(TRACE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[shim] #{idx} {self.path} -> {status} "
              f"(system畳み込み {folded}件{', model固定' if forced else ''}, "
              f"{'stream' if stream else 'buffered'}, {dur:.1f}s)", flush=True)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


print(f"ccshim: :{PORT} -> {UPSTREAM}  (log: {LOGDIR}"
      + (f", model固定: {FORCE_MODEL}" if FORCE_MODEL else "") + ")", flush=True)
Server(("0.0.0.0", PORT), Handler).serve_forever()
