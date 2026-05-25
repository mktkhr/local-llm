"""Ollama リクエストボディに think:false を注入する薄いリバースプロキシ。

Qwen3.5 など thinking モードを持つモデルで、Modelfile での無効化が
未対応の Ollama バージョン向けの暫定回避策。
クライアント側が think を明示的に指定していた場合はそれを尊重する(上書きしない)。
"""

import json
import logging
import os

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:11434")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "11500"))

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("think-injector")


def maybe_inject_think(body: bytes, content_type: str) -> tuple[bytes, bool]:
    if not body or not content_type.lower().startswith("application/json"):
        return body, False
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, False
    if not isinstance(data, dict) or "think" in data:
        return body, False
    data["think"] = False
    return json.dumps(data).encode(), True


async def proxy(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    body, injected = maybe_inject_think(body, request.headers.get("content-type", ""))

    req_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() != "host"
    }
    if body:
        req_headers["Content-Length"] = str(len(body))

    url = f"{UPSTREAM}{request.path_qs}"
    timeout = ClientTimeout(total=None, sock_read=600, sock_connect=10)

    async with ClientSession(timeout=timeout) as session:
        async with session.request(
            method=request.method,
            url=url,
            data=body if body else None,
            headers=req_headers,
            allow_redirects=False,
        ) as upstream:
            resp_headers = {
                k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
            }
            resp = web.StreamResponse(status=upstream.status, headers=resp_headers)
            await resp.prepare(request)
            async for chunk in upstream.content.iter_any():
                if chunk:
                    await resp.write(chunk)
            await resp.write_eof()
            log.info(
                "%s %s -> %s (think_injected=%s)",
                request.method,
                request.path,
                upstream.status,
                injected,
            )
            return resp


def make_app() -> web.Application:
    # 100MB まで(画像入力などの大きめペイロード想定)
    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=LISTEN_PORT, access_log=None)
