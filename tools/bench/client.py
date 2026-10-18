"""Ollama HTTP クライアント。

責務:
- /api/generate, /api/ps, /api/tags, /api/version を叩く
- ストリーミング応答から TTFT を計測
- 非ストリーミング応答から eval_count / eval_duration 等を取得
- モデルの即時アンロード(keep_alive: 0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


@dataclass
class GenerateResult:
    """非ストリーミング generate の生のメトリクス。"""

    response_text: str
    eval_count: int
    eval_duration_ns: int
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    load_duration_ns: int
    total_duration_ns: int

    @property
    def decode_tokens_per_sec(self) -> float:
        if self.eval_duration_ns == 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)

    @property
    def prefill_tokens_per_sec(self) -> float:
        if self.prompt_eval_duration_ns == 0:
            return 0.0
        return self.prompt_eval_count / (self.prompt_eval_duration_ns / 1e9)


@dataclass
class StreamResult:
    """ストリーミング generate の結果。TTFT のために存在する。"""

    response_text: str
    ttft_sec: float
    total_elapsed_sec: float


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 600.0):
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def version(self) -> str:
        r = self._client.get("/api/version")
        r.raise_for_status()
        return r.json().get("version", "")

    def ps(self) -> list[dict[str, Any]]:
        r = self._client.get("/api/ps")
        r.raise_for_status()
        return r.json().get("models", [])

    def tags(self) -> list[dict[str, Any]]:
        r = self._client.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        think: bool | None = None,
        keep_alive: str | int = "5m",
    ) -> GenerateResult:
        """非ストリーミングで /api/generate を呼ぶ。eval_* 系メトリクスを返す。"""
        options: dict[str, Any] = {}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if num_predict is not None:
            options["num_predict"] = num_predict

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if options:
            body["options"] = options
        if think is not None:
            body["think"] = think

        r = self._client.post("/api/generate", json=body)
        r.raise_for_status()
        data = r.json()
        return GenerateResult(
            response_text=data.get("response", ""),
            eval_count=int(data.get("eval_count", 0)),
            eval_duration_ns=int(data.get("eval_duration", 0)),
            prompt_eval_count=int(data.get("prompt_eval_count", 0)),
            prompt_eval_duration_ns=int(data.get("prompt_eval_duration", 0)),
            load_duration_ns=int(data.get("load_duration", 0)),
            total_duration_ns=int(data.get("total_duration", 0)),
        )

    def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        think: bool | None = None,
        keep_alive: str | int = "5m",
    ) -> StreamResult:
        """ストリーミングで /api/generate を呼び、TTFT を計測。"""
        options: dict[str, Any] = {}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if num_predict is not None:
            options["num_predict"] = num_predict

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
        }
        if options:
            body["options"] = options
        if think is not None:
            body["think"] = think

        start = time.monotonic()
        ttft: float | None = None
        chunks: list[str] = []

        with self._client.stream("POST", "/api/generate", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                import json

                obj = json.loads(line)
                token = obj.get("response", "")
                if token and ttft is None:
                    ttft = time.monotonic() - start
                chunks.append(token)
                if obj.get("done"):
                    break

        end = time.monotonic()
        return StreamResult(
            response_text="".join(chunks),
            ttft_sec=ttft if ttft is not None else (end - start),
            total_elapsed_sec=end - start,
        )

    def unload(self, model: str) -> None:
        """keep_alive=0 で即時アンロード。"""
        self._client.post(
            "/api/generate",
            json={"model": model, "keep_alive": 0},
        )

    def wait_unloaded(self, model: str, timeout_sec: float = 30.0) -> bool:
        """ps から model が消えるまで待つ。timeout 内にアンロード完了したら True。"""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if not any(m.get("name") == model or m.get("model") == model for m in self.ps()):
                return True
            time.sleep(0.5)
        return False

    def model_info_in_ps(self, model: str) -> dict[str, Any] | None:
        """ロード済みモデルの ps エントリを返す。未ロードなら None。"""
        for m in self.ps():
            if m.get("name") == model or m.get("model") == model:
                return m
        return None
