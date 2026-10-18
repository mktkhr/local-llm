#!/usr/bin/env python3
"""KV キャッシュタイプを systemd 経由で切り替える。

`OLLAMA_KV_CACHE_TYPE` は Ollama 起動時環境変数なので、変更には service の再起動が
必要。override.conf を書き換えて daemon-reload + restart する。sudo 権限が要る。

事前に `sudo -v` で credentials をキャッシュしてから本スクリプトを呼ぶこと。

使い方:
    python kv.py --type q8_0   # KV を q8_0 に設定
    python kv.py --type q4_0   # KV を q4_0 に設定
    python kv.py --type f16    # KV を明示的に f16 に設定
    python kv.py --type none   # OLLAMA_KV_CACHE_TYPE を削除(= Ollama 既定の f16)
"""

from __future__ import annotations

import argparse
import subprocess
import time

import httpx

OVERRIDE_PATH = "/etc/systemd/system/ollama.service.d/override.conf"
SUPPORTED = {"f16", "q8_0", "q4_0"}


class SudoUnavailableError(RuntimeError):
    pass


def _sudo_run(argv: list[str], *, input_bytes: bytes | None = None) -> bytes:
    """sudo -n で実行。credentials がキャッシュされていないなら例外。"""
    res = subprocess.run(
        ["sudo", "-n", *argv],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        if b"a password is required" in res.stderr or b"sudo:" in res.stderr:
            raise SudoUnavailableError(
                "sudo credentials が未キャッシュ。先に `sudo -v` を実行してください。"
            )
        raise RuntimeError(
            f"sudo {' '.join(argv)} failed: {res.stderr.decode(errors='replace')}"
        )
    return res.stdout


def _read_override() -> str:
    try:
        with open(OVERRIDE_PATH, encoding="utf-8") as f:
            return f.read()
    except PermissionError:
        return _sudo_run(["cat", OVERRIDE_PATH]).decode()


def _patch_override(content: str, kv_type: str | None) -> str:
    lines = [
        line for line in content.splitlines() if "OLLAMA_KV_CACHE_TYPE" not in line
    ]
    if kv_type is None:
        return "\n".join(lines).rstrip("\n") + "\n"

    new_env = f'Environment="OLLAMA_KV_CACHE_TYPE={kv_type}"'
    inserted = False
    out: list[str] = []
    for line in lines:
        out.append(line)
        if not inserted and line.strip() == "[Service]":
            out.append(new_env)
            inserted = True
    if not inserted:
        out = ["[Service]", new_env, *lines]
    return "\n".join(out).rstrip("\n") + "\n"


def _write_override(content: str) -> None:
    _sudo_run(["tee", OVERRIDE_PATH], input_bytes=content.encode())


def _reload_and_restart() -> None:
    _sudo_run(["systemctl", "daemon-reload"])
    _sudo_run(["systemctl", "restart", "ollama"])


def _wait_ready(timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:11434/api/version", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("Ollama did not become ready within timeout")


def set_kv_cache_type(kv_type: str | None) -> None:
    """KV キャッシュタイプを設定し、Ollama を再起動する。

    kv_type が None の場合は `OLLAMA_KV_CACHE_TYPE` 行を削除する
    (= Ollama 既定値の f16 で起動する)。
    """
    if kv_type is not None and kv_type not in SUPPORTED:
        raise ValueError(f"Unsupported kv_type: {kv_type}. Use {SUPPORTED} or None.")

    current = _read_override()
    new_content = _patch_override(current, kv_type)
    if new_content == current:
        print("[kv] no change needed")
        return
    _write_override(new_content)
    _reload_and_restart()
    _wait_ready()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["f16", "q8_0", "q4_0", "none"],
        required=True,
        help="`none` で OLLAMA_KV_CACHE_TYPE を削除(= Ollama 既定 f16)",
    )
    args = parser.parse_args()

    kv = None if args.type == "none" else args.type
    print(f"[kv] set OLLAMA_KV_CACHE_TYPE={args.type}")
    set_kv_cache_type(kv)

    env = subprocess.check_output(
        ["systemctl", "show", "ollama", "-p", "Environment"]
    ).decode()
    print(env.strip())
    print("[kv] done")


if __name__ == "__main__":
    main()
