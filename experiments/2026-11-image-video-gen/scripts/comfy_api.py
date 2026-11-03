#!/usr/bin/env python3
"""ComfyUI の HTTP API で 1 枚生成し、所要時間・VRAM ピーク・出力ファイルを JSONL に記録する。

ワークフローは API 形式(/prompt に渡す dict)を Python で組み立てる。UI の JSON とは形が違う。
  python3 comfy_api.py --url http://192.168.0.144:8188 --graph z-image-turbo \
      --prompt "..." --seed 42 --width 1024 --height 1024 --out results/raw/smoke.jsonl --download results/raw/img

GPU のサンプリングは 2026-10-speculative-decoding/scripts/spec_bench.py の GpuSampler を流用する(nvidia-smi が
ある機械で動かすこと。別機から叩くときは --nvidia-smi 0)。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "2026-10-speculative-decoding" / "scripts"))
from spec_bench import GpuSampler  # noqa: E402


def graph_z_image_turbo(prompt: str, seed: int, width: int, height: int, steps: int = 8,
                        unet: str = "z_image_turbo_bf16.safetensors", clip: str = "qwen_3_4b.safetensors",
                        vae: str = "ae.safetensors", prefix: str = "z-image-turbo") -> dict:
    """公式テンプレート image_z_image_turbo.json を API 形式に平坦化したもの。"""
    return {
        "28": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "30": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "lumina2", "device": "default"}},
        "29": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3, "model": ["28", 0]}},
        "27": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["30", 0]}},
        "33": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["27", 0]}},
        "13": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 1, "sampler_name": "res_multistep",
                                                   "scheduler": "simple", "denoise": 1, "model": ["11", 0],
                                                   "positive": ["27", 0], "negative": ["33", 0], "latent_image": ["13", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["29", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


GRAPHS = {"z-image-turbo": graph_z_image_turbo}


def _json(url: str, data: dict | None = None, timeout: float = 60):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


def run(url: str, graph: dict, timeout: float = 1800) -> tuple[dict, float]:
    """/prompt に投げ、/history で完了を待つ。戻りは (history entry, wall_s)。"""
    client_id = uuid.uuid4().hex
    t0 = time.monotonic()
    pid = _json(f"{url}/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]
    while True:
        hist = _json(f"{url}/history/{pid}")
        if pid in hist and hist[pid].get("status", {}).get("completed") is not None or (pid in hist and hist[pid].get("outputs")):
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(entry["status"], ensure_ascii=False)[:800])
            if entry.get("outputs"):
                return entry, time.monotonic() - t0
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"prompt {pid} did not finish in {timeout}s")
        time.sleep(1.0)


def outputs_of(entry: dict) -> list[dict]:
    out = []
    for node_id, o in (entry.get("outputs") or {}).items():
        for kind in ("images", "gifs", "videos"):
            for f in o.get(kind, []) or []:
                out.append({"node": node_id, "kind": kind, **f})
    return out


def download(url: str, f: dict, dest_dir: pathlib.Path) -> pathlib.Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
    dest = dest_dir / f["filename"]
    with urllib.request.urlopen(f"{url}/view?{q}", timeout=300) as res, dest.open("wb") as w:
        w.write(res.read())
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--graph", choices=sorted(GRAPHS), required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=pathlib.Path, help="JSONL の追記先")
    ap.add_argument("--download", type=pathlib.Path, help="出力画像の保存先ディレクトリ")
    ap.add_argument("--nvidia-smi", type=int, default=1)
    args = ap.parse_args(argv)

    kwargs = {"prompt": args.prompt, "seed": args.seed, "width": args.width, "height": args.height}
    if args.steps is not None:
        kwargs["steps"] = args.steps
    graph = GRAPHS[args.graph](**kwargs)
    url = args.url.rstrip("/")
    status, error, entry, wall = "ok", "", None, None
    with GpuSampler(enabled=bool(args.nvidia_smi)) as gpu:
        try:
            entry, wall = run(url, graph)
        except Exception as e:  # noqa: BLE001
            status, error = "error", repr(e)[:800]
    files = outputs_of(entry) if entry else []
    saved = [str(download(url, f, args.download)) for f in files] if args.download and files else []
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": args.label, "graph": args.graph, "prompt": args.prompt,
           "seed": args.seed, "width": args.width, "height": args.height, "steps": args.steps,
           "status": status, "error": error, "wall_s": round(wall, 2) if wall else None,
           "outputs": [f["filename"] for f in files], "saved": saved, "gpu": gpu.summary()}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: row[k] for k in ("status", "error", "wall_s", "outputs", "saved", "gpu")}, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
