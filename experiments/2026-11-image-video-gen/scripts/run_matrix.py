#!/usr/bin/env python3
"""モデル構成 × 固定プロンプト × seed の行列を ComfyUI で回し、画像と計測行(JSONL)を残す。

  python3 run_matrix.py --url http://gpu:8188 --models data/phase1_models.json --prompts data/prompts.json \
      --seeds 1,2,3 --out ../results/raw/phase1/runs.jsonl --download ../results/raw/phase1/img

同じ (label, prompt id, seed) で status=ok の行が --out にあれば飛ばす(再開可能)。
--check は実行せず、各構成のモデルファイル名がサーバの選択肢(/object_info の COMBO)にあるかだけ確かめる。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from comfy_api import GpuSampler, _json, download, graph_from_template, object_info, outputs_of, run  # noqa: E402
from tpl2api import swap_unet_loader_gguf  # noqa: E402

LOADER_INPUTS = {"UNETLoader": "unet_name", "UnetLoaderGGUF": "unet_name", "CLIPLoader": "clip_name",
                 "DualCLIPLoader": ("clip_name1", "clip_name2"), "VAELoader": "vae_name",
                 "CheckpointLoaderSimple": "ckpt_name", "LoraLoaderModelOnly": "lora_name", "LoraLoader": "lora_name"}


def build(model: dict, info: dict, prompt: str, seed: int, prefix: str) -> dict:
    api = graph_from_template(model["template"], info, prompt, seed, model.get("width"), model.get("height"),
                              model.get("steps"), model.get("sets") or [])
    if model.get("gguf"):
        swap_unet_loader_gguf(api, model["gguf"])
    for v in api.values():
        if "filename_prefix" in v["inputs"]:
            v["inputs"]["filename_prefix"] = prefix
    return api


def missing_files(api: dict, info: dict) -> list[str]:
    """ローダーのファイル名がサーバの選択肢に無いものを列挙する。"""
    out = []
    for v in api.values():
        names = LOADER_INPUTS.get(v["class_type"])
        if not names:
            continue
        for name in (names if isinstance(names, tuple) else (names,)):
            val = v["inputs"].get(name)
            spec = info[v["class_type"]]["input"]["required"].get(name) or info[v["class_type"]]["input"].get("optional", {}).get(name)
            choices = spec[0] if spec and isinstance(spec[0], list) else None
            if isinstance(val, str) and choices is not None and val not in choices:
                out.append(f"{v['class_type']}.{name}={val}")
    return out


def gpu_power_limit() -> float | None:
    """nvidia-smi の電力上限(W)。計測条件の記録用。取れなければ None。"""
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                                      stderr=subprocess.DEVNULL, timeout=5)
        return float(out.decode().strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def free_models(url: str) -> None:
    """読み込み済みモデルを VRAM/RAM から外す。RAM 32 GB の機では、モデル構成を切り替えるたびに呼ばないと
    前のモデルの CPU オフロード分が残ってスワップする。"""
    try:
        _json(url + "/free", {"unload_models": True, "free_memory": True})
    except Exception as e:  # noqa: BLE001
        print(f"/free failed: {e!r}", file=sys.stderr)


def done_keys(jsonl: pathlib.Path) -> set[tuple[str, str, int]]:
    keys = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                keys.add((r["label"], r["prompt_id"], r["seed"]))
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--models", type=pathlib.Path, required=True)
    ap.add_argument("--prompts", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--only", help="label をカンマ区切りで限定")
    ap.add_argument("--prompt-ids", help="prompt id をカンマ区切りで限定")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--download", type=pathlib.Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refresh-object-info", action="store_true")
    ap.add_argument("--nvidia-smi", type=int, default=1)
    args = ap.parse_args(argv)

    url = args.url.rstrip("/")
    info = object_info(url, refresh=args.refresh_object_info)
    models = [m for m in json.loads(args.models.read_text()) if not args.only or m["label"] in args.only.split(",")]
    prompts = [p for p in json.loads(args.prompts.read_text()) if not args.prompt_ids or p["id"] in args.prompt_ids.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.check:
        bad = 0
        for m in models:
            miss = missing_files(build(m, info, "check", 0, "check"), info)
            print(f"{'NG' if miss else 'ok'} {m['label']}: {', '.join(miss)}")
            bad += bool(miss)
        return 1 if bad else 0

    done = done_keys(args.out)
    power_limit = gpu_power_limit()
    total = len(models) * len(prompts) * len(seeds)
    i = 0
    for m in models:
        freed = False
        for p in prompts:
            for seed in seeds:
                i += 1
                key = (m["label"], p["id"], seed)
                if key in done:
                    continue
                if not freed:
                    free_models(url)
                    freed = True
                prefix = f"{m['label']}/{p['id']}_s{seed}"
                graph = build(m, info, p["prompt"], seed, prefix)
                status, error, entry, wall = "ok", "", None, None
                with GpuSampler(enabled=bool(args.nvidia_smi)) as gpu:
                    try:
                        entry, wall = run(url, graph)
                    except Exception as e:  # noqa: BLE001
                        status, error = "error", repr(e)[:1500]
                files = outputs_of(entry) if entry else []
                saved = [str(download(url, f, args.download / m["label"],
                                      name=f"{p['id']}_s{seed}{'' if j == 0 else f'-{j}'}{pathlib.Path(f['filename']).suffix}"))
                         for j, f in enumerate(files)]
                row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": m["label"], "template": m["template"],
                       "sets": m.get("sets") or [], "gguf": m.get("gguf"), "width": m.get("width"), "height": m.get("height"),
                       "steps": m.get("steps"), "prompt_id": p["id"], "seed": seed, "status": status, "error": error,
                       "power_limit_w": power_limit,
                       "wall_s": round(wall, 2) if wall else None, "saved": saved, "gpu": gpu.summary()}
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{i}/{total}] {m['label']} {p['id']} s{seed}: {status} {row['wall_s']}s "
                      f"vram={row['gpu'].get('vram_peak_mib')} {error[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
