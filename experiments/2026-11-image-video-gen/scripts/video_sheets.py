#!/usr/bin/env python3
"""run_matrix.py が保存した動画(構成/プロンプト_seed.mp4)から、プロンプトごとに
「行 = 構成、列 = 等間隔の N フレーム」の比較シート(JPEG)を ffmpeg で作る。

  python3 video_sheets.py --vid ../results/raw/phase2/vid --out ../results/raw/phase2/sheets --frames 6
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def plan(files: list[pathlib.Path], labels: list[str] | None = None) -> dict[str, list[pathlib.Path]]:
    """{prompt_seed: [構成ごとの動画パス(labels の順)]}。パスは <label>/<prompt>_s<seed>.mp4 の形。"""
    out: dict[str, list[pathlib.Path]] = {}
    for f in files:
        out.setdefault(f.stem, []).append(f)
    order = {l: i for i, l in enumerate(labels or [])}
    for k in out:
        out[k].sort(key=lambda p: (order.get(p.parts[-2] if len(p.parts) > 1 else p.parts[0], 10**6), str(p)))
    return dict(sorted(out.items()))


def ffmpeg_args(src: pathlib.Path, dst: pathlib.Path, frames: int, total_frames: int, width: int) -> list[str]:
    step = max(1, (total_frames - 1) // max(1, frames - 1))  # 最初と最後のフレームを含める
    vf = f"select='not(mod(n,{step}))',scale={width}:-1,tile={frames}x1"
    return ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-frames:v", "1", str(dst)]


def nb_frames(src: pathlib.Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=nb_read_frames", "-of", "json", str(src)])
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vid", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--width", type=int, default=320, help="1 フレームの幅(px)")
    ap.add_argument("--labels", help="行の順序(カンマ区切り)")
    args = ap.parse_args(argv)
    from PIL import Image, ImageDraw

    files = sorted(args.vid.glob("*/*.mp4"))
    groups = plan([f.relative_to(args.vid) for f in files], args.labels.split(",") if args.labels else None)
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "_rows"
    tmp.mkdir(exist_ok=True)
    for key, rel_paths in groups.items():
        rows = []
        for rel in rel_paths:
            src = args.vid / rel
            row = tmp / f"{rel.parts[0]}__{key}.jpg"
            subprocess.check_call(ffmpeg_args(src, row, args.frames, nb_frames(src), args.width))
            rows.append((rel.parts[0], Image.open(row).convert("RGB")))
        w = max(im.width for _, im in rows)
        h = sum(im.height + 18 for _, im in rows)
        sheet = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(sheet)
        y = 0
        for label, im in rows:
            d.text((4, y + 2), label, fill="black")
            sheet.paste(im, (0, y + 18))
            y += im.height + 18
        dest = args.out / f"{key}.jpg"
        sheet.save(dest, quality=88)
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
