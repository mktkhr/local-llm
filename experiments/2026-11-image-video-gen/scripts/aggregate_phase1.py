#!/usr/bin/env python3
"""run_matrix.py の runs.jsonl を集計して Markdown 表にし、モデル × プロンプトの比較シート(PNG)を作る。

  python3 aggregate_phase1.py --runs ../results/raw/phase1/runs.jsonl --img ../results/raw/phase1/img \
      --sheets ../results/phase1-sheets --md ../results/phase1-summary.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def load_rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def summarize(rows: list[dict]) -> dict[str, dict]:
    """label ごとに: 件数、最初の 1 回(コールド)の時間、2 回目以降の中央値、VRAM ピーク、平均電力。"""
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["label"], {"n_ok": 0, "n_error": 0, "cold_s": None, "_warm": [], "vram_peak_mib": None,
                                        "_power": [], "power_limit_w": r.get("power_limit_w"), "errors": []})
        if r["status"] != "ok":
            s["n_error"] += 1
            s["errors"].append(f"{r.get('prompt_id')} s{r.get('seed')}: {r.get('error', '')[:160]}")
            continue
        s["n_ok"] += 1
        if s["cold_s"] is None:
            s["cold_s"] = r["wall_s"]
        else:
            s["_warm"].append(r["wall_s"])
        g = r.get("gpu") or {}
        if g.get("vram_peak_mib") is not None:
            s["vram_peak_mib"] = max(s["vram_peak_mib"] or 0, g["vram_peak_mib"])
        if g.get("power_mean_w") is not None:
            s["_power"].append(g["power_mean_w"])
    for s in out.values():
        s["warm_median_s"] = round(statistics.median(s["_warm"]), 2) if s["_warm"] else None
        s["power_mean_w"] = round(statistics.mean(s["_power"]), 1) if s["_power"] else None
        del s["_warm"], s["_power"]
    return out


def to_markdown(summary: dict[str, dict]) -> str:
    lines = ["| 構成 | 成功/失敗 | 初回(モデル読込込み) | 2 回目以降 中央値 | VRAM ピーク | 平均電力 | 電力制限 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    skipped = [label for label, s in summary.items() if s["n_ok"] == 0]
    for label, s in summary.items():
        if s["n_ok"] == 0:
            continue  # 1 枚も成功していない構成(ファイル未取得など)は表に出さない
        f = lambda v, u: f"{v} {u}" if v is not None else "-"  # noqa: E731
        lines.append(f"| {label} | {s['n_ok']}/{s['n_error']} | {f(s['cold_s'], 's')} | {f(s['warm_median_s'], 's')} | "
                     f"{f(s['vram_peak_mib'], 'MiB')} | {f(s['power_mean_w'], 'W')} | {f(s['power_limit_w'], 'W')} |")
    if skipped:
        lines += ["", "成功 0 件(未取得・未実行): " + ", ".join(skipped)]
    errs = [f"- {label}: {e}" for label, s in summary.items() if s["n_ok"] for e in s["errors"]]
    if errs:
        lines += ["", "失敗:", *errs]
    return "\n".join(lines) + "\n"


def make_sheets(rows: list[dict], img_dir: pathlib.Path, out_dir: pathlib.Path, seed: int, cell: int = 512,
                labels: list[str] | None = None) -> list[pathlib.Path]:
    """プロンプトごとに 1 枚: 横にモデル構成を並べる(指定 seed)。成功 0 件の構成は省く。PIL が無ければ何もしない。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_labels = list(dict.fromkeys(r["label"] for r in rows if r["status"] == "ok"))
    labels = [l for l in (labels or ok_labels) if l in ok_labels]
    pids = list(dict.fromkeys(r["prompt_id"] for r in rows))
    made = []
    for pid in pids:
        cells = []
        for label in labels:
            hits = sorted((img_dir / label).glob(f"{pid}_s{seed}.png")) or sorted((img_dir / label).glob(f"{pid}_s{seed}_*.png"))
            cells.append((label, hits[0] if hits else None))  # ComfyUI が付ける連番(_00001_)の有無どちらでも
        sheet = Image.new("RGB", (cell * len(cells), cell + 24), "white")
        d = ImageDraw.Draw(sheet)
        for i, (label, p) in enumerate(cells):
            d.text((i * cell + 4, 4), label, fill="black")
            if p:
                im = Image.open(p).convert("RGB")
                im.thumbnail((cell, cell))
                sheet.paste(im, (i * cell + (cell - im.width) // 2, 24))
        dest = out_dir / f"{pid}_s{seed}.jpg"
        sheet.save(dest, quality=88)
        made.append(dest)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=pathlib.Path, required=True)
    ap.add_argument("--img", type=pathlib.Path, help="run_matrix --download のディレクトリ")
    ap.add_argument("--sheets", type=pathlib.Path, help="比較シートの出力先")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cell", type=int, default=512, help="シートの 1 枚あたりの辺(px)。リポジトリに置く縮小版は 256")
    ap.add_argument("--labels", help="シートに載せる構成をカンマ区切りで限定(順序もこの通り)")
    ap.add_argument("--md", type=pathlib.Path, help="Markdown 表の出力先(省略時は標準出力)")
    args = ap.parse_args(argv)
    rows = load_rows(args.runs)
    md = to_markdown(summarize(rows))
    if args.md:
        args.md.write_text(md)
    else:
        print(md)
    if args.img and args.sheets:
        for p in make_sheets(rows, args.img, args.sheets, args.seed, cell=args.cell,
                             labels=args.labels.split(",") if args.labels else None):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
