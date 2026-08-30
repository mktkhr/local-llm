#!/usr/bin/env python3
"""計測結果を CSV に整形し、PD分離の損益分岐を算出する。

入力:
  results/baseline.jsonl  単体構成 (A: M1 のみ / B: 4080 のみ)
  results/pd_runs.jsonl   PD分離構成 (C ほか)
出力:
  results/benchmark.csv   全試行の平坦な表
  results/breakeven.csv   ネットワーク速度別の損益分岐
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# 実効スループットの仮定値。Wi-Fi のみ実測、他は推定。
# 10GbE を実測したら EFFECTIVE_MIB_S を差し替える。
NETWORKS = {
    "wifi-measured": None,      # 実測値を使う
    "1gbe-est": 112.0,          # 1GbE の実効を 940Mbps 相当と仮定
    "10gbe-est": 1100.0,        # 10GbE の実効を 9.2Gbps 相当と仮定
}


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    baseline = load(RESULTS / "baseline.jsonl")
    pd_runs = load(RESULTS / "pd_runs.jsonl")
    if not baseline or not pd_runs:
        print("計測結果が足りません", file=sys.stderr)
        return 1

    # --- benchmark.csv ---
    rows = []
    for r in baseline:
        rows.append({
            "model": r.get("model", "?"), "kv_type": r.get("kv_type", "q8_0"), "config": r["label"], "prompt_tokens": r["prompt_tokens"],
            "prefill_ms": round(r["prefill_ms"], 1),
            "prefill_tok_per_s": round(r["prefill_tok_per_s"], 1),
            "kv_save_ms": "", "kv_mib": "", "transfer_ms": "", "transfer_mib_per_s": "",
            "restore_ms": "", "transfer_conns": "", "residual_prefill_tokens": "",
            "decode_tok_per_s": round(r["decode_tok_per_s"], 2),
            "ttft_ms": round(r["ttft_ms"], 1),
            "prefill_path_ms": round(r["prefill_ms"], 1),
            "pd_ok": "",
        })
    for r in pd_runs:
        rows.append({
            "model": r.get("model", "?"), "kv_type": r.get("kv_type", "q8_0"), "config": r["label"], "prompt_tokens": r["prompt_tokens"],
            "prefill_ms": round(r["prefill"]["prompt_ms"], 1),
            "prefill_tok_per_s": round(r["prefill"]["prompt_tok_per_s"], 1),
            "kv_save_ms": round(r["save"]["save_ms"], 1),
            "kv_mib": round(r["save"]["mib"], 1),
            "transfer_ms": round(r["transfer"]["ms"], 1),
            "transfer_mib_per_s": round(r["transfer"]["mib_per_s"], 1),
            "restore_ms": round(r["restore"]["restore_ms"], 1),
            "transfer_conns": r["transfer"].get("conns", 1),
            "residual_prefill_tokens": r["residual_prefill_tokens"],
            "decode_tok_per_s": round(r["decode"]["decode_tok_per_s"], 2),
            "ttft_ms": round(r["e2e_prefill_path_ms"], 1),
            "prefill_path_ms": round(r["e2e_prefill_path_ms"], 1),
            "pd_ok": r["pd_ok"],
        })

    fields = list(rows[0].keys())
    with open(RESULTS / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # --- breakeven.csv ---
    # 単体構成 A の基準値。モデルが異なれば別物なので (model, tokens) で引く。
    a = {(r.get("model", "?"), r.get("kv_type", "q8_0"), r["prompt_tokens"]): r
         for r in baseline if r["label"].startswith("A-")}

    # PD分離の構成ごとに損益分岐を出す。転送方式(単一接続 / 並列)の違いを
    # そのまま比較できるようにするため、ラベルとモデルを行に残す。
    pd_configs = {}
    for r in pd_runs:
        if r["label"].startswith("C"):
            pd_configs.setdefault((r.get("model", "?"), r.get("kv_type", "q8_0"), r["label"]), {})[r["prompt_tokens"]] = r

    be_rows = []
    for (model, kv_type, label), pd_by_tok in pd_configs.items():
      for tok in sorted(pd_by_tok):
        if (model, kv_type, tok) not in a:
            continue
        p = pd_by_tok[tok]
        m1_prefill = a[(model, kv_type, tok)]["prefill_ms"]
        kv_mib = p["save"]["mib"]
        fixed = p["prefill"]["prompt_ms"] + p["save"]["save_ms"] + p["restore"]["restore_ms"] + p["decode"]["prompt_ms"]

        for net, mibps in NETWORKS.items():
            xfer = p["transfer"]["ms"] if mibps is None else kv_mib / mibps * 1000.0
            total = fixed + xfer
            be_rows.append({
                "model": model,
                "kv_type": kv_type,
                "config": label,
                "prompt_tokens": tok,
                "network": net,
                "transfer_mib_per_s": round(p["transfer"]["mib_per_s"], 1) if mibps is None else mibps,
                "kv_mib": round(kv_mib, 1),
                "m1_only_prefill_ms": round(m1_prefill, 1),
                "pd_prefill_path_ms": round(total, 1),
                "transfer_ms": round(xfer, 1),
                "speedup_x": round(m1_prefill / total, 2),
                "pd_wins": m1_prefill > total,
            })
    be_rows.sort(key=lambda r: (r["model"], r["kv_type"], r["config"], r["prompt_tokens"], r["network"]))

    with open(RESULTS / "breakeven.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(be_rows[0].keys()))
        w.writeheader()
        w.writerows(be_rows)

    print(f"wrote {RESULTS/'benchmark.csv'} ({len(rows)} rows)")
    print(f"wrote {RESULTS/'breakeven.csv'} ({len(be_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
