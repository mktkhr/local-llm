import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from aggregate import summarize  # noqa: E402


def row(config, pid, block, prompt_n, prefill, decode, n, sda, vram=15000, status=200):
    return {"config": config, "prompt_id": pid, "block": block, "status": status, "wall_s": 10.0,
            "metrics": {"prompt_n": prompt_n, "prefill_tok_s": prefill, "decode_tok_s": decode, "predicted_n": n},
            "sda_read_mib": sda, "gpu": {"vram_peak_mib": vram}}


class Summarize(unittest.TestCase):
    def test_medians_per_config_split_by_fresh_and_cached_prefix(self):
        rows = [row("a", "W4-01", 1, 44, 2.0, 6.0, 300, 12000),
                row("a", "W4-02", 1, 112, 4.0, 7.0, 300, 18000),
                row("a", "W4-01", 2, 4, 14.0, 40.0, 300, 50),      # 接頭辞キャッシュ(prompt_n < 10)
                row("a", "W4-03", 1, 40, 0, 0, 0, 0, status=500)]  # 失敗は除外
        s = summarize(rows)["a"]
        self.assertEqual(s["n_ok"], 3)
        self.assertEqual(s["fresh"]["n"], 2)
        self.assertEqual(s["fresh"]["decode_med"], 6.5)
        self.assertEqual(s["fresh"]["prefill_med"], 3.0)
        self.assertEqual(s["fresh"]["sda_mib_per_100tok_med"], 5000.0)  # (4000+6000)/2
        self.assertEqual(s["cached"]["n"], 1)
        self.assertEqual(s["cached"]["decode_med"], 40.0)
        self.assertEqual(s["vram_peak_mib"], 15000)
