import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from aggregate_laguna import summarize_speed  # noqa: E402


def row(config, block, pid, tok_s, pre, vram, w="W3"):
    return {"sweep": "s", "block": block, "config": config, "spec": "none", "workload": w, "prompt_id": pid, "status": 200,
            "metrics": {"decode_tok_s": tok_s, "prefill_tok_s": pre, "prompt_n": 1000},
            "gpu": {"vram_peak_mib": vram, "power_mean_w": 200}}


class Speed(unittest.TestCase):
    def test_ratio_vs_reference(self):
        rows = [row("current-C", 1, "p1", 150, 3000, 15000), row("L-iq3xxs", 1, "p1", 120, 3600, 14000),
                row("current-C", 2, "p1", 148, 3000, 15000), row("L-iq3xxs", 2, "p1", 126, 3600, 14000)]
        s = {r["config"]: r for r in summarize_speed(rows, reference="current-C")}
        self.assertAlmostEqual(s["L-iq3xxs"]["decode_vs_ref"], (120 / 150 + 126 / 148) / 2)
        self.assertAlmostEqual(s["L-iq3xxs"]["prefill_vs_ref"], 1.2)
        self.assertEqual(s["current-C"]["decode_vs_ref"], 1.0)
        self.assertEqual(s["L-iq3xxs"]["vram_peak_mib_max"], 14000)
        self.assertEqual(s["L-iq3xxs"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
