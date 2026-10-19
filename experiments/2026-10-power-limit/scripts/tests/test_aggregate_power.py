import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from aggregate_power import limit_from_name, summarize  # noqa: E402


def row(limit, block, spec, pid, tok_s, pre, power, clock, w="W3"):
    return {"limit_w": limit, "sweep": f"sweep-x-pl{limit}", "block": block, "config": "C-iq3xxs", "spec": spec,
            "workload": w, "prompt_id": pid, "status": 200,
            "metrics": {"decode_tok_s": tok_s, "prefill_tok_s": pre, "acceptance": 0.8 if spec != "none" else None},
            "gpu": {"power_mean_w": power, "power_max_w": power + 10, "temp_max_c": 70, "clock_sm_mean_mhz": clock}}


class Names(unittest.TestCase):
    def test_limit_from_name(self):
        self.assertEqual(limit_from_name("sweep-20261019-pl150.jsonl"), 150)
        self.assertIsNone(limit_from_name("sweep-20261018-142340.jsonl"))


class Summarize(unittest.TestCase):
    def test_groups_and_relative(self):
        rows = [row(220, 1, "none", "p1", 150, 3000, 200, 2600), row(220, 1, "mtp2", "p1", 210, 2900, 180, 2650),
                row(150, 1, "none", "p1", 120, 2000, 150, 1900), row(150, 1, "mtp2", "p1", 175, 1950, 148, 1950)]
        s = {(r["limit_w"], r["spec"]): r for r in summarize(rows)}
        self.assertAlmostEqual(s[(220, "mtp2")]["ratio_vs_none"], 1.4)
        self.assertAlmostEqual(s[(150, "none")]["decode_vs_220w"], 0.8)
        self.assertEqual(s[(220, "none")]["decode_vs_220w"], 1.0)
        self.assertAlmostEqual(s[(220, "none")]["decode_tok_per_w"], 0.75)
        self.assertAlmostEqual(s[(150, "mtp2")]["decode_tok_per_w"], 175 / 148, places=2)
        self.assertEqual(s[(150, "none")]["clock_sm_mean_mhz"], 1900)
        self.assertEqual(s[(220, "none")]["workload"], "W3")


if __name__ == "__main__":
    unittest.main()
