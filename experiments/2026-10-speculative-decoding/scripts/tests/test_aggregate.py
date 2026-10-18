import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from aggregate import per_pos, summarize  # noqa: E402


def row(block, spec, pid, tok_s, sha, acc=None, draft=0, status=200):
    return {"block": block, "config": "C", "spec": spec, "workload": "W1", "prompt_id": pid, "status": status,
            "metrics": {"decode_tok_s": tok_s, "acceptance": acc, "draft_per_generated": draft,
                        "prefill_tok_s": 1000, "cached_n": 5, "accepted_per_pos": [3, 1]},
            "text_sha256": sha, "gpu": {"vram_peak_mib": 15000, "power_mean_w": 200}}


class Summarize(unittest.TestCase):
    def test_paired_ratio_and_match(self):
        rows = [row(1, "none", "p1", 100, "a"), row(1, "mtp2", "p1", 120, "a", 0.8, 0.7),
                row(2, "none", "p1", 110, "b"), row(2, "mtp2", "p1", 110, "c", 0.6, 0.7)]
        s = {r["spec"]: r for r in summarize(rows)}
        self.assertAlmostEqual(s["mtp2"]["ratio_vs_none_mean"], (1.2 + 1.0) / 2)
        self.assertAlmostEqual(s["mtp2"]["text_match_rate"], 0.5)
        self.assertEqual(s["none"]["ratio_vs_none_mean"], 1.0)
        self.assertEqual(s["mtp2"]["n"], 2)
        self.assertAlmostEqual(s["mtp2"]["acceptance_mean"], 0.7)
        self.assertAlmostEqual(s["mtp2"]["decode_tok_s_mean"], 115.0)
        self.assertEqual(s["mtp2"]["vram_peak_mib_max"], 15000)

    def test_errors_excluded(self):
        rows = [row(1, "none", "p1", 100, "a"), row(1, "none", "p2", 0, "", status=500)]
        s = summarize(rows)[0]
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["errors"], 1)

    def test_unpaired_ratio_is_empty(self):
        s = summarize([row(1, "mtp2", "p1", 120, "a", 0.8, 0.7)])[0]
        self.assertEqual(s["ratio_vs_none_mean"], "")

    def test_per_pos(self):
        pp = per_pos([row(1, "mtp2", "p1", 1, "a"), row(1, "mtp2", "p2", 1, "a")])
        self.assertEqual(pp[0]["config"], "C")
        self.assertEqual(pp[0]["spec"], "mtp2")
        self.assertEqual(pp[0]["pos_0"], 6)
        self.assertEqual(pp[0]["pos_1"], 2)


if __name__ == "__main__":
    unittest.main()
