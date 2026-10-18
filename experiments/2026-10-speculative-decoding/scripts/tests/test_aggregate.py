import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from aggregate import per_pos, summarize, summarize_cc  # noqa: E402


def row(block, spec, pid, tok_s, sha, acc=None, draft=0, status=200, sweep="sweep-1"):
    return {"sweep": sweep, "block": block, "config": "C", "spec": spec, "workload": "W1", "prompt_id": pid, "status": status,
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

    def test_pairs_only_within_same_sweep(self):
        rows = [row(1, "none", "p1", 100, "a", sweep="sweep-1"), row(1, "mtp2", "p1", 150, "a", sweep="sweep-2")]
        s = {r["spec"]: r for r in summarize(rows)}
        self.assertEqual(s["mtp2"]["ratio_vs_none_mean"], "")

    def test_missing_metrics_is_error(self):
        r = row(1, "none", "p1", 100, "a")
        r["metrics"] = None
        s = summarize([r])[0]
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["errors"], 1)

    def test_summarize_cc(self):
        sums = [{"task": "M1", "label": "C-mtp2-run1", "pass": True, "wall_s": 84, "exit_code": 0, "num_turns": 30,
                 "changed_files": 3, "cc_metrics": {"session": {"requests": 31, "tool_use_total": 40, "json_broken": 0,
                                                                "raw_token_leak": 0, "cache_n_list": [0, 25000, None]},
                                                    "metrics": {"acceptance": 0.75, "draft_n": 900, "predicted_n": 1200,
                                                                "prompt_n": 2000, "cached_n": 8000}}}]
        r = summarize_cc(sums)[0]
        self.assertEqual(r["prompt_n"], 2000)
        self.assertEqual(r["cached_n"], 8000)
        self.assertAlmostEqual(r["cache_hit"], 0.8)
        self.assertEqual(r["task"], "M1")
        self.assertEqual(r["pass"], 1)
        self.assertEqual(r["tool_use_total"], 40)
        self.assertAlmostEqual(r["acceptance"], 0.75)
        self.assertAlmostEqual(r["cache_n_mean"], 12500.0)

    def test_per_pos(self):
        pp = per_pos([row(1, "mtp2", "p1", 1, "a"), row(1, "mtp2", "p2", 1, "a")])
        self.assertEqual(pp[0]["config"], "C")
        self.assertEqual(pp[0]["spec"], "mtp2")
        self.assertEqual(pp[0]["pos_0"], 6)
        self.assertEqual(pp[0]["pos_1"], 2)


if __name__ == "__main__":
    unittest.main()
