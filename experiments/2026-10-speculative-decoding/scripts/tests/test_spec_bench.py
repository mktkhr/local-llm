import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from spec_bench import extract_text, extract_timings, metrics_delta, parse_metrics  # noqa: E402

METRICS = """# HELP llamacpp:tokens_predicted_total x
llamacpp:tokens_predicted_total 100
llamacpp:tokens_predicted_seconds_total 2.5
llamacpp:prompt_tokens_total 1000
llamacpp:prompt_tokens_cached_total 900
llamacpp:prompt_seconds_total 0.5
llamacpp:spec_decode_num_draft_tokens_total 60
llamacpp:spec_decode_num_accepted_tokens_total 45
llamacpp:spec_decode_num_drafts_total 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 15
"""


class ParseMetrics(unittest.TestCase):
    def test_parses_labels(self):
        m = parse_metrics(METRICS)
        self.assertEqual(m["llamacpp:tokens_predicted_total"], 100.0)
        self.assertEqual(m['llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"}'], 15.0)


class MetricsDelta(unittest.TestCase):
    def test_delta_and_derived(self):
        before = parse_metrics(METRICS)
        after = dict(before)
        after["llamacpp:tokens_predicted_total"] = 300
        after["llamacpp:tokens_predicted_seconds_total"] = 4.5
        after["llamacpp:spec_decode_num_draft_tokens_total"] = 160
        after["llamacpp:spec_decode_num_accepted_tokens_total"] = 125
        after['llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}'] = 80
        d = metrics_delta(before, after)
        self.assertEqual(d["predicted_n"], 200)
        self.assertAlmostEqual(d["decode_tok_s"], 100.0)
        self.assertEqual(d["draft_n"], 100)
        self.assertEqual(d["accepted_n"], 80)
        self.assertAlmostEqual(d["acceptance"], 0.8)
        self.assertAlmostEqual(d["draft_per_generated"], 0.5)
        self.assertEqual(d["accepted_per_pos"], [50, 0])

    def test_no_spec_counters(self):
        d = metrics_delta({}, {"llamacpp:tokens_predicted_total": 10.0, "llamacpp:tokens_predicted_seconds_total": 1.0})
        self.assertEqual(d["draft_n"], 0)
        self.assertIsNone(d["acceptance"])
        self.assertEqual(d["accepted_per_pos"], [])


class Timings(unittest.TestCase):
    def test_absent(self):
        self.assertIsNone(extract_timings({"choices": []}))

    def test_present(self):
        t = extract_timings({"timings": {"prompt_n": 5, "prompt_ms": 1.0, "cache_n": 100, "predicted_n": 7,
                                         "predicted_ms": 70.0, "predicted_per_second": 100.0}})
        self.assertEqual(t["cache_n"], 100)


class Text(unittest.TestCase):
    def test_messages(self):
        resp = {"content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "a"},
                            {"type": "tool_use"}, {"type": "text", "text": "b"}]}
        self.assertEqual(extract_text(resp), "ab")

    def test_chat(self):
        self.assertEqual(extract_text({"choices": [{"message": {"content": "hi"}}]}), "hi")


if __name__ == "__main__":
    unittest.main()
