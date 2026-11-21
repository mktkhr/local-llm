import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from decode_run import row_from  # noqa: E402


class RowFrom(unittest.TestCase):
    def test_keeps_timings_and_tokens(self):
        resp = {"tokens": [5, 6, 7], "timings": {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 3,
                                                  "predicted_ms": 30.0, "predicted_per_token_ms": 10.0}}
        r = row_from(resp, {"cache_mib": "4096"}, "E01")
        self.assertEqual(r["prompt_id"], "E01")
        self.assertEqual(r["cache_mib"], "4096")
        self.assertEqual(r["tokens"], [5, 6, 7])
        self.assertEqual(r["predicted_n"], 3)
        self.assertAlmostEqual(r["ms_per_tok"], 10.0)
        self.assertAlmostEqual(r["prompt_ms"], 50.0)


if __name__ == "__main__":
    unittest.main()
