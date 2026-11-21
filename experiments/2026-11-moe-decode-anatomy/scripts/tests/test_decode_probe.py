import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decode_probe import summarize  # noqa: E402


class Summarize(unittest.TestCase):
    def test_drops_warmup_and_takes_median(self):
        s = summarize([20.0, 6.0, 6.5, 6.2])
        self.assertEqual(s["n_reps"], 3)
        self.assertEqual(s["ms_median"], 6.2)
        self.assertEqual(s["ms_min"], 6.0)
        self.assertAlmostEqual(s["tok_s_median"], 161.29, places=2)
