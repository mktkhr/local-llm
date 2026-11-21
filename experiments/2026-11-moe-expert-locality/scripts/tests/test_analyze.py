import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analyze import bw_fixed_t0, fit_time, temporal_overlap  # noqa: E402


class Overlap(unittest.TestCase):
    def test_consecutive_token_overlap_vs_independent_expectation(self):
        # 1 層、k=2、4 トークン。連続トークン間で共有する expert の数: {0,1}&{0,2}=1, {0,2}&{0,2}=2, {0,2}&{1,3}=0
        recs = [(0, 2, (0, 1)), (0, 2, (0, 2)), (0, 2, (0, 2)), (0, 2, (1, 3))]
        got = temporal_overlap(recs, layers=[0], n_expert=4)
        self.assertAlmostEqual(got["observed"], 1.0)
        # 選択確率 q = [3/4, 2/4, 2/4, 1/4](expert 0..3)。独立なら期待値 Σ q^2 = 0.5625+0.25+0.25+0.0625
        self.assertAlmostEqual(got["independent"], 1.125)

    def test_per_layer_breakdown(self):
        recs = [(0, 2, (0, 1)), (1, 2, (2, 3)), (0, 2, (0, 1)), (1, 2, (0, 1))]
        got = temporal_overlap(recs, layers=[0, 1], n_expert=4)
        self.assertEqual(got["per_layer"][0]["observed"], 2.0)
        self.assertEqual(got["per_layer"][1]["observed"], 0.0)

    def test_prefill_batches_are_ignored(self):
        big = (0, 2, tuple(range(2)) * 40)  # 40 トークンのバッチ
        got = temporal_overlap([(0, 2, (0, 1)), big, (0, 2, (0, 1))], layers=[0], n_expert=4)
        self.assertAlmostEqual(got["observed"], 2.0)


class FitTime(unittest.TestCase):
    def test_recovers_base_and_bandwidth(self):
        # t = 6 ms + bytes / (10 GB/s)。100 MB/トークン → 10 ms
        pts = [(0.0, 6.0), (100e6, 16.0), (200e6, 26.0)]
        base, bw, r2 = fit_time(pts)
        self.assertAlmostEqual(base, 6.0)
        self.assertAlmostEqual(bw / 1e9, 10.0)
        self.assertAlmostEqual(r2, 1.0)


class FixedT0(unittest.TestCase):
    def test_bandwidth_with_known_base(self):
        # t0 = 6 ms、100 MB で 11 ms → 5 ms で 100 MB = 20 GB/s
        self.assertAlmostEqual(bw_fixed_t0(100e6, 11.0, 6.0) / 1e9, 20.0)


if __name__ == "__main__":
    unittest.main()
