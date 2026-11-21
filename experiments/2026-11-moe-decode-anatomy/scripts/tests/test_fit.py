import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fit import linfit, two_point_bw  # noqa: E402


class Fit(unittest.TestCase):
    def test_linfit_exact_line(self):
        a, b, r2 = linfit([1, 2, 4, 8], [5.5, 6.0, 7.0, 9.0])
        self.assertAlmostEqual(a, 5.0)
        self.assertAlmostEqual(b, 0.5)
        self.assertAlmostEqual(r2, 1.0)

    def test_two_point_bw_recovers_bandwidth_and_fixed_cost(self):
        # BW 500 GB/s、固定費 1 ms: 8 GB → 17 ms、4 GB → 9 ms
        bw, fix = two_point_bw(8e9, 17.0, 4e9, 9.0)
        self.assertAlmostEqual(bw, 500.0)
        self.assertAlmostEqual(fix, 1.0)
