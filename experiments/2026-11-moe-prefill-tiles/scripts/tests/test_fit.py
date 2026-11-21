import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fit import lstsq  # noqa: E402


class Lstsq(unittest.TestCase):
    def test_recovers_exact_coefficients_and_r2(self):
        # y = 2 + 3 x1 - 1 x2
        X = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 2, 3]]
        y = [2, 5, 1, 5]
        coef, r2 = lstsq(X, y)
        for a, b in zip(coef, [2, 3, -1]):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(r2, 1.0)


if __name__ == "__main__":
    unittest.main()
