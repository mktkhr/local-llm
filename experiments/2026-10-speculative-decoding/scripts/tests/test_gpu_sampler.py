import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from spec_bench import GpuSampler  # noqa: E402


class Sampler(unittest.TestCase):
    def test_summary_includes_clock(self):
        samples = iter(["15000, 200.5, 70, 2700", "15100, 210.5, 72, 2500"])
        with GpuSampler(interval_s=0.01, query=lambda: next(samples, None)) as g:
            time.sleep(0.1)
        s = g.summary()
        self.assertEqual(s["vram_peak_mib"], 15100)
        self.assertAlmostEqual(s["power_mean_w"], 205.5)
        self.assertEqual(s["power_max_w"], 210.5)
        self.assertEqual(s["temp_max_c"], 72)
        self.assertEqual(s["clock_sm_mean_mhz"], 2600)
        self.assertEqual(s["samples"], 2)

    def test_no_samples(self):
        with GpuSampler(enabled=False) as g:
            pass
        self.assertIsNone(g.summary()["clock_sm_mean_mhz"])


if __name__ == "__main__":
    unittest.main()
