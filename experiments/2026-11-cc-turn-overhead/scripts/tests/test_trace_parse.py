import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trace_parse import phases  # noqa: E402

LOG = """\
0.08.187.296 I slot       select: id  0 | task -1 |  - checking sim = 1.000 (10788/10788) > 0.100
0.08.187.300 I slot       select: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold), f_keep = 1.000
0.08.187.381 I slot launch_slot_: id  0 | task 10 | processing task, is_child = 0
0.08.193.344 I slot   operator(): id  0 | task 10 | restored context checkpoint (pos_min = 10783, pos_max = 10783, n_tokens = 10784, n_past = 10784, size = 62.813 MiB)
0.08.194.681 I slot init_sampler: id  0 | task 10 | init sampler, took 1.31 ms, tokens: text = 10788, total = 10788
0.08.194.684 I slot create_check: id  0 | task 10 | superseding context checkpoint at n_tokens = 10784
0.08.233.104 I slot create_check: id  0 | task 10 | created context checkpoint 4 of 32 (pos_min = 10783, pos_max = 10783, n_tokens = 10784, size = 62.813 MiB)
0.08.244.727 I slot print_timing: id  0 | task 10 | prompt eval time =      57.33 ms /     4 tokens (   14.33 ms per token,    69.78 tokens per second)
0.08.245.325 I slot      release: id  0 | task 10 | stop processing: n_tokens = 10788, truncated = 0
"""


class TestPhases(unittest.TestCase):
    def test_warm_resend_phases_in_ms(self):
        (p,) = phases(LOG.splitlines())
        self.assertEqual(p["task"], 10)
        self.assertEqual(p["n_tokens"], 10788)
        self.assertAlmostEqual(p["select_to_launch"], 0.081, places=3)
        self.assertAlmostEqual(p["restore"], 5.963, places=3)
        self.assertAlmostEqual(p["sampler_init"], 1.337, places=3)
        self.assertAlmostEqual(p["checkpoint_create"], 38.423, places=3)
        self.assertAlmostEqual(p["eval_and_sample"], 11.623, places=3)
        self.assertAlmostEqual(p["finish"], 0.598, places=3)

    def test_requests_without_restore_are_skipped(self):
        cold = [l for l in LOG.splitlines() if "restored" not in l]
        self.assertEqual(phases(cold), [])


if __name__ == "__main__":
    unittest.main()
