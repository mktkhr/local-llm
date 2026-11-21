import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from parse_kld import parse  # noqa: E402

LOG = """\
Mean PPL(Q)                   :   5.123456 ±   0.012345
Mean ln(PPL(Q)/PPL(base))     :   0.031234 ±   0.000456
Mean    KLD:   0.045678 ±   0.000321
99.0%   KLD:   0.512345
Median  KLD:   0.012345
RMS Δp    :  5.432 ± 0.012 %
Same top p: 91.234 ± 0.056 %
"""


class Parse(unittest.TestCase):
    def test_extracts_summary_values(self):
        d = parse(LOG)
        self.assertAlmostEqual(d["ppl"], 5.123456)
        self.assertAlmostEqual(d["ln_ppl_ratio"], 0.031234)
        self.assertAlmostEqual(d["kld"], 0.045678)
        self.assertAlmostEqual(d["kld_unc"], 0.000321)
        self.assertAlmostEqual(d["kld_p99"], 0.512345)
        self.assertAlmostEqual(d["kld_median"], 0.012345)
        self.assertAlmostEqual(d["rms_dp"], 5.432)
        self.assertAlmostEqual(d["same_top"], 91.234)

    def test_missing_fields_are_none(self):
        self.assertIsNone(parse("nothing here")["kld"])


if __name__ == "__main__":
    unittest.main()
