import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hazard import Ecdf, analyze, bucket, delta_by_margin, deltas, first_divergence, margins, reference_deltas, top_pairs  # noqa: E402


def tok(tid, top):
    return {"id": tid, "lp": top[0][1], "top": top}


class Parts(unittest.TestCase):
    def test_margins_and_divergence(self):
        a = [tok(1, [[1, -0.1], [2, -2.0]]), tok(3, [[3, -0.5], [4, -0.6]])]
        b = [tok(1, [[1, -0.1], [2, -2.0]]), tok(4, [[4, -0.55], [3, -0.56]])]
        self.assertAlmostEqual(margins(a)[1], 0.1)
        self.assertEqual(first_divergence(a, b), 1)
        self.assertIsNone(first_divergence(a, a))

    def test_delta_is_change_of_none_top1_minus_top2_gap(self):
        n = [tok(3, [[3, -0.5], [4, -0.6]])]
        s = [tok(4, [[4, -0.55], [3, -0.56]])]
        # none の gap 0.1、spec での同じ 2 トークンの gap −0.01 → δ = −0.11
        self.assertAlmostEqual(deltas(n, s, 0)[0], -0.11)

    def test_ecdf_and_bucket(self):
        F = Ecdf([-0.2, -0.1, 0.1, 0.2])
        self.assertEqual(F.below(-0.15), 0.25)
        self.assertEqual(bucket(0.0), 0)
        self.assertEqual(bucket(0.05), 2)
        self.assertEqual(bucket(5.0), 5)

    def test_analyze_counts_divergence_at_small_margin(self):
        none = [tok(1, [[1, -0.1], [2, -3.0]]), tok(1, [[1, -0.1], [2, -3.0]]), tok(3, [[3, -0.50], [4, -0.52]])]
        spec = [tok(1, [[1, -0.1], [2, -3.0]]), tok(1, [[1, -0.11], [2, -3.0]]), tok(4, [[4, -0.50], [3, -0.53]])]
        rows = [{"build": "x", "spec": "none", "prompt_id": "p", "kind": "edit", "tokens": none, "drafts": 0, "accepted": 0},
                {"build": "x", "spec": "ng", "prompt_id": "p", "kind": "edit", "tokens": spec, "drafts": 3, "accepted": 1}]
        r = analyze(rows, "x", "ng")
        self.assertEqual(r["observed_divergences"], 1)
        self.assertEqual(r["runs"][0]["D"], 2)
        self.assertEqual(r["runs"][0]["p0"], 1)
        self.assertEqual(r["margin_at_D"], [0.02])


def _rows():
    none = [tok(1, [[1, -0.1], [2, -3.0]]), tok(1, [[1, -0.1], [2, -3.0]]), tok(3, [[3, -0.50], [4, -0.52]])]
    spec = [tok(1, [[1, -0.1], [2, -3.0]]), tok(1, [[1, -0.11], [2, -3.0]]), tok(4, [[4, -0.50], [3, -0.53]])]
    return [{"build": "x", "spec": "none", "prompt_id": "p", "kind": "edit", "tokens": none, "drafts": 0, "accepted": 0},
            {"build": "x", "spec": "ng", "prompt_id": "p", "kind": "edit", "tokens": spec, "drafts": 3, "accepted": 1}]


class Scaled(unittest.TestCase):
    def test_top_pairs_are_none_top1_top2(self):
        self.assertEqual(top_pairs(_rows()[0]["tokens"]), [(1, 2), (1, 2), (3, 4)])

    def test_missing_scale_equals_unscaled(self):
        self.assertEqual(analyze(_rows(), "x", "ng", scale={})["expected_divergences"],
                         analyze(_rows(), "x", "ng")["expected_divergences"])

    def test_pair_norm_rescales_delta_and_margin(self):
        # δ = [-0.01 (1,2), -0.05 (3,4)]。素の ECDF では位置 2 の h = P(δ < -0.02) = 1/2。
        # (3,4) のノルム 10 で z = [-0.01, -0.005]、閾値 -0.002 → h = 1
        self.assertEqual(analyze(_rows(), "x", "ng")["expected_divergences"], 0.5)
        self.assertEqual(analyze(_rows(), "x", "ng", scale={(3, 4): 10.0})["expected_divergences"], 1.0)


class Reference(unittest.TestCase):
    def test_reference_keeps_only_low_margin_positions_across_specs(self):
        rows = _rows() + [dict(_rows()[1], spec="ng2")]
        # 位置 1 は余裕 2.9(δ=-0.01)、位置 2 は余裕 0.02(δ=-0.05)。余裕 1 未満だけ残す
        got = reference_deltas(rows, "x", max_margin=1.0)
        self.assertEqual(len(got), 2)
        self.assertTrue(all(abs(d + 0.05) < 1e-9 for d in got))

    def test_analyze_uses_given_reference(self):
        # 参照分布が [-0.05] だけなら位置 2 の h = P(δ < -0.02) = 1、位置 1 は 0
        r = analyze(_rows(), "x", "ng", ref=[-0.05])
        self.assertEqual(r["expected_divergences"], 1.0)
        self.assertEqual(r["n_ref"], 1)


class ByMargin(unittest.TestCase):
    def test_bins_abs_delta_by_none_margin(self):
        got = {b["margin"]: b for b in delta_by_margin(_rows(), "x", edges=[0, 1, 4])}
        self.assertEqual(got["[0, 1)"]["n"], 1)
        self.assertAlmostEqual(got["[0, 1)"]["abs_p50"], 0.05)
        self.assertEqual(got["[1, 4)"]["n"], 1)
        self.assertAlmostEqual(got["[1, 4)"]["abs_p50"], 0.01)


if __name__ == "__main__":
    unittest.main()
