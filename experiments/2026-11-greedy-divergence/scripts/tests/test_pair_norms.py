import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pair_norms import collect_pairs, pair_norms  # noqa: E402


def tok(tid, top):
    return {"id": tid, "lp": top[0][1], "top": top}


class PairNorms(unittest.TestCase):
    def test_collect_pairs_only_from_none_runs_of_build(self):
        rows = [
            {"build": "a", "spec": "none", "tokens": [tok(1, [[1, -0.1], [2, -1.0]])]},
            {"build": "a", "spec": "ng", "tokens": [tok(5, [[5, -0.1], [6, -1.0]])]},
            {"build": "b", "spec": "none", "tokens": [tok(7, [[7, -0.1], [8, -1.0]])]},
        ]
        self.assertEqual(collect_pairs(rows, "a"), {(1, 2)})

    def test_pair_norm_is_euclidean_distance_of_rows(self):
        rows = {1: [0.0, 0.0], 2: [3.0, 4.0], 3: [1.0, 0.0]}
        got = pair_norms({(1, 2), (1, 3)}, lambda ids: [rows[i] for i in ids])
        self.assertAlmostEqual(got[(1, 2)], 5.0)
        self.assertAlmostEqual(got[(1, 3)], 1.0)


if __name__ == "__main__":
    unittest.main()
