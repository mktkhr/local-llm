import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from imatrix_cos import cosine, kind_of, summarize  # noqa: E402


class Cos(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine([1, 1], [2, 2]), 1.0)

    def test_kind_strips_layer_and_suffix(self):
        self.assertEqual(kind_of("blk.12.ffn_up.weight"), "ffn_up")
        self.assertEqual(kind_of("output.weight"), "output")

    def test_summary_groups_by_kind(self):
        rows = [{"name": "blk.0.ffn_up.weight", "cos": 0.9}, {"name": "blk.1.ffn_up.weight", "cos": 0.7},
                {"name": "blk.0.attn_qkv.weight", "cos": 0.99}]
        s = {r["kind"]: r for r in summarize(rows)}
        self.assertAlmostEqual(s["ffn_up"]["median"], 0.8)
        self.assertAlmostEqual(s["ffn_up"]["min"], 0.7)
        self.assertEqual(s["attn_qkv"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
