import math
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from locality import belady_hits, che_hit_rate, hit_curve, read_trace, stack_distances  # noqa: E402


class StackDistance(unittest.TestCase):
    def test_cold_misses_are_inf_and_reuse_counts_distinct_keys_between(self):
        # a b c a b b → a: ∞, b: ∞, c: ∞, a: 2 (b, c の間), b: 2 (c, a), b: 0
        self.assertEqual(stack_distances(["a", "b", "c", "a", "b", "b"]), [math.inf, math.inf, math.inf, 2, 2, 0])

    def test_lru_hit_curve_from_distances(self):
        d = [math.inf, math.inf, math.inf, 2, 2, 0]
        # 容量 C の LRU は距離 < C でヒット
        self.assertEqual(hit_curve(d, [1, 2, 3]), [1 / 6, 1 / 6, 3 / 6])


class Belady(unittest.TestCase):
    def test_optimal_evicts_farthest_next_use(self):
        # 容量 2: a b c a b。c を入れるとき次に使うのが遠い方(b)を捨てる → a がヒット、b はミス
        self.assertEqual(belady_hits(["a", "b", "c", "a", "b"], 2), 1)
        self.assertEqual(belady_hits(["a", "b", "c", "a", "b"], 3), 2)


class Che(unittest.TestCase):
    def test_uniform_popularity_gives_capacity_fraction(self):
        p = [0.25] * 4
        self.assertAlmostEqual(che_hit_rate(p, 2), 0.5, places=3)

    def test_skewed_popularity_beats_uniform(self):
        self.assertGreater(che_hit_rate([0.7, 0.1, 0.1, 0.1], 1), 0.25)

    def test_full_capacity_hits_everything(self):
        self.assertAlmostEqual(che_hit_rate([0.5, 0.3, 0.2], 3), 1.0, places=6)


class Trace(unittest.TestCase):
    def test_reads_header_and_records(self):
        with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
            f.write(b"MCT1" + struct.pack("<3I", 2, 2, 1))
            f.write(struct.pack("<3I", 0, 4, 2) + struct.pack("<2I", 0, 1))  # id, n_expert, n_layers, layers
            f.write(struct.pack("<2d", 1.0, 1.0) + struct.pack("<3Q", 100, 100, 256))
            f.write(struct.pack("<I", 1) + struct.pack("<Q", 50))
            f.write(struct.pack("<3i", 0, 2, 2) + struct.pack("<2i", 3, 1))
            f.write(struct.pack("<3i", 1, 2, 4) + struct.pack("<4i", 0, 2, 2, 3))
        n_layer, n_used, groups, recs = read_trace(f.name)
        self.assertEqual((n_layer, n_used), (2, 2))
        self.assertEqual(groups[0]["layers"], [0, 1])
        self.assertEqual(groups[0]["n_expert"], 4)
        self.assertEqual(groups[0]["expert_bytes"], 50)
        self.assertEqual(recs, [(0, 2, (3, 1)), (1, 2, (0, 2, 2, 3))])


if __name__ == "__main__":
    unittest.main()
