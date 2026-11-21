import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tiles import chunk_stats, prompt_only, tile_width, token_experts  # noqa: E402


class TileWidth(unittest.TestCase):
    def test_smallest_multiple_of_8_covering_mean_rows_capped_at_128(self):
        # 平均行数 = ceil(ub * k / E)
        self.assertEqual(tile_width(512, 8, 256), 16)
        self.assertEqual(tile_width(100, 8, 256), 8)    # ceil(3.125) = 4 → 8
        self.assertEqual(tile_width(2048, 8, 256), 64)
        self.assertEqual(tile_width(8192, 8, 256), 128)  # 256 → 上限 128


class TokenExperts(unittest.TestCase):
    def test_flattens_traced_ubatches_per_layer_in_order(self):
        recs = [(0, 2, (1, 2, 3, 4)), (1, 2, (5, 6, 7, 8)), (0, 2, (9, 10))]
        got = token_experts(recs)
        self.assertEqual(got[0], [(1, 2), (3, 4), (9, 10)])
        self.assertEqual(got[1], [(5, 6), (7, 8)])


class PromptOnly(unittest.TestCase):
    def test_drops_warmup_records_before_first_full_batch(self):
        recs = [(0, 2, (1, 2)), (0, 2, tuple(range(64))), (0, 2, (3, 4))]
        self.assertEqual(prompt_only(recs, full=32), recs[1:])


class ChunkStats(unittest.TestCase):
    def test_padded_rows_and_utilization(self):
        # 1 層、k=2、E=4、ub=4: 平均行数 ceil(4*2/4)=2 → J=8
        toks = {0: [(0, 1), (0, 1), (0, 2), (0, 3)]}
        s = chunk_stats(toks, ub=4, k=2, n_expert=4)
        # 行数 n = [4, 2, 1, 1] → タイル 1 枚ずつ、詰め物込み 8*4 = 32 行、実 8 行
        self.assertEqual(s["J"], 8)
        self.assertEqual(s["rows"], 8)
        self.assertEqual(s["padded"], 32)
        self.assertAlmostEqual(s["util"], 0.25)
        self.assertAlmostEqual(s["active_experts"], 4.0)

    def test_expert_weight_bytes_read_per_tile(self):
        # 上と同じ 4 タイル。1 expert の重みが 100 バイトなら、タイルごとに読み直して 400 バイト
        toks = {0: [(0, 1), (0, 1), (0, 2), (0, 3)]}
        s = chunk_stats(toks, ub=4, k=2, n_expert=4, expert_bytes={0: 100})
        self.assertEqual(s["tiles"], 4)
        self.assertEqual(s["bytes_read"], 400)

    def test_layers_without_full_chunks_do_not_count(self):
        toks = {0: [(0, 1)] * 4, 1: [(0, 1)]}
        self.assertEqual(chunk_stats(toks, ub=4, k=2, n_expert=4)["chunks"], 1)

    def test_drops_incomplete_last_chunk(self):
        toks = {0: [(0, 1)] * 5}
        s = chunk_stats(toks, ub=4, k=2, n_expert=4)
        self.assertEqual(s["chunks"], 1)


if __name__ == "__main__":
    unittest.main()
