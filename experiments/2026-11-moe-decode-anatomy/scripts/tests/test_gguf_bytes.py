import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gguf_bytes import bytes_per_token, classify, summarize  # noqa: E402


class Bytes(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("blk.0.ffn_up_exps.weight"), "expert")
        self.assertEqual(classify("blk.0.ffn_up_shexp.weight"), "dense")
        self.assertEqual(classify("blk.0.attn_q.weight"), "dense")
        self.assertEqual(classify("token_embd.weight"), "embd")
        self.assertEqual(classify("output.weight"), "dense")

    def test_bytes_per_token_counts_one_embedding_row_and_k_experts(self):
        tensors = [("token_embd.weight", 1000, [10, 100]),      # 100 行 → 1 行 10 B
                   ("blk.0.ffn_up_exps.weight", 800, [4, 4, 8]),  # E=8 → 1 本 100 B
                   ("blk.0.attn_q.weight", 50, [5, 10]),
                   ("output.weight", 200, [10, 20])]
        s = summarize(tensors, n_expert=8)
        self.assertEqual(s["dense_bytes"], 250)
        self.assertEqual(s["embd_row_bytes"], 10)
        self.assertEqual(s["expert_bytes_per_k"], 100)
        self.assertEqual(bytes_per_token(s, 0), 260)
        self.assertEqual(bytes_per_token(s, 4), 660)


if __name__ == "__main__":
    unittest.main()
