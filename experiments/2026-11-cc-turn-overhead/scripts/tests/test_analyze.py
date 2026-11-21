import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analyze import decompose, linfit  # noqa: E402


def rows(kind, walls, **extra):
    return [{"build": "b", "axis": "tools", "n_tools": 8, "n_msgs": 3, "chars_tools": 1000,
             "chars_msgs": 10, "kind": kind, "rep": i, "status": 200, "wall_s": w, **extra}
            for i, w in enumerate(walls)]


class TestDecompose(unittest.TestCase):
    def setUp(self):
        rs = (rows("health", [0.001, 0.001, 0.003])
              + rows("apply", [0.051, 0.041, 0.041], prompt_chars=5000)
              + rows("tokenize", [0.011, 0.011, 0.020], n_tokens=1200)
              + rows("count", [0.061, 0.061, 0.070], input_tokens=1200)
              + rows("msg_warm", [0.300, 0.200, 0.200], prompt_s=0.05, predicted_s=0.01, prompt_n=1)
              + rows("dec_auto", [1.0], predicted_s=0.8, predicted_n=100))
        self.g = decompose(rs)[0]

    def test_components_are_medians_minus_http_baseline(self):
        self.assertAlmostEqual(self.g["apply_ms"], 40.0)
        self.assertAlmostEqual(self.g["tokenize_ms"], 10.0)
        self.assertAlmostEqual(self.g["count_ms"], 60.0)
        self.assertAlmostEqual(self.g["conv_ms"], 10.0)

    def test_residual_removes_preprocess_prefill_and_decode(self):
        # 200 - 1(base) - 60(count) - 50(prompt) - 10(decode)
        self.assertAlmostEqual(self.g["warm_ms"], 200.0)
        self.assertAlmostEqual(self.g["resid_ms"], 79.0)

    def test_token_count_agreement_flag(self):
        self.assertTrue(self.g["tokens_match"])
        self.assertEqual(self.g["n_tokens"], 1200)

    def test_decode_ms_per_token(self):
        self.assertAlmostEqual(self.g["dec_auto_ms"], 8.0)
        self.assertIsNone(self.g["dec_any_ms"])

    def test_failed_rows_ignored(self):
        rs = rows("health", [0.001]) + rows("apply", [9.9], prompt_chars=1)
        rs[-1]["status"] = 500
        self.assertIsNone(decompose(rs)[0]["apply_ms"])


class TestGrammarSplit(unittest.TestCase):
    def setUp(self):
        rs = (rows("health", [0.001])
              + rows("apply", [0.051], prompt_chars=5000)
              + rows("apply_none", [0.021], prompt_chars=5000)
              + rows("tokenize", [0.011], n_tokens=1200)
              + rows("chat_warm", [0.200], prompt_s=0.05, predicted_s=0.01)
              + rows("chat_none_warm", [0.150], prompt_s=0.05, predicted_s=0.01)
              + rows("dec_chat_auto", [1.0], predicted_s=1.2, predicted_n=100)
              + rows("dec_chat_none", [1.0], predicted_s=1.0, predicted_n=100))
        self.g = decompose(rs)[0]

    def test_grammar_string_cost_is_apply_minus_apply_none(self):
        self.assertAlmostEqual(self.g["grammar_build_ms"], 30.0)
        self.assertTrue(self.g["same_prompt_none"])

    def test_grammar_init_cost_is_residual_difference(self):
        # chat: 200-1-50-10-50-10 = 79、none: 150-1-20-10-50-10 = 59 → 20
        self.assertAlmostEqual(self.g["chat_resid_ms"], 79.0)
        self.assertAlmostEqual(self.g["chat_none_resid_ms"], 59.0)
        self.assertAlmostEqual(self.g["grammar_init_ms"], 20.0)

    def test_grammar_decode_tax_per_token(self):
        self.assertAlmostEqual(self.g["grammar_tok_ms"], 2.0)


class TestLinfit(unittest.TestCase):
    def test_exact_line(self):
        a, b, r2 = linfit([0, 1, 2], [1, 3, 5])
        self.assertAlmostEqual(a, 1.0)
        self.assertAlmostEqual(b, 2.0)
        self.assertAlmostEqual(r2, 1.0)


if __name__ == "__main__":
    unittest.main()
