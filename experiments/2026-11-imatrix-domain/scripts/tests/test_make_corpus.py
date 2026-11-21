import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from make_corpus import latest_per_session, render_messages, render_preamble, session_key, shuffled_pieces, split  # noqa: E402

BODY = {
    "system": [{"type": "text", "text": "You are Claude Code."}],
    "tools": [{"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "fix the bug"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"},
                                          {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "print(1)"}]},
        {"role": "assistant", "content": "done"},
    ],
}


class Render(unittest.TestCase):
    def test_messages_include_text_thinking_tool_calls_and_results_in_order(self):
        txt = render_messages(BODY)
        for a, b in (("fix the bug", "hmm"), ("hmm", '"path": "a.py"'), ('"path": "a.py"', "print(1)"), ("print(1)", "done")):
            self.assertLess(txt.index(a), txt.index(b))
        self.assertNotIn("You are Claude Code.", txt)

    def test_preamble_has_system_and_tool_definitions(self):
        pre = render_preamble(BODY)
        self.assertIn("You are Claude Code.", pre)
        self.assertIn("read a file", pre)


class Sessions(unittest.TestCase):
    def test_key_is_first_user_text(self):
        self.assertEqual(session_key(BODY), session_key({"messages": BODY["messages"][:1]}))
        self.assertNotEqual(session_key(BODY), session_key({"messages": [{"role": "user", "content": "other"}]}))

    def test_latest_per_session_keeps_longest_request(self):
        short = {"messages": BODY["messages"][:2]}
        got = latest_per_session([short, BODY, {"messages": [{"role": "user", "content": "other"}]}])
        self.assertEqual(len(got), 2)
        self.assertIn(BODY, got)

    def test_split_is_deterministic_and_disjoint(self):
        bodies = [{"messages": [{"role": "user", "content": f"task {i}"}]} for i in range(30)]
        cal, ev = split(bodies, eval_mod=3)
        self.assertEqual(len(cal) + len(ev), 30)
        self.assertTrue(ev)
        self.assertEqual(split(bodies, eval_mod=3), (cal, ev))
        self.assertFalse({session_key(b) for b in cal} & {session_key(b) for b in ev})


class Shuffle(unittest.TestCase):
    TEXT = "\n".join(f"line {i} " + "x" * (i % 7) for i in range(200))

    def test_pieces_keep_all_lines_and_respect_size(self):
        ps = shuffled_pieces(self.TEXT, size=100, seed=1)
        self.assertEqual(sorted("\n".join(ps).split("\n")), sorted(self.TEXT.split("\n")))
        self.assertTrue(all(len(p) <= 100 for p in ps))

    def test_deterministic_and_actually_shuffled(self):
        a = shuffled_pieces(self.TEXT, size=100, seed=1)
        self.assertEqual(a, shuffled_pieces(self.TEXT, size=100, seed=1))
        self.assertNotEqual(a, shuffled_pieces(self.TEXT, size=100, seed=None))


if __name__ == "__main__":
    unittest.main()
