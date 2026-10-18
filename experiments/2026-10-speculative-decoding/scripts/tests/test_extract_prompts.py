import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from extract_prompts import build_w3, classify, select, tool_uses_in_resp  # noqa: E402

SSE = (
    b'event: content_block_start\n'
    b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"Edit","input":{}}}\n\n'
    b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\":\\"a.go\\",\\"new_string\\":\\""}}\n\n'
    b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"' + b"x" * 1600 + b'\\"}"}}\n\n'
    b'event: content_block_stop\n'
    b'data: {"type":"content_block_stop","index":1}\n\n'
)


class ToolUses(unittest.TestCase):
    def test_sse_concatenates_partial_json(self):
        tus = tool_uses_in_resp(SSE)
        self.assertEqual(len(tus), 1)
        self.assertEqual(tus[0]["name"], "Edit")
        self.assertGreater(len(tus[0]["input_json"]), 1500)
        json.loads(tus[0]["input_json"])

    def test_plain_json(self):
        raw = json.dumps({"content": [{"type": "text", "text": "x"},
                                      {"type": "tool_use", "name": "Read", "input": {"file_path": "x"}}]}).encode()
        tus = tool_uses_in_resp(raw)
        self.assertEqual(tus[0]["name"], "Read")
        self.assertEqual(json.loads(tus[0]["input_json"]), {"file_path": "x"})

    def test_garbage_is_empty(self):
        self.assertEqual(tool_uses_in_resp(b"not json at all"), [])


class Classify(unittest.TestCase):
    def test_w2_requires_tools_and_big_edit(self):
        self.assertEqual(classify({"tools": [1]}, [{"name": "Edit", "input_json": "x" * 1600}]), "W2")
        self.assertEqual(classify({"tools": [1]}, [{"name": "Write", "input_json": "x" * 1600}]), "W2")
        self.assertEqual(classify({"tools": [1]}, [{"name": "Edit", "input_json": "x" * 100}]), "W1")
        self.assertEqual(classify({"tools": [1]}, [{"name": "Read", "input_json": "{}"}]), "W1")
        self.assertIsNone(classify({}, [{"name": "Read", "input_json": "{}"}]))
        self.assertIsNone(classify({"tools": [1]}, []))


class Select(unittest.TestCase):
    def test_picks_percentiles(self):
        c = {"W1": [{"bytes": b, "req": {}} for b in range(100, 1100, 100)]}
        picked = select(c, k=3)
        self.assertEqual([p["bytes"] for p in picked["W1"]], [200, 600, 1000])

    def test_fewer_candidates_than_k(self):
        picked = select({"W2": [{"bytes": 5, "req": {}}]}, k=3)
        self.assertEqual(len(picked["W2"]), 1)


class W3(unittest.TestCase):
    def test_three_prompts(self):
        ps = build_w3([{"transcript": "aaa"}, {"transcript": "bbb"}])
        self.assertEqual(len(ps), 3)
        self.assertEqual([p["id"] for p in ps], ["W3-01", "W3-02", "W3-03"])
        self.assertIn("aaa", ps[0]["messages"][0]["content"])
        self.assertIn("aaa", ps[2]["messages"][0]["content"])
        self.assertIn("bbb", ps[2]["messages"][0]["content"])
        self.assertEqual(ps[0]["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
