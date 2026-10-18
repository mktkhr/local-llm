import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cc_metrics import session_stats, text_blocks_in_resp  # noqa: E402
from test_extract_prompts import SSE  # noqa: E402  (正常な Edit tool_use の SSE)

BROKEN = json.dumps({
    "content": [
        {"type": "text", "text": "ok <tool_call> leaked"},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
    ],
    "timings": {"cache_n": 25000, "prompt_n": 40},
}).encode()


class SessionStats(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        trace = [
            {"idx": 1, "path": "/v1/messages", "status": 200, "dur_s": 10.0},
            {"idx": 2, "path": "/v1/models?limit=1000", "status": 200, "dur_s": 0.0},
            {"idx": 3, "path": "/v1/messages", "status": 200, "dur_s": 5.5},
        ]
        (self.dir / "trace.jsonl").write_text("\n".join(json.dumps(t) for t in trace) + "\n")
        (self.dir / "001-resp.bin").write_bytes(SSE)
        # 003: SSE。tool_use の input_json が閉じない + text に生トークン露出 + message_delta に timings
        broken_sse = (
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok <tool_call> leaked"}}\n\n'
            b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t2","name":"Bash","input":{}}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"ls"}}\n\n'
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9},"timings":{"cache_n":25000,"prompt_n":40}}\n\n'
        )
        (self.dir / "003-resp.bin").write_bytes(broken_sse)

    def tearDown(self):
        self._tmp.cleanup()

    def test_counts(self):
        s = session_stats(self.dir, 1, 3)
        self.assertEqual(s["requests"], 2)
        self.assertEqual(s["tool_use_total"], 2)
        self.assertEqual(s["json_broken"], 1)
        self.assertEqual(s["raw_token_leak"], 1)
        self.assertEqual(s["cache_n_list"], [None, 25000])
        self.assertAlmostEqual(s["wall_s_sum"], 15.5)

    def test_range_filter(self):
        self.assertEqual(session_stats(self.dir, 2, 2)["requests"], 0)
        self.assertEqual(session_stats(self.dir, 3, 3)["requests"], 1)


class TextBlocks(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(text_blocks_in_resp(BROKEN), ["ok <tool_call> leaked"])

    def test_sse_text_delta(self):
        sse = (b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
               b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}\n\n'
               b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n')
        self.assertEqual(text_blocks_in_resp(sse), ["hello"])


if __name__ == "__main__":
    unittest.main()
