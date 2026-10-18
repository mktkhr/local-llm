import http.server
import json
import pathlib
import socket
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from spec_bench import build_body, main  # noqa: E402


class _Fake(http.server.BaseHTTPRequestHandler):
    """/metrics は呼ばれるたびに predicted が 20 増える。POST は chat 形式の応答を返す。"""

    calls = 0

    def log_message(self, *a):
        pass

    fail_metrics_once = False

    def do_GET(self):
        if self.path == "/props":
            body = json.dumps({"build_info": "b10920-eafe15a5e"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics" and _Fake.fail_metrics_once:
            _Fake.fail_metrics_once = False
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/metrics":
            _Fake.calls += 1
            n = 100 + 20 * _Fake.calls
            body = (f"llamacpp:tokens_predicted_total {n}\n"
                    f"llamacpp:tokens_predicted_seconds_total {n / 50}\n"
                    f"llamacpp:prompt_tokens_cached_total 100\n").encode()
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n))
        assert req["max_tokens"] == 64 and req["temperature"] == 0
        body = json.dumps({"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                           "timings": {"prompt_n": 5, "prompt_ms": 1.0, "cache_n": 100, "predicted_n": 20,
                                       "predicted_ms": 200.0, "predicted_per_second": 100.0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CliRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", cls.port), _Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _run(self, url: str, out: pathlib.Path, prompt: pathlib.Path) -> int:
        return main(["--url", url, "--endpoint", "chat", "--prompt-file", str(prompt), "--n-predict", "64",
                     "--out", str(out), "--config", "C", "--spec", "mtp2", "--workload", "W3",
                     "--prompt-id", "W3-01", "--block", "1", "--nvidia-smi", "0"])

    def test_writes_one_jsonl_row(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = pathlib.Path(d, "p.json")
            prompt.write_text(json.dumps({"id": "W3-01", "messages": [{"role": "user", "content": "hi"}]}))
            out = pathlib.Path(d, "out.jsonl")
            _Fake.calls = 0
            self.assertEqual(self._run(f"http://127.0.0.1:{self.port}", out, prompt), 0)
            row = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(row["status"], 200)
            self.assertEqual(row["metrics"]["predicted_n"], 20)
            self.assertEqual(row["text_len"], 5)
            self.assertEqual(row["timings"]["cache_n"], 100)
            self.assertEqual(row["workload"], "W3")
            self.assertEqual(row["config"], "C")
            self.assertIsNone(row["gpu"]["vram_peak_mib"])
            self.assertEqual(row["build"], "b10920-eafe15a5e")
            self.assertEqual(row["stop_reason"], "stop")
            self.assertEqual(row["block_types"], ["text"])
            self.assertTrue(row["payload_sha256"])

    def test_metrics_fetch_failure_blanks_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = pathlib.Path(d, "p.json")
            prompt.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
            out = pathlib.Path(d, "out.jsonl")
            _Fake.fail_metrics_once = True
            self.assertNotEqual(self._run(f"http://127.0.0.1:{self.port}", out, prompt), 0)
            row = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(row["status"], 200)
            self.assertIsNone(row["metrics"])
            self.assertIn("metrics", row["error"])

    def test_connection_refused_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = pathlib.Path(d, "p.json")
            prompt.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
            out = pathlib.Path(d, "out.jsonl")
            self.assertNotEqual(self._run(f"http://127.0.0.1:{_free_port()}", out, prompt), 0)
            row = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(row["status"], 0)
            self.assertTrue(row["error"])


class BuildBody(unittest.TestCase):
    def test_messages(self):
        b = build_body("messages", {"model": "x", "max_tokens": 8192, "stream": True, "messages": [],
                                    "system": "s", "thinking": {"type": "enabled", "budget_tokens": 1000}}, 256, False)
        self.assertEqual(b["max_tokens"], 256)
        self.assertFalse(b["stream"])
        self.assertEqual(b["temperature"], 0)
        self.assertEqual(b["thinking"], {"type": "disabled"})
        self.assertEqual(b["system"], "s")
        self.assertFalse(b["chat_template_kwargs"]["enable_thinking"])

    def test_chat_think(self):
        b = build_body("chat", {"messages": [{"role": "user", "content": "q"}]}, 300, True)
        self.assertEqual(b["max_tokens"], 300)
        self.assertTrue(b["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(b["timings_per_token"])


if __name__ == "__main__":
    unittest.main()
