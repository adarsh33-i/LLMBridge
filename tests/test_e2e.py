"""End-to-end tests: a real LLMBridge server subprocess against a mock upstream.

Unlike the other suites, which patch ``urlopen`` in-process, these tests run
``llmbridge.server`` as a real process speaking real HTTP over localhost, with
a scripted OpenAI-compatible upstream. They exercise the full interrupt ->
inspect -> resume/fork flow plus the ledger's keep-last-10 cap.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent


class _MockUpstream(BaseHTTPRequestHandler):
    """Scriptable OpenAI-compatible upstream shared with the bridge process."""

    mode = "fail"  # "fail" | "ok" | "stream_partial"

    def log_message(self, *args):  # silence request logging
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if _MockUpstream.mode == "fail":
            self.send_response(429)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if _MockUpstream.mode == "stream_partial" and body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n')
            self.wfile.flush()
            # A malformed event aborts the bridge's stream mid-flight, which
            # is what records partial output and marks the generation
            # interrupted.
            self.wfile.write(b"data: {broken json\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        payload = {
            "id": "mock-1",
            "choices": [{"message": {"content": "mock says ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock = ThreadingHTTPServer(("127.0.0.1", 0), _MockUpstream)
        threading.Thread(target=cls.mock.serve_forever, daemon=True).start()
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.ledger_dir = Path(cls._tmp.name) / "generations"
        config = Path(cls._tmp.name) / "llmbridge.toml"
        config.write_text(
            f'[models.mock]\n'
            f'base_url = "http://127.0.0.1:{cls.mock.server_address[1]}/v1"\n'
            f'model = "mock-model"\n'
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {
            **os.environ,
            "LLMBRIDGE_CONFIG": str(config),
            "LLMBRIDGE_HOST": "127.0.0.1",
            "LLMBRIDGE_PORT": str(port),
            "LLMBRIDGE_LEDGER_PATH": str(cls.ledger_dir),
            "LLMBRIDGE_TRACK_GENERATIONS": "1",
            "LLMBRIDGE_TRACK_INTERRUPTED": "1",
        }
        cls.server = subprocess.Popen(
            [sys.executable, "-c", "from llmbridge.server import main; main()"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cls.base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                cls.req("GET", "/v1/models")
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("LLMBridge server did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - cleanup only
            cls.server.kill()
        cls.mock.shutdown()

    @classmethod
    def req(cls, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            cls.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read()), dict(response.headers)
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw": raw.decode(errors="replace")}
            return error.code, parsed, dict(error.headers)

    def _record_count(self):
        return len(list(self.ledger_dir.glob("gen_*.json")))

    def test_end_to_end_flow(self):
        # 1. An upstream 429 saves the request as interrupted work in the ledger.
        _MockUpstream.mode = "fail"
        status, body, _ = self.req(
            "POST",
            "/v1/messages",
            {"model": "mock", "max_tokens": 50, "messages": [{"role": "user", "content": "Write a poem"}]},
        )
        self.assertEqual(status, 502)
        self.assertIn("gen_", body["error"]["message"])
        status, state, _ = self.req("GET", "/v1/interrupted")
        self.assertEqual(status, 200)
        self.assertEqual(state["prompt"], "Write a poem")
        self.assertEqual(state["model_name"], "mock")
        self.assertEqual(state["error"], "Upstream returned HTTP 429")
        self.assertEqual(self._record_count(), 1)

        # 2. The same prompt succeeding clears the interruption; its own
        # record is completed and the stale one is gone.
        _MockUpstream.mode = "ok"
        status, _, headers = self.req(
            "POST",
            "/v1/messages",
            {"model": "mock", "max_tokens": 50, "messages": [{"role": "user", "content": "Write a poem"}]},
        )
        self.assertEqual(status, 200)
        self.assertTrue(headers.get("X-LLMBridge-Generation", "").startswith("gen_"))
        self.assertEqual(self.req("GET", "/v1/interrupted")[0], 404)
        completed = self.req("GET", "/v1/generations/" + headers["X-LLMBridge-Generation"])[1]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.req("GET", "/v1/generations/" + state["generation_id"])[0], 404)

        # 3. Fourteen creates later the ledger holds only the newest ten.
        for index in range(12):
            status, _, _ = self.req(
                "POST",
                "/v1/messages",
                {"model": "mock", "max_tokens": 50, "messages": [{"role": "user", "content": f"cap test {index}"}]},
            )
            self.assertEqual(status, 200)
        self.assertEqual(self._record_count(), 10)
        listed = self.req("GET", "/v1/generations")[1]["data"]
        self.assertEqual(len(listed), 10)
        self.assertEqual(listed[0]["prompt_preview"], "cap test 11")

        # 4. A mid-stream upstream failure records partial output and progress.
        _MockUpstream.mode = "stream_partial"
        request = urllib.request.Request(
            self.base + "/v1/messages",
            data=json.dumps(
                {
                    "model": "mock",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Stream story"}],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                stream_body = response.read().decode()
        except urllib.error.HTTPError as error:
            stream_body = error.read().decode()
        self.assertIn("event: error", stream_body)
        status, state, _ = self.req("GET", "/v1/interrupted")
        self.assertEqual(status, 200)
        self.assertEqual(state["partial_output"], "Hel")
        self.assertGreater(state["progress_percent"], 0)
        self.assertIn("streaming", state["error"] or "")

        # 5. Forking the interrupted generation continues it to completion and
        # clears the interruption it grew out of.
        _MockUpstream.mode = "ok"
        stream_id = state["generation_id"]
        status, body, _ = self.req("POST", f"/v1/generations/{stream_id}/fork", {"model": "mock"})
        self.assertEqual(status, 200)
        self.assertEqual(body["content"][0]["text"], "mock says ok")
        child = self.req("GET", "/v1/generations")[1]["data"][0]
        self.assertEqual(child["parent"], stream_id)
        self.assertEqual(self.req("GET", "/v1/interrupted")[0], 404)

        # 6. POST /v1/interrupted/resume re-issues the saved work.
        _MockUpstream.mode = "fail"
        self.req(
            "POST",
            "/v1/messages",
            {"model": "mock", "max_tokens": 50, "messages": [{"role": "user", "content": "Resume me"}]},
        )
        _MockUpstream.mode = "ok"
        self.assertEqual(self.req("POST", "/v1/interrupted/resume", {})[0], 200)
        self.assertEqual(self.req("GET", "/v1/interrupted")[0], 404)

        # 7. DELETE /v1/interrupted clears saved work on demand. Clearing and
        # deleting may take the ledger below the cap; never above it.
        _MockUpstream.mode = "fail"
        self.req(
            "POST",
            "/v1/messages",
            {"model": "mock", "max_tokens": 50, "messages": [{"role": "user", "content": "Delete me"}]},
        )
        status, body, _ = self.req("DELETE", "/v1/interrupted")
        self.assertEqual(status, 200)
        self.assertTrue(body["cleared"])
        self.assertEqual(self.req("GET", "/v1/interrupted")[0], 404)
        self.assertLessEqual(self._record_count(), 10)


class _HealUpstream(BaseHTTPRequestHandler):
    """Breaks the first stream; any request continuing an assistant turn succeeds."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        continuing = bool(messages) and messages[-1].get("role") == "assistant"
        if body.get("stream") and not continuing:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n')
            self.wfile.flush()
            self.wfile.write(b"data: {broken json\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"delta":{"content":"lo world"},"finish_reason":null}]}\n\n'
            )
            self.wfile.flush()
            self.wfile.write(
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5}}\n\n'
            )
            self.wfile.flush()
            self.close_connection = True
            return
        payload = {
            "id": "mock-1",
            "choices": [{"message": {"content": "mock says ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3},
        }
        self._json(payload)

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _QuotaUpstream(BaseHTTPRequestHandler):
    """Returns 429 (with Retry-After) until its failure budget runs out."""

    failures = 1

    def log_message(self, *args):
        pass

    def do_POST(self):
        if _QuotaUpstream.failures > 0:
            _QuotaUpstream.failures -= 1
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = {
            "id": "mock-1",
            "choices": [{"message": {"content": "mock says ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_HEAL_CONFIG = """
[models.heal]
base_url = "{upstream}"
model = "heal-primary"
api_key_env = "LLMBRIDGE_TEST_API_KEY"
fallbacks = ["heal-backup"]

[models.heal-backup]
base_url = "{upstream}"
model = "heal-backup-model"
api_key_env = "LLMBRIDGE_TEST_API_KEY"
"""

_QUEUE_CONFIG = """
[models.quota]
base_url = "{upstream}"
model = "quota-model"
"""


class _BridgeProcess:
    """Shared harness: one mock upstream + one real bridge subprocess."""

    @classmethod
    def _start(cls, handler, config_template, extra_env=None):
        cls.mock = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.mock.serve_forever, daemon=True).start()
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.ledger_dir = Path(cls._tmp.name) / "generations"
        config = Path(cls._tmp.name) / "llmbridge.toml"
        config.write_text(
            config_template.format(upstream=f"http://127.0.0.1:{cls.mock.server_address[1]}/v1")
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {
            **os.environ,
            "LLMBRIDGE_CONFIG": str(config),
            "LLMBRIDGE_HOST": "127.0.0.1",
            "LLMBRIDGE_PORT": str(port),
            "LLMBRIDGE_LEDGER_PATH": str(cls.ledger_dir),
            "LLMBRIDGE_TRACK_GENERATIONS": "1",
            "LLMBRIDGE_TRACK_INTERRUPTED": "1",
            **(extra_env or {}),
        }
        cls.server = subprocess.Popen(
            [sys.executable, "-c", "from llmbridge.server import main; main()"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cls.base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                cls.req("GET", "/v1/models")
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("LLMBridge server did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - cleanup only
            cls.server.kill()
        cls.mock.shutdown()

    @classmethod
    def req(cls, method, path, body=None, timeout=10, raw_reply=False):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            cls.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                parsed = raw if raw_reply else json.loads(raw)
                return response.status, parsed, dict(response.headers)
        except urllib.error.HTTPError as error:
            raw = error.read()
            if raw_reply:
                parsed = raw
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"raw": raw.decode(errors="replace")}
            return error.code, parsed, dict(error.headers)

    def _wait_status(self, generation_id, wanted, timeout=10):
        deadline = time.time() + timeout
        record = {}
        while time.time() < deadline:
            status, record, _ = self.req("GET", f"/v1/generations/{generation_id}")
            if status == 200 and record.get("status") == wanted:
                return record
            time.sleep(0.1)
        self.fail(f"generation {generation_id} never reached status {wanted!r}")

    def _sse_texts(self, payload):
        texts = []
        for frame in payload.split("\n\n"):
            for line in frame.splitlines():
                if line.startswith("data: "):
                    delta = json.loads(line[6:]).get("delta", {})
                    if "text" in delta:
                        texts.append(delta["text"])
        return texts

    def _sse_kinds(self, payload):
        kinds = []
        for frame in payload.split("\n\n"):
            for line in frame.splitlines():
                if line.startswith("event: "):
                    kinds.append(line[7:].strip())
        return kinds


class SelfHealEndToEndTests(_BridgeProcess, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._start(
            _HealUpstream,
            _HEAL_CONFIG,
            extra_env={"LLMBRIDGE_TEST_API_KEY": "test-key"},
        )

    def test_broken_stream_is_healed_invisibly(self):
        status, payload, headers = self.req(
            "POST",
            "/v1/messages",
            {
                "model": "heal",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            raw_reply=True,
        )
        self.assertEqual(status, 200)
        body = payload.decode()
        self.assertEqual("".join(self._sse_texts(body)), "Hello world")
        kinds = self._sse_kinds(body)
        self.assertEqual(kinds.count("message_start"), 1)
        self.assertNotIn("error", kinds)
        self.assertEqual(kinds[-1], "message_stop")

        generation_id = headers["X-LLMBridge-Generation"]
        record = self._wait_status(generation_id, "completed")
        self.assertEqual(record["output"], "Hello world")
        self.assertEqual(record["output_tokens"], 5)
        heals = [event for event in record["events"] if event["type"] == "heal"]
        self.assertEqual(len(heals), 1)
        self.assertEqual(heals[0]["data"]["to_model"], "heal-backup-model")
        self.assertEqual(heals[0]["data"]["resumed_chars"], 3)


class QueueEndToEndTests(_BridgeProcess, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._start(
            _QuotaUpstream,
            _QUEUE_CONFIG,
            extra_env={
                "LLMBRIDGE_QUEUE_ON_QUOTA": "1",
                "LLMBRIDGE_QUEUE_RETRY_SECONDS": "1",
                "LLMBRIDGE_QUEUE_MAX_ATTEMPTS": "3",
            },
        )

    def setUp(self):
        _QuotaUpstream.failures = 1

    def test_quota_parks_the_request_and_auto_resumes(self):
        status, body, headers = self.req(
            "POST",
            "/v1/messages",
            {"model": "quota", "max_tokens": 50, "messages": [{"role": "user", "content": "Park me"}]},
        )
        self.assertEqual(status, 429)
        self.assertTrue(body["queued"])
        self.assertTrue(body["generation_id"].startswith("gen_"))
        self.assertEqual(headers.get("Retry-After"), "2")

        status, listing, _ = self.req("GET", "/v1/queue")
        self.assertEqual(status, 200)
        self.assertIn(body["generation_id"], [item["id"] for item in listing["data"]])

        record = self._wait_status(body["generation_id"], "completed", timeout=20)
        self.assertEqual(record["output"], "mock says ok")
        self.assertIn("queue", [event["type"] for event in record["events"]])

        status, listing, _ = self.req("GET", "/v1/queue")
        self.assertEqual(listing["data"], [])

    def test_queued_work_can_be_cancelled(self):
        status, body, _ = self.req(
            "POST",
            "/v1/messages",
            {"model": "quota", "max_tokens": 50, "messages": [{"role": "user", "content": "Cancel me"}]},
        )
        self.assertEqual(status, 429)
        generation_id = body["generation_id"]
        status, reply, _ = self.req("DELETE", f"/v1/queue/{generation_id}")
        self.assertEqual(status, 200)
        self.assertTrue(reply["cancelled"])
        self.assertEqual(self.req("GET", "/v1/queue")[1]["data"], [])
        # Give the dispatcher a full tick; a live entry would have been retried.
        time.sleep(2.5)
        record = self._wait_status(generation_id, "failed")
        self.assertIn("cancelled", record["events"][-1]["data"].get("message", ""))


if __name__ == "__main__":
    unittest.main()
