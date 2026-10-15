import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llmbridge.ledger import GenerationLedger
from llmbridge.routing import ModelRoute
from llmbridge.server import (
    LLMBridgeHandler,
    LLMBridgeServer,
    _anthropic_text,
    _upstream_output_tokens,
)


def _decode(payload):
    """Parse JSON bodies, but tolerate SSE (streaming) responses."""
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload.decode("utf-8", "replace")


def _json_response(body):
    class _Response:
        def __init__(self, payload):
            self.body = json.dumps(payload).encode("utf-8")
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=-1):
            if size is None or size < 0:
                chunk = self.body[self._offset :]
            else:
                chunk = self.body[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    return _Response(body)


def _completion(content="Hello from Nemotron", completion_tokens=4):
    return {
        "id": "chatcmpl-test",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": completion_tokens},
    }


class _StreamScript:
    def __init__(self, lines, fail=None):
        self._lines = lines
        self._fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for line in self._lines:
            yield line
        if self._fail is not None:
            raise self._fail


class GenerationServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "test-key"})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                ),
                "qwen-local": ModelRoute("http://127.0.0.1:11434/v1", model="qwen3:8b"),
            },
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _call(self, method, path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request) as response:
                payload = response.read()
                headers = {key.lower(): value for key, value in response.headers.items()}
                return response.status, headers, _decode(payload)
        except HTTPError as error:
            payload = error.read()
            headers = {key.lower(): value for key, value in error.headers.items()}
            status = error.code
            error.close()
            return status, headers, _decode(payload)

    def _message(self, prompt="Hello", model="nemotron", stream=False):
        return {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

    def _generation_id(self, headers):
        return headers.get("x-llmbridge-generation")

    def test_non_stream_success_records_completed_generation(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            status, headers, _ = self._call("POST", "/v1/messages", self._message())

        self.assertEqual(status, 200)
        generation_id = self._generation_id(headers)
        self.assertTrue(generation_id and generation_id.startswith("gen_"))

        status, _, detail = self._call("GET", f"/v1/generations/{generation_id}")
        self.assertEqual(status, 200)
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["output"], "Hello from Nemotron")
        self.assertEqual(detail["output_tokens"], 4)
        self.assertEqual([event["type"] for event in detail["events"]], ["start", "complete"])

    def test_listing_reports_recorded_generations(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            self._call("POST", "/v1/messages", self._message(prompt="first"))
            self._call("POST", "/v1/messages", self._message(prompt="second"))

        status, headers, listing = self._call("GET", "/v1/generations")
        self.assertEqual(status, 200)
        self.assertEqual(listing["object"], "list")
        self.assertEqual(len(listing["data"]), 2)
        self.assertEqual(listing["data"][0]["model"], "nemotron")
        self.assertIn("prompt_preview", listing["data"][0])

    def test_empty_ledger_lists_nothing(self):
        status, _, listing = self._call("GET", "/v1/generations")
        self.assertEqual(status, 200)
        self.assertEqual(listing["data"], [])

    def test_streaming_success_records_output(self):
        stream = _StreamScript(
            [
                b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":9}}\n',
            ]
        )
        with patch("llmbridge.server.urlopen", return_value=stream):
            status, headers, _ = self._call("POST", "/v1/messages", self._message(stream=True))

        self.assertEqual(status, 200)
        generation_id = self._generation_id(headers)
        _, _, detail = self._call("GET", f"/v1/generations/{generation_id}")
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["output"], "done")
        self.assertEqual(detail["output_tokens"], 9)

    def test_streaming_failure_records_interrupted_generation(self):
        stream = _StreamScript(
            [
                b'data: {"choices":[{"delta":{"content":"Half"},"finish_reason":null}]}\n',
            ],
            fail=TimeoutError(),
        )
        with patch("llmbridge.server.urlopen", return_value=stream):
            status, headers, _ = self._call("POST", "/v1/messages", self._message(stream=True))

        self.assertEqual(status, 200)
        generation_id = self._generation_id(headers)
        _, _, detail = self._call("GET", f"/v1/generations/{generation_id}")
        self.assertEqual(detail["status"], "interrupted")
        self.assertEqual(detail["output"], "Half")
        self.assertEqual(detail["events"][-1]["type"], "interrupt")

    def test_non_resumable_upstream_error_is_recorded_as_failed(self):
        upstream_error = HTTPError("https://upstream.test", 400, "Bad Request", {}, None)
        with patch("llmbridge.server.urlopen", side_effect=upstream_error):
            status, _, _ = self._call("POST", "/v1/messages", self._message())

        self.assertEqual(status, 502)
        records = self.ledger.list()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "failed")

    def test_interrupted_error_body_includes_fork_hint(self):
        upstream_error = HTTPError("https://upstream.test", 429, "Too Many Requests", {}, None)
        with patch("llmbridge.server.urlopen", side_effect=upstream_error):
            status, _, body = self._call("POST", "/v1/messages", self._message())

        self.assertEqual(status, 502)
        message = body["error"]["message"]
        self.assertIn("/v1/generations/", message)
        self.assertIn("/fork", message)
        record = self.ledger.list()[0]
        self.assertEqual(record.status, "interrupted")
        self.assertIn(record.id, message)

    def test_stream_error_includes_fork_hint(self):
        stream = _StreamScript(
            [b'data: {"choices":[{"delta":{"content":"Half"},"finish_reason":null}]}\n'],
            fail=TimeoutError(),
        )
        with patch("llmbridge.server.urlopen", return_value=stream):
            status, _, body = self._call("POST", "/v1/messages", self._message(stream=True))
        self.assertEqual(status, 200)
        self.assertIn("/fork", body)

    def test_failed_generation_error_has_no_recovery_hint(self):
        upstream_error = HTTPError("https://upstream.test", 400, "Bad Request", {}, None)
        with patch("llmbridge.server.urlopen", side_effect=upstream_error):
            _, _, body = self._call("POST", "/v1/messages", self._message())
        self.assertNotIn("/fork", body["error"]["message"])

    def test_fork_creates_child_generation_with_parent_lineage(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message(prompt="Original"))
        parent_id = self._generation_id(headers)

        with patch(
            "llmbridge.server.urlopen",
            return_value=_json_response(_completion(content="Forked continuation")),
        ):
            status, child_headers, _ = self._call(
                "POST",
                f"/v1/generations/{parent_id}/fork",
                {"model": "qwen-local", "max_tokens": 50},
            )

        self.assertEqual(status, 200)
        child_id = self._generation_id(child_headers)
        self.assertNotEqual(child_id, parent_id)
        _, _, child = self._call("GET", f"/v1/generations/{child_id}")
        self.assertEqual(child["parent"], parent_id)
        self.assertEqual(child["model"], "qwen-local")
        self.assertEqual(child["status"], "completed")
        self.assertEqual(child["output"], "Forked continuation")
        self.assertEqual(child["events"][0]["type"], "fork")

    def test_fork_defaults_to_source_model_when_unspecified(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message(prompt="Base"))
            parent_id = self._generation_id(headers)
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            status, child_headers, _ = self._call(
                "POST", f"/v1/generations/{parent_id}/fork", {}
            )
        self.assertEqual(status, 200)
        child_id = self._generation_id(child_headers)
        _, _, child = self._call("GET", f"/v1/generations/{child_id}")
        self.assertEqual(child["model"], "nemotron")
        self.assertEqual(child["parent"], parent_id)

    def test_fork_rejects_non_object_body(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message())
        parent_id = self._generation_id(headers)

        status, _, body = self._call(
            "POST", f"/v1/generations/{parent_id}/fork", [1, 2, 3]
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_fork_of_unknown_generation_returns_404(self):
        status, _, body = self._call("POST", "/v1/generations/gen_missing/fork", {})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["type"], "not_found")

    def test_fork_with_unknown_model_returns_400(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message())
        parent_id = self._generation_id(headers)

        status, _, body = self._call(
            "POST", f"/v1/generations/{parent_id}/fork", {"model": "nonexistent"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_delete_removes_generation(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message())
        generation_id = self._generation_id(headers)

        status, _, body = self._call("DELETE", f"/v1/generations/{generation_id}")
        self.assertEqual(status, 200)
        self.assertTrue(body["deleted"])
        status, _, _ = self._call("GET", f"/v1/generations/{generation_id}")
        self.assertEqual(status, 404)

    def test_get_unknown_generation_returns_404(self):
        status, _, body = self._call("GET", "/v1/generations/gen_missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["type"], "not_found")

    def test_generation_tracking_disabled_returns_404(self):
        server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {"nemotron": ModelRoute("http://upstream.test/v1", model="nvidia/nemotron")},
            ledger=None,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/v1/generations") as response:
                status = response.status
                response.read()
        except HTTPError as error:
            status = error.code
            error.close()
        self.assertEqual(status, 404)


class GenerationHelperTests(unittest.TestCase):
    def test_anthropic_text_concatenates_text_blocks(self):
        response = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "tool_use", "id": "x"},
                {"type": "text", "text": "world"},
            ]
        }
        self.assertEqual(_anthropic_text(response), "Hello world")

    def test_anthropic_text_tolerates_missing_content(self):
        self.assertEqual(_anthropic_text(None), "")
        self.assertEqual(_anthropic_text({}), "")
        self.assertEqual(_anthropic_text({"content": "not-a-list"}), "")

    def test_upstream_output_tokens_reads_completion_tokens(self):
        self.assertEqual(
            _upstream_output_tokens({"usage": {"completion_tokens": 3}}), 3
        )

    def test_upstream_output_tokens_tolerates_bad_input(self):
        self.assertIsNone(_upstream_output_tokens(None))
        self.assertIsNone(_upstream_output_tokens({}))
        self.assertIsNone(_upstream_output_tokens({"usage": "nope"}))
        self.assertIsNone(_upstream_output_tokens({"usage": {"completion_tokens": True}}))
        self.assertIsNone(_upstream_output_tokens({"usage": {"completion_tokens": -1}}))


if __name__ == "__main__":
    unittest.main()
