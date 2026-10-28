import io
import json
import os
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llmbridge.ledger import GenerationLedger
from llmbridge.protocol import ProtocolError
from llmbridge.routing import ModelRoute
from llmbridge.server import (
    LLMBridgeHandler,
    LLMBridgeServer,
    _anthropic_text,
    _upstream_output_tokens,
    score_seam,
    settle_stale_generations,
    stale_generation_seconds,
)
from llmbridge.work_tracker import WorkTracker


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

    def test_stats_endpoint_reports_ledger_counts(self):
        record = self.ledger.create(
            {
                "model": "nemotron",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            "nemotron",
        )
        self.ledger.complete(record, "Hello", 3)

        status, _, body = self._call("GET", "/v1/stats")

        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "stats")
        self.assertEqual(body["generations"]["total"], 1)
        self.assertEqual(body["generations"]["by_status"], {"completed": 1})
        self.assertEqual(body["tokens"]["output"], 3)

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

    def test_tree_endpoint_returns_lineage(self):
        parent = self.ledger.create(self._message(prompt="seed"), "nemotron")
        self.ledger.complete(parent, "root", 2)
        child = self.ledger.create(
            self._message(prompt="seed"),
            "qwen-local",
            parent=parent.id,
            fork_event_seq=1,
        )

        status, _, tree = self._call("GET", f"/v1/generations/{child.id}/tree")
        self.assertEqual(status, 200)
        self.assertEqual(tree["id"], child.id)
        self.assertEqual([node["id"] for node in tree["ancestors"]], [parent.id])
        self.assertEqual(tree["node"]["record"]["id"], child.id)

        status, _, _ = self._call("GET", "/v1/generations/gen_missing/tree")
        self.assertEqual(status, 404)

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

    def test_fork_of_an_interrupted_generation_settles_it(self):
        upstream_error = HTTPError("https://upstream.test", 429, "Too Many Requests", {}, None)
        with patch("llmbridge.server.urlopen", side_effect=upstream_error):
            self._call("POST", "/v1/messages", self._message(prompt="Dead run"))
        dead = self.ledger.list()[0]
        self.assertEqual(dead.status, "interrupted")

        with patch(
            "llmbridge.server.urlopen",
            return_value=_json_response(_completion(content="Recovered")),
        ):
            status, child_headers, _ = self._call(
                "POST",
                f"/v1/generations/{dead.id}/fork",
                {"model": "nemotron", "max_tokens": 50},
            )

        self.assertEqual(status, 200)
        settled = self.ledger.load(dead.id)
        self.assertEqual(settled.status, "resumed")
        self.assertEqual(
            settled.events[-1].data["resumed_by"], self._generation_id(child_headers)
        )

    def test_fork_that_fails_leaves_the_source_recoverable(self):
        upstream_error = HTTPError("https://upstream.test", 429, "Too Many Requests", {}, None)
        with patch("llmbridge.server.urlopen", side_effect=upstream_error):
            self._call("POST", "/v1/messages", self._message(prompt="Dead run"))
        dead = self.ledger.list()[0]

        with patch("llmbridge.server.urlopen", side_effect=upstream_error):
            status, _, _ = self._call(
                "POST",
                f"/v1/generations/{dead.id}/fork",
                {"model": "nemotron", "max_tokens": 50},
            )

        self.assertEqual(status, 502)
        self.assertEqual(self.ledger.load(dead.id).status, "interrupted")
        self.assertEqual(len(self.ledger.interrupted()), 2)

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

    def test_fork_rejects_an_at_event_that_does_not_exist(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message())
        parent_id = self._generation_id(headers)

        status, _, body = self._call(
            "POST", f"/v1/generations/{parent_id}/fork", {"at_event": 99}
        )
        self.assertEqual(status, 400)
        self.assertIn("at_event", body["error"]["message"])
        # Lineage was not recorded for a step that never happened.
        self.assertEqual(len(self.ledger.children(parent_id)), 0)

    def test_fork_rejects_a_non_integer_at_event(self):
        with patch("llmbridge.server.urlopen", return_value=_json_response(_completion())):
            _, headers, _ = self._call("POST", "/v1/messages", self._message())
        parent_id = self._generation_id(headers)

        for value in ("1", True):
            status, _, body = self._call(
                "POST", f"/v1/generations/{parent_id}/fork", {"at_event": value}
            )
            self.assertEqual(status, 400, value)
            self.assertIn("at_event", body["error"]["message"])
        self.assertEqual(len(self.ledger.children(parent_id)), 0)

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


class QueueServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        env_patcher = patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "test-key"})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        self.tracker = WorkTracker(self.ledger)
        # Built with the queue disabled so no dispatcher thread races the
        # tests; each test enables the queue explicitly and dispatches by hand.
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                )
            },
            work_tracker=self.tracker,
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)

    def enable_queue(self, **extra):
        patcher = patch.dict(
            os.environ, {"LLMBRIDGE_QUEUE_ON_QUOTA": "1", **extra}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def _message(self, prompt="Hello", model="nemotron"):
        return {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
        }

    def _post_with_quota_error(self):
        error = HTTPError(
            "http://upstream.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {"Retry-After": "7"},
            io.BytesIO(b""),
        )
        with patch("llmbridge.server.urlopen", side_effect=error):
            return self._call("POST", "/v1/messages", self._message())

    def test_quota_failure_queues_the_generation(self):
        self.enable_queue()
        status, headers, body = self._post_with_quota_error()

        self.assertEqual(status, 429)
        self.assertTrue(body["queued"])
        generation_id = body["generation_id"]
        self.assertTrue(generation_id.startswith("gen_"))
        self.assertEqual(headers["retry-after"], "8")
        self.assertIn("queued", body["error"]["message"])

        record = self.ledger.load(generation_id)
        self.assertEqual(record.status, "queued")
        self.assertEqual(record.events[-1].type, "queue")
        self.assertEqual(record.events[-1].data["attempt"], 1)
        self.assertAlmostEqual(
            GenerationLedger.queue_retry_at(record), time.time() + 7, delta=3
        )

    def test_queue_is_opt_in(self):
        status, _, body = self._post_with_quota_error()
        self.assertEqual(status, 502)
        self.assertIsNone(body.get("queued"))
        records = self.ledger.list(limit=None)
        self.assertEqual(records[0].status, "interrupted")

    def test_dispatcher_completes_a_due_generation(self):
        self.enable_queue()
        _, _, body = self._post_with_quota_error()
        generation_id = body["generation_id"]
        record = self.ledger.load(generation_id)
        # Pull the retry window into the past so the dispatcher picks it up.
        self.ledger.queue(record, "quota", retry_at=time.time() - 1, attempt=1)

        with patch(
            "llmbridge.server.urlopen", return_value=_json_response(_completion())
        ):
            self.server._dispatch_due_queue()

        reloaded = self.ledger.load(generation_id)
        self.assertEqual(reloaded.status, "completed")
        self.assertEqual(reloaded.output, "Hello from Nemotron")
        _, _, detail = self._call("GET", f"/v1/generations/{generation_id}")
        self.assertEqual(detail["status"], "completed")
        _, _, listing = self._call("GET", "/v1/queue")
        self.assertEqual(listing["data"], [])

    def test_dispatcher_requeues_on_repeated_quota_errors(self):
        self.enable_queue()
        _, _, body = self._post_with_quota_error()
        generation_id = body["generation_id"]
        record = self.ledger.load(generation_id)
        self.ledger.queue(record, "quota", retry_at=time.time() - 1, attempt=1)

        error = HTTPError(
            "http://upstream.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b""),
        )
        with patch("llmbridge.server.urlopen", side_effect=error):
            self.server._dispatch_due_queue()

        requeued = self.ledger.load(generation_id)
        self.assertEqual(requeued.status, "queued")
        self.assertEqual(GenerationLedger.queue_attempt(requeued), 2)
        self.assertGreater(
            GenerationLedger.queue_retry_at(requeued), time.time() + 5
        )

    def test_attempt_budget_exhaustion_fails_the_record(self):
        self.enable_queue(**{"LLMBRIDGE_QUEUE_MAX_ATTEMPTS": "1"})
        _, _, body = self._post_with_quota_error()
        generation_id = body["generation_id"]
        record = self.ledger.load(generation_id)
        self.ledger.queue(record, "quota", retry_at=time.time() - 1, attempt=1)

        error = HTTPError(
            "http://upstream.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b""),
        )
        with patch("llmbridge.server.urlopen", side_effect=error):
            self.server._dispatch_due_queue()

        self.assertEqual(self.ledger.load(generation_id).status, "failed")

    def test_client_side_upstream_errors_are_not_queued(self):
        self.enable_queue()
        error = HTTPError(
            "http://upstream.test/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b""),
        )
        with patch("llmbridge.server.urlopen", side_effect=error):
            status, _, body = self._call("POST", "/v1/messages", self._message())
        self.assertEqual(status, 502)
        self.assertEqual(self.ledger.list(limit=None)[0].status, "failed")

    def test_queue_listing_and_cancellation(self):
        self.enable_queue()
        _, _, body = self._post_with_quota_error()
        generation_id = body["generation_id"]

        _, _, listing = self._call("GET", "/v1/queue")
        self.assertEqual(len(listing["data"]), 1)
        entry = listing["data"][0]
        self.assertEqual(entry["id"], generation_id)
        self.assertEqual(entry["attempt"], 1)
        self.assertGreaterEqual(entry["retry_in"], 0.0)

        status, _, cancel = self._call("DELETE", f"/v1/queue/{generation_id}")
        self.assertEqual(status, 200)
        self.assertTrue(cancel["cancelled"])
        self.assertEqual(self.ledger.load(generation_id).status, "failed")
        self.assertIn("cancelled", self.ledger.load(generation_id).events[-1].data["message"])

        _, _, listing = self._call("GET", "/v1/queue")
        self.assertEqual(listing["data"], [])

    def test_cancel_unknown_generation_returns_404(self):
        status, _, body = self._call("DELETE", "/v1/queue/gen_missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["type"], "not_found")

    def _queue_due(self):
        _, _, body = self._post_with_quota_error()
        generation_id = body["generation_id"]
        record = self.ledger.load(generation_id)
        self.ledger.queue(record, "quota", retry_at=time.time() - 1, attempt=1)
        return generation_id

    def test_queued_retry_follows_the_configured_fallbacks(self):
        self.enable_queue()
        generation_id = self._queue_due()
        # Configured after the request was parked, so the retry is the only run
        # that sees the secondary route.
        self.server.model_routes["backup"] = ModelRoute(
            "http://backup.test/v1", model="backup-model"
        )
        self.server.model_routes["nemotron"] = replace(
            self.server.model_routes["nemotron"], fallbacks=("backup",)
        )
        quota = HTTPError(
            "http://upstream.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b""),
        )
        with patch(
            "llmbridge.server.urlopen",
            side_effect=[quota, _json_response(_completion())],
        ) as call:
            self.server._dispatch_due_queue()

        self.assertEqual(call.call_count, 2)
        self.assertIn("backup.test", call.call_args.args[0].full_url)
        self.assertEqual(self.ledger.load(generation_id).status, "completed")

    def test_untranslatable_queued_request_fails_instead_of_looping(self):
        self.enable_queue()
        generation_id = self._queue_due()
        with patch(
            "llmbridge.server.anthropic_to_openai",
            side_effect=ProtocolError("unsupported content"),
        ) as translate:
            self.server._dispatch_due_queue()

        translate.assert_called_once()
        reloaded = self.ledger.load(generation_id)
        self.assertEqual(reloaded.status, "failed")
        self.assertIn("can no longer be sent", reloaded.events[-1].data["message"])

    def test_queued_retry_requeues_an_untranslatable_reply(self):
        self.enable_queue()
        generation_id = self._queue_due()
        malformed = {
            "choices": [
                {"message": {"content": "hi"}, "finish_reason": "made_up_reason"}
            ]
        }
        with patch(
            "llmbridge.server.urlopen", return_value=_json_response(malformed)
        ):
            self.server._dispatch_due_queue()

        requeued = self.ledger.load(generation_id)
        self.assertEqual(requeued.status, "queued")
        self.assertEqual(GenerationLedger.queue_attempt(requeued), 2)

    def test_queued_retry_requeues_a_connect_failure(self):
        self.enable_queue()
        generation_id = self._queue_due()
        with patch("llmbridge.server.urlopen", side_effect=URLError("refused")):
            self.server._dispatch_due_queue()

        requeued = self.ledger.load(generation_id)
        self.assertEqual(requeued.status, "queued")
        self.assertEqual(GenerationLedger.queue_attempt(requeued), 2)

    def test_queued_retry_requeues_a_server_error(self):
        self.enable_queue()
        generation_id = self._queue_due()
        error = HTTPError(
            "http://upstream.test/v1/chat/completions",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b""),
        )
        with patch("llmbridge.server.urlopen", side_effect=error):
            self.server._dispatch_due_queue()

        self.assertEqual(self.ledger.load(generation_id).status, "queued")

    def test_queued_retry_fails_fast_on_a_client_error(self):
        self.enable_queue()
        generation_id = self._queue_due()
        error = HTTPError(
            "http://upstream.test/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b""),
        )
        with patch("llmbridge.server.urlopen", side_effect=error):
            self.server._dispatch_due_queue()

        # Retrying a malformed request only burns the budget.
        self.assertEqual(self.ledger.load(generation_id).status, "failed")

    def test_queued_retry_marks_the_earlier_interruption_resumed(self):
        self.enable_queue()
        _, _, body = self._post_with_quota_error()
        retry = self.ledger.load(body["generation_id"])
        earlier = self.ledger.create(
            retry.request, retry.model, fingerprint=retry.fingerprint
        )
        self.ledger.interrupt(earlier, "quota", output="half", progress=5.0)
        self.ledger.queue(retry, "quota", retry_at=time.time() - 1, attempt=1)

        with patch(
            "llmbridge.server.urlopen", return_value=_json_response(_completion())
        ):
            self.server._dispatch_due_queue()

        self.assertEqual(self.ledger.load(earlier.id).status, "resumed")
        self.assertEqual(self.ledger.load(retry.id).status, "completed")

    def test_queued_retry_settles_only_after_it_is_completed(self):
        self.enable_queue()
        generation_id = self._queue_due()
        seen = {}

        def observe(fingerprint, *, by, model):
            seen["status"] = self.ledger.load(by).status
            return None

        with patch.object(self.tracker, "resolve_matching", side_effect=observe):
            with patch(
                "llmbridge.server.urlopen", return_value=_json_response(_completion())
            ):
                self.server._dispatch_due_queue()

        # Settling first would mark work done that a crash could still lose.
        self.assertEqual(seen["status"], "completed")
        self.assertEqual(self.ledger.load(generation_id).status, "completed")


def _heal_routes():
    return {
        "nemotron": ModelRoute(
            "http://primary.test/v1",
            model="nvidia/nemotron",
            api_key_env="LLMBRIDGE_TEST_API_KEY",
            fallbacks=("backup",),
        ),
        "backup": ModelRoute(
            "http://backup.test/v1",
            model="backup-model",
            api_key_env="LLMBRIDGE_TEST_API_KEY",
        ),
    }


def _broken_after_text():
    return _StreamScript(
        [
            b'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n',
            b'data: {"error": {"message": "quota exhausted"}}\n',
        ]
    )


def _continuation(text="lo world"):
    return _StreamScript(
        [
            f'data: {{"choices":[{{"delta":{{"content":"{text}"}},"finish_reason":null}}]}}\n'.encode(),
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5}}\n',
        ]
    )


def _sse_events(payload) -> list[dict]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", "replace")
    events = []
    for chunk in payload.split("\n\n"):
        for line in chunk.splitlines():
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except json.JSONDecodeError:
                    events.append({"raw": line})
    return events


class HealServerTests(QueueServerTests):
    def setUp(self):
        super().setUp()
        # Rebuild the server with a fallback chain; QueueServerTests uses one
        # route without fallbacks.
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            _heal_routes(),
            work_tracker=self.tracker,
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)

    def _message(self, prompt="Hello", model="nemotron"):
        return {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

    def _generation_id(self, headers):
        return headers.get("x-llmbridge-generation")

    def test_mid_stream_failure_heals_onto_the_fallback(self):
        with patch(
            "llmbridge.server.urlopen",
            side_effect=[_broken_after_text(), _continuation()],
        ) as upstream_calls:
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(status, 200)
        events = _sse_events(payload)
        kinds = [event.get("type") for event in events]
        self.assertNotIn("error", kinds)
        self.assertEqual(kinds.count("message_start"), 1)
        self.assertEqual(kinds[-1], "message_stop")

        text = "".join(
            event["delta"]["text"]
            for event in events
            if event.get("type") == "content_block_delta"
        )
        self.assertEqual(text, "Hello world")

        generation_id = self._generation_id(headers)
        record = self.ledger.load(generation_id)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.output, "Hello world")
        self.assertEqual(record.output_tokens, 5)
        heal_events = [event for event in record.events if event.type == "heal"]
        self.assertEqual(len(heal_events), 1)
        self.assertEqual(heal_events[0].data["to_model"], "backup-model")
        self.assertEqual(heal_events[0].data["resumed_chars"], 3)

        # The continuation request must carry the partial text as an
        # assistant turn so the fallback model continues rather than restarts.
        second_request = upstream_calls.call_args_list[1][0][0]
        body = json.loads(second_request.data)
        self.assertEqual(body["messages"][-1], {"role": "assistant", "content": "Hel"})

    def test_heal_records_a_seam_audit(self):
        with patch(
            "llmbridge.server.urlopen",
            side_effect=[_broken_after_text(), _continuation()],
        ):
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(status, 200)
        record = self.ledger.load(self._generation_id(headers))
        audits = [event for event in record.events if event.type == "heal_seam"]
        self.assertEqual(len(audits), 1)
        data = audits[0].data
        self.assertEqual(data["seam_at"], 3)
        self.assertEqual(data["continuation_chars"], len("lo world"))
        self.assertFalse(data["suspect"])

        heal = next(event for event in record.events if event.type == "heal")
        self.assertEqual(data["from_model"], heal.data["from_model"])
        self.assertEqual(data["to_model"], heal.data["to_model"])

    def test_a_heal_with_no_continuation_text_is_not_audited(self):
        with patch(
            "llmbridge.server.urlopen",
            side_effect=[_broken_after_text(), _continuation("")],
        ):
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(status, 200)
        record = self.ledger.load(self._generation_id(headers))
        # The heal is still recorded; only the seam score is skipped. Scoring an
        # empty continuation would count a join that never happened as a pass.
        self.assertEqual([event for event in record.events if event.type == "heal_seam"], [])
        self.assertEqual(len([event for event in record.events if event.type == "heal"]), 1)
        self.assertEqual(record.output, "Hel")

    def test_a_continuation_that_restarts_is_flagged(self):
        with patch(
            "llmbridge.server.urlopen",
            side_effect=[_broken_after_text(), _continuation("Sure here is the whole answer")],
        ):
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(status, 200)
        record = self.ledger.load(self._generation_id(headers))
        audit = next(event for event in record.events if event.type == "heal_seam").data
        self.assertTrue(audit["restarted"])
        self.assertTrue(audit["suspect"])
        # Flagging is an audit only: the stitched answer still reaches the client.
        self.assertEqual(record.status, "completed")
        self.assertTrue(record.output.endswith("Sure here is the whole answer"))

    def test_heal_can_be_disabled(self):
        env = patch.dict(os.environ, {"LLMBRIDGE_SELF_HEAL": "0"})
        env.start()
        self.addCleanup(env.stop)
        with patch(
            "llmbridge.server.urlopen", side_effect=[_broken_after_text()]
        ) as upstream_calls:
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(status, 200)
        events = _sse_events(payload)
        self.assertIn("error", [event.get("type") for event in events])
        self.assertEqual(upstream_calls.call_count, 1)
        record = self.ledger.load(self._generation_id(headers))
        self.assertEqual(record.status, "interrupted")
        self.assertEqual(record.output, "Hel")

    def test_heal_failure_still_saves_the_interruption(self):
        empty_failure = _StreamScript(
            [b'data: {"error": {"message": "heal route died too"}}\n']
        )
        with patch(
            "llmbridge.server.urlopen",
            side_effect=[_broken_after_text(), empty_failure],
        ):
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(status, 200)
        events = _sse_events(payload)
        self.assertIn("error", [event.get("type") for event in events])
        text = "".join(
            event["delta"]["text"]
            for event in events
            if event.get("type") == "content_block_delta"
        )
        self.assertEqual(text, "Hel")
        record = self.ledger.load(self._generation_id(headers))
        self.assertEqual(record.status, "interrupted")
        self.assertEqual(record.output, "Hel")

    def test_heal_is_skipped_without_partial_text(self):
        empty_failure = _StreamScript(
            [b'data: {"error": {"message": "instant death"}}\n']
        )
        with patch(
            "llmbridge.server.urlopen", side_effect=[empty_failure]
        ) as upstream_calls:
            status, headers, payload = self._call(
                "POST", "/v1/messages", self._message()
            )

        self.assertEqual(upstream_calls.call_count, 1)
        events = _sse_events(payload)
        self.assertIn("error", [event.get("type") for event in events])
        record = self.ledger.load(self._generation_id(headers))
        self.assertEqual(record.status, "interrupted")
        self.assertEqual(record.output, "")


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


class RunningGenerationSettlementTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _create(self, prompt="Hello"):
        return self.ledger.create(
            {
                "model": "nemotron",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": prompt}],
            },
            "nemotron",
        )

    def test_stop_settles_records_left_running(self):
        server = LLMBridgeServer(
            ("127.0.0.1", 0), LLMBridgeHandler, {}, ledger=self.ledger
        )
        self.addCleanup(server.server_close)
        record = self._create()

        server.stop()

        reloaded = self.ledger.load(record.id)
        self.assertEqual(reloaded.status, "interrupted")
        self.assertIn("Bridge stopped", reloaded.events[-1].data["message"])

    def test_startup_sweep_skips_records_younger_than_the_margin(self):
        record = self._create()

        self.assertEqual(settle_stale_generations(self.ledger), [])
        self.assertEqual(self.ledger.load(record.id).status, "running")

        settled = settle_stale_generations(
            self.ledger, now=time.time() + stale_generation_seconds() + 1
        )
        self.assertEqual([item.id for item in settled], [record.id])
        self.assertEqual(self.ledger.load(record.id).status, "interrupted")


class SeamScoringTests(unittest.TestCase):
    def test_a_plain_join_is_not_suspect(self):
        audit = score_seam(
            "The capital of France is",
            " Paris, and its metro carries about 10 million riders a week.",
        )
        self.assertFalse(audit["duplicated"])
        self.assertFalse(audit["restarted"])
        self.assertFalse(audit["suspect"])

    def test_a_repeated_tail_is_flagged_as_duplicated(self):
        partial = "Below are the quarterly figures for fiscal 2024"
        audit = score_seam(partial, "figures for fiscal 2024. Revenue rose 12 percent.")
        self.assertTrue(audit["duplicated"])
        self.assertTrue(audit["suspect"])
        self.assertGreaterEqual(audit["overlap_chars"], 12)

    def test_a_fresh_preamble_is_flagged_as_a_restart(self):
        for continuation in (
            "Sure! The capital of France is Paris.",
            "Certainly, here is a summary of the ledger:",
            "Here's the answer you asked for.",
            "I can help with that. The total is 42.",
        ):
            with self.subTest(continuation=continuation):
                self.assertTrue(score_seam("totally different prefix text", continuation)["restarted"])

    def test_a_shared_phrase_inside_the_continuation_is_not_duplicated(self):
        # The repeated words sit mid-continuation, which is ordinary phrasing
        # reuse; only a match that starts at the continuation cannot be that.
        audit = score_seam(
            "In 2023 revenue increased sharply and then the market cooled",
            "Overall revenue increased by 12 percent last year.",
        )
        self.assertGreaterEqual(audit["overlap_chars"], 12)
        self.assertGreaterEqual(audit["overlap_ratio"], 0.25)
        self.assertFalse(audit["duplicated"])
        self.assertFalse(audit["suspect"])

    def test_a_decorated_fresh_preamble_is_flagged_as_a_restart(self):
        for continuation in (
            "**Sure!** The capital of France is Paris.",
            "- Certainly, continuing where it stopped.",
            "> Of course!",
        ):
            with self.subTest(continuation=continuation):
                self.assertTrue(
                    score_seam("totally different prefix text", continuation)["restarted"]
                )

    def test_a_normal_word_continuation_is_not_a_restart(self):
        self.assertFalse(score_seam("The total rev", "enue for the quarter was 12 percent.")["restarted"])

    def test_a_short_boundary_cannot_trip_the_overlap_floor(self):
        # Ratio alone would be misleading on tiny windows; the char floor guards it.
        audit = score_seam("Hel", "lo world")
        self.assertGreaterEqual(audit["overlap_ratio"], 0.25)
        self.assertLess(audit["overlap_chars"], 12)
        self.assertFalse(audit["suspect"])

    def test_only_the_boundary_window_is_compared(self):
        repeated = "the quarterly figures for fiscal 2024 were reviewed"
        partial = "Chapter one opens with " + repeated + " and then the narrative moves on"
        audit = score_seam(partial, " The next paragraph introduces the accounts.")
        self.assertFalse(audit["duplicated"])

    def test_empty_inputs_are_scored_without_error(self):
        self.assertEqual(score_seam("", "")["continuation_chars"], 0)
        self.assertFalse(score_seam("some text", "")["suspect"])
        self.assertFalse(score_seam("", "Paris is the capital")["duplicated"])


if __name__ == "__main__":
    unittest.main()
