import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from llmbridge.ledger import (
    EVENT_RESUME,
    MAX_TRACKED_INTERRUPTED,
    GenerationLedger,
    GenerationRecord,
    _fingerprint,
)
from llmbridge.routing import ModelRoute, RoutingError
from llmbridge.server import (
    LLMBridgeHandler,
    LLMBridgeServer,
    _build_heal_body,
    _client_timeout_seconds,
    _coalescible,
    _extract_prompt_and_system,
    _is_loopback_host,
    _is_retryable_status,
    _parse_retry_after,
    _retry_sleep,
    _route_attempts,
    _route_retry_backoff_seconds,
    _upstream_timeout_seconds,
    _validate_listener_auth,
    load_server_settings,
)
from llmbridge.work_tracker import WorkTracker


class _UpstreamResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")
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


class _TimedOutStream:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}\n'
        raise TimeoutError


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                ),
                "qwen-local": ModelRoute(
                    "http://127.0.0.1:11434/v1",
                    model="qwen3:8b",
                ),
            },
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_messages_endpoint_translates_upstream_request_and_response(self):
        upstream_response = _UpstreamResponse(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {"content": "Hello from Nemotron"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        env = {
            "LLMBRIDGE_TEST_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env):
            with patch("llmbridge.server.urlopen", return_value=upstream_response) as call:
                with urlopen(request) as response:
                    result = json.loads(response.read())

        forwarded_request = call.call_args.args[0]
        self.assertEqual(
            forwarded_request.full_url,
            "http://upstream.test/v1/chat/completions",
        )
        self.assertEqual(forwarded_request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(forwarded_request.data)["model"], "nvidia/nemotron")
        self.assertEqual(result["model"], "nemotron")
        self.assertEqual(
            result["content"],
            [{"type": "text", "text": "Hello from Nemotron"}],
        )
        self.assertEqual(result["usage"]["input_tokens"], 12)

    def test_model_list_returns_configured_route_aliases(self):
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/v1/models?limit=1000") as response:
            result = json.loads(response.read())

        self.assertEqual(
            [model["id"] for model in result["data"]],
            ["nemotron", "qwen-local"],
        )

    def test_configured_client_key_protects_api_but_not_health(self):
        self.server.client_api_key = "client-secret"
        try:
            with urlopen(f"http://127.0.0.1:{self.server.server_port}/health") as response:
                self.assertEqual(response.status, 200)

            with self.assertRaises(HTTPError) as error:
                urlopen(f"http://127.0.0.1:{self.server.server_port}/v1/models")
            self.assertEqual(error.exception.code, 401)
            error.exception.close()

            request = Request(
                f"http://127.0.0.1:{self.server.server_port}/v1/models",
                headers={"Authorization": "Bearer client-secret"},
            )
            with urlopen(request) as response:
                result = json.loads(response.read())
            self.assertEqual(len(result["data"]), 2)

            request = Request(
                f"http://127.0.0.1:{self.server.server_port}/v1/models",
                headers={"x-api-key": "client-secret"},
            )
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
        finally:
            self.server.client_api_key = None

    def test_non_loopback_listeners_require_client_authentication(self):
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host("::1"))
        self.assertTrue(_is_loopback_host("localhost"))
        self.assertFalse(_is_loopback_host("0.0.0.0"))
        self.assertFalse(_is_loopback_host("192.0.2.10"))
        with self.assertRaisesRegex(SystemExit, "LLMBRIDGE_API_KEY is required"):
            _validate_listener_auth("0.0.0.0", None)
        with self.assertRaisesRegex(SystemExit, "at least 32 characters"):
            _validate_listener_auth("0.0.0.0", "x" * 31)
        _validate_listener_auth("0.0.0.0", "x" * 32)

    def test_claude_desktop_gateway_probe_supports_head_api_hello(self):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/api/hello",
            method="HEAD",
        )
        with urlopen(request) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")

    def test_count_tokens_route_accepts_beta_query(self):
        body = {
            "model": "nemotron",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages/count_tokens?beta=true",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.loads(response.read())

        self.assertEqual(response.status, 200)
        self.assertGreater(result["input_tokens"], 0)

    def test_count_tokens_uses_route_configured_tiktoken_encoding(self):
        self.server.model_routes["nemotron"] = ModelRoute(
            "http://upstream.test/v1",
            model="nvidia/nemotron",
            tokenizer_encoding="cl100k_base",
        )
        body = {
            "model": "nemotron",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages/count_tokens",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        tokenizer = MagicMock()
        tokenizer.get_encoding.return_value.encode.return_value = [1, 2, 3, 4]
        with patch.dict("sys.modules", {"tiktoken": tokenizer}):
            with urlopen(request) as response:
                result = json.loads(response.read())

        self.assertEqual(result["input_tokens"], 4)
        tokenizer.get_encoding.assert_called_once_with("cl100k_base")

    def test_messages_route_accepts_beta_query(self):
        upstream_response = _UpstreamResponse(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {"content": "Hello from Nemotron"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages?beta=true",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "test-key"}):
            with patch("llmbridge.server.urlopen", return_value=upstream_response):
                with urlopen(request) as response:
                    result = json.loads(response.read())

        self.assertEqual(response.status, 200)
        self.assertEqual(result["content"][0]["text"], "Hello from Nemotron")

    def test_stream_timeout_is_reported_as_sse_error_after_response_starts(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "test-key"}):
            with patch("llmbridge.server.urlopen", return_value=_TimedOutStream()):
                with urlopen(request) as response:
                    result = response.read().decode("utf-8")

        self.assertIn("event: message_start", result)
        self.assertIn("event: error", result)
        self.assertIn("Upstream stream timed out", result)
        self.assertNotIn("event: message_stop", result)

    def test_upstream_error_details_are_not_written_to_logs(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        upstream_error = HTTPError(
            "https://upstream.test",
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b"private upstream detail"),
        )
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "test-key"}):
            with patch("llmbridge.server.urlopen", side_effect=upstream_error):
                with patch.object(LLMBridgeHandler, "log_error") as logged:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request)

        self.assertEqual(error.exception.code, 502)
        error.exception.close()
        self.assertNotIn("private upstream detail", str(logged.call_args))

    def test_unknown_model_is_rejected_without_calling_upstream(self):
        body = {
            "model": "unconfigured-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with patch("llmbridge.server.urlopen") as upstream:
            with self.assertRaises(HTTPError) as error:
                with urlopen(request) as response:
                    response.read()

        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        upstream.assert_not_called()


class _StreamFailure:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        raise TimeoutError


class _PartialThenFailingStream:
    """Streams some text, then fails mid-generation."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"Hello wor"},"finish_reason":null}]}\n'
        yield b'data: {"choices":[{"delta":{"content":"ld!"},"finish_reason":null}]}\n'
        raise TimeoutError


class _CompletingStream:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n'


class InterruptedWorkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.tracker = WorkTracker(self.ledger)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                ),
            },
            work_tracker=self.tracker,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def _post_message(self, body: dict, **kwargs) -> Request:
        return Request(
            self._url("/v1/messages"),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _get_interrupted(self):
        with urlopen(self._url("/v1/interrupted")) as response:
            return response.status, json.loads(response.read())

    def test_quota_error_records_interrupted_work(self):
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        upstream_error = HTTPError("https://upstream.test", 429, "Too Many Requests", {}, None)
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=upstream_error):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

        status, state = self._get_interrupted()
        self.assertEqual(status, 200)
        self.assertEqual(state["prompt"], "Hello")
        self.assertEqual(state["model_name"], "nemotron")
        self.assertEqual(state["error"], "Upstream returned HTTP 429")

    def test_client_error_is_not_recorded(self):
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        upstream_error = HTTPError("https://upstream.test", 400, "Bad Request", {}, None)
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=upstream_error):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        error.exception.close()
        self.assertEqual(self.ledger.interrupted(), [])

    def test_stream_failure_records_interrupted_work(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=_StreamFailure()):
                with urlopen(self._post_message(body)) as response:
                    response.read()

        status, state = self._get_interrupted()
        self.assertEqual(status, 200)
        self.assertEqual(state["error"], "Upstream stream timed out")

    def test_stream_failure_saves_partial_output_and_progress(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=_PartialThenFailingStream()):
                with urlopen(self._post_message(body)) as response:
                    response.read()

        _, state = self._get_interrupted()
        self.assertEqual(state["partial_output"], "Hello world!")
        # 12 characters ~ 3 estimated tokens of 100 allowed.
        self.assertEqual(state["progress_percent"], 3.0)
        self.assertEqual(state["error"], "Upstream stream timed out")

    def test_stream_failure_without_output_reports_zero_progress(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=_StreamFailure()):
                with urlopen(self._post_message(body)) as response:
                    response.read()

        _, state = self._get_interrupted()
        self.assertEqual(state["partial_output"], "")
        self.assertEqual(state["progress_percent"], 0.0)

    def test_successful_stream_does_not_save_interrupted_work(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=_CompletingStream()):
                with urlopen(self._post_message(body)) as response:
                    result = response.read().decode("utf-8")

        self.assertIn("event: message_stop", result)
        self.assertEqual(self.ledger.interrupted(), [])

    def _interrupt(self, body):
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=URLError("refused")):
                with self.assertRaises(HTTPError) as failure:
                    urlopen(self._post_message(body))
        failure.exception.close()

    def _complete(self, body):
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=_UpstreamResponse(
                {"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            )):
                with urlopen(self._post_message(body)) as response:
                    self.assertEqual(response.status, 200)

    def test_resending_the_failed_request_resolves_its_dead_run(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self._interrupt(body)
        _, state = self._get_interrupted()
        self.assertEqual(state["prompt"], "Hello")

        self._complete(body)

        self.assertEqual(self.ledger.interrupted(), [])
        resolved = self.ledger.load(state["generation_id"])
        self.assertEqual(resolved.status, "resumed")
        self.assertEqual(resolved.events[-1].type, EVENT_RESUME)
        # Evidence stays on disk: the interruption is still measurable.
        self.assertTrue(self.ledger._path(resolved.id).exists())

    def test_a_different_request_leaves_another_runs_dead_work_alone(self):
        dead = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Some earlier work"}],
        }
        self._interrupt(dead)

        other = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Something else"}],
        }
        self._complete(other)

        status, state = self._get_interrupted()
        self.assertEqual(status, 200)
        self.assertEqual(state["prompt"], "Some earlier work")
        self.assertEqual(self.ledger.interrupted()[0].status, "interrupted")

    def test_successful_request_keeps_unrelated_interrupted_work(self):
        self.tracker.start_tracking(prompt="Some earlier work", model_name="nemotron")
        self.tracker.save_interrupted()

        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Something else"}]}
        self._complete(body)

        status, state = self._get_interrupted()
        self.assertEqual(status, 200)
        self.assertEqual(state["prompt"], "Some earlier work")

    def test_delete_clears_interrupted_work(self):
        self.tracker.start_tracking(prompt="Hello", model_name="nemotron")
        self.tracker.save_interrupted()

        with urlopen(Request(self._url("/v1/interrupted"), method="DELETE")) as response:
            result = json.loads(response.read())
        self.assertTrue(result["cleared"])
        self.assertEqual(self.ledger.interrupted(), [])

    def test_interrupted_endpoint_lists_every_recoverable_job(self):
        first = self.tracker.save(prompt="First dead run", model_name="nemotron")
        second = self.tracker.save(prompt="Second dead run", model_name="nemotron")

        status, state = self._get_interrupted()
        self.assertEqual(status, 200)
        self.assertEqual(state["generation_id"], second["generation_id"])
        self.assertEqual(state["recoverable_count"], 2)
        self.assertEqual(state["other_recoverable"], [first["generation_id"]])

    def test_delete_reports_the_work_still_waiting(self):
        first = self.tracker.save(prompt="First dead run", model_name="nemotron")
        self.tracker.save(prompt="Second dead run", model_name="nemotron")

        with urlopen(Request(self._url("/v1/interrupted"), method="DELETE")) as response:
            result = json.loads(response.read())
        self.assertTrue(result["cleared"])
        self.assertEqual(result["remaining_recoverable"], [first["generation_id"]])

    def test_interrupted_endpoint_requires_authentication(self):
        self.server.client_api_key = "client-secret"
        try:
            self.tracker.start_tracking(prompt="Hello", model_name="nemotron")
            self.tracker.save_interrupted()
            with self.assertRaises(HTTPError) as error:
                urlopen(self._url("/v1/interrupted"))
            self.assertEqual(error.exception.code, 401)
            error.exception.close()
        finally:
            self.server.client_api_key = None

    def test_saved_prompt_is_not_affected_by_shared_session(self):
        self.tracker.start_tracking(prompt="STALE", model_name="nemotron")
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        upstream_error = HTTPError("https://upstream.test", 429, "Too Many Requests", {}, None)
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=upstream_error):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        error.exception.close()

        _, state = self._get_interrupted()
        self.assertEqual(state["prompt"], "Hello")

    def test_records_system_prompt(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "system": "be terse",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        upstream_error = HTTPError("https://upstream.test", 429, "Too Many Requests", {}, None)
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=upstream_error):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        error.exception.close()

        _, state = self._get_interrupted()
        self.assertEqual(state["system"], "be terse")

    def test_request_without_a_fingerprint_leaves_saved_work_alone(self):
        state = self.tracker.save(prompt="Hello", model_name="nemotron")
        handler = MagicMock()
        handler._resolves = None
        handler._request_fingerprint = ""
        LLMBridgeHandler._settle_interrupted_work(handler, self.tracker, None, "nemotron")
        self.assertEqual(
            [record.id for record in self.ledger.interrupted()], [state["generation_id"]]
        )

    def test_connection_error_records_interrupted_work(self):
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=URLError("refused")):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

        _, state = self._get_interrupted()
        self.assertEqual(state["error"], "Could not connect to the configured upstream")

    def test_invalid_upstream_response_records_interrupted_work(self):
        class _BadResponse:
            def __init__(self):
                self._sent = False

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, size=-1):
                if self._sent:
                    return b""
                self._sent = True
                return b"not json"

        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=_BadResponse()):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

        _, state = self._get_interrupted()
        self.assertTrue(state["error"])

    def test_non_stream_timeout_records_interrupted_work(self):
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=TimeoutError()):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post_message(body))
        self.assertEqual(error.exception.code, 504)
        error.exception.close()

        _, state = self._get_interrupted()
        self.assertEqual(state["error"], "Upstream request timed out")

    def test_oversized_upstream_response_is_rejected(self):
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        upstream_response = _UpstreamResponse(
            {"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server._MAX_UPSTREAM_RESPONSE_BYTES", 5):
                with patch("llmbridge.server.urlopen", return_value=upstream_response):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(self._post_message(body))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

        _, state = self._get_interrupted()
        self.assertIn("size limit", state["error"])

    def test_missing_interrupted_work_returns_404(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self._url("/v1/interrupted"))
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def test_delete_unknown_endpoint_returns_404(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self._url("/v1/nope"), method="DELETE"))
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def test_delete_interrupted_requires_authentication(self):
        self.server.client_api_key = "client-secret"
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self._url("/v1/interrupted"), method="DELETE"))
            self.assertEqual(error.exception.code, 401)
            error.exception.close()
        finally:
            self.server.client_api_key = None

    def test_save_interrupted_swallows_tracker_failures(self):
        handler = MagicMock()
        tracker = MagicMock()
        tracker.save.side_effect = OSError("disk")
        LLMBridgeHandler._save_interrupted(handler, tracker, "p", "m", None, "boom")
        handler.log_error.assert_called_once()

    def test_save_interrupted_skips_when_generation_record_exists(self):
        handler = MagicMock()
        tracker = MagicMock()
        record = GenerationRecord(
            id="gen_abc",
            fingerprint="fp",
            model="m",
            prompt="p",
            system=None,
            params={},
            status="running",
            created_at=0.0,
            updated_at=0.0,
        )
        LLMBridgeHandler._save_interrupted(
            handler, tracker, "p", "m", None, "boom", record=record
        )
        tracker.save.assert_not_called()
        handler.log_error.assert_not_called()

    def test_save_interrupted_forwards_replayable_request(self):
        handler = MagicMock()
        handler._request_fingerprint = ""
        tracker = MagicMock()
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be terse",
            "tools": [{"name": "search"}],
        }
        LLMBridgeHandler._save_interrupted(
            handler, tracker, "hi", "m", "be terse", "boom", body=body
        )
        _, kwargs = tracker.save.call_args
        self.assertEqual(kwargs["request"]["messages"], body["messages"])
        self.assertEqual(kwargs["request"]["tools"], body["tools"])
        self.assertEqual(kwargs["request"]["system"], "be terse")

    def test_save_interrupted_keeps_the_request_fingerprint(self):
        handler = MagicMock()
        handler._request_fingerprint = "fp-of-the-client-request"
        tracker = MagicMock()
        LLMBridgeHandler._save_interrupted(
            handler,
            tracker,
            "hi",
            "m",
            None,
            "boom",
            body={"messages": [{"role": "user", "content": "hi"}]},
        )
        # The record is rebuilt from the replayable parts, so hashing it again
        # would give a different identity than the request that died.
        self.assertEqual(
            tracker.save.call_args.kwargs["fingerprint"],
            "fp-of-the-client-request",
        )

    def test_settle_interrupted_delegates_to_fingerprint_match(self):
        handler = MagicMock()
        handler._resolves = None
        handler._request_fingerprint = "fp-current"
        handler._generation_id = "gen_current"
        tracker = MagicMock()
        LLMBridgeHandler._settle_interrupted_work(handler, tracker, None, "m")
        tracker.resolve_matching.assert_called_once_with(
            "fp-current", by="gen_current", model="m"
        )

    def test_settle_interrupted_logs_failures(self):
        handler = MagicMock()
        handler._resolves = None
        handler._request_fingerprint = "fp-current"
        tracker = MagicMock()
        tracker.resolve_matching.side_effect = OSError("disk")
        LLMBridgeHandler._settle_interrupted_work(handler, tracker, None, "p")
        handler.log_error.assert_called_once()

    def test_settle_interrupted_ignores_missing_tracker(self):
        handler = MagicMock()
        handler._resolves = None
        handler._request_fingerprint = "fp-current"
        LLMBridgeHandler._settle_interrupted_work(handler, None, None, "p")
        handler.log_error.assert_not_called()

    def test_settle_interrupted_without_a_fingerprint_is_a_no_op(self):
        handler = MagicMock()
        handler._resolves = None
        handler._request_fingerprint = ""
        tracker = MagicMock()
        LLMBridgeHandler._settle_interrupted_work(handler, tracker, None, "p")
        tracker.resolve_matching.assert_not_called()

    def test_resume_settles_its_source_by_id(self):
        handler = MagicMock()
        handler._resolves = MagicMock(id="gen_dead")
        handler._request_fingerprint = ""
        handler._work_ledger.return_value = MagicMock()
        tracker = MagicMock()
        child = MagicMock()
        child.id = "gen_retry"
        child.status = "completed"
        LLMBridgeHandler._settle_interrupted_work(handler, tracker, child, "m")
        handler._work_ledger.return_value.resume.assert_called_once_with(
            handler._resolves, by="gen_retry", model="m"
        )
        tracker.resolve_matching.assert_not_called()

    def test_unfinished_resume_leaves_the_source_recoverable(self):
        handler = MagicMock()
        handler._resolves = MagicMock()
        handler._request_fingerprint = ""
        tracker = MagicMock()
        child = MagicMock()
        child.status = "interrupted"
        LLMBridgeHandler._settle_interrupted_work(handler, tracker, child, "m")
        tracker.resolve_matching.assert_not_called()
        handler._work_ledger.assert_not_called()
        handler.log_error.assert_not_called()

    def test_different_request_does_not_resolve_saved_work(self):
        state = self.tracker.save(prompt="Hello", model_name="nemotron", max_tokens=100)
        handler = MagicMock()
        handler._request_fingerprint = _fingerprint(
            {
                "model": "nemotron",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Something else"}],
                "stream": False,
            }
        )
        handler._resolves = None
        LLMBridgeHandler._settle_interrupted_work(handler, self.tracker, None, "nemotron")
        self.assertEqual(
            [record.id for record in self.ledger.interrupted()], [state["generation_id"]]
        )

    def test_identical_request_resolves_its_own_dead_run(self):
        state = self.tracker.save(
            prompt="Hello",
            model_name="nemotron",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
        handler = MagicMock()
        # The same body the bridge recorded when the run died.
        handler._request_fingerprint = _fingerprint(
            {
                "model": "nemotron",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "max_tokens": 100,
            }
        )
        handler._generation_id = "gen_the_retry"
        handler._resolves = None
        LLMBridgeHandler._settle_interrupted_work(handler, self.tracker, None, "nemotron")
        self.assertEqual(self.ledger.interrupted(), [])
        record = self.ledger.load(state["generation_id"])
        self.assertEqual(record.status, "resumed")
        self.assertEqual(record.events[-1].data["resumed_by"], "gen_the_retry")


class StreamResponseUnitTests(unittest.TestCase):
    def test_client_disconnect_persists_partial_output(self):
        class _TextStream:
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n'

        handler = MagicMock()
        # message_start, ping, content_block_start, then the text delta breaks
        handler.wfile.write.side_effect = [
            None,
            None,
            None,
            BrokenPipeError("client gone"),
        ]
        LLMBridgeHandler._stream_response(
            handler,
            _TextStream(),
            "nemotron",
            MagicMock(),
            "prompt",
            None,
            max_tokens=100,
        )

        handler._save_interrupted.assert_called_once()
        _, kwargs = handler._save_interrupted.call_args
        self.assertEqual(kwargs["partial_output"], "Hello")
        self.assertGreater(kwargs["progress"], 0.0)

    def test_handler_declares_client_socket_timeout(self):
        self.assertEqual(LLMBridgeHandler.timeout, 300)

    def test_client_timeout_honours_environment(self):
        with patch.dict(os.environ, {"LLMBRIDGE_CLIENT_TIMEOUT_SECONDS": "45"}):
            self.assertEqual(_client_timeout_seconds(), 45.0)
            handler = object.__new__(LLMBridgeHandler)
            handler.request = MagicMock()
            LLMBridgeHandler.setup(handler)
            self.assertEqual(handler.timeout, 45.0)

    def test_client_timeout_ignores_invalid_values(self):
        for value in ("0", "-1", "abc"):
            with patch.dict(os.environ, {"LLMBRIDGE_CLIENT_TIMEOUT_SECONDS": value}):
                self.assertEqual(_client_timeout_seconds(), 300)

    def test_upstream_timeout_honours_environment(self):
        with patch.dict(os.environ, {"LLMBRIDGE_UPSTREAM_TIMEOUT_SECONDS": "12.5"}):
            self.assertEqual(_upstream_timeout_seconds(), 12.5)

    def test_upstream_timeout_ignores_invalid_values(self):
        for value in ("0", "-3", "nope"):
            with patch.dict(os.environ, {"LLMBRIDGE_UPSTREAM_TIMEOUT_SECONDS": value}):
                self.assertEqual(_upstream_timeout_seconds(), 180)

    def test_broken_pipe_during_request_is_swallowed(self):
        handler = object.__new__(LLMBridgeHandler)
        handler.close_connection = False
        handler.log_error = MagicMock()
        with patch.object(
            BaseHTTPRequestHandler, "handle_one_request", side_effect=BrokenPipeError
        ):
            LLMBridgeHandler.handle_one_request(handler)
        self.assertTrue(handler.close_connection)
        handler.log_error.assert_called_once()

    def test_connection_reset_during_request_is_swallowed(self):
        handler = object.__new__(LLMBridgeHandler)
        handler.close_connection = False
        handler.log_error = MagicMock()
        with patch.object(
            BaseHTTPRequestHandler, "handle_one_request", side_effect=ConnectionResetError
        ):
            LLMBridgeHandler.handle_one_request(handler)
        self.assertTrue(handler.close_connection)

    def test_other_errors_still_propagate(self):
        handler = object.__new__(LLMBridgeHandler)
        handler.close_connection = False
        handler.log_error = MagicMock()
        with patch.object(
            BaseHTTPRequestHandler, "handle_one_request", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                LLMBridgeHandler.handle_one_request(handler)


class ShutdownSignalTests(unittest.TestCase):
    def test_sigterm_is_routed_through_keyboard_interrupt(self):
        import signal

        from llmbridge.server import install_shutdown_signals

        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        install_shutdown_signals()
        handler = signal.getsignal(signal.SIGTERM)
        self.assertTrue(callable(handler))
        # The same path SIGINT takes, so main()'s finally still runs stop().
        with self.assertRaises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)


class ServerSettingsTests(unittest.TestCase):
    def setUp(self):
        self._routes = {
            "nemotron": ModelRoute("http://upstream.test/v1", model="nvidia/nemotron")
        }
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _load(self, env):
        env = {"LLMBRIDGE_LEDGER_PATH": str(Path(self._tmp.name) / "generations"), **env}
        with patch("llmbridge.server.load_model_routes", return_value=self._routes):
            with patch.dict(os.environ, env, clear=True):
                return load_server_settings()

    def test_defaults_to_loopback_and_enables_tracking(self):
        settings = self._load({})
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8787)
        self.assertIsNone(settings.client_api_key)
        self.assertIsNotNone(settings.work_tracker)
        self.assertIsNotNone(settings.ledger)

    def test_reads_host_port_and_client_key(self):
        settings = self._load(
            {
                "LLMBRIDGE_HOST": "127.0.0.2",
                "LLMBRIDGE_PORT": "9999",
                "LLMBRIDGE_API_KEY": "x" * 32,
            }
        )
        self.assertEqual(settings.host, "127.0.0.2")
        self.assertEqual(settings.port, 9999)
        self.assertEqual(settings.client_api_key, "x" * 32)

    def test_tracking_can_be_disabled(self):
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value):
                settings = self._load({"LLMBRIDGE_TRACK_INTERRUPTED": value})
                self.assertIsNone(settings.work_tracker)

    def test_generation_tracking_can_be_disabled(self):
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value):
                settings = self._load({"LLMBRIDGE_TRACK_GENERATIONS": value})
                self.assertIsNone(settings.ledger)

    def test_custom_ledger_path_is_used(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ledger"
            settings = self._load({"LLMBRIDGE_LEDGER_PATH": str(path)})
            self.assertIsNotNone(settings.ledger)
            self.assertEqual(settings.ledger.root, path)

    def test_custom_partial_work_path_is_no_longer_used(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            settings = self._load({"LLMBRIDGE_PARTIAL_WORK_PATH": str(path)})
            self.assertIsNotNone(settings.work_tracker)
            # Interrupted work lives in the ledger, not a partial-work file.
            self.assertEqual(
                settings.work_tracker.ledger.root, Path(self._tmp.name) / "generations"
            )

    def test_work_tracker_shares_the_server_ledger(self):
        settings = self._load({})
        self.assertIsNotNone(settings.work_tracker)
        self.assertIsNotNone(settings.ledger)
        self.assertIs(settings.work_tracker.ledger, settings.ledger)

    def test_interrupted_tracking_without_generations_uses_own_ledger(self):
        settings = self._load({"LLMBRIDGE_TRACK_GENERATIONS": "0"})
        self.assertIsNone(settings.ledger)
        self.assertIsNotNone(settings.work_tracker)
        self.assertEqual(
            settings.work_tracker.ledger.root, Path(self._tmp.name) / "generations"
        )

    def test_ledger_retention_cap_is_env_configurable(self):
        settings = self._load({"LLMBRIDGE_LEDGER_MAX_RECORDS": "3"})
        self.assertEqual(settings.ledger.max_records, 3)

    def test_invalid_ledger_retention_cap_falls_back_to_default(self):
        settings = self._load({"LLMBRIDGE_LEDGER_MAX_RECORDS": "nope"})
        self.assertEqual(settings.ledger.max_records, MAX_TRACKED_INTERRUPTED)

    def test_non_loopback_without_key_exits(self):
        with self.assertRaisesRegex(SystemExit, "LLMBRIDGE_API_KEY is required"):
            self._load({"LLMBRIDGE_HOST": "0.0.0.0"})

    def test_routing_error_becomes_system_exit(self):
        with patch(
            "llmbridge.server.load_model_routes", side_effect=RoutingError("bad config")
        ):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(SystemExit, "bad config"):
                    load_server_settings()


class PromptExtractionTests(unittest.TestCase):
    def test_extracts_last_user_text_and_system(self):
        body = {
            "system": [{"type": "text", "text": "be terse"}],
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ],
        }
        prompt, system = _extract_prompt_and_system(body)
        self.assertEqual(prompt, "second")
        self.assertEqual(system, "be terse")

    def test_missing_fields_yield_empty_prompt(self):
        prompt, system = _extract_prompt_and_system({})
        self.assertEqual(prompt, "")
        self.assertIsNone(system)


class CountTokensUnitTests(unittest.TestCase):
    def _run(self, content_length, body=b""):
        handler = MagicMock()
        handler.headers = {"Content-Length": str(content_length)}
        handler.rfile = io.BytesIO(body)
        handler.server = SimpleNamespace(
            model_routes={
                "nemotron": ModelRoute("http://upstream.test/v1", model="nemotron")
            }
        )
        LLMBridgeHandler._count_tokens(handler)
        return handler

    def test_rejects_body_outside_size_bounds(self):
        for size in (0, 11 * 1024 * 1024):
            with self.subTest(size=size):
                handler = self._run(size)
                handler._send_error.assert_called_once()
                self.assertEqual(handler._send_error.call_args.args[0], 400)
                self.assertIn("MiB", handler._send_error.call_args.args[2])

    def test_rejects_malformed_body_variants(self):
        cases = {
            "must include a messages list": b"[1, 2]",
            "model must be a non-empty string": b'{"model": 5, "messages": []}',
        }
        for message, body in cases.items():
            with self.subTest(message=message):
                handler = self._run(len(body), body)
                handler._send_error.assert_called_once()
                self.assertEqual(handler._send_error.call_args.args[0], 400)
                self.assertIn(message, handler._send_error.call_args.args[2])

    def test_rejects_invalid_json(self):
        body = b"{oops"
        handler = self._run(len(body), body)
        self.assertEqual(handler._send_error.call_args.args[0], 400)

    def test_rejects_empty_messages(self):
        body = b'{"model": "nemotron"}'
        handler = self._run(len(body), body)
        self.assertEqual(handler._send_error.call_args.args[0], 400)

    def test_rejects_unknown_model(self):
        body = json.dumps({"model": "missing", "messages": []}).encode()
        handler = self._run(len(body), body)
        self.assertEqual(handler._send_error.call_args.args[0], 400)

    def test_estimates_input_tokens(self):
        body = json.dumps(
            {"model": "nemotron", "messages": [{"role": "user", "content": "hello world"}]}
        ).encode()
        handler = self._run(len(body), body)
        handler._send_error.assert_not_called()
        handler._send_json.assert_called_once()
        status, payload = handler._send_json.call_args.args
        self.assertEqual(status, 200)
        self.assertGreater(payload["input_tokens"], 0)


class RouteRetrySettingTests(unittest.TestCase):
    def test_attempts_default_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_route_attempts(), 1)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_ATTEMPTS": "3"}):
            self.assertEqual(_route_attempts(), 3)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_ATTEMPTS": "0"}):
            self.assertEqual(_route_attempts(), 1)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_ATTEMPTS": "999"}):
            self.assertEqual(_route_attempts(), 10)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_ATTEMPTS": "lots"}):
            self.assertEqual(_route_attempts(), 1)

    def test_retry_backoff_default_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_route_retry_backoff_seconds(), 0.5)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_RETRY_BACKOFF": "2.5"}):
            self.assertEqual(_route_retry_backoff_seconds(), 2.5)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_RETRY_BACKOFF": "-1"}):
            self.assertEqual(_route_retry_backoff_seconds(), 0.0)
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_RETRY_BACKOFF": "soon"}):
            self.assertEqual(_route_retry_backoff_seconds(), 0.5)

    def test_parse_retry_after_accepts_seconds_and_http_dates(self):
        self.assertEqual(_parse_retry_after("7"), 7.0)
        self.assertEqual(_parse_retry_after(" 12 "), 12.0)
        self.assertIsNone(_parse_retry_after(None))
        self.assertIsNone(_parse_retry_after(""))
        self.assertIsNone(_parse_retry_after("soon"))
        self.assertIsNone(_parse_retry_after("-3"))
        with patch("llmbridge.server.time.time", return_value=784111767.0):
            # RFC 9110 example date is 784111777 in epoch seconds.
            self.assertEqual(
                _parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT"), 10.0
            )
        self.assertEqual(_parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT"), 0.0)

    def test_retry_sleep_waits_at_least_retry_after(self):
        with patch.dict(os.environ, {"LLMBRIDGE_ROUTE_RETRY_BACKOFF": "0.5"}):
            with patch("llmbridge.server.time.sleep") as sleep:
                _retry_sleep(1, retry_after=4.0)
                _retry_sleep(1, retry_after=0.1)
                _retry_sleep(2)
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list], [4.0, 0.5, 1.0]
        )

    def test_retryable_statuses(self):
        self.assertTrue(_is_retryable_status(429))
        self.assertTrue(_is_retryable_status(503))
        self.assertTrue(_is_retryable_status(599))
        self.assertFalse(_is_retryable_status(400))
        self.assertFalse(_is_retryable_status(401))


class FallbackRouteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.tracker = WorkTracker(self.ledger)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "primary": ModelRoute(
                    "http://primary.test/v1",
                    model="primary-model",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                    fallbacks=("backup",),
                ),
                "backup": ModelRoute(
                    "http://backup.test/v1",
                    model="backup-model",
                ),
            },
            work_tracker=self.tracker,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def _post(self, body: dict) -> Request:
        return Request(
            self._url("/v1/messages"),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _ok_response(self, content="from backup"):
        return _UpstreamResponse(
            {
                "id": "x",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            }
        )

    def test_falls_back_after_resumable_primary_error(self):
        quota = HTTPError("http://primary.test", 429, "Too Many", {}, None)
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", side_effect=[quota, self._ok_response()]
            ) as call:
                with urlopen(self._post(body)) as response:
                    result = json.loads(response.read())

        self.assertEqual(result["content"][0]["text"], "from backup")
        urls = [c.args[0].full_url for c in call.call_args_list]
        self.assertEqual(
            urls,
            [
                "http://primary.test/v1/chat/completions",
                "http://backup.test/v1/chat/completions",
            ],
        )
        self.assertEqual(self.ledger.interrupted(), [])

    def test_falls_back_after_non_resumable_primary_error(self):
        bad = HTTPError("http://primary.test", 400, "Bad Request", {}, None)
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", side_effect=[bad, self._ok_response()]
            ):
                with urlopen(self._post(body)) as response:
                    self.assertEqual(response.status, 200)

    def test_untranslatable_primary_reply_falls_back_instead_of_dropping(self):
        # A 200 whose body is not a completion used to raise out of the handler,
        # so the client saw a dropped connection even with a healthy fallback.
        malformed = _UpstreamResponse(
            {
                "id": "x",
                "choices": [
                    {"message": {"content": "hi"}, "finish_reason": "made_up_reason"}
                ],
            }
        )
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen",
                side_effect=[malformed, self._ok_response()],
            ) as call:
                with urlopen(self._post(body)) as response:
                    result = json.loads(response.read())

        self.assertEqual(result["content"][0]["text"], "from backup")
        self.assertEqual(call.call_count, 2)

    def test_untranslatable_reply_on_every_route_is_a_502(self):
        malformed = _UpstreamResponse(
            {"id": "x", "choices": [{"message": {"content": "hi"}, "finish_reason": None}]}
        )
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", side_effect=[malformed, malformed]
            ):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post(body))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

    def test_stream_falls_back_when_primary_open_fails(self):
        quota = HTTPError("http://primary.test", 429, "Too Many", {}, None)
        body = {
            "model": "primary",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen",
                side_effect=[quota, _CompletingStream()],
            ) as call:
                with urlopen(self._post(body)) as response:
                    rendered = response.read().decode("utf-8")

        self.assertIn("event: message_stop", rendered)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(
            call.call_args_list[1].args[0].full_url,
            "http://backup.test/v1/chat/completions",
        )
        self.assertEqual(self.ledger.interrupted(), [])

    def test_all_routes_failing_records_the_last_error(self):
        quota = HTTPError("http://primary.test", 429, "Too Many", {}, None)
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen",
                side_effect=[quota, URLError("refused")],
            ):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post(body))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

        with urlopen(self._url("/v1/interrupted")) as response:
            state = json.loads(response.read())
        self.assertEqual(
            state["error"], "Could not connect to the configured upstream"
        )

    def test_payload_failure_on_primary_falls_through_to_capable_route(self):
        self.server.model_routes["primary"] = ModelRoute(
            "http://primary.test/v1",
            model="primary-model",
            api_key_env="LLMBRIDGE_TEST_API_KEY",
            supports_images=False,
            fallbacks=("backup",),
        )
        body = {
            "model": "primary",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "http://x/i.png"}}
                    ],
                }
            ],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", return_value=self._ok_response()
            ) as call:
                with urlopen(self._post(body)) as response:
                    self.assertEqual(response.status, 200)

        self.assertEqual(call.call_count, 1)
        self.assertEqual(
            call.call_args.args[0].full_url,
            "http://backup.test/v1/chat/completions",
        )

    def test_payload_failure_on_every_route_returns_400(self):
        self.server.model_routes["primary"] = ModelRoute(
            "http://primary.test/v1", model="p", supports_images=False
        )
        self.server.model_routes["backup"] = ModelRoute(
            "http://backup.test/v1", model="b", supports_images=False
        )
        body = {
            "model": "primary",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "http://x/i.png"}}
                    ],
                }
            ],
        }
        with patch("llmbridge.server.urlopen") as call:
            with self.assertRaises(HTTPError) as error:
                urlopen(self._post(body))
        self.assertEqual(error.exception.code, 400)
        body_text = json.loads(error.exception.read())
        self.assertIn("does not support images", body_text["error"]["message"])
        error.exception.close()
        call.assert_not_called()

    def test_retries_the_same_route_before_falling_through(self):
        env = {
            "LLMBRIDGE_TEST_API_KEY": "k",
            "LLMBRIDGE_ROUTE_ATTEMPTS": "2",
            "LLMBRIDGE_ROUTE_RETRY_BACKOFF": "0",
        }
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, env):
            with patch(
                "llmbridge.server.urlopen",
                side_effect=[URLError("flake"), self._ok_response()],
            ) as call:
                with urlopen(self._post(body)) as response:
                    self.assertEqual(response.status, 200)

        urls = [c.args[0].full_url for c in call.call_args_list]
        self.assertEqual(
            urls,
            [
                "http://primary.test/v1/chat/completions",
                "http://primary.test/v1/chat/completions",
            ],
        )


    def _rate_limited(self, retry_after: str):
        return HTTPError(
            "http://primary.test", 429, "Too Many", {"Retry-After": retry_after}, None
        )

    def test_same_route_retry_honours_retry_after(self):
        env = {
            "LLMBRIDGE_TEST_API_KEY": "k",
            "LLMBRIDGE_ROUTE_ATTEMPTS": "2",
            "LLMBRIDGE_ROUTE_RETRY_BACKOFF": "0",
        }
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, env):
            with patch("llmbridge.server.time.sleep") as sleep, patch(
                "llmbridge.server.urlopen",
                side_effect=[self._rate_limited("3"), self._ok_response()],
            ) as call:
                with urlopen(self._post(body)) as response:
                    self.assertEqual(response.status, 200)

        sleep.assert_called_once_with(3.0)
        urls = [c.args[0].full_url for c in call.call_args_list]
        self.assertEqual(urls, ["http://primary.test/v1/chat/completions"] * 2)

    def test_long_retry_after_skips_to_fallback_without_waiting(self):
        env = {
            "LLMBRIDGE_TEST_API_KEY": "k",
            "LLMBRIDGE_ROUTE_ATTEMPTS": "3",
            "LLMBRIDGE_ROUTE_RETRY_BACKOFF": "0",
        }
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, env):
            with patch("llmbridge.server.time.sleep") as sleep, patch(
                "llmbridge.server.urlopen",
                side_effect=[self._rate_limited("600"), self._ok_response()],
            ) as call:
                with urlopen(self._post(body)) as response:
                    self.assertEqual(response.status, 200)

        sleep.assert_not_called()
        urls = [c.args[0].full_url for c in call.call_args_list]
        self.assertEqual(
            urls,
            [
                "http://primary.test/v1/chat/completions",
                "http://backup.test/v1/chat/completions",
            ],
        )

    def test_streaming_retry_honours_retry_after(self):
        env = {
            "LLMBRIDGE_TEST_API_KEY": "k",
            "LLMBRIDGE_ROUTE_ATTEMPTS": "2",
            "LLMBRIDGE_ROUTE_RETRY_BACKOFF": "0",
        }
        body = {
            "model": "primary",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with patch.dict(os.environ, env):
            with patch("llmbridge.server.time.sleep") as sleep, patch(
                "llmbridge.server.urlopen",
                side_effect=[self._rate_limited("2")] + [URLError("down")] * 3,
            ):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post(body))
                error.exception.close()

        self.assertIn(2.0, [c.args[0] for c in sleep.call_args_list])


class ResumeInterruptedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.tracker = WorkTracker(self.ledger)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                ),
                "other": ModelRoute("http://other.test/v1", model="other-model"),
            },
            work_tracker=self.tracker,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def _resume(self, body: dict | None = None) -> Request:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        return Request(
            self._url("/v1/interrupted/resume"),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _ok_response(self):
        return _UpstreamResponse(
            {
                "id": "x",
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            }
        )

    def test_resume_without_saved_work_returns_404(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self._resume({}))
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def _use_ledger_backed_server(self):
        """Rebuild the server with the ledger attached (lineage + GET /v1/generations)."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                ),
                "other": ModelRoute("http://other.test/v1", model="other-model"),
            },
            work_tracker=self.tracker,
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _interrupted_record(self, text="half", prompt="Continue this"):
        body = {
            "model": "nemotron", "max_tokens": 256, "system": "be terse",
            "messages": [{"role": "user", "content": prompt}],
        }
        record = self.ledger.create(body, "nemotron")
        self.ledger.interrupt(record, "Upstream stream timed out", output=text, output_tokens=2)
        return record

    def test_handoff_token_round_trips_and_is_tamper_evident(self):
        from llmbridge.server import mint_handoff_token, verify_handoff_token
        record = self._interrupted_record()
        token = mint_handoff_token(record, secret=b"s" * 16, now=1000.0)
        self.assertNotIn("=", token)
        payload = verify_handoff_token(token, secret=b"s" * 16, now=1001.0)
        self.assertEqual(payload["generation_id"], record.id)
        self.assertEqual(payload["partial_output"], "half")
        self.assertEqual(payload["request"]["messages"][0]["content"], "Continue this")
        self.assertEqual(payload["fingerprint"], record.fingerprint)
        with self.assertRaises(ValueError):
            verify_handoff_token(token, secret=b"t" * 16, now=1001.0)
        head, _, sig = token.partition(".")
        with self.assertRaises(ValueError):
            verify_handoff_token(head + "x." + sig, secret=b"s" * 16, now=1001.0)
        with self.assertRaises(ValueError):
            verify_handoff_token("not-a-token", secret=b"s" * 16)
        with self.assertRaises(ValueError) as expired:
            verify_handoff_token(token, secret=b"s" * 16, now=1000.0 + 25 * 3600)
        self.assertIn("expired", str(expired.exception))
        with patch.dict(os.environ, {"LLMBRIDGE_HANDOFF_TTL_SECONDS": "10"}):
            with self.assertRaises(ValueError):
                verify_handoff_token(token, secret=b"s" * 16, now=1020.0)
        with self.assertRaises(ValueError):
            verify_handoff_token(token, secret=b"s" * 16, now=100.0)  # issued in the future

    def test_get_interrupted_carries_a_token_only_when_a_secret_is_set(self):
        self._use_ledger_backed_server()
        self._interrupted_record()
        with urlopen(self._url("/v1/interrupted")) as response:
            state = json.loads(response.read())
        self.assertNotIn("handoff_token", state)
        with patch.dict(os.environ, {"LLMBRIDGE_HANDOFF_SECRET": "shared-secret-value"}):
            with urlopen(self._url("/v1/interrupted")) as response:
                state = json.loads(response.read())
            self.assertIn("handoff_token", state)
            with urlopen(self._url(f"/v1/generations/{state['generation_id']}")) as response:
                full = json.loads(response.read())
            # Each mint carries its own issue time, so compare what the tokens say.
            from llmbridge.server import verify_handoff_token
            a = verify_handoff_token(state["handoff_token"], secret=b"shared-secret-value")
            b = verify_handoff_token(full["handoff_token"], secret=b"shared-secret-value")
            self.assertEqual(a["generation_id"], b["generation_id"])
            self.assertEqual(a["partial_output"], b["partial_output"])
        with patch.dict(os.environ, {"LLMBRIDGE_HANDOFF_SECRET": "short"}):
            with urlopen(self._url("/v1/interrupted")) as response:
                self.assertNotIn("handoff_token", json.loads(response.read()))

    def test_resume_accepts_a_handoff_token_from_another_bridge(self):
        # Bridge A minted the token; this bridge (B) has never seen gen_remote.
        self._use_ledger_backed_server()
        from llmbridge.ledger import GenerationRecord
        from llmbridge.server import mint_handoff_token
        import time as _time
        remote = GenerationRecord(
            id="gen_remote000001", fingerprint="fp-remote", model="claude-sonnet-4-5",
            prompt="Continue this", system="be terse", params={"max_tokens": 256},
            status="interrupted", created_at=_time.time(), updated_at=_time.time(),
            request={"messages": [{"role": "user", "content": "Continue this"}], "system": "be terse"},
            output="half", output_tokens=2,
        )
        token = mint_handoff_token(remote, secret=b"shared-secret-value")
        env = {"LLMBRIDGE_HANDOFF_SECRET": "shared-secret-value", "LLMBRIDGE_TEST_API_KEY": "k"}
        with patch.dict(os.environ, env):
            with patch("llmbridge.server.urlopen", return_value=self._ok_response()) as call:
                with urlopen(self._resume({"handoff_token": token, "model": "other"})) as response:
                    self.assertEqual(response.status, 200)
                    child_id = response.headers.get("X-LLMBridge-Generation")
        forwarded = json.loads(call.call_args.args[0].data)
        self.assertEqual(forwarded["model"], "other-model")
        self.assertEqual(forwarded["max_tokens"], 256)
        self.assertEqual(
            forwarded["messages"],
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "Continue this"},
                {"role": "assistant", "content": "half"},
            ],
        )
        child = self.ledger.load(child_id)
        self.assertEqual(child.status, "completed")
        self.assertIsNone(child.parent)
        self.assertEqual(child.fingerprint, "fp-remote")
        handoff = next(e for e in child.events if e.type == "handoff")
        self.assertEqual(handoff.data["source_generation"], "gen_remote000001")
        self.assertEqual(handoff.data["source_model"], "claude-sonnet-4-5")
        self.assertEqual(handoff.data["resumed_chars"], 4)
        self.assertEqual(self.ledger.interrupted(), [])  # nothing local was touched

    def test_resume_with_token_but_no_secret_is_a_clear_400(self):
        from llmbridge.server import mint_handoff_token
        token = mint_handoff_token(self._interrupted_record(), secret=b"shared-secret-value")
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("LLMBRIDGE_HANDOFF_SECRET", "LLMBRIDGE_API_KEY")}
        with patch.dict(os.environ, clean, clear=True):
            with self.assertRaises(HTTPError) as error:
                urlopen(self._resume({"handoff_token": token}))
        self.assertEqual(error.exception.code, 400)
        self.assertIn("LLMBRIDGE_HANDOFF_SECRET", error.exception.read().decode())
        error.exception.close()
        with patch.dict(os.environ, {"LLMBRIDGE_HANDOFF_SECRET": "a-different-secret!"}):
            with self.assertRaises(HTTPError) as error:
                urlopen(self._resume({"handoff_token": token}))
        self.assertEqual(error.exception.code, 400)
        self.assertIn("signature", error.exception.read().decode())
        error.exception.close()

    def test_resume_reissues_partial_output_as_assistant_turn(self):
        self.tracker.save(
            prompt="Continue this",
            model_name="nemotron",
            system="be terse",
            partial_output="half",
            max_tokens=256,
        )
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", return_value=self._ok_response()
            ) as call:
                with urlopen(self._resume({})) as response:
                    self.assertEqual(response.status, 200)

        forwarded = json.loads(call.call_args.args[0].data)
        self.assertEqual(forwarded["model"], "nvidia/nemotron")
        self.assertEqual(forwarded["max_tokens"], 256)
        self.assertEqual(
            forwarded["messages"],
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "Continue this"},
                {"role": "assistant", "content": "half"},
            ],
        )
        self.assertEqual(forwarded["stream"], False)
        self.assertEqual(self.ledger.interrupted(), [])

    def test_resume_replays_full_recorded_conversation(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        self.tracker.save(
            prompt="second",
            model_name="nemotron",
            system="be terse",
            partial_output="half",
            max_tokens=256,
            request={
                "messages": history,
                "system": "be terse",
                "tools": [{"name": "search", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "auto"},
            },
        )
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", return_value=self._ok_response()
            ) as call:
                with urlopen(self._resume({})) as response:
                    self.assertEqual(response.status, 200)

        forwarded = json.loads(call.call_args.args[0].data)
        roles = [message["role"] for message in forwarded["messages"]]
        # system + full history + the replayed partial assistant turn
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(forwarded["messages"][-1]["content"], "half")
        self.assertIn("tools", forwarded)
        self.assertEqual(forwarded["tool_choice"], "auto")

    def test_resume_accepts_an_explicit_model_and_max_tokens(self):
        self.tracker.save(
            prompt="Continue", model_name="nemotron", max_tokens=256
        )
        with patch("llmbridge.server.urlopen", return_value=self._ok_response()) as call:
            with urlopen(self._resume({"model": "other", "max_tokens": 42})) as response:
                self.assertEqual(response.status, 200)

        forwarded = json.loads(call.call_args.args[0].data)
        self.assertEqual(forwarded["model"], "other-model")
        self.assertEqual(forwarded["max_tokens"], 42)
        self.assertEqual(
            forwarded["messages"], [{"role": "user", "content": "Continue"}]
        )
        self.assertNotIn("system", forwarded)

    def test_resume_defaults_max_tokens_when_unrecorded(self):
        self.tracker.save(prompt="Continue", model_name="nemotron")
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", return_value=self._ok_response()
            ) as call:
                with urlopen(self._resume({})) as response:
                    self.assertEqual(response.status, 200)
        forwarded = json.loads(call.call_args.args[0].data)
        self.assertEqual(forwarded["max_tokens"], 4096)

    def test_resume_rejects_unknown_model(self):
        self.tracker.save(prompt="Continue", model_name="nemotron")
        with self.assertRaises(HTTPError) as error:
            urlopen(self._resume({"model": "missing"}))
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_resume_rejects_state_without_a_prompt(self):
        self.tracker.save(prompt="", model_name="nemotron")
        with self.assertRaises(HTTPError) as error:
            urlopen(self._resume({}))
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_resume_rejects_malformed_json(self):
        self.tracker.save(prompt="Continue", model_name="nemotron")
        request = Request(
            self._url("/v1/interrupted/resume"),
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_resume_requires_authentication(self):
        self.server.client_api_key = "client-secret"
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(self._resume({}))
            self.assertEqual(error.exception.code, 401)
            error.exception.close()
        finally:
            self.server.client_api_key = None


class _ContinuationStream:
    """Streams a continuation the way a model does when given a prefill."""

    def __init__(self, text: str = " world"):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        yield b"data: " + json.dumps(
            {"choices": [{"delta": {"content": self.text}, "finish_reason": None}]}
        ).encode() + b"\n"
        yield (
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"completion_tokens":3}}\n'
        )

    def close(self):
        pass


class AutoResumeTests(unittest.TestCase):
    """LLMBRIDGE_AUTO_RESUME: a re-sent request continues its interrupted run."""

    BODY = {
        "model": "nemotron",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello world"}],
    }

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.tracker = WorkTracker(self.ledger)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    fallbacks=["other"],
                ),
                "other": ModelRoute("http://other.test/v1", model="other-model"),
            },
            work_tracker=self.tracker,
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def _post(self, body: dict) -> Request:
        return Request(
            self._url("/v1/messages"),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _interrupt(self, body: dict, partial: str = "Hello") -> GenerationRecord:
        """Record an interrupted run of ``body`` that saved ``partial``."""
        record = self.ledger.create(body, body["model"])
        self.ledger.interrupt(record, "Upstream stream timed out", output=partial, output_tokens=2)
        return record

    def _ok(self, text: str = " world"):
        return _UpstreamResponse(
            {
                "id": "x",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )

    def test_off_by_default_reruns_from_scratch(self):
        source = self._interrupt(self.BODY)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMBRIDGE_AUTO_RESUME", None)
            with patch("llmbridge.server.urlopen", return_value=self._ok("fresh")) as call:
                with urlopen(self._post(self.BODY)) as response:
                    result = json.loads(response.read())
                    resumed_header = response.headers.get("X-LLMBridge-Resumed")
        self.assertEqual(result["content"][0]["text"], "fresh")
        self.assertIsNone(resumed_header)
        sent = json.loads(call.call_args.args[0].data)
        self.assertEqual(sent["messages"][-1]["role"], "user")
        # The identical re-send still settles the old record, as before.
        self.assertEqual(self.ledger.load(source.id).status, "resumed")

    def test_non_stream_resume_prepends_saved_text(self):
        source = self._interrupt(self.BODY)
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok(" world")) as call:
                with urlopen(self._post(self.BODY)) as response:
                    result = json.loads(response.read())
                    child_id = response.headers.get("X-LLMBridge-Generation")
                    resumed = response.headers.get("X-LLMBridge-Resumed")
        # The client receives one whole answer, saved prefix first.
        self.assertEqual(result["content"], [{"type": "text", "text": "Hello world"}])
        self.assertEqual(result["usage"]["output_tokens"], 5)
        self.assertEqual(resumed, source.id)
        # The upstream was asked to continue, not to start over.
        sent = json.loads(call.call_args.args[0].data)
        self.assertEqual(sent["messages"][-1], {"role": "assistant", "content": "Hello"})
        # Lineage: the child points at the source, both are settled.
        child = self.ledger.load(child_id)
        self.assertEqual(child.parent, source.id)
        self.assertEqual(child.status, "completed")
        self.assertEqual(child.output, "Hello world")
        self.assertEqual(child.fingerprint, source.fingerprint)
        types = [event.type for event in child.events]
        self.assertEqual(types, ["fork", "heal", "heal_seam", "complete"])
        heal = next(e for e in child.events if e.type == "heal")
        self.assertEqual(heal.data["reason"], "auto_resume")
        self.assertEqual(heal.data["resumed_chars"], 5)
        self.assertEqual(self.ledger.load(source.id).status, "resumed")
        self.assertEqual(self.ledger.interrupted(), [])

    def test_stream_resume_replays_prefix_then_continuation(self):
        body = {**self.BODY, "stream": True}
        source = self._interrupt(body)
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", return_value=_ContinuationStream(" world")):
                with urlopen(self._post(body)) as response:
                    raw = response.read().decode()
                    child_id = response.headers.get("X-LLMBridge-Generation")
                    resumed = response.headers.get("X-LLMBridge-Resumed")
        self.assertEqual(resumed, source.id)
        self.assertEqual(raw.count("event: message_start"), 1)
        self.assertEqual(raw.count("event: content_block_start"), 1)
        self.assertEqual(raw.count("event: message_stop"), 1)
        self.assertNotIn("event: error", raw)
        texts = [
            json.loads(line[6:])["delta"]["text"]
            for line in raw.splitlines()
            if line.startswith("data: ") and '"content_block_delta"' in line
        ]
        self.assertEqual(texts, ["Hello", " world"])
        child = self.ledger.load(child_id)
        self.assertEqual(child.status, "completed")
        self.assertEqual(child.output, "Hello world")
        self.assertIn("heal_seam", [event.type for event in child.events])
        self.assertEqual(self.ledger.load(source.id).status, "resumed")

    def test_different_request_is_not_resumed(self):
        self._interrupt(self.BODY)
        other = {**self.BODY, "messages": [{"role": "user", "content": "Something else"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok("fresh")) as call:
                with urlopen(self._post(other)) as response:
                    result = json.loads(response.read())
                    resumed = response.headers.get("X-LLMBridge-Resumed")
        self.assertEqual(result["content"][0]["text"], "fresh")
        self.assertIsNone(resumed)
        self.assertEqual(json.loads(call.call_args.args[0].data)["messages"][-1]["role"], "user")
        self.assertEqual(len(self.ledger.interrupted()), 1)

    def test_interrupted_record_without_text_runs_fresh(self):
        source = self.ledger.create(self.BODY, "nemotron")
        self.ledger.interrupt(source, "Upstream returned HTTP 429")  # no output saved
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok("fresh")) as call:
                with urlopen(self._post(self.BODY)) as response:
                    result = json.loads(response.read())
        self.assertEqual(result["content"][0]["text"], "fresh")
        self.assertEqual(json.loads(call.call_args.args[0].data)["messages"][-1]["role"], "user")
        self.assertEqual(self.ledger.load(source.id).status, "resumed")

    def test_unreachable_upstream_falls_back_to_the_normal_path(self):
        source = self._interrupt(self.BODY)
        failure = URLError("refused")
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", side_effect=failure) as call:
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post(self.BODY))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()
        # Both routes tried for the resume, then both again for the fresh run.
        self.assertEqual(call.call_count, 4)
        # The original interruption is still offered; a failed resume must not
        # settle work it did not finish.
        self.assertEqual(self.ledger.load(source.id).status, "interrupted")

    def test_uses_fallback_route_when_primary_is_down(self):
        source = self._interrupt(self.BODY)
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request.full_url)
            if "upstream.test" in request.full_url:
                raise URLError("down")
            return self._ok(" world")

        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", side_effect=fake_urlopen):
                with urlopen(self._post(self.BODY)) as response:
                    result = json.loads(response.read())
                    child_id = response.headers.get("X-LLMBridge-Generation")
        self.assertEqual(result["content"][0]["text"], "Hello world")
        self.assertEqual(calls, [
            "http://upstream.test/v1/chat/completions",
            "http://other.test/v1/chat/completions",
        ])
        heal = next(e for e in self.ledger.load(child_id).events if e.type == "heal")
        self.assertEqual(heal.data["to_model"], "other-model")
        self.assertEqual(self.ledger.load(source.id).status, "resumed")

    def test_simultaneous_resends_share_one_resume(self):
        # Audit finding: two identical re-sends arriving together must not
        # both resume the same source and both pay for an upstream call.
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        source = self._interrupt(self.BODY)
        calls = []

        class _SlowOk(_UpstreamResponse):
            def read(self, size=-1):
                _time.sleep(0.3)
                return super().read(size)

        def fake(request, timeout=None):
            calls.append(request.full_url)
            return _SlowOk({
                "id": "x",
                "choices": [{"message": {"content": " world"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            })

        def send():
            with urlopen(self._post(self.BODY)) as response:
                return json.loads(response.read()), dict(response.headers)

        env = {"LLMBRIDGE_AUTO_RESUME": "1", "LLMBRIDGE_COALESCE_REQUESTS": "always"}
        with patch.dict(os.environ, env):
            with patch("llmbridge.server.urlopen", side_effect=fake):
                with ThreadPoolExecutor(2) as pool:
                    a, b = pool.map(lambda _: send(), range(2))
        self.assertEqual(len(calls), 1, "the duplicate must ride on the leader's resume")
        self.assertEqual(a[0]["content"][0]["text"], "Hello world")
        self.assertEqual(b[0]["content"][0]["text"], "Hello world")
        coalesced = [h for h in (a[1], b[1]) if h.get("X-LLMBridge-Coalesced") == "1"]
        self.assertEqual(len(coalesced), 1)
        self.assertEqual(a[1]["X-LLMBridge-Generation"], b[1]["X-LLMBridge-Generation"])
        children = [r for r in self.ledger.list(limit=None) if r.parent == source.id]
        self.assertEqual(len(children), 1)
        self.assertEqual(self.ledger.load(source.id).status, "resumed")

    def test_non_stream_resume_record_carries_the_summed_token_count(self):
        source = self._interrupt(self.BODY)  # output_tokens=2
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok(" world")):
                with urlopen(self._post(self.BODY)) as response:
                    result = json.loads(response.read())
                    child_id = response.headers.get("X-LLMBridge-Generation")
        self.assertEqual(result["usage"]["output_tokens"], 5)
        self.assertEqual(self.ledger.load(child_id).output_tokens, 5)

    def test_non_stream_resume_reports_seam_verdict_in_a_header_when_enabled(self):
        self._interrupt(self.BODY)
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1", "LLMBRIDGE_SEAM_EVENTS": "1"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok(" world")):
                with urlopen(self._post(self.BODY)) as response:
                    self.assertEqual(response.headers.get("X-LLMBridge-Seam"), "clean")
        self._interrupt(self.BODY)
        with patch.dict(os.environ, {"LLMBRIDGE_AUTO_RESUME": "1"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok(" world")):
                with urlopen(self._post(self.BODY)) as response:
                    self.assertIsNone(response.headers.get("X-LLMBridge-Seam"))

    def test_resumable_lookup_prefers_newest_with_text(self):
        older = self._interrupt(self.BODY, partial="old half")
        self.ledger.interrupt(self.ledger.create(self.BODY, "nemotron"), "quota")  # newest, no text
        found = self.ledger.resumable(_fingerprint(self.BODY))
        self.assertEqual(found.id, older.id)
        self.assertIsNone(self.ledger.resumable(""))
        self.assertIsNone(self.ledger.resumable("nope"))


class ResumeLineageTests(unittest.TestCase):
    """With generation tracking on, a resume is a child run, not a deletion."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.tracker = WorkTracker(self.ledger)
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {
                "nemotron": ModelRoute(
                    "http://upstream.test/v1",
                    model="nvidia/nemotron",
                    api_key_env="LLMBRIDGE_TEST_API_KEY",
                ),
            },
            work_tracker=self.tracker,
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def _post(self, path: str, body: dict) -> Request:
        return Request(
            self._url(path),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _ok(self):
        return _UpstreamResponse(
            {
                "id": "x",
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            }
        )

    def _dead_run(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Write a long answer"}],
        }
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=URLError("refused")):
                with self.assertRaises(HTTPError) as failure:
                    urlopen(self._post("/v1/messages", body))
        failure.exception.close()
        with urlopen(self._url("/v1/interrupted")) as response:
            return body, json.loads(response.read())

    def test_resume_is_a_child_of_the_interruption(self):
        body, state = self._dead_run()
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok()):
                with urlopen(
                    self._post("/v1/interrupted/resume", {"model": "nemotron"})
                ) as response:
                    self.assertEqual(response.status, 200)

        source = self.ledger.load(state["generation_id"])
        self.assertEqual(source.status, "resumed")
        self.assertEqual(source.events[-1].type, EVENT_RESUME)
        children = [
            record for record in self.ledger.list(limit=None) if record.parent == source.id
        ]
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].status, "completed")
        self.assertEqual(source.events[-1].data["resumed_by"], children[0].id)
        self.assertEqual(
            children[0].fork_event_seq, max(event.seq for event in source.events) - 1
        )

    def test_resume_stops_offering_the_work_it_finished(self):
        _, state = self._dead_run()
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=self._ok()):
                with urlopen(
                    self._post("/v1/interrupted/resume", {"model": "nemotron"})
                ) as response:
                    response.read()

        self.assertEqual(self.ledger.interrupted(), [])
        with self.assertRaises(HTTPError) as error:
            urlopen(self._url("/v1/interrupted"))
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def test_a_failed_resume_leaves_both_runs_recoverable(self):
        body, state = self._dead_run()
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", side_effect=URLError("refused")):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self._post("/v1/interrupted/resume", {"model": "nemotron"}))
        self.assertEqual(error.exception.code, 502)
        error.exception.close()

        self.assertEqual(self.ledger.load(state["generation_id"]).status, "interrupted")
        pending = self.ledger.interrupted()
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0].parent, state["generation_id"])
        with urlopen(self._url("/v1/interrupted")) as response:
            offered = json.loads(response.read())
        self.assertEqual(offered["recoverable_count"], 2)
        self.assertEqual(offered["generation_id"], pending[0].id)


class CoalescePolicyUnitTests(unittest.TestCase):
    """Which requests may ride on an identical one already in flight."""

    def _mode(self, value):
        if value is None:
            return patch.dict(os.environ, {})
        return patch.dict(os.environ, {"LLMBRIDGE_COALESCE_REQUESTS": value})

    def test_off_by_default(self):
        with self._mode(None):
            os.environ.pop("LLMBRIDGE_COALESCE_REQUESTS", None)
            self.assertFalse(_coalescible({"temperature": 0}, stream=False))

    def test_unknown_values_fail_safe_to_off(self):
        with self._mode("maybe"):
            self.assertFalse(_coalescible({"temperature": 0}, stream=False))

    def test_deterministic_mode_needs_an_explicit_zero_temperature(self):
        with self._mode("deterministic"):
            self.assertTrue(_coalescible({"temperature": 0}, stream=False))
            self.assertFalse(_coalescible({"temperature": 0.5}, stream=False))
            self.assertFalse(_coalescible({}, stream=False))
            self.assertFalse(_coalescible({"temperature": True}, stream=False))

    def test_always_mode_accepts_any_temperature(self):
        with self._mode("1"):
            self.assertTrue(_coalescible({"temperature": 1.0}, stream=False))
            self.assertTrue(_coalescible({}, stream=False))

    def test_streaming_is_never_coalesced(self):
        with self._mode("always"):
            self.assertFalse(_coalescible({"temperature": 0}, stream=True))


class HealBodyTests(unittest.TestCase):
    def test_prefill_is_merged_not_duplicated(self):
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "write it"},
                {"role": "assistant", "content": "def f():"},
            ],
        }
        healed = _build_heal_body(body, "More")
        self.assertEqual(len(healed["messages"]), 2)
        self.assertEqual(healed["messages"][-1]["content"], "def f():More")
        # The client's request is reused by later fallback attempts, so healing
        # must not mutate it.
        self.assertEqual(body["messages"][-1]["content"], "def f():")


class CoalesceTests(unittest.TestCase):
    """Two identical non-streaming requests in flight at once."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))
        self.server = LLMBridgeServer(
            ("127.0.0.1", 0),
            LLMBridgeHandler,
            {"nemotron": ModelRoute("http://upstream.test/v1", model="nemotron")},
            ledger=self.ledger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)

    def _call(self, body):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request) as response:
                headers = {key.lower(): value for key, value in response.headers.items()}
                return response.status, headers, json.loads(response.read())
        except HTTPError as error:
            payload = error.read()
            status = error.code
            headers = {key.lower(): value for key, value in error.headers.items()}
            error.close()
            return status, headers, json.loads(payload)

    def _message(self, temperature=None):
        body = {
            "model": "nemotron",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        return body

    @staticmethod
    def _reply():
        return _UpstreamResponse(
            {
                "id": "chatcmpl-coalesce",
                "choices": [
                    {"message": {"content": "Shared answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )

    def _run_pair(self, body, *, mode, expect_follower=True, failure=None):
        """Send the same body twice while the first is still in flight."""
        release = threading.Event()
        calls: list[int] = []
        calls_lock = threading.Lock()
        first_started = threading.Event()
        second_started = threading.Event()

        def upstream(*_args, **_kwargs):
            with calls_lock:
                calls.append(1)
                index = len(calls)
            (first_started if index == 1 else second_started).set()
            release.wait(5)
            if failure is not None:
                raise failure
            return self._reply()

        results = {}

        def send(name):
            results[name] = self._call(body)

        follower_waiting = threading.Event()
        original = LLMBridgeHandler._serve_coalesced

        def marked(handler, entry):
            follower_waiting.set()
            return original(handler, entry)

        env = {} if mode is None else {"LLMBRIDGE_COALESCE_REQUESTS": mode}
        with patch.dict(os.environ, env):
            if mode is None:
                os.environ.pop("LLMBRIDGE_COALESCE_REQUESTS", None)
            # urlopen is patched inside the server module only, so the test's own
            # client still reaches the real listener.
            with patch.object(LLMBridgeHandler, "_serve_coalesced", marked):
                with patch("llmbridge.server.urlopen", side_effect=upstream):
                    first = threading.Thread(target=send, args=("first",))
                    first.start()
                    self.assertTrue(first_started.wait(5), "first request never started")
                    second = threading.Thread(target=send, args=("second",))
                    second.start()
                    if expect_follower:
                        self.assertTrue(
                            follower_waiting.wait(5), "duplicate never attached"
                        )
                    else:
                        self.assertTrue(
                            second_started.wait(5), "second request never started"
                        )
                    release.set()
                    first.join(10)
                    second.join(10)
        return results, len(calls)

    def test_duplicate_in_flight_request_shares_one_upstream_call(self):
        results, calls = self._run_pair(self._message(), mode="always")
        self.assertEqual(calls, 1)
        self.assertEqual(results["first"][0], 200)
        self.assertEqual(results["second"][0], 200)
        self.assertEqual(
            results["first"][2]["content"], results["second"][2]["content"]
        )
        self.assertIn("x-llmbridge-coalesced", results["second"][1])
        self.assertEqual(results["second"][1]["x-llmbridge-coalesced"], "1")
        self.assertIn("x-llmbridge-generation", results["second"][1])
        self.assertNotIn("x-llmbridge-coalesced", results["first"][1])
        # The duplicate is an answer, not another generation.
        self.assertEqual(len(self.ledger.list(limit=None)), 1)

    def test_coalescing_is_off_by_default(self):
        results, calls = self._run_pair(
            self._message(), mode=None, expect_follower=False
        )
        self.assertEqual(calls, 2)
        self.assertEqual(results["first"][0], 200)
        self.assertEqual(results["second"][0], 200)
        self.assertNotIn("x-llmbridge-coalesced", results["second"][1])
        self.assertEqual(len(self.ledger.list(limit=None)), 2)

    def test_deterministic_mode_waits_for_an_explicit_zero(self):
        _, calls = self._run_pair(
            self._message(temperature=1.0),
            mode="deterministic",
            expect_follower=False,
        )
        self.assertEqual(calls, 2)
        _, calls = self._run_pair(
            self._message(temperature=0), mode="deterministic"
        )
        self.assertEqual(calls, 1)

    def test_a_failed_leader_replays_its_error_to_the_duplicate(self):
        failure = HTTPError(
            "http://upstream.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b""),
        )
        results, calls = self._run_pair(
            self._message(), mode="always", failure=failure
        )
        self.assertEqual(calls, 1)
        self.assertEqual(results["first"][0], 502)
        self.assertEqual(results["second"][0], results["first"][0])
        self.assertEqual(
            results["second"][2]["error"]["message"],
            results["first"][2]["error"]["message"],
        )
        records = self.ledger.list(limit=None)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "interrupted")


if __name__ == "__main__":
    unittest.main()


class HealHelperUnitTests(unittest.TestCase):
    """Pure-function coverage for the tool-heal, shadow, and ranking helpers."""

    def _sse(self, event_type, payload):
        from llmbridge.server import _sse_bytes
        return _sse_bytes(event_type, {"type": event_type, **payload})

    def test_line_has_token_recognises_text_tool_and_finish(self):
        from llmbridge.server import _line_has_token
        self.assertTrue(_line_has_token(b'data: {"choices":[{"delta":{"content":"x"}}]}\n'))
        self.assertTrue(_line_has_token(b'data: {"choices":[{"delta":{"tool_calls":[{}]}}]}\n'))
        self.assertTrue(_line_has_token(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'))
        self.assertFalse(_line_has_token(b'data: {"choices":[{"delta":{"content":""}}]}\n'))
        self.assertFalse(_line_has_token(b'data: [DONE]\n'))
        self.assertFalse(_line_has_token(b': keepalive\n'))
        self.assertFalse(_line_has_token(b'data: not json\n'))
        self.assertFalse(_line_has_token(b'data: [1,2]\n'))
        self.assertFalse(_line_has_token(b'data: {"choices":["bad"]}\n'))

    def test_primed_stream_replays_buffer_then_live_lines(self):
        from llmbridge.server import _PrimedStream

        class _Live:
            def __init__(self):
                self.closed = False
                self._lines = iter([b"c\n", b"d\n"])
            def readline(self, limit=-1):
                return next(self._lines, b"")
            def close(self):
                self.closed = True

        live = _Live()
        with _PrimedStream(live, [b"a\n", b"b\n"]) as primed:
            self.assertEqual([primed.readline() for _ in range(5)], [b"a\n", b"b\n", b"c\n", b"d\n", b""])
        self.assertTrue(live.closed)

    def test_primed_stream_close_swallows_errors(self):
        from llmbridge.server import _PrimedStream

        class _Bad:
            def readline(self, limit=-1):
                return b""
            def close(self):
                raise OSError("already closed")

        _PrimedStream(_Bad(), []).close()  # must not raise

    def test_racer_reports_connect_errors_and_cancellation(self):
        from llmbridge.server import _Racer
        route = ModelRoute("http://x.test/v1")
        with patch("llmbridge.server.urlopen", side_effect=URLError("down")):
            racer = _Racer(route, "m", Request("http://x.test/v1/chat/completions")).start()
            racer.ready.wait(2)
        self.assertIsInstance(racer.error, URLError)
        self.assertFalse(racer.has_token)
        racer.close()  # upstream is None: nothing to close, must not raise

        class _Lines:
            closed = False
            def __init__(self):
                self._it = iter([b": ping\n", b'data: {"choices":[{"delta":{"content":"t"}}]}\n', b"more\n"])
            def readline(self, limit=-1):
                return next(self._it, b"")
            def close(self):
                self.closed = True

        upstream = _Lines()
        with patch("llmbridge.server.urlopen", return_value=upstream):
            racer = _Racer(route, "m", Request("http://x.test/v1/chat/completions")).start()
            racer.ready.wait(2)
        self.assertTrue(racer.has_token)
        self.assertEqual(len(racer.buffered), 2)  # stopped at the first token line
        racer.close()
        self.assertTrue(upstream.closed)

    def test_hold_and_release_tool_events(self):
        from llmbridge.server import (
            _hold_tool_event, _new_tool_state, _release_tool_block, _split_held_tools,
            _tool_block_complete,
        )
        state = _new_tool_state()
        start = self._sse("content_block_start", {"index": 1, "content_block": {"type": "tool_use", "id": "c1", "name": "bash", "input": {}}})
        text_start = self._sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
        d1 = self._sse("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'}})
        d2 = self._sse("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": ' "ls"}'}})
        other = self._sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}})
        self.assertFalse(_hold_tool_event(text_start, state))
        self.assertTrue(_hold_tool_event(start, state))
        self.assertTrue(_hold_tool_event(d1, state))
        self.assertFalse(_hold_tool_event(other, state))
        self.assertFalse(_hold_tool_event(b"garbage", state))
        done, cut = _split_held_tools(state)
        self.assertEqual([i for i, _ in cut], [1])
        self.assertEqual(done, [])
        self.assertTrue(_hold_tool_event(d2, state))
        done, cut = _split_held_tools(state)
        self.assertEqual([i for i, _ in done], [1])
        self.assertTrue(_tool_block_complete(state["held"][1]))
        released = _release_tool_block(state, 1)
        self.assertEqual(released, [start, d1, d2])
        self.assertEqual(_release_tool_block(state, 1), [])
        self.assertEqual(state["order"], [])

    def test_tool_block_complete_rejects_empty_and_non_object_json(self):
        from llmbridge.server import _tool_block_complete
        self.assertFalse(_tool_block_complete({"json": ""}))
        self.assertFalse(_tool_block_complete({"json": "   "}))
        self.assertFalse(_tool_block_complete({"json": "[1]"}))
        self.assertFalse(_tool_block_complete({"json": '{"a":'}))
        self.assertTrue(_tool_block_complete({"json": '{"a": 1}'}))

    def test_tool_heal_hint_and_system_addendum_shapes(self):
        from llmbridge.server import _build_tool_heal_body, _tool_heal_hint, _with_system_addendum
        hint = _tool_heal_hint("", [{"id": "c1", "name": "read", "input": "{}"}],
                               [{"name": "bash", "json": '{"cmd":'}, {"name": None, "json": ""}])
        self.assertIn("read id=c1", hint)
        self.assertIn("calling tool `bash`", hint)
        self.assertIn("beginning a call to tool `<unknown>`", hint)
        self.assertEqual(_with_system_addendum({"system": "A"}, "B")["system"], "A\n\nB")
        self.assertEqual(_with_system_addendum({"system": [{"type": "text", "text": "A"}]}, "B")["system"][-1]["text"], "B")
        self.assertEqual(_with_system_addendum({}, "B")["system"], "B")
        self.assertIsNone(_build_tool_heal_body("nope", "", [], []))
        self.assertIsNone(_build_tool_heal_body({"messages": "nope"}, "", [], []))
        body = _build_tool_heal_body({"messages": [{"role": "user", "content": "q"}]}, "partial", [], [{"name": "bash", "json": ""}])
        self.assertEqual(body["messages"][-1], {"role": "assistant", "content": "partial"})
        self.assertIn("bash", body["system"])

    def test_heal_text_index_and_block_tracking(self):
        from llmbridge.server import _heal_text_index, _track_open_blocks
        state = {"open": {}, "finished": False, "seen": set()}
        _track_open_blocks(b"garbage", state)
        _track_open_blocks(self._sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}), state)
        self.assertEqual(_heal_text_index(state), 0)
        _track_open_blocks(self._sse("content_block_start", {"index": 1, "content_block": "weird"}), state)
        self.assertIsNone(_heal_text_index(state))  # unknown block type blocks a text heal
        _track_open_blocks(self._sse("content_block_stop", {"index": 1}), state)
        self.assertEqual(_heal_text_index(state), 0)
        _track_open_blocks(self._sse("message_stop", {}), state)
        self.assertIsNone(_heal_text_index(state))
        self.assertEqual(state["seen"], {0, 1})
        self.assertIsNone(_heal_text_index({"open": {}, "finished": False}))

    def test_env_parsers_fall_back_on_bad_input(self):
        from llmbridge.server import (
            _adaptive_fallbacks_enabled, _adaptive_min_samples, _heal_tool_calls_enabled,
            _shadow_after_seconds,
        )
        with patch.dict(os.environ, {"LLMBRIDGE_ADAPTIVE_MIN_SAMPLES": "x", "LLMBRIDGE_SHADOW_AFTER_SECONDS": "nope"}):
            self.assertEqual(_adaptive_min_samples(), 3)
            self.assertEqual(_shadow_after_seconds(), 0.0)
        with patch.dict(os.environ, {"LLMBRIDGE_ADAPTIVE_MIN_SAMPLES": "0", "LLMBRIDGE_SHADOW_AFTER_SECONDS": "-2"}):
            self.assertEqual(_adaptive_min_samples(), 3)
            self.assertEqual(_shadow_after_seconds(), 0.0)
        with patch.dict(os.environ, {"LLMBRIDGE_ADAPTIVE_MIN_SAMPLES": "7", "LLMBRIDGE_SHADOW_AFTER_SECONDS": "1.5",
                                     "LLMBRIDGE_ADAPTIVE_FALLBACKS": "yes", "LLMBRIDGE_HEAL_TOOL_CALLS": "0"}):
            self.assertEqual(_adaptive_min_samples(), 7)
            self.assertEqual(_shadow_after_seconds(), 1.5)
            self.assertTrue(_adaptive_fallbacks_enabled())
            self.assertFalse(_heal_tool_calls_enabled())

    def test_rank_handles_empty_quality(self):
        from llmbridge.server import rank_heal_candidates
        r = ModelRoute("http://x/v1")
        ranked, note = rank_heal_candidates([(r, "a"), (r, "b")], {})
        self.assertEqual([m for _, m in ranked], ["a", "b"])
        self.assertEqual([n["samples"] for n in note], [0, 0])
