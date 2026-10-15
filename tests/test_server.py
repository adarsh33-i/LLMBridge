import io
import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from llmbridge.routing import ModelRoute, RoutingError
from llmbridge.server import (
    LLMBridgeHandler,
    LLMBridgeServer,
    _extract_prompt_and_system,
    _is_loopback_host,
    _is_retryable_status,
    _route_attempts,
    _route_retry_backoff_seconds,
    _validate_listener_auth,
    load_server_settings,
)
from llmbridge.work_tracker import WorkTracker


class _UpstreamResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size=-1):
        return self.body[:size] if size is not None and size >= 0 else self.body


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
        self.state_path = Path(self._tmp.name) / ".partial_work.json"
        self.tracker = WorkTracker(storage_path=str(self.state_path))
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
        self.assertFalse(self.state_path.exists())

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
        self.assertFalse(self.state_path.exists())

    def test_successful_request_clears_matching_interrupted_work(self):
        self.tracker.start_tracking(prompt="Hello", model_name="nemotron")
        self.tracker.save_interrupted()

        upstream_response = _UpstreamResponse(
            {
                "id": "x",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
        )
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=upstream_response):
                with urlopen(self._post_message(body)) as response:
                    self.assertEqual(response.status, 200)

        self.assertFalse(self.state_path.exists())

    def test_successful_request_keeps_unrelated_interrupted_work(self):
        self.tracker.start_tracking(prompt="Some earlier work", model_name="nemotron")
        self.tracker.save_interrupted()

        upstream_response = _UpstreamResponse(
            {
                "id": "x",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
        )
        body = {"model": "nemotron", "max_tokens": 100, "messages": [{"role": "user", "content": "Something else"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch("llmbridge.server.urlopen", return_value=upstream_response):
                with urlopen(self._post_message(body)) as response:
                    self.assertEqual(response.status, 200)

        status, state = self._get_interrupted()
        self.assertEqual(status, 200)
        self.assertEqual(state["prompt"], "Some earlier work")

    def test_delete_clears_interrupted_work(self):
        self.tracker.start_tracking(prompt="Hello", model_name="nemotron")
        self.tracker.save_interrupted()

        with urlopen(Request(self._url("/v1/interrupted"), method="DELETE")) as response:
            result = json.loads(response.read())
        self.assertTrue(result["cleared"])
        self.assertFalse(self.state_path.exists())

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

    def test_empty_prompt_does_not_clear_saved_work(self):
        self.tracker.save(prompt="Hello", model_name="nemotron")
        LLMBridgeHandler._clear_interrupted_on_match(object(), self.tracker, "")
        self.assertTrue(self.state_path.exists())

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
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, size=-1):
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

    def test_clear_interrupted_ignores_load_failures(self):
        handler = MagicMock()
        tracker = MagicMock()
        tracker.load_state.side_effect = OSError("disk")
        LLMBridgeHandler._clear_interrupted_on_match(handler, tracker, "p")
        tracker.clear_state.assert_not_called()

    def test_clear_interrupted_logs_clear_failures(self):
        handler = MagicMock()
        tracker = MagicMock()
        tracker.load_state.return_value = {"prompt": "p"}
        tracker.clear_state.side_effect = OSError("disk")
        LLMBridgeHandler._clear_interrupted_on_match(handler, tracker, "p")
        handler.log_error.assert_called_once()

    def test_clear_interrupted_ignores_missing_tracker(self):
        handler = MagicMock()
        LLMBridgeHandler._clear_interrupted_on_match(handler, None, "p")
        handler.log_error.assert_not_called()

    def test_mismatched_prompt_does_not_clear_saved_work(self):
        self.tracker.save(prompt="Hello", model_name="nemotron")
        handler = MagicMock()
        LLMBridgeHandler._clear_interrupted_on_match(handler, self.tracker, "Other")
        self.assertTrue(self.state_path.exists())


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


class ServerSettingsTests(unittest.TestCase):
    def setUp(self):
        self._routes = {
            "nemotron": ModelRoute("http://upstream.test/v1", model="nvidia/nemotron")
        }

    def _load(self, env):
        with patch("llmbridge.server.load_model_routes", return_value=self._routes):
            with patch.dict(os.environ, env, clear=True):
                return load_server_settings()

    def test_defaults_to_loopback_and_enables_tracking(self):
        settings = self._load({})
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8787)
        self.assertIsNone(settings.client_api_key)
        self.assertIsNotNone(settings.work_tracker)

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

    def test_custom_partial_work_path_is_used(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            settings = self._load({"LLMBRIDGE_PARTIAL_WORK_PATH": str(path)})
            self.assertIsNotNone(settings.work_tracker)
            self.assertEqual(settings.work_tracker.storage_path, path)

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
        self.state_path = Path(self._tmp.name) / ".partial_work.json"
        self.tracker = WorkTracker(storage_path=str(self.state_path))
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
        self.assertFalse(self.state_path.exists())

    def test_falls_back_after_non_resumable_primary_error(self):
        bad = HTTPError("http://primary.test", 400, "Bad Request", {}, None)
        body = {"model": "primary", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}
        with patch.dict(os.environ, {"LLMBRIDGE_TEST_API_KEY": "k"}):
            with patch(
                "llmbridge.server.urlopen", side_effect=[bad, self._ok_response()]
            ):
                with urlopen(self._post(body)) as response:
                    self.assertEqual(response.status, 200)

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
        self.assertFalse(self.state_path.exists())

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


class ResumeInterruptedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / ".partial_work.json"
        self.tracker = WorkTracker(storage_path=str(self.state_path))
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
        self.assertFalse(self.state_path.exists())

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


if __name__ == "__main__":
    unittest.main()
