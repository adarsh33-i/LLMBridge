import io
import json
import os
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from llmbridge.routing import ModelRoute
from llmbridge.server import (
    LLMBridgeHandler,
    LLMBridgeServer,
    _is_loopback_host,
    _validate_listener_auth,
)


class _UpstreamResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


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


if __name__ == "__main__":
    unittest.main()
