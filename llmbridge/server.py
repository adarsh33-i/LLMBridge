import hmac
import ipaddress
import json
import os
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .protocol import (
    ProtocolError,
    anthropic_to_openai,
    openai_to_anthropic,
    stream_events,
)
from .routing import ModelRoute, RoutingError, load_model_routes, resolve_model_route
from .work_tracker import WorkTracker

# Upstream statuses that usually indicate quota exhaustion, rate limiting, an
# overloaded backend, or a transient server fault — i.e. work worth resuming
# with a different model rather than a client mistake.
_RESUMABLE_HTTP_STATUS = frozenset({402, 403, 408, 429, 500, 502, 503, 504, 529})


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _block_text(content: Any) -> str:
    """Flatten Anthropic message/system content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_prompt_and_system(body: dict[str, Any]) -> tuple[str, str | None]:
    """Pull the last user prompt and the system text out of an Anthropic request."""
    system = body.get("system")
    if isinstance(system, str):
        system_text: str | None = system or None
    elif isinstance(system, list):
        system_text = _block_text(system) or None
    else:
        system_text = None

    prompt = ""
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                prompt = _block_text(message.get("content"))
                if prompt:
                    break
    return prompt, system_text


def _sse_event_data(event: bytes) -> dict[str, Any] | None:
    """Extract the JSON payload from an SSE event emitted by stream_events()."""
    try:
        decoded = event.decode("utf-8")
    except UnicodeDecodeError:
        return None
    payload = None
    for line in decoded.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            break
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _event_output_tokens(event: bytes) -> int | None:
    """Return the exact output-token count reported by a message_delta event."""
    data = _sse_event_data(event)
    if not data or data.get("type") != "message_delta":
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("output_tokens")
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
        return tokens
    return None


def _collect_text_delta(event: bytes, partial_chunks: list[str]) -> None:
    """Append streamed assistant text from an SSE event to the partial output."""
    data = _sse_event_data(event)
    if not data or data.get("type") != "content_block_delta":
        return
    delta = data.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return
    text = delta.get("text")
    if isinstance(text, str):
        partial_chunks.append(text)


def _stream_progress(
    partial_output: str, output_tokens: int | None, max_tokens: Any
) -> float:
    """Estimate completion percentage for an interrupted stream.

    Uses the exact output-token count when the upstream reported one before
    the failure, otherwise estimates from the characters already streamed.
    Capped below 100% because an interrupted run is by definition incomplete.
    """
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        return 0.0
    tokens = output_tokens
    if not tokens:
        if not partial_output:
            return 0.0
        tokens = max(1, (len(partial_output) + 3) // 4)
    return min(99.9, round(100.0 * tokens / max_tokens, 1))


def _upstream_request(payload: dict[str, Any], route: ModelRoute) -> Request:
    headers = {"Content-Type": "application/json"}
    api_key = route.api_key()
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return Request(
        route.completions_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def _is_loopback_host(host: str) -> bool:
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _validate_listener_auth(host: str, client_api_key: str | None) -> None:
    if _is_loopback_host(host):
        return
    if client_api_key is not None and len(client_api_key) < 32:
        raise ValueError(
            "LLMBRIDGE_API_KEY must be at least 32 characters for non-loopback "
            "listeners"
        )
    if client_api_key is None and not _is_loopback_host(host):
        import warnings
        warnings.warn(
            "LLMBRIDGE_API_KEY not set and host is not loopback; "
            "anyone can connect and use your upstream API keys",
            RuntimeWarning,
            stacklevel=2,
        )


class LLMBridgeServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        model_routes: dict[str, ModelRoute],
        client_api_key: str | None = None,
        work_tracker: WorkTracker | None = None,
    ) -> None:
        super().__init__(address, handler)
        self.model_routes = model_routes
        self.client_api_key = client_api_key
        self.work_tracker = work_tracker


class LLMBridgeHandler(BaseHTTPRequestHandler):
    server_version = "LLMBridge/0.1"

    def _models_response(self) -> dict[str, Any]:
        routes = self.server.model_routes
        models = [
            {"id": alias, "object": "model"}
            for alias in sorted(routes)
            if alias != "*"
        ]
        return {"object": "list", "data": models}

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/health", "/api/hello"):
            self._send_json(200, {"status": "ok"})
            return
        if not self._authenticate_client():
            return
        if path == "/v1/models":
            self._send_json(200, self._models_response())
            return
        if path == "/v1/interrupted":
            self._send_interrupted()
            return
        self._send_error(404, "not_found", "Endpoint not found")

    def do_HEAD(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/health", "/api/hello"):
            self._send_json(200, {"status": "ok"})
            return
        if not self._authenticate_client():
            return
        if path == "/v1/models":
            self._send_json(200, self._models_response())
            return
        self._send_error(404, "not_found", "Endpoint not found")

    def do_DELETE(self) -> None:
        if not self._authenticate_client():
            return
        path = urlsplit(self.path).path
        if path != "/v1/interrupted":
            self._send_error(404, "not_found", "Endpoint not found")
            return
        tracker = self.server.work_tracker
        cleared = bool(tracker is not None and tracker.clear_state())
        self._send_json(200, {"cleared": cleared})

    def _send_interrupted(self) -> None:
        tracker = self.server.work_tracker
        state = tracker.load_state() if tracker is not None else None
        if state is None:
            self._send_error(404, "not_found", "No interrupted work is recorded")
            return
        self._send_json(200, state)

    def do_POST(self) -> None:
        if not self._authenticate_client():
            return
        path = urlsplit(self.path).path
        if path == "/v1/messages/count_tokens":
            self._count_tokens()
            return
        if path != "/v1/messages":
            self._send_error(404, "not_found", "Endpoint not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 10 * 1024 * 1024:
                raise ProtocolError("Request body must be between 1 byte and 10 MiB")
            body = json.loads(self.rfile.read(size))
            model = body.get("model") if isinstance(body, dict) else None
            if not isinstance(model, str) or not model:
                raise ProtocolError("A client model name is required")
            route, upstream_model = resolve_model_route(
                model, self.server.model_routes
            )
            payload = anthropic_to_openai(
                body,
                upstream_model,
                reasoning_effort=route.reasoning_effort,
                supports_images=route.supports_images,
                supports_tools=route.supports_tools,
            )
            request = _upstream_request(payload, route)
        except (ValueError, RoutingError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return

        tracker = self.server.work_tracker
        prompt = ""
        system_prompt: str | None = None
        if tracker is not None:
            prompt, system_prompt = _extract_prompt_and_system(body)

        try:
            with urlopen(request, timeout=180) as upstream:
                if payload["stream"]:
                    self._stream_response(
                        upstream,
                        model or payload["model"],
                        tracker,
                        prompt,
                        system_prompt,
                        max_tokens=body.get("max_tokens"),
                    )
                    return
                response = json.loads(upstream.read())
            anthropic_response = openai_to_anthropic(response, model or payload["model"])
            self._clear_interrupted_on_match(tracker, prompt)
            self._send_json(200, anthropic_response)
        except HTTPError as error:
            status = error.code
            error.close()
            self.log_error("Upstream returned HTTP %s", status)
            if status in _RESUMABLE_HTTP_STATUS:
                self._save_interrupted(
                    tracker, prompt, model or payload["model"], system_prompt,
                    f"Upstream returned HTTP {status}",
                )
            self._send_error(502, "upstream_error", f"Upstream returned HTTP {status}")
        except URLError as error:
            self.log_error("Could not connect to upstream: %s", error.reason)
            self._save_interrupted(
                tracker, prompt, model or payload["model"], system_prompt,
                "Could not connect to the configured upstream",
            )
            self._send_error(502, "upstream_error", "Could not connect to the configured upstream")
        except (json.JSONDecodeError, ProtocolError) as error:
            self.log_error("Invalid upstream response: %s", error)
            self._save_interrupted(
                tracker, prompt, model or payload["model"], system_prompt, str(error)
            )
            self._send_error(502, "upstream_error", str(error))
        except TimeoutError:
            self.log_error("Upstream request timed out")
            self._save_interrupted(
                tracker, prompt, model or payload["model"], system_prompt,
                "Upstream request timed out",
            )
            self._send_error(504, "upstream_timeout", "Upstream request timed out")

    def _save_interrupted(
        self,
        tracker: WorkTracker | None,
        prompt: str,
        model: str,
        system: str | None,
        message: str,
        progress: float = 0.0,
        partial_output: str = "",
    ) -> None:
        if tracker is None:
            return
        try:
            tracker.save(
                prompt=prompt,
                model_name=model,
                system=system,
                error=message,
                progress_percent=progress,
                partial_output=partial_output,
            )
        except Exception:
            self.log_error("Could not record interrupted work")

    def _clear_interrupted_on_match(self, tracker: WorkTracker | None, prompt: str) -> None:
        """Clear a saved interruption once the same prompt completes successfully."""
        if tracker is None or not prompt:
            return
        try:
            state = tracker.load_state()
        except Exception:
            return
        if isinstance(state, dict) and state.get("prompt") == prompt:
            try:
                tracker.clear_state()
            except Exception:
                self.log_error("Could not clear completed interrupted work")

    def _count_tokens(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 10 * 1024 * 1024:
                raise ProtocolError("Request body must be between 1 byte and 10 MiB")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                raise ProtocolError("Request must include a messages list")
            tokenizer_encoding = None
            model = body.get("model")
            if model is not None:
                if not isinstance(model, str) or not model:
                    raise ProtocolError("model must be a non-empty string")
                route, _ = resolve_model_route(model, self.server.model_routes)
                tokenizer_encoding = route.tokenizer_encoding
            serialized_request = json.dumps(body, ensure_ascii=False)
            if tokenizer_encoding:
                try:
                    import tiktoken
                except ImportError as error:
                    raise ProtocolError(
                        "This model route configures tokenizer_encoding; install "
                        "LLMBridge with its token-count extra"
                    ) from error
                estimated_tokens = len(
                    tiktoken.get_encoding(tokenizer_encoding).encode(
                        serialized_request, disallowed_special=()
                    )
                )
                estimated_tokens = max(1, estimated_tokens)
            else:
                estimated_tokens = max(
                    1, (len(serialized_request) + 3) // 4
                )
        except (ValueError, ProtocolError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return
        self._send_json(200, {"input_tokens": estimated_tokens})

    def _stream_response(
        self,
        upstream: Any,
        model: str,
        tracker: WorkTracker | None = None,
        prompt: str = "",
        system: str | None = None,
        max_tokens: Any = None,
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        events = iter(stream_events(upstream, model))
        partial_chunks: list[str] = []
        reported_tokens: int | None = None
        while True:
            try:
                event = next(events)
            except StopIteration:
                self._clear_interrupted_on_match(tracker, prompt)
                return
            except (ProtocolError, TimeoutError, URLError, OSError, HTTPException) as error:
                if isinstance(error, ProtocolError):
                    message = str(error)
                elif isinstance(error, TimeoutError):
                    message = "Upstream stream timed out"
                else:
                    message = "Upstream stream failed"
                self.log_error("Invalid upstream stream: %s", message)
                partial_output = "".join(partial_chunks)
                self._save_interrupted(
                    tracker,
                    prompt,
                    model,
                    system,
                    message,
                    progress=_stream_progress(partial_output, reported_tokens, max_tokens),
                    partial_output=partial_output,
                )
                event = json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "upstream_error", "message": message},
                    }
                )
                try:
                    self.wfile.write(f"event: error\ndata: {event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.log_error("Client disconnected during streaming response")
                return
            _collect_text_delta(event, partial_chunks)
            if reported_tokens is None:
                tokens = _event_output_tokens(event)
                if tokens is not None:
                    reported_tokens = tokens
            try:
                self.wfile.write(event)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.log_error("Client disconnected during streaming response")
                return

    def _authenticate_client(self) -> bool:
        expected_key = self.server.client_api_key
        if expected_key is None:
            return True
        authorization = self.headers.get("Authorization", "")
        token = (
            authorization[7:]
            if authorization.lower().startswith("bearer ")
            else self.headers.get("x-api-key", "")
        )
        if hmac.compare_digest(token.encode("utf-8"), expected_key.encode("utf-8")):
            return True
        self._send_error(401, "authentication_error", "Valid client API key required")
        return False

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_error(self, status: int, error_type: str, message: str) -> None:
        self._send_json(
            status,
            {"type": "error", "error": {"type": error_type, "message": message}},
        )

    def log_message(self, format_string: str, *args: Any) -> None:
        super().log_message(format_string, *args)


def main() -> None:
    host = os.environ.get("LLMBRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("LLMBRIDGE_PORT", "8787"))
    client_api_key = os.environ.get("LLMBRIDGE_API_KEY") or None
    _validate_listener_auth(host, client_api_key)
    try:
        model_routes = load_model_routes()
    except RoutingError as error:
        raise SystemExit(str(error)) from error
    work_tracker = None
    if _truthy(os.environ.get("LLMBRIDGE_TRACK_INTERRUPTED", "1")):
        work_tracker = WorkTracker(
            storage_path=os.environ.get("LLMBRIDGE_PARTIAL_WORK_PATH") or None
        )
    server = LLMBridgeServer(
        (host, port), LLMBridgeHandler, model_routes, client_api_key, work_tracker
    )
    print(f"LLMBridge listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down LLMBridge")
    finally:
        server.server_close()
