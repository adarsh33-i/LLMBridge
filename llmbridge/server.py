import hmac
import ipaddress
import json
import os
import time
from dataclasses import dataclass
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

# How many times a single route is attempted before giving up on it, and how
# long to wait before the first retry. Both are overridable from the
# environment so a deployment can trade latency for resilience.
_DEFAULT_ROUTE_ATTEMPTS = 1
_DEFAULT_ROUTE_RETRY_BACKOFF_SECONDS = 0.5

# A failed upstream attempt: (kind, message, http_status). ``kind`` decides how
# the failure is reported to the client and whether it is worth saving.
_Failure = tuple[str, str, int | None]

# Bound how much we will ever buffer from a single upstream reply, and how long
# a client socket may stay idle mid-request, so a misbehaving peer cannot
# exhaust memory or tie up a worker thread indefinitely.
_MAX_UPSTREAM_RESPONSE_BYTES = 32 * 1024 * 1024
_CLIENT_SOCKET_TIMEOUT_SECONDS = 300

# Fallback max_tokens for a resumed generation whose original limit was not
# recorded (it always is for runs this server saved, but older state files
# predate that field).
_DEFAULT_RESUME_MAX_TOKENS = 4096


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _route_attempts() -> int:
    """How many times one route is tried before falling through to the next."""
    raw = os.environ.get("LLMBRIDGE_ROUTE_ATTEMPTS")
    if raw is None:
        return _DEFAULT_ROUTE_ATTEMPTS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_ROUTE_ATTEMPTS
    return max(1, min(value, 10))


def _route_retry_backoff_seconds() -> float:
    """Base delay between attempts at the same route; doubles per attempt."""
    raw = os.environ.get("LLMBRIDGE_ROUTE_RETRY_BACKOFF")
    if raw is None:
        return _DEFAULT_ROUTE_RETRY_BACKOFF_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_ROUTE_RETRY_BACKOFF_SECONDS
    return max(0.0, value)


def _retry_sleep(attempt: int) -> None:
    backoff = _route_retry_backoff_seconds()
    if backoff > 0:
        time.sleep(backoff * (2 ** (attempt - 1)))


def _is_retryable_status(status: int) -> bool:
    return status in _RESUMABLE_HTTP_STATUS or status >= 500



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


def _build_resume_request(
    state: dict[str, Any], model: str, max_tokens: int
) -> dict[str, Any]:
    """Rebuild a non-streaming Anthropic request from saved interrupted work.

    Any partial output already produced is supplied as an assistant turn so the
    upstream continues from where it stopped rather than starting over.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": state["prompt"]}]
    partial_output = state.get("partial_output")
    if isinstance(partial_output, str) and partial_output:
        messages.append({"role": "assistant", "content": partial_output})
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": False,
    }
    system = state.get("system")
    if isinstance(system, str) and system:
        request["system"] = system
    return request


def _read_bounded(stream: Any, limit: int) -> bytes:
    """Read an upstream body, refusing to buffer more than ``limit`` bytes."""
    data = stream.read(limit + 1)
    if data is None:
        return b""
    if len(data) > limit:
        raise ProtocolError("Upstream response exceeded the size limit")
    return data


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
    if client_api_key is None:
        raise SystemExit(
            "LLMBRIDGE_API_KEY is required when binding to a non-loopback "
            "address; set it to a secret of at least 32 characters"
        )
    if len(client_api_key) < 32:
        raise SystemExit(
            "LLMBRIDGE_API_KEY must be at least 32 characters for non-loopback "
            "listeners"
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
    timeout = _CLIENT_SOCKET_TIMEOUT_SECONDS

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

    def _resume_interrupted(self) -> None:
        """Re-issue saved interrupted work to a (possibly different) model."""
        tracker = self.server.work_tracker
        state = tracker.load_state() if tracker is not None else None
        if state is None:
            self._send_error(404, "not_found", "No interrupted work is recorded")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > 10 * 1024 * 1024:
                raise ProtocolError("Request body must be at most 10 MiB")
            body = json.loads(self.rfile.read(size)) if size > 0 else {}
        except (ValueError, ProtocolError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return
        if not isinstance(body, dict):
            self._send_error(
                400, "invalid_request_error", "Request body must be a JSON object"
            )
            return
        if not isinstance(state.get("prompt"), str) or not state["prompt"]:
            self._send_error(
                400, "invalid_request_error", "The saved work has no prompt to resume"
            )
            return
        model = body.get("model") or state.get("model_name")
        if not isinstance(model, str) or not model:
            self._send_error(
                400, "invalid_request_error", "A model name is required to resume"
            )
            return
        try:
            route, upstream_model = resolve_model_route(
                model, self.server.model_routes
            )
        except (ValueError, RoutingError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return
        max_tokens = body.get("max_tokens")
        if not _is_positive_int(max_tokens):
            max_tokens = state.get("max_tokens")
        if not _is_positive_int(max_tokens):
            max_tokens = _DEFAULT_RESUME_MAX_TOKENS
        resume_body = _build_resume_request(state, model, max_tokens)
        self._handle_messages(resume_body, model, route, upstream_model)

    def do_POST(self) -> None:
        if not self._authenticate_client():
            return
        path = urlsplit(self.path).path
        if path == "/v1/messages/count_tokens":
            self._count_tokens()
            return
        if path == "/v1/interrupted/resume":
            self._resume_interrupted()
            return
        if path != "/v1/messages":
            self._send_error(404, "not_found", "Endpoint not found")
            return
        try:
            body = self._read_json_body()
            model = body.get("model") if isinstance(body, dict) else None
            if not isinstance(model, str) or not model:
                raise ProtocolError("A client model name is required")
            route, upstream_model = resolve_model_route(
                model, self.server.model_routes
            )
        except (ValueError, RoutingError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return
        self._handle_messages(body, model, route, upstream_model)

    def _read_json_body(self) -> Any:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > 10 * 1024 * 1024:
            raise ProtocolError("Request body must be between 1 byte and 10 MiB")
        return json.loads(self.rfile.read(size))

    def _route_candidates(
        self, primary_route: ModelRoute, primary_model: str
    ) -> list[tuple[ModelRoute, str]]:
        """Primary route followed by its configured fallbacks, in order."""
        candidates: list[tuple[ModelRoute, str]] = [(primary_route, primary_model)]
        for alias in primary_route.fallbacks:
            try:
                route, upstream_model = resolve_model_route(
                    alias, self.server.model_routes
                )
            except RoutingError as error:
                self.log_error("Skipping unusable fallback %s: %s", alias, error)
                continue
            candidate = (route, upstream_model)
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _handle_messages(
        self,
        body: Any,
        model: str,
        primary_route: ModelRoute,
        primary_model: str,
    ) -> None:
        """Translate, call upstream (following fallbacks), and answer the client."""
        tracker = self.server.work_tracker
        prompt = ""
        system_prompt: str | None = None
        if tracker is not None and isinstance(body, dict):
            prompt, system_prompt = _extract_prompt_and_system(body)
        display_model = model or primary_model
        candidates = self._route_candidates(primary_route, primary_model)
        max_tokens = body.get("max_tokens") if isinstance(body, dict) else None
        stream = isinstance(body, dict) and bool(body.get("stream"))
        if stream:
            self._stream_with_fallback(
                candidates,
                body,
                display_model,
                tracker,
                prompt,
                system_prompt,
                max_tokens,
            )
            return
        self._respond_with_fallback(
            candidates,
            body,
            display_model,
            tracker,
            prompt,
            system_prompt,
            max_tokens,
        )

    def _build_upstream_request(
        self, route: ModelRoute, upstream_model: str, body: Any
    ) -> Request:
        payload = anthropic_to_openai(
            body,
            upstream_model,
            reasoning_effort=route.reasoning_effort,
            supports_images=route.supports_images,
            supports_tools=route.supports_tools,
            cache_control=route.cache_control,
        )
        return _upstream_request(payload, route)

    def _respond_with_fallback(
        self,
        candidates: list[tuple[ModelRoute, str]],
        body: Any,
        display_model: str,
        tracker: WorkTracker | None,
        prompt: str,
        system_prompt: str | None,
        max_tokens: Any = None,
    ) -> None:
        payload_error: ValueError | None = None
        failure: _Failure | None = None
        for route, upstream_model in candidates:
            for attempt in range(1, _route_attempts() + 1):
                try:
                    request = self._build_upstream_request(
                        route, upstream_model, body
                    )
                except ValueError as error:
                    payload_error = error
                    break
                try:
                    upstream = urlopen(request, timeout=180)
                except HTTPError as error:
                    failure = self._record_http_failure(error)
                    if (
                        failure[2] is not None
                        and _is_retryable_status(failure[2])
                        and attempt < _route_attempts()
                    ):
                        _retry_sleep(attempt)
                        continue
                    break
                except URLError as error:
                    self.log_error("Could not connect to upstream: %s", error.reason)
                    failure = (
                        "connect",
                        "Could not connect to the configured upstream",
                        None,
                    )
                    if attempt < _route_attempts():
                        _retry_sleep(attempt)
                        continue
                    break
                except TimeoutError:
                    self.log_error("Upstream request timed out")
                    failure = ("timeout", "Upstream request timed out", None)
                    if attempt < _route_attempts():
                        _retry_sleep(attempt)
                        continue
                    break
                response, failure = self._read_json_reply(upstream)
                if failure is not None:
                    break
                anthropic_response = openai_to_anthropic(response, display_model)
                self._clear_interrupted_on_match(tracker, prompt)
                self._send_json(200, anthropic_response)
                return
        self._report_failure(
            failure,
            payload_error,
            tracker,
            prompt,
            display_model,
            system_prompt,
            max_tokens,
        )

    def _stream_with_fallback(
        self,
        candidates: list[tuple[ModelRoute, str]],
        body: Any,
        display_model: str,
        tracker: WorkTracker | None,
        prompt: str,
        system_prompt: str | None,
        max_tokens: Any = None,
    ) -> None:
        payload_error: ValueError | None = None
        failure: _Failure | None = None
        for route, upstream_model in candidates:
            for attempt in range(1, _route_attempts() + 1):
                try:
                    request = self._build_upstream_request(
                        route, upstream_model, body
                    )
                except ValueError as error:
                    payload_error = error
                    break
                try:
                    upstream = urlopen(request, timeout=180)
                except HTTPError as error:
                    failure = self._record_http_failure(error)
                    if (
                        failure[2] is not None
                        and _is_retryable_status(failure[2])
                        and attempt < _route_attempts()
                    ):
                        _retry_sleep(attempt)
                        continue
                    break
                except URLError as error:
                    self.log_error("Could not connect to upstream: %s", error.reason)
                    failure = (
                        "connect",
                        "Could not connect to the configured upstream",
                        None,
                    )
                    if attempt < _route_attempts():
                        _retry_sleep(attempt)
                        continue
                    break
                except TimeoutError:
                    self.log_error("Upstream request timed out")
                    failure = ("timeout", "Upstream request timed out", None)
                    if attempt < _route_attempts():
                        _retry_sleep(attempt)
                        continue
                    break
                # The connection opened, so this route is good enough to start
                # streaming on. Once the client has bytes it cannot be
                # re-routed, so mid-stream failures stay on the interrupt path.
                with upstream:
                    self._stream_response(
                        upstream,
                        display_model,
                        tracker,
                        prompt,
                        system_prompt,
                        max_tokens=max_tokens,
                    )
                return
        self._report_failure(
            failure,
            payload_error,
            tracker,
            prompt,
            display_model,
            system_prompt,
            max_tokens,
        )

    def _record_http_failure(self, error: HTTPError) -> _Failure:
        status = error.code
        error.close()
        self.log_error("Upstream returned HTTP %s", status)
        return ("http", f"Upstream returned HTTP {status}", status)

    def _read_json_reply(self, upstream: Any) -> tuple[Any, _Failure | None]:
        try:
            with upstream:
                payload = json.loads(
                    _read_bounded(upstream, _MAX_UPSTREAM_RESPONSE_BYTES)
                )
        except (json.JSONDecodeError, ProtocolError) as error:
            self.log_error("Invalid upstream response: %s", error)
            return None, ("invalid", str(error), None)
        except TimeoutError:
            self.log_error("Upstream request timed out")
            return None, ("timeout", "Upstream request timed out", None)
        return payload, None

    def _report_failure(
        self,
        failure: _Failure | None,
        payload_error: ValueError | None,
        tracker: WorkTracker | None,
        prompt: str,
        model: str,
        system_prompt: str | None,
        max_tokens: Any = None,
    ) -> None:
        """Answer the client after every candidate route has been exhausted."""
        if failure is None:
            message = (
                str(payload_error)
                if payload_error is not None
                else "No usable model route is configured"
            )
            self._send_error(400, "invalid_request_error", message)
            return
        kind, message, status = failure
        if kind == "http" and status not in _RESUMABLE_HTTP_STATUS:
            # A client-side mistake from the upstream would just repeat on the
            # next attempt, so it is not worth saving for resumption.
            self._send_error(502, "upstream_error", message)
            return
        self._save_interrupted(
            tracker, prompt, model, system_prompt, message, max_tokens=max_tokens
        )
        if kind == "timeout":
            self._send_error(504, "upstream_timeout", message)
        else:
            self._send_error(502, "upstream_error", message)

    def _save_interrupted(
        self,
        tracker: WorkTracker | None,
        prompt: str,
        model: str,
        system: str | None,
        message: str,
        progress: float = 0.0,
        partial_output: str = "",
        max_tokens: Any = None,
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
                max_tokens=max_tokens if isinstance(max_tokens, int) else None,
            )
        except Exception as error:
            self.log_error("Could not record interrupted work: %s", error)

    def _clear_interrupted_on_match(self, tracker: WorkTracker | None, prompt: str) -> None:
        """Clear a saved interruption once the same prompt completes successfully."""
        if tracker is None or not prompt:
            return
        try:
            state = tracker.load_state()
        except Exception as error:
            self.log_error("Could not read interrupted work before clearing: %s", error)
            return
        if isinstance(state, dict) and state.get("prompt") == prompt:
            try:
                tracker.clear_state()
            except Exception as error:
                self.log_error("Could not clear completed interrupted work: %s", error)

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
                    max_tokens=max_tokens,
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
                partial_output = "".join(partial_chunks)
                if partial_output:
                    self._save_interrupted(
                        tracker,
                        prompt,
                        model,
                        system,
                        "Client disconnected during streaming response",
                        progress=_stream_progress(
                            partial_output, reported_tokens, max_tokens
                        ),
                        partial_output=partial_output,
                        max_tokens=max_tokens,
                    )
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


@dataclass(frozen=True)
class ServerSettings:
    """Resolved server configuration derived from the environment."""

    host: str
    port: int
    client_api_key: str | None
    model_routes: dict[str, ModelRoute]
    work_tracker: WorkTracker | None


def load_server_settings() -> "ServerSettings":
    """Resolve listener, routing, and interrupted-work settings from the environment."""
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
    return ServerSettings(
        host=host,
        port=port,
        client_api_key=client_api_key,
        model_routes=model_routes,
        work_tracker=work_tracker,
    )


def main() -> None:  # pragma: no cover - process entrypoint, blocks in serve_forever
    settings = load_server_settings()
    server = LLMBridgeServer(
        (settings.host, settings.port),
        LLMBridgeHandler,
        settings.model_routes,
        settings.client_api_key,
        settings.work_tracker,
    )
    print(f"LLMBridge listening on http://{settings.host}:{settings.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down LLMBridge")
    finally:
        server.server_close()
