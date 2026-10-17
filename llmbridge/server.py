import hmac
import ipaddress
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .ledger import GenerationLedger, GenerationRecord, request_replay
from .protocol import (
    ProtocolError,
    anthropic_to_openai,
    openai_to_anthropic,
    stream_events,
)
from .routing import ModelRoute, RoutingError, load_model_routes, resolve_model_route
from .work_tracker import WorkTracker

logger = logging.getLogger(__name__)

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
# exhaust memory or tie up a worker thread indefinitely. Both timeouts are
# overridable from the environment.
_MAX_UPSTREAM_RESPONSE_BYTES = 32 * 1024 * 1024
_DEFAULT_CLIENT_TIMEOUT_SECONDS = 300
_DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 180

# Fallback max_tokens for a resumed generation whose original limit was not
# recorded (it always is for runs this server saved, but older state files
# predate that field).
_DEFAULT_RESUME_MAX_TOKENS = 4096

# Upstream statuses that mean "quota/rate limit exhausted" rather than "the
# request was wrong": with LLMBRIDGE_QUEUE_ON_QUOTA=1 these park the request
# in the ledger for an automatic retry instead of failing immediately.
_QUEUEABLE_STATUS = frozenset({402, 429})
_DEFAULT_QUEUE_RETRY_SECONDS = 5.0
_DEFAULT_QUEUE_MAX_ATTEMPTS = 5
_QUEUE_POLL_SECONDS = 1.0


def _queue_on_quota_enabled() -> bool:
    return _truthy(os.environ.get("LLMBRIDGE_QUEUE_ON_QUOTA"))


def _queue_retry_seconds() -> float:
    return _env_positive_number(
        "LLMBRIDGE_QUEUE_RETRY_SECONDS", _DEFAULT_QUEUE_RETRY_SECONDS
    )


def _queue_max_attempts() -> int:
    raw = os.environ.get("LLMBRIDGE_QUEUE_MAX_ATTEMPTS")
    if raw is None:
        return _DEFAULT_QUEUE_MAX_ATTEMPTS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_QUEUE_MAX_ATTEMPTS
    return value if value >= 1 else _DEFAULT_QUEUE_MAX_ATTEMPTS


def _self_heal_enabled() -> bool:
    # Self-healing a broken stream onto a fallback model is the default; it
    # only changes behaviour when a fallback route is configured anyway.
    return _truthy(os.environ.get("LLMBRIDGE_SELF_HEAL", "1"))


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _env_positive_number(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back on bad input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _client_timeout_seconds() -> float:
    return _env_positive_number(
        "LLMBRIDGE_CLIENT_TIMEOUT_SECONDS", _DEFAULT_CLIENT_TIMEOUT_SECONDS
    )


def _upstream_timeout_seconds() -> float:
    return _env_positive_number(
        "LLMBRIDGE_UPSTREAM_TIMEOUT_SECONDS", _DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    )


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


def _anthropic_text(response: Any) -> str:
    """Flatten the text blocks of an Anthropic response into plain text."""
    if not isinstance(response, dict):
        return ""
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _upstream_output_tokens(response: Any) -> int | None:
    """Read the completion-token count from an OpenAI-style upstream response."""
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("completion_tokens")
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
        return tokens
    return None


def _generation_recovery_hint(record: GenerationRecord | None) -> str:
    """Tell the client how to continue an interrupted generation.

    Without this, durable generations are invisible: the work is saved but the
    client only sees a bare upstream error. The hint names the recorded
    generation and the endpoint that resumes it on any model.
    """
    if record is None:
        return ""
    return (
        f". Interrupted work was saved as {record.id}; continue it (optionally "
        f"on another model) with POST /v1/generations/{record.id}/fork"
    )


def _build_resume_request(
    state: dict[str, Any], model: str, max_tokens: int
) -> dict[str, Any]:
    """Rebuild a non-streaming Anthropic request from saved interrupted work.

    When the saved state kept the original request's replayable parts
    (``messages``/``tools``/``system``), the full conversation is restored so a
    multi-turn or tool-using generation resumes faithfully. Otherwise it falls
    back to the recorded prompt. Any partial output already produced is supplied
    as an assistant turn so the upstream continues from where it stopped rather
    than starting over.
    """
    replay = state.get("request")
    replay = replay if isinstance(replay, dict) else {}
    recorded_messages = replay.get("messages")
    if isinstance(recorded_messages, list) and recorded_messages:
        messages: list[dict[str, Any]] = [
            message for message in recorded_messages if isinstance(message, dict)
        ]
    else:
        messages = [{"role": "user", "content": state["prompt"]}]
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
    if not isinstance(system, str) or not system:
        system = replay.get("system")
    if isinstance(system, str) and system:
        request["system"] = system
    tools = replay.get("tools")
    if isinstance(tools, list) and tools:
        request["tools"] = tools
    if replay.get("tool_choice") is not None:
        request["tool_choice"] = replay["tool_choice"]
    return request


def _build_heal_body(body: Any, partial_output: str) -> dict[str, Any] | None:
    """Rebuild a streaming request so a fallback model continues mid-answer.

    The partial text becomes an assistant turn, so the next upstream writes a
    continuation instead of starting over. Returns None when the original body
    is not a well-formed Anthropic request.
    """
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None
    healed = dict(body)
    healed["messages"] = [
        message for message in body["messages"] if isinstance(message, dict)
    ] + [{"role": "assistant", "content": partial_output}]
    return healed


def _read_bounded(stream: Any, limit: int) -> bytes:
    """Read an upstream body, refusing to buffer more than ``limit`` bytes.

    Loops until EOF or limit is reached to handle chunked responses where
    a single ``read()`` call may not return the complete body.
    """
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        chunk = stream.read(min(8192, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise ProtocolError("Upstream response exceeded the size limit")
    return b"".join(chunks)


def _sse_bytes(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


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
        ledger: GenerationLedger | None = None,
    ) -> None:
        super().__init__(address, handler)
        self.model_routes = model_routes
        self.client_api_key = client_api_key
        self.work_tracker = work_tracker
        self.ledger = ledger
        self._queue_stop = threading.Event()
        self._queue_thread: threading.Thread | None = None
        if ledger is not None and _queue_on_quota_enabled():
            # The queue lives in the ledger, so this also re-arms retries that
            # were pending when the process last exited.
            self._queue_thread = threading.Thread(
                target=self._dispatch_queue_loop, daemon=True
            )
            self._queue_thread.start()

    def shutdown(self) -> None:
        self._queue_stop.set()
        super().shutdown()

    def _dispatch_queue_loop(self) -> None:
        while not self._queue_stop.wait(_QUEUE_POLL_SECONDS):
            try:
                self._dispatch_due_queue()
            except Exception as error:  # pragma: no cover - defensive
                logger.error("Queue dispatch failed: %s", error)

    def _dispatch_due_queue(self) -> None:
        ledger = self.ledger
        if ledger is None:
            return
        now = time.time()
        for record in ledger.queued():
            if GenerationLedger.queue_retry_at(record) > now:
                continue
            self._run_queued_generation(record)

    def _run_queued_generation(self, record: GenerationRecord) -> None:
        """Re-send one queued generation to its model route and finalize it."""
        ledger = self.ledger
        if ledger is None or record.status != "queued":
            return
        try:
            route, upstream_model = resolve_model_route(
                record.model, self.model_routes
            )
        except RoutingError as error:
            ledger.fail(record, f"queued model is no longer configured: {error}")
            return
        attempt = GenerationLedger.queue_attempt(record)
        body: dict[str, Any] = dict(record.request)
        body["model"] = record.model
        body["stream"] = False
        if isinstance(record.params, dict) and record.params.get("max_tokens"):
            body.setdefault("max_tokens", record.params["max_tokens"])
        payload = anthropic_to_openai(
            body,
            upstream_model,
            reasoning_effort=route.reasoning_effort,
            supports_images=route.supports_images,
            supports_tools=route.supports_tools,
            cache_control=route.cache_control,
        )
        try:
            request = _upstream_request(payload, route)
        except RoutingError as error:
            ledger.fail(record, str(error))
            return
        try:
            with urlopen(request, timeout=_upstream_timeout_seconds()) as upstream:
                reply = json.loads(
                    _read_bounded(upstream, _MAX_UPSTREAM_RESPONSE_BYTES)
                )
        except HTTPError as error:
            failure = ("http", f"Upstream returned HTTP {error.code}", error.code)
            error.close()
            logger.info("Queued generation %s got HTTP %s", record.id, error.code)
            self._requeue_or_fail(record, attempt, failure)
            return
        except (URLError, TimeoutError, OSError) as error:
            logger.info("Queued generation %s failed: %s", record.id, error)
            self._requeue_or_fail(
                record, attempt, ("connect", "Could not connect to the configured upstream", None)
            )
            return
        except (json.JSONDecodeError, ProtocolError) as error:
            logger.info(
                "Queued generation %s got an invalid reply: %s", record.id, error
            )
            ledger.fail(record, f"upstream returned an invalid response: {error}")
            return
        anthropic_response = openai_to_anthropic(reply, record.model)
        output = _anthropic_text(anthropic_response)
        tracker = self.work_tracker
        if tracker is not None and record.prompt:
            try:
                tracker.clear_if_matches(record.prompt)
            except Exception:  # pragma: no cover - defensive
                pass
        ledger.complete(record, output, _upstream_output_tokens(reply))
        logger.info(
            "Queued generation %s completed (attempt %d)", record.id, attempt
        )

    def _requeue_or_fail(
        self, record: GenerationRecord, attempt: int, failure: _Failure
    ) -> None:
        """After a failed retry, park it again or give up for good."""
        ledger = self.ledger
        if ledger is None:
            return
        kind, message, status = failure
        if (
            kind == "http"
            and status in _QUEUEABLE_STATUS
            and attempt < _queue_max_attempts()
        ):
            retry_after = _queue_retry_seconds() * (attempt + 1)
            ledger.queue(record, message, time.time() + retry_after, attempt + 1)
            return
        ledger.fail(record, message)


class LLMBridgeHandler(BaseHTTPRequestHandler):
    server_version = "LLMBridge/0.1"
    timeout = _DEFAULT_CLIENT_TIMEOUT_SECONDS

    def setup(self) -> None:
        # Resolve the idle-socket timeout per connection so
        # LLMBRIDGE_CLIENT_TIMEOUT_SECONDS takes effect without a restart at
        # import time.
        self.timeout = _client_timeout_seconds()
        super().setup()

    def handle_one_request(self) -> None:
        # A client (e.g. an editor or agent) can drop the connection before we
        # finish writing a response. A broken pipe there is benign and common,
        # so catch it rather than letting socketserver print a full traceback.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            self.log_error("Client disconnected before the response completed")

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
        if path == "/v1/generations":
            self._list_generations()
            return
        if path == "/v1/queue":
            self._list_queue()
            return
        if path.startswith("/v1/generations/"):
            if path.endswith("/tree"):
                self._generation_tree(path)
            else:
                self._send_generation(path)
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
        if path.startswith("/v1/queue/"):
            self._cancel_queued(path)
            return
        if path.startswith("/v1/generations/"):
            self._delete_generation(path)
            return
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

    def _ledger(self) -> GenerationLedger | None:
        ledger = self.server.ledger
        if ledger is None:
            self._send_error(
                404, "not_found", "Generation tracking is disabled on this server"
            )
        return ledger

    @staticmethod
    def _generation_id_from_path(path: str, suffix: str = "") -> str | None:
        prefix = "/v1/generations/"
        if not path.startswith(prefix):
            return None
        rest = path[len(prefix):]
        if suffix:
            if not rest.endswith(suffix):
                return None
            rest = rest[: -len(suffix)]
        if not rest or "/" in rest:
            return None
        return rest

    def _list_generations(self) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        records = ledger.list()
        self._send_json(
            200,
            {"object": "list", "data": [record.summary() for record in records]},
        )

    def _send_generation(self, path: str) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        generation_id = self._generation_id_from_path(path)
        record = ledger.load(generation_id) if generation_id else None
        if record is None:
            self._send_error(404, "not_found", "Generation not found")
            return
        self._send_json(200, record.to_dict())

    def _generation_tree(self, path: str) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        generation_id = self._generation_id_from_path(path, "/tree")
        tree = ledger.tree(generation_id) if generation_id else None
        if tree is None:
            self._send_error(404, "not_found", "Generation not found")
            return
        self._send_json(200, tree)

    @staticmethod
    def _queue_id_from_path(path: str) -> str | None:
        prefix = "/v1/queue/"
        if not path.startswith(prefix):
            return None
        rest = path[len(prefix):]
        if not rest or "/" in rest:
            return None
        return rest

    def _list_queue(self) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        now = time.time()
        items = []
        for record in ledger.queued():
            retry_at = GenerationLedger.queue_retry_at(record)
            items.append(
                {
                    **record.summary(),
                    "attempt": GenerationLedger.queue_attempt(record),
                    "retry_at": retry_at,
                    "retry_in": max(0.0, retry_at - now),
                }
            )
        self._send_json(200, {"object": "list", "data": items})

    def _cancel_queued(self, path: str) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        generation_id = self._queue_id_from_path(path)
        record = ledger.load(generation_id) if generation_id else None
        if record is None or record.status != "queued":
            self._send_error(404, "not_found", "No queued generation with that id")
            return
        ledger.fail(record, "cancelled from the queue")
        self._send_json(200, {"cancelled": True, "id": record.id})

    def _delete_generation(self, path: str) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        generation_id = self._generation_id_from_path(path)
        if not generation_id or ledger.load(generation_id) is None:
            self._send_error(404, "not_found", "Generation not found")
            return
        deleted = ledger.delete(generation_id)
        self._send_json(200, {"deleted": deleted, "id": generation_id})

    def _fork_generation(self, path: str) -> None:
        """Continue a recorded generation as a new one, optionally on a new model."""
        ledger = self._ledger()
        if ledger is None:
            return
        generation_id = self._generation_id_from_path(path, "/fork")
        record = ledger.load(generation_id) if generation_id else None
        if record is None:
            self._send_error(404, "not_found", "Generation not found")
            return
        body: dict[str, Any] = {}
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > 10 * 1024 * 1024:
                raise ProtocolError("Request body must be at most 10 MiB")
            raw = json.loads(self.rfile.read(size)) if size > 0 else {}
        except (ValueError, ProtocolError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return
        if isinstance(raw, dict):
            body = raw
        elif raw:
            self._send_error(
                400, "invalid_request_error", "Fork body must be a JSON object"
            )
            return
        model = body.get("model") or record.model
        if not isinstance(model, str) or not model:
            self._send_error(400, "invalid_request_error", "A model name is required")
            return
        try:
            route, upstream_model = resolve_model_route(model, self.server.model_routes)
        except (ValueError, RoutingError) as error:
            self._send_error(400, "invalid_request_error", str(error))
            return
        max_tokens = body.get("max_tokens")
        if not _is_positive_int(max_tokens):
            max_tokens = None
        at_event = body.get("at_event")
        if not isinstance(at_event, int) or isinstance(at_event, bool):
            at_event = record.events[-1].seq if record.events else None
        fork_body = GenerationLedger.fork_request(
            record,
            model=model,
            max_tokens=max_tokens,
            stream=bool(body.get("stream")),
        )
        child = ledger.create(
            fork_body, model, parent=record.id, fork_event_seq=at_event
        )
        self._handle_messages(fork_body, model, route, upstream_model, record=child)

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
        def _coerce_positive_int(value: Any, default: int) -> int:
            """Coerce value to positive int, returning default on failure."""
            if _is_positive_int(value):
                return value
            if isinstance(value, str):
                try:
                    value = int(value)
                    if value > 0:
                        return value
                except ValueError:
                    pass
            return default

        max_tokens = _coerce_positive_int(body.get("max_tokens"), 0)
        if max_tokens == 0:
            max_tokens = _coerce_positive_int(state.get("max_tokens"), _DEFAULT_RESUME_MAX_TOKENS)
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
        if path.startswith("/v1/generations/") and path.endswith("/fork"):
            self._fork_generation(path)
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
        record: GenerationRecord | None = None,
    ) -> None:
        """Translate, call upstream (following fallbacks), and answer the client."""
        tracker = self.server.work_tracker
        prompt = ""
        system_prompt: str | None = None
        if tracker is not None and isinstance(body, dict):
            prompt, system_prompt = _extract_prompt_and_system(body)
        display_model = model or primary_model
        if record is None:
            record = self._begin_generation(body, display_model)
        self._generation_id = record.id if record is not None else None
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
                record,
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
            record,
        )

    def _begin_generation(
        self, body: Any, display_model: str
    ) -> GenerationRecord | None:
        """Open a durable generation record, or return None when tracking is off."""
        ledger = self.server.ledger
        if ledger is None or not isinstance(body, dict):
            return None
        try:
            return ledger.create(body, display_model)
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not record generation: %s", error)
            return None

    def _finalize_generation(
        self,
        record: GenerationRecord | None,
        status: str,
        *,
        output: str = "",
        output_tokens: int | None = None,
        message: str = "",
        progress: float = 0.0,
    ) -> None:
        """Close out a generation record with its terminal status."""
        ledger = self.server.ledger
        if record is None or ledger is None:
            return
        try:
            if status == "completed":
                ledger.complete(record, output, output_tokens)
            elif status == "interrupted":
                ledger.interrupt(
                    record,
                    message,
                    output=output,
                    output_tokens=output_tokens,
                    progress=progress,
                )
            else:
                ledger.fail(record, message)
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not finalize generation %s: %s", record.id, error)

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
        record: GenerationRecord | None = None,
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
                    upstream = urlopen(request, timeout=_upstream_timeout_seconds())
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
                self._finalize_generation(
                    record,
                    "completed",
                    output=_anthropic_text(anthropic_response),
                    output_tokens=_upstream_output_tokens(response),
                )
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
            record,
            body=body,
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
        record: GenerationRecord | None = None,
    ) -> None:
        payload_error: ValueError | None = None
        failure: _Failure | None = None
        for candidate_index, (route, upstream_model) in enumerate(candidates):
            for attempt in range(1, _route_attempts() + 1):
                try:
                    request = self._build_upstream_request(
                        route, upstream_model, body
                    )
                except ValueError as error:
                    payload_error = error
                    break
                try:
                    upstream = urlopen(request, timeout=_upstream_timeout_seconds())
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
                # streaming on. If it dies mid-stream with text already sent,
                # _stream_response can still heal onto a later candidate.
                with upstream:
                    self._stream_response(
                        upstream,
                        display_model,
                        tracker,
                        prompt,
                        system_prompt,
                        max_tokens=max_tokens,
                        record=record,
                        body=body,
                        candidates=candidates,
                        route_index=candidate_index,
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
            record,
            body=body,
        )

    def _record_http_failure(self, error: HTTPError) -> _Failure:
        status = error.code
        retry_after = None
        if error.headers is not None:
            raw = error.headers.get("Retry-After")
            if raw is not None and raw.strip().isdigit():
                retry_after = float(raw.strip())
        self._upstream_retry_after = retry_after
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
        record: GenerationRecord | None = None,
        body: Any = None,
    ) -> None:
        """Answer the client after every candidate route has been exhausted."""
        if failure is None:
            message = (
                str(payload_error)
                if payload_error is not None
                else "No usable model route is configured"
            )
            self._finalize_generation(record, "failed", message=message)
            self._send_error(400, "invalid_request_error", message)
            return
        kind, message, status = failure
        if kind == "http" and status not in _RESUMABLE_HTTP_STATUS:
            # A client-side mistake from the upstream would just repeat on the
            # next attempt, so it is not worth saving for resumption.
            self._finalize_generation(record, "failed", message=message)
            self._send_error(502, "upstream_error", message)
            return
        if (
            kind == "http"
            and status in _QUEUEABLE_STATUS
            and record is not None
            and self.server.ledger is not None
            and _queue_on_quota_enabled()
        ):
            if self._queue_for_retry(record, message, status):
                return
        self._save_interrupted(
            tracker,
            prompt,
            model,
            system_prompt,
            message,
            max_tokens=max_tokens,
            body=body,
            record=record,
        )
        self._finalize_generation(record, "interrupted", message=message)
        client_message = message + _generation_recovery_hint(record)
        if kind == "timeout":
            self._send_error(504, "upstream_timeout", client_message)
        else:
            self._send_error(502, "upstream_error", client_message)

    def _queue_for_retry(self, record: GenerationRecord, message: str, status: int) -> bool:
        """Park a quota-failed generation for automatic retry.

        Returns True when the request was queued and answered; False when the
        attempt budget is exhausted so the caller falls back to the interrupt
        path. The queue itself lives in the ledger (status "queued"), so a
        server restart re-arms pending retries automatically.
        """
        ledger = self.server.ledger
        if ledger is None:
            return False
        attempt = GenerationLedger.queue_attempt(record) + 1
        if attempt > _queue_max_attempts():
            return False
        retry_after = self._upstream_retry_after or _queue_retry_seconds()
        retry_at = time.time() + retry_after
        try:
            ledger.queue(record, message, retry_at, attempt)
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not queue generation %s: %s", record.id, error)
            return False
        self.log_message(
            "queued generation %s (attempt %d, retry in %.1fs)",
            record.id,
            attempt,
            retry_after,
        )
        self._send_json(
            429,
            {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": (
                        message
                        + f". The request is queued (attempt {attempt}) and will "
                        + "retry automatically; poll GET /v1/generations/"
                        + record.id
                        + " for the result."
                    ),
                },
                "generation_id": record.id,
                "queued": True,
                "attempt": attempt,
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
        return True

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
        body: Any = None,
        record: GenerationRecord | None = None,
    ) -> None:
        if tracker is None:
            return
        if record is not None:
            # The durable generation record already holds this interruption
            # (finalized right after us); saving again would duplicate it.
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
                request=request_replay(body) or None,
            )
        except Exception as error:
            self.log_error("Could not record interrupted work: %s", error)

    def _clear_interrupted_on_match(self, tracker: WorkTracker | None, prompt: str) -> None:
        """Clear a saved interruption once the same prompt completes successfully."""
        if tracker is None or not prompt:
            return
        try:
            tracker.clear_if_matches(prompt)
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
        record: GenerationRecord | None = None,
        body: Any = None,
        candidates: list[tuple[ModelRoute, str]] | None = None,
        route_index: int = -1,
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if record is not None:
            self.send_header("X-LLMBridge-Generation", record.id)
        self.end_headers()
        events = iter(stream_events(upstream, model))
        partial_chunks: list[str] = []
        reported_tokens: int | None = None
        while True:
            try:
                event = next(events)
            except StopIteration:
                self._clear_interrupted_on_match(tracker, prompt)
                self._finalize_generation(
                    record,
                    "completed",
                    output="".join(partial_chunks),
                    output_tokens=reported_tokens,
                )
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
                remaining = (
                    candidates[route_index + 1:]
                    if candidates is not None and route_index >= 0
                    else []
                )
                if (
                    partial_output
                    and remaining
                    and _self_heal_enabled()
                ):
                    if self._heal_stream(
                        remaining,
                        body,
                        partial_chunks,
                        reported_tokens,
                        model,
                        record,
                        tracker,
                        prompt,
                        system,
                        max_tokens,
                    ):
                        return
                self._save_interrupted(
                    tracker,
                    prompt,
                    model,
                    system,
                    message,
                    progress=_stream_progress(partial_output, reported_tokens, max_tokens),
                    partial_output=partial_output,
                    max_tokens=max_tokens,
                    body=body,
                    record=record,
                )
                self._finalize_generation(
                    record,
                    "interrupted",
                    output=partial_output,
                    output_tokens=reported_tokens,
                    message=message,
                    progress=_stream_progress(partial_output, reported_tokens, max_tokens),
                )
                event = json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "upstream_error",
                            "message": message + _generation_recovery_hint(record),
                        },
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
                        body=body,
                        record=record,
                    )
                self._finalize_generation(
                    record,
                    "interrupted",
                    output=partial_output,
                    output_tokens=reported_tokens,
                    message="Client disconnected during streaming response",
                )
                return

    def _heal_stream(
        self,
        remaining: list[tuple[ModelRoute, str]],
        body: Any,
        partial_chunks: list[str],
        reported_tokens: int | None,
        model: str,
        record: GenerationRecord | None,
        tracker: WorkTracker | None,
        prompt: str,
        system: str | None,
        max_tokens: Any,
    ) -> bool:
        """Resume a broken stream on a fallback route, invisibly to the client.

        Tries each remaining candidate once; the first one that opens carries
        the continuation. Returns True when the outcome (success or a further
        failure) was fully handled, False when nothing was attempted and the
        caller should run the normal interrupt path.
        """
        partial_output = "".join(partial_chunks)
        heal_body = _build_heal_body(body, partial_output)
        if heal_body is None:
            return False
        for route, upstream_model in remaining:
            try:
                request = self._build_upstream_request(route, upstream_model, heal_body)
            except ValueError as error:
                self.log_error(
                    "Could not rebuild the request for %s: %s", upstream_model, error
                )
                continue
            try:
                upstream = urlopen(request, timeout=_upstream_timeout_seconds())
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                self.log_error("Heal route %s unavailable: %s", upstream_model, error)
                continue
            ledger = self.server.ledger
            if ledger is not None and record is not None:
                try:
                    ledger.heal(
                        record,
                        from_model=model,
                        to_model=upstream_model,
                        resumed_chars=len(partial_output),
                    )
                except Exception as error:  # pragma: no cover - defensive
                    self.log_error("Could not record heal: %s", error)
            self.log_message(
                "healing stream onto %s after %d chars", upstream_model, len(partial_output)
            )
            with upstream:
                self._relay_healed_stream(
                    upstream,
                    upstream_model,
                    partial_chunks,
                    reported_tokens,
                    model,
                    record,
                    tracker,
                    prompt,
                    system,
                    max_tokens,
                    body,
                )
            return True
        return False

    def _relay_healed_stream(
        self,
        upstream: Any,
        upstream_model: str,
        partial_chunks: list[str],
        reported_tokens: int | None,
        model: str,
        record: GenerationRecord | None,
        tracker: WorkTracker | None,
        prompt: str,
        system: str | None,
        max_tokens: Any,
        body: Any,
    ) -> None:
        """Stream a continuation into the client's still-open response.

        Only text deltas are forwarded (the original content block is still
        open); envelope events from the continuation are swallowed and the
        bridge emits its own closing message_delta/message_stop.
        """
        healed_tokens: int | None = None
        events = iter(stream_events(upstream, upstream_model))
        while True:
            try:
                event = next(events)
            except StopIteration:
                usage: dict[str, Any] = {}
                if reported_tokens is not None:
                    usage["output_tokens"] = reported_tokens + (healed_tokens or 0)
                elif healed_tokens is not None:
                    usage["output_tokens"] = healed_tokens
                else:
                    usage["output_tokens"] = 0
                closing = [
                    _sse_bytes(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": usage,
                        },
                    ),
                    _sse_bytes("message_stop", {"type": "message_stop"}),
                ]
                try:
                    for chunk in closing:
                        self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self._interrupt_after_client_loss(
                        partial_chunks,
                        reported_tokens if healed_tokens is None else healed_tokens,
                        "Client disconnected during streaming response",
                        model,
                        record,
                        tracker,
                        prompt,
                        system,
                        max_tokens,
                        body,
                    )
                    return
                self._clear_interrupted_on_match(tracker, prompt)
                self._finalize_generation(
                    record,
                    "completed",
                    output="".join(partial_chunks),
                    output_tokens=usage.get("output_tokens"),
                )
                return
            except (ProtocolError, TimeoutError, URLError, OSError, HTTPException) as error:
                if isinstance(error, ProtocolError):
                    message = str(error)
                elif isinstance(error, TimeoutError):
                    message = "Upstream stream timed out"
                else:
                    message = "Upstream stream failed"
                self.log_error("Healed stream failed: %s", message)
                partial_output = "".join(partial_chunks)
                progress = _stream_progress(
                    partial_output, healed_tokens or reported_tokens, max_tokens
                )
                self._save_interrupted(
                    tracker,
                    prompt,
                    model,
                    system,
                    message,
                    progress=progress,
                    partial_output=partial_output,
                    max_tokens=max_tokens,
                    body=body,
                    record=record,
                )
                self._finalize_generation(
                    record,
                    "interrupted",
                    output=partial_output,
                    output_tokens=healed_tokens or reported_tokens,
                    message=message,
                    progress=progress,
                )
                error_event = json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "upstream_error",
                            "message": message + _generation_recovery_hint(record),
                        },
                    }
                )
                try:
                    self.wfile.write(f"event: error\ndata: {error_event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.log_error("Client disconnected during streaming response")
                return
            data = _sse_event_data(event)
            if data is None:
                continue
            event_type = data.get("type")
            if event_type == "content_block_delta":
                delta = data.get("delta")
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    _collect_text_delta(event, partial_chunks)
                    try:
                        self.wfile.write(event)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        self._interrupt_after_client_loss(
                            partial_chunks,
                            healed_tokens or reported_tokens,
                            "Client disconnected during streaming response",
                            model,
                            record,
                            tracker,
                            prompt,
                            system,
                            max_tokens,
                            body,
                        )
                        return
            elif event_type == "message_delta":
                tokens = _event_output_tokens(event)
                if tokens is not None:
                    healed_tokens = tokens
            # message_start / ping / block start/stop events are swallowed: the
            # client's original message is still open and must not be restarted.

    def _interrupt_after_client_loss(
        self,
        partial_chunks: list[str],
        reported_tokens: int | None,
        message: str,
        model: str,
        record: GenerationRecord | None,
        tracker: WorkTracker | None,
        prompt: str,
        system: str | None,
        max_tokens: Any,
        body: Any,
    ) -> None:
        partial_output = "".join(partial_chunks)
        if partial_output:
            self._save_interrupted(
                tracker,
                prompt,
                model,
                system,
                message,
                progress=_stream_progress(partial_output, reported_tokens, max_tokens),
                partial_output=partial_output,
                max_tokens=max_tokens,
                body=body,
                record=record,
            )
        self._finalize_generation(
            record,
            "interrupted",
            output=partial_output,
            output_tokens=reported_tokens,
            message=message,
        )

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

    def _send_json(
        self, status: int, value: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, header_value in (headers or {}).items():
            self.send_header(key, header_value)
        generation_id = getattr(self, "_generation_id", None)
        if generation_id:
            self.send_header("X-LLMBridge-Generation", generation_id)
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
    ledger: GenerationLedger | None


def load_server_settings() -> "ServerSettings":
    """Resolve listener, routing, and interrupted-work settings from the environment."""
    host = os.environ.get("LLMBRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("LLMBRIDGE_PORT", "8787"))
    client_api_key = os.environ.get("LLMBRIDGE_API_KEY") or None
    _validate_listener_auth(host, client_api_key)
    try:
        model_routes = load_model_routes()
    except RoutingError as error:
        raise SystemExit(
            f"{error}\n\nNext steps:\n"
            "  llmbridge setup    # guided configuration\n"
            "  llmbridge doctor   # diagnose the environment"
        ) from error
    ledger = None
    if _truthy(os.environ.get("LLMBRIDGE_TRACK_GENERATIONS", "1")):
        ledger = GenerationLedger(os.environ.get("LLMBRIDGE_LEDGER_PATH") or None)
    work_tracker = None
    if _truthy(os.environ.get("LLMBRIDGE_TRACK_INTERRUPTED", "1")):
        # Interrupted work is stored as ledger records; share the server's
        # ledger so a generation is saved exactly once.
        work_tracker = WorkTracker(ledger)
    return ServerSettings(
        host=host,
        port=port,
        client_api_key=client_api_key,
        model_routes=model_routes,
        work_tracker=work_tracker,
        ledger=ledger,
    )


def main() -> None:  # pragma: no cover - process entrypoint, blocks in serve_forever
    settings = load_server_settings()
    server = LLMBridgeServer(
        (settings.host, settings.port),
        LLMBridgeHandler,
        settings.model_routes,
        settings.client_api_key,
        settings.work_tracker,
        settings.ledger,
    )
    if settings.ledger is not None:
        print(
            f"Recording generations to {settings.ledger.root} "
            "(set LLMBRIDGE_TRACK_GENERATIONS=0 to disable)"
        )
    print(f"LLMBridge listening on http://{settings.host}:{settings.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down LLMBridge")
    finally:
        server.server_close()
