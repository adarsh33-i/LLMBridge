import base64
import difflib
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .ledger import (
    GenerationLedger,
    GenerationRecord,
    _fingerprint,
    append_assistant_turn,
    request_replay,
)
from .protocol import (
    _MAX_STREAM_LINE_BYTES,
    _iter_stream_lines,
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

# The longest an upstream ``Retry-After`` may hold a request open before a
# same-route retry. Anything longer is not worth blocking the client for, so
# the route is abandoned in favour of the next fallback (or the queue).
_MAX_RETRY_AFTER_WAIT_SECONDS = 30.0

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

# How often a running server looks for generations orphaned by a dead process.
_STALE_SWEEP_SECONDS = 60.0

# Two identical non-streaming requests in flight at once normally cost two
# upstream calls. ``LLMBRIDGE_COALESCE_REQUESTS`` lets the second attach to the
# first and share its answer. ``deterministic`` only merges requests that pin
# temperature to 0 - the only case where two identical requests must agree -
# while ``1``/``always`` also merges retries at higher temperatures.
_COALESCE_OFF = "off"
_COALESCE_DETERMINISTIC = "deterministic"
_COALESCE_ALWAYS = "always"


def _coalesce_mode() -> str:
    """Read the duplicate-request policy; anything unrecognised means off."""
    raw = os.environ.get("LLMBRIDGE_COALESCE_REQUESTS")
    if raw is None:
        return _COALESCE_OFF
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on", "always"):
        return _COALESCE_ALWAYS
    if value in ("deterministic", "temp0", "temperature0"):
        return _COALESCE_DETERMINISTIC
    return _COALESCE_OFF


def _pins_temperature(body: Any) -> bool:
    """Whether this request asks for a temperature that makes output repeatable."""
    if not isinstance(body, dict):
        return False
    temperature = body.get("temperature")
    return (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and temperature == 0
    )


def _coalescible(body: Any, *, stream: bool) -> bool:
    """Whether a duplicate of this request may ride on the in-flight one."""
    mode = _coalesce_mode()
    if mode == _COALESCE_OFF or stream:
        return False
    if mode == _COALESCE_DETERMINISTIC and not _pins_temperature(body):
        return False
    return True


def _route_candidates(
    model_routes: dict[str, ModelRoute],
    primary_route: ModelRoute,
    primary_model: str,
) -> list[tuple[ModelRoute, str]]:
    """Primary route followed by its configured fallbacks, in order."""
    candidates: list[tuple[ModelRoute, str]] = [(primary_route, primary_model)]
    for alias in primary_route.fallbacks:
        try:
            route, upstream_model = resolve_model_route(alias, model_routes)
        except RoutingError as error:
            logger.warning("Skipping unusable fallback %s: %s", alias, error)
            continue
        candidate = (route, upstream_model)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


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


def _auto_resume_enabled() -> bool:
    """Opt-in: continue a re-sent request from its saved partial output.

    Off by default. A client that aborted on purpose and re-sends the same
    request may want a fresh answer, and the bridge cannot tell that apart
    from a retry after a timeout; the operator decides which is more common.
    """
    return _truthy(os.environ.get("LLMBRIDGE_AUTO_RESUME"))


_DEFAULT_ADAPTIVE_MIN_SAMPLES = 3


def _adaptive_fallbacks_enabled() -> bool:
    """Opt-in: order heal candidates by their measured seam quality.

    Off by default so a config's fallback order stays authoritative until the
    operator asks the bridge to learn from its own digest.
    """
    return _truthy(os.environ.get("LLMBRIDGE_ADAPTIVE_FALLBACKS"))


def _adaptive_min_samples() -> int:
    raw = os.environ.get("LLMBRIDGE_ADAPTIVE_MIN_SAMPLES")
    if raw is None:
        return _DEFAULT_ADAPTIVE_MIN_SAMPLES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_ADAPTIVE_MIN_SAMPLES
    return value if value >= 1 else _DEFAULT_ADAPTIVE_MIN_SAMPLES


def rank_heal_candidates(
    remaining: list[tuple[ModelRoute, str]],
    quality: dict[str, dict[str, Any]],
) -> tuple[list[tuple[ModelRoute, str]], list[dict[str, Any]]]:
    """Reorder heal candidates by evidence; keep config order as the tiebreak.

    A candidate with enough recorded seams sorts by its suspect rate, lowest
    first. Candidates with too few samples keep their config position relative
    to each other and sort after every trusted one, so the chain only moves
    where the digest can justify it. Returns the new order and a per-candidate
    explanation that is written into the ``heal`` event.
    """
    def key(item: tuple[int, tuple[ModelRoute, str]]) -> tuple[int, float, int]:
        position, (_, upstream_model) = item
        bucket = quality.get(upstream_model)
        if bucket is None or not bucket.get("trusted"):
            return (1, 0.0, position)
        return (0, float(bucket.get("suspect_rate", 0.0)), position)

    ordered = sorted(enumerate(remaining), key=key)
    explanation = []
    for position, (_, upstream_model) in ordered:
        bucket = quality.get(upstream_model) or {}
        explanation.append(
            {
                "model": upstream_model,
                "config_position": position,
                "samples": bucket.get("samples", 0),
                "suspect_rate": bucket.get("suspect_rate"),
                "trusted": bool(bucket.get("trusted")),
            }
        )
    return [candidate for _, candidate in ordered], explanation


def _shadow_after_seconds() -> float:
    """Seconds without a first token before a fallback is raced; 0 = off.

    Off by default: the race spends a second upstream call on purpose, so the
    operator chooses the delay that makes that trade worth it for a route.
    """
    raw = os.environ.get("LLMBRIDGE_SHADOW_AFTER_SECONDS")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


class _PrimedStream:
    """An upstream SSE stream whose first lines were read ahead by the race.

    ``stream_events`` pulls lines through ``readline``; this replays the lines
    the race consumed while waiting for a first token, then hands off to the
    live socket, so the winner's output reaches the client unchanged.
    """

    def __init__(self, upstream: Any, buffered: list[bytes], lines: Any = None) -> None:
        self._upstream = upstream
        self._buffered = list(buffered)
        self._lines = lines if lines is not None else iter(
            _iter_stream_lines(upstream, _MAX_STREAM_LINE_BYTES)
        )

    def readline(self, limit: int = -1) -> bytes:
        if self._buffered:
            return self._buffered.pop(0)
        return next(self._lines, b"")

    def read(self, size: int = -1) -> bytes:  # pragma: no cover - not used by SSE
        return self._upstream.read(size)

    def close(self) -> None:
        try:
            self._upstream.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> "_PrimedStream":
        return self

    def __exit__(self, *_: Any) -> bool:
        self.close()
        return False


def _line_has_token(line: bytes) -> bool:
    """Whether an upstream SSE line carries visible output (text or a tool call)."""
    stripped = line.strip()
    if not stripped.startswith(b"data:"):
        return False
    payload = stripped[5:].strip()
    if not payload or payload == b"[DONE]":
        return False
    try:
        chunk = json.loads(payload)
    except ValueError:
        return False
    if not isinstance(chunk, dict):
        return False
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and (delta.get("content") or delta.get("tool_calls")):
            return True
        if choice.get("finish_reason"):
            return True
    return False


class _Racer:
    """Open one upstream and read until its first visible token, in a thread."""

    def __init__(self, route: ModelRoute, upstream_model: str, request: Request) -> None:
        self.route = route
        self.upstream_model = upstream_model
        self.request = request
        self.upstream: Any = None
        self.buffered: list[bytes] = []
        self.error: BaseException | None = None
        self.ready = threading.Event()
        self.cancelled = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_Racer":
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            self.upstream = urlopen(self.request, timeout=_upstream_timeout_seconds())
            self._lines = iter(_iter_stream_lines(self.upstream, _MAX_STREAM_LINE_BYTES))
            while not self.cancelled:
                line = next(self._lines, b"")
                if not line:
                    break
                self.buffered.append(line)
                if _line_has_token(line):
                    break
        except Exception as error:  # reported to the chooser, never raised here
            self.error = error
        finally:
            self.ready.set()
            if self.cancelled:
                self.close()

    def close(self) -> None:
        self.cancelled = True
        upstream = self.upstream
        if upstream is not None:
            try:
                upstream.close()
            except Exception:  # pragma: no cover - defensive
                pass

    @property
    def has_token(self) -> bool:
        return self.error is None and self.upstream is not None and bool(self.buffered)


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


def stale_generation_seconds() -> float:
    """How old a ``running`` record must be before a restart may claim it.

    Twice the upstream timeout, never under five minutes: a generation that is
    genuinely in flight updates its record at most once, so the margin has to be
    wider than any one upstream call before a sweep treats the record as orphaned.
    """
    return max(2 * _upstream_timeout_seconds(), 300.0)


def settle_stale_generations(
    ledger: GenerationLedger, *, now: float | None = None
) -> list[Any]:
    """Interrupt ``running`` records left behind by a process that exited."""
    try:
        return ledger.settle_running(
            "Bridge restarted before the generation finished",
            older_than=stale_generation_seconds(),
            now=now,
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.error("Could not settle stale generations: %s", error)
        return []


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


def _retry_sleep(attempt: int, retry_after: float | None = None) -> None:
    """Wait before re-trying a route: exponential backoff, but never less
    than the delay the upstream asked for with ``Retry-After``."""
    delay = _route_retry_backoff_seconds() * (2 ** (attempt - 1))
    if retry_after is not None:
        delay = max(delay, retry_after)
    if delay > 0:
        time.sleep(delay)


def _parse_retry_after(raw: Any) -> float | None:
    """Seconds to wait from a ``Retry-After`` header, or None if unusable.

    RFC 9110 allows either delay-seconds or an HTTP-date; both are accepted.
    A date in the past means "retry now" (0 seconds).
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, when.timestamp() - time.time())


def _is_retryable_status(status: int) -> bool:
    return status in _RESUMABLE_HTTP_STATUS or status >= 500


# A retry that failed for one of these reasons can plausibly succeed later
# without the request changing: quota, a server-side 5xx, a dropped connection,
# a timeout, or a reply the upstream mangled. A deterministic 4xx (bad key,
# bad request) cannot, so it stops the retry loop instead of burning the budget.
_QUEUE_TRANSIENT_KINDS = frozenset({"connect", "timeout", "invalid"})


def _queue_retryable(failure: "_Failure") -> bool:
    kind, _, status = failure
    if kind == "http":
        return (
            isinstance(status, int)
            and not isinstance(status, bool)
            and (status in _QUEUEABLE_STATUS or status >= 500 or status == 408)
        )
    return kind in _QUEUE_TRANSIENT_KINDS


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


_HANDOFF_VERSION = 1
_HANDOFF_MAX_BYTES = 2 * 1024 * 1024
_DEFAULT_HANDOFF_TTL_SECONDS = 24 * 3600.0


def _handoff_secret() -> bytes | None:
    """Key that signs handoff tokens: ``LLMBRIDGE_HANDOFF_SECRET`` or the API key.

    None means no secret is configured, so tokens cannot be minted or accepted.
    Two bridges that should hand work to each other share one of these values.
    """
    secret = os.environ.get("LLMBRIDGE_HANDOFF_SECRET") or os.environ.get("LLMBRIDGE_API_KEY")
    if not secret or len(secret) < 16:
        return None
    return secret.encode("utf-8")


def _handoff_ttl_seconds() -> float:
    return _env_positive_number("LLMBRIDGE_HANDOFF_TTL_SECONDS", _DEFAULT_HANDOFF_TTL_SECONDS)


def mint_handoff_token(record: GenerationRecord, *, secret: bytes, now: float | None = None) -> str:
    """Pack an interrupted record into a compact signed blob another bridge can resume.

    The payload is the replayable request, the partial text, the source model,
    the fingerprint, and the source bridge's generation id, so the receiving
    bridge records lineage back to it. Signed with HMAC-SHA256; the signature
    covers the exact payload bytes.
    """
    payload = {
        "v": _HANDOFF_VERSION,
        "iat": now if now is not None else time.time(),
        "generation_id": record.id,
        "fingerprint": record.fingerprint,
        "model": record.model,
        "prompt": record.prompt,
        "system": record.system,
        "max_tokens": (record.params or {}).get("max_tokens"),
        "request": record.request or {},
        "partial_output": record.output or "",
        "output_tokens": record.output_tokens,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    )


def _b64decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def verify_handoff_token(token: str, *, secret: bytes, now: float | None = None) -> dict[str, Any]:
    """Check a handoff token's signature, age and shape; return its payload.

    Raises ``ValueError`` with a reason safe to show the caller.
    """
    if not isinstance(token, str) or token.count(".") != 1:
        raise ValueError("handoff token is malformed")
    if len(token) > _HANDOFF_MAX_BYTES:
        raise ValueError("handoff token is too large")
    raw_part, sig_part = token.split(".", 1)
    try:
        raw = _b64decode(raw_part)
        sig = _b64decode(sig_part)
    except (ValueError, TypeError) as error:
        raise ValueError("handoff token is malformed") from error
    expected = hmac.new(secret, raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("handoff token signature does not match this bridge's secret")
    try:
        payload = json.loads(raw)
    except ValueError as error:
        raise ValueError("handoff token payload is not JSON") from error
    if not isinstance(payload, dict) or payload.get("v") != _HANDOFF_VERSION:
        raise ValueError("handoff token version is not supported")
    issued = payload.get("iat")
    current = now if now is not None else time.time()
    if not isinstance(issued, (int, float)) or issued > current + 60:
        raise ValueError("handoff token has an invalid issue time")
    if current - issued > _handoff_ttl_seconds():
        raise ValueError("handoff token has expired")
    if not isinstance(payload.get("partial_output"), str):
        raise ValueError("handoff token has no partial output")
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"]:
        raise ValueError("handoff token has no prompt")
    return payload


def _state_from_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    """Shape a verified handoff payload like the tracker's interrupted state."""
    max_tokens = payload.get("max_tokens")
    return {
        "generation_id": payload.get("generation_id"),
        "prompt": payload["prompt"],
        "model_name": payload.get("model"),
        "system": payload.get("system"),
        "partial_output": payload["partial_output"],
        "max_tokens": max_tokens if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) else None,
        "request": payload.get("request") or None,
        "fingerprint": payload.get("fingerprint"),
        "output_tokens": payload.get("output_tokens"),
        "handoff": True,
    }


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
        messages = append_assistant_turn(messages, partial_output)
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
    healed["messages"] = append_assistant_turn(
        [message for message in body["messages"] if isinstance(message, dict)],
        partial_output,
    )
    if _heal_hint_enabled():
        healed = _with_system_addendum(healed, _HEAL_HINT)
    return healed


# A healed stream joins two models inside one response, so a bad join is invisible
# to the client by design. These checks score the boundary and flag the suspicious
# ones in the ledger; they never block or alter a heal.
_SEAM_WINDOW = 160
_SEAM_MIN_OVERLAP_CHARS = 12
_SEAM_MIN_OVERLAP_RATIO = 0.25
# Models decorate the start of a fresh answer with markdown, quotes or list
# markers. Strip that layer before asking whether the text opens like a restart.
_SEAM_DECORATION_RE = re.compile(r"^[\s>*_#`~\"'\-\u2022\d.)\]]*")
_SEAM_RESTART_RE = re.compile(
    r"^(?:sure\b|certainly\b|of course\b|no problem\b|absolutely\b|"
    r"great question\b|happy to help\b|okay\b|alright\b|as an ai language model\b|"
    r"here(?:'s| is)\b|below is\b|continuing\b|resuming\b|picking up\b|"
    r"to continue\b|next\b|in summary\b|"
    r"i can (?:help|absolutely|do that)\b|"
    r"i'?ll (?:help|continue|start|begin|answer|pick)\b|"
    r"let me (?:help|start|look|continue|pick))",
    re.IGNORECASE,
)


def _strip_seam_decoration(text: str) -> str:
    return _SEAM_DECORATION_RE.sub("", text, count=1)


def is_restart_opener(continuation: str) -> bool:
    """Whether text opens like a fresh answer rather than a continuation."""
    return bool(_SEAM_RESTART_RE.match(_strip_seam_decoration(continuation)))


def score_seam(partial_output: str, continuation: str) -> dict[str, Any]:
    """Score the joint between a truncated answer and its healed continuation.

    ``duplicated`` catches the classic stitching artifact: the fallback re-emits text
    the first model already produced. ``restarted`` catches a fallback that ignored
    the appended assistant turn and began a fresh answer instead.
    """
    before = partial_output[-_SEAM_WINDOW:].lower()
    after = continuation[:_SEAM_WINDOW].lower()
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    match = matcher.find_longest_match(0, len(before), 0, len(after))
    overlap = match.size
    shortest = min(len(before), len(after))
    ratio = round(overlap / shortest, 3) if shortest else 0.0
    # Only a match that joins the tail of the partial text to the head of the
    # continuation is evidence of re-emission. A shared phrase in the middle of
    # a short continuation is ordinary language, not a bad seam.
    anchored = match.a + match.size == len(before) and match.b == 0
    duplicated = (
        anchored
        and overlap >= _SEAM_MIN_OVERLAP_CHARS
        and ratio >= _SEAM_MIN_OVERLAP_RATIO
    )
    restarted = is_restart_opener(continuation)
    return {
        "continuation_chars": len(continuation),
        "overlap_chars": overlap,
        "overlap_ratio": ratio,
        "duplicated": duplicated,
        "restarted": restarted,
        "suspect": duplicated or restarted,
    }


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


def _track_open_blocks(event: bytes, state: dict[str, Any]) -> None:
    """Track which content blocks the client still has open.

    The heal path appends into the client's live message, so it has to know
    whether the answer stopped inside text or inside a tool call. ``state`` is
    ``{"open": {index: block_type}, "finished": bool}``.
    """
    data = _sse_event_data(event)
    if not data:
        return
    kind = data.get("type")
    index = data.get("index")
    if kind == "content_block_start" and isinstance(index, int):
        block = data.get("content_block")
        block_type = block.get("type") if isinstance(block, dict) else None
        state["open"][index] = (
            block_type if isinstance(block_type, str) else "unknown"
        )
        state.setdefault("seen", set()).add(index)
    elif kind == "content_block_stop" and isinstance(index, int):
        state["open"].pop(index, None)
    elif kind == "message_stop":
        state["finished"] = True


def _heal_text_index(state: dict[str, Any]) -> int | None:
    """Index of the open text block a continuation may append to, else None.

    A continuation is only safe while the answer is open inside text. Appending
    after a tool_use block opened would hand the client a tool call whose
    arguments stop mid-JSON, and appending after message_stop would reopen a
    message the client already finished.
    """
    if state["finished"]:
        return None
    open_blocks: dict[int, str] = state["open"]
    if any(block_type != "text" for block_type in open_blocks.values()):
        return None
    text_indices = [
        index for index, block_type in open_blocks.items() if block_type == "text"
    ]
    return max(text_indices) if text_indices else None


def _heal_tool_calls_enabled() -> bool:
    """Whether a stream that dies inside a tool call may be finished by a fallback.

    On by default alongside self-heal. It changes one thing for every tool
    stream: tool_use blocks are delivered to the client whole, at the moment
    they complete, instead of argument by argument. Clients act on a tool call
    only once its JSON is complete, so nothing observable is lost, and a model
    that dies mid-call has then shown the client nothing it has to retract.
    """
    return _truthy(os.environ.get("LLMBRIDGE_HEAL_TOOL_CALLS", "1"))


def _new_tool_state() -> dict[str, Any]:
    return {"held": {}, "order": []}


def _hold_tool_event(event: bytes, tool_state: dict[str, Any]) -> bool:
    """Buffer tool_use block events until the block is known to be complete.

    Returns True when the event was held and must not be written yet. Text
    events and envelope events are never held. A ``content_block_stop`` for a
    held block is not held either: the caller flushes the block before writing
    it, which is the moment the block becomes visible to the client.
    """
    data = _sse_event_data(event)
    if not data:
        return False
    kind = data.get("type")
    index = data.get("index")
    held = tool_state["held"]
    if kind == "content_block_start" and isinstance(index, int):
        block = data.get("content_block")
        if isinstance(block, dict) and block.get("type") == "tool_use":
            held[index] = {
                "events": [event],
                "json": "",
                "id": block.get("id"),
                "name": block.get("name"),
            }
            tool_state["order"].append(index)
            return True
        return False
    if kind == "content_block_delta" and isinstance(index, int) and index in held:
        delta = data.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                held[index]["json"] += partial
        held[index]["events"].append(event)
        return True
    return False


def _tool_block_complete(entry: dict[str, Any]) -> bool:
    """A held tool block is complete when its arguments parse as a JSON object.

    An empty argument string is *not* complete: a stream that died right after
    the block opened has not finished the call, even though ``{}`` would parse.
    A block the upstream closed never reaches this check: it is released from
    the hold the moment its ``content_block_stop`` arrives.
    """
    raw = entry["json"].strip()
    if not raw:
        return False
    try:
        return isinstance(json.loads(raw), dict)
    except ValueError:
        return False


def _release_tool_block(tool_state: dict[str, Any], index: int) -> list[bytes]:
    """Take a held block's buffered events out of the hold, in arrival order."""
    entry = tool_state["held"].pop(index, None)
    if entry is None:
        return []
    if index in tool_state["order"]:
        tool_state["order"].remove(index)
    return list(entry["events"])


def _split_held_tools(tool_state: dict[str, Any]) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    """Held blocks that completed vs. the ones cut off, both in stream order."""
    done: list[tuple[int, dict[str, Any]]] = []
    cut: list[tuple[int, dict[str, Any]]] = []
    for index in list(tool_state["order"]):
        entry = tool_state["held"][index]
        (done if _tool_block_complete(entry) else cut).append((index, entry))
    return done, cut


def _tool_heal_hint(
    partial_text: str,
    issued: list[dict[str, Any]],
    cut: list[dict[str, Any]],
) -> str:
    """System text that tells the fallback exactly where the first model stopped."""
    lines = [
        "The previous assistant turn was cut off by a connection failure while it "
        "was issuing a tool call. Continue that same turn; do not start a new answer "
        "and do not repeat text or tool calls that were already produced.",
    ]
    if issued:
        lines.append("Tool calls already issued in this turn (do not repeat them):")
        for call in issued:
            lines.append(f"  - {call['name']} id={call['id']} input={call['input']}")
    for call in cut:
        name = call.get("name") or "<unknown>"
        prefix = (call.get("json") or "").strip()
        if prefix:
            lines.append(
                f"The turn stopped while calling tool `{name}`; the arguments written "
                f"so far were: {prefix}"
            )
        else:
            lines.append(f"The turn stopped while beginning a call to tool `{name}`.")
    lines.append(
        "Finish that tool call now, with complete arguments, using the same tool. "
        "Add any further tool calls the turn still needs."
    )
    return "\n".join(lines)


_HEAL_HINT = (
    "The previous assistant turn was cut off mid-answer by a connection failure. "
    "Continue it from exactly where it stops: do not restate, summarise, or "
    "acknowledge the interruption, and do not begin with a greeting or preamble. "
    "Your first characters should read as the next characters of that text."
)


def _seam_events_enabled() -> bool:
    """Opt-in: tell the client, in the stream, how its heal was scored.

    Emits one ``llmbridge_seam`` SSE event after ``message_stop``, so a strict
    Anthropic parser that stops at message_stop never sees it, while a client
    that opted in can decide to re-ask rather than trust a suspect join at the
    moment it matters. Off by default: it is a non-standard event.
    """
    return _truthy(os.environ.get("LLMBRIDGE_SEAM_EVENTS"))


def _seam_event_bytes(
    record: GenerationRecord | None, audit: dict[str, Any] | None, from_model: str, to_model: str, seam_at: int,
) -> bytes:
    """The verdict as the client sees it: what was joined, where, and whether it looks wrong."""
    payload: dict[str, Any] = {
        "type": "llmbridge_seam",
        "generation_id": record.id if record is not None else None,
        "from_model": from_model,
        "to_model": to_model,
        "seam_at": seam_at,
        "audited": audit is not None,
    }
    if audit is not None:
        payload.update({
            key: audit.get(key)
            for key in ("suspect", "duplicated", "restarted", "overlap_chars", "overlap_ratio",
                        "continuation_chars", "tool_call", "tool_finished", "hinted")
            if key in audit
        })
    return _sse_bytes("llmbridge_seam", payload)


def _heal_hint_enabled() -> bool:
    """Opt-in: tell the fallback, in the system prompt, to continue mid-sentence.

    Off by default so the digest keeps measuring un-hinted joins; turn it on to
    compare restart rates per model with and without the hint (`llmbridge stats`
    reports both sides).
    """
    return _truthy(os.environ.get("LLMBRIDGE_HEAL_HINT"))


def _with_system_addendum(body: dict[str, Any], addendum: str) -> dict[str, Any]:
    """Return a copy of ``body`` with ``addendum`` appended to its system prompt."""
    patched = dict(body)
    system = patched.get("system")
    if isinstance(system, list):
        patched["system"] = [*system, {"type": "text", "text": addendum}]
    elif isinstance(system, str) and system:
        patched["system"] = system + "\n\n" + addendum
    else:
        patched["system"] = addendum
    return patched


def _build_tool_heal_body(
    body: Any,
    partial_text: str,
    issued: list[dict[str, Any]],
    cut: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Rebuild a request so a fallback finishes a turn cut off inside a tool call.

    Text already streamed becomes an assistant prefill, as in a text heal.
    Completed tool calls are *described* in the system addendum rather than
    replayed as ``tool_use`` blocks: an assistant turn ending in tool calls
    with no results is rejected by most OpenAI-compatible upstreams, and the
    client already holds those calls, so they only need to be not repeated.
    """
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None
    healed = dict(body)
    messages = [message for message in body["messages"] if isinstance(message, dict)]
    if partial_text:
        messages = append_assistant_turn(messages, partial_text)
    healed["messages"] = messages
    return _with_system_addendum(healed, _tool_heal_hint(partial_text, issued, cut))


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


@dataclass
class _Inflight:
    """One in-flight request that an identical duplicate may attach to."""

    done: threading.Event = field(default_factory=threading.Event)
    generation_id: str | None = None
    status: int = 502
    payload: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


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
        self._inflight: dict[str, _Inflight] = {}
        self._inflight_lock = threading.Lock()
        if ledger is not None:
            # A process that died mid-stream leaves ``running`` records behind.
            # Sweep for them while serving too, so one created just before a
            # crash is settled once it is old enough instead of staying stuck.
            self._sweep_thread = threading.Thread(
                target=self._stale_sweep_loop, daemon=True
            )
            self._sweep_thread.start()
        else:
            self._sweep_thread = None
        if ledger is not None and _queue_on_quota_enabled():
            # The queue lives in the ledger, so this also re-arms retries that
            # were pending when the process last exited.
            self._queue_thread = threading.Thread(
                target=self._dispatch_queue_loop, daemon=True
            )
            self._queue_thread.start()

    def join_inflight(self, fingerprint: str) -> tuple[bool, _Inflight]:
        """Lead this request, or attach to the identical one already running."""
        with self._inflight_lock:
            entry = self._inflight.get(fingerprint)
            if entry is not None:
                return False, entry
            entry = _Inflight()
            self._inflight[fingerprint] = entry
            return True, entry

    def finish_inflight(
        self,
        fingerprint: str,
        entry: _Inflight,
        result: tuple[int, dict[str, Any], dict[str, str], str | None] | None,
    ) -> None:
        """Publish the leader's answer to waiting duplicates and stop accepting joins."""
        with self._inflight_lock:
            if self._inflight.get(fingerprint) is entry:
                del self._inflight[fingerprint]
        if result is None:
            entry.status = 502
            entry.payload = {
                "type": "error",
                "error": {
                    "type": "upstream_error",
                    "message": "The original request failed before producing a response",
                },
            }
        else:
            entry.status, entry.payload, entry.headers, entry.generation_id = result
        entry.done.set()

    def consolidate_audits(self) -> dict[str, Any] | None:
        """Flush seam audits into the durable digest; never raises."""
        ledger = self.ledger
        if ledger is None:
            return None
        try:
            return ledger.consolidate()
        except Exception as error:  # pragma: no cover - defensive
            logger.error("Could not consolidate heal audits: %s", error)
            return None

    def stop(self) -> None:
        """Stop background work, settle unfinished records, flush the digest."""
        self._queue_stop.set()
        ledger = self.ledger
        if ledger is not None:
            # The process is exiting, so anything still ``running`` will never be
            # finished by this one; leaving it that way hides the work from every
            # resume surface. A handler that does finish afterwards re-completes
            # the record, which is the truthful outcome.
            try:
                ledger.settle_running("Bridge stopped before the generation finished")
            except Exception as error:  # pragma: no cover - defensive
                logger.error("Could not settle running generations: %s", error)
        self.consolidate_audits()

    def shutdown(self) -> None:
        self.stop()
        super().shutdown()

    def _stale_sweep_loop(self) -> None:
        while not self._queue_stop.wait(_STALE_SWEEP_SECONDS):
            ledger = self.ledger
            if ledger is not None:
                settle_stale_generations(ledger)

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
        """Re-send one queued generation, following its fallbacks, and finalize it."""
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
        failure: _Failure | None = None
        quota_failure: _Failure | None = None
        for candidate_route, candidate_model in _route_candidates(
            self.model_routes, route, upstream_model
        ):
            try:
                payload = anthropic_to_openai(
                    body,
                    candidate_model,
                    reasoning_effort=candidate_route.reasoning_effort,
                    supports_images=candidate_route.supports_images,
                    supports_tools=candidate_route.supports_tools,
                    cache_control=candidate_route.cache_control,
                )
                request = _upstream_request(payload, candidate_route)
            except (ValueError, RoutingError) as error:
                # The saved request can no longer be translated (or the route's
                # key is gone). Another poll cannot fix that, so fail once
                # instead of retrying every second forever.
                ledger.fail(record, f"queued request can no longer be sent: {error}")
                return
            try:
                with urlopen(request, timeout=_upstream_timeout_seconds()) as upstream:
                    reply = json.loads(
                        _read_bounded(upstream, _MAX_UPSTREAM_RESPONSE_BYTES)
                    )
            except HTTPError as error:
                failure = ("http", f"Upstream returned HTTP {error.code}", error.code)
                if error.code in _QUEUEABLE_STATUS:
                    quota_failure = failure
                error.close()
                logger.info(
                    "Queued generation %s got HTTP %s from %s",
                    record.id,
                    error.code,
                    candidate_model,
                )
                continue
            except (URLError, TimeoutError, OSError) as error:
                logger.info("Queued generation %s failed: %s", record.id, error)
                failure = (
                    "connect",
                    "Could not connect to the configured upstream",
                    None,
                )
                continue
            except (json.JSONDecodeError, ProtocolError) as error:
                logger.info(
                    "Queued generation %s got an invalid reply: %s", record.id, error
                )
                failure = (
                    "invalid",
                    f"upstream returned an invalid response: {error}",
                    None,
                )
                continue
            try:
                anthropic_response = openai_to_anthropic(reply, record.model)
            except ValueError as error:
                # A reply the gateway cannot translate is not an answer. Treat it
                # like a broken stream so the attempt budget still applies to it.
                logger.info(
                    "Queued generation %s got an untranslatable reply: %s",
                    record.id,
                    error,
                )
                failure = (
                    "invalid",
                    f"upstream returned an invalid response: {error}",
                    None,
                )
                continue
            output = _anthropic_text(anthropic_response)
            ledger.complete(record, output, _upstream_output_tokens(reply))
            # Only after the work is finished: a failure between the two writes
            # must leave it recoverable, never marked resumed but incomplete.
            tracker = self.work_tracker
            if tracker is not None:
                try:
                    tracker.resolve_matching(
                        record.fingerprint, by=record.id, model=record.model
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            logger.info(
                "Queued generation %s completed on %s (attempt %d)",
                record.id,
                candidate_model,
                attempt,
            )
            return
        self._requeue_or_fail(
            record,
            attempt,
            quota_failure
            or failure
            or ("invalid", "no usable model route is configured", None),
        )

    def _requeue_or_fail(
        self, record: GenerationRecord, attempt: int, failure: _Failure
    ) -> None:
        """After a failed retry, park it again or give up for good."""
        ledger = self.ledger
        if ledger is None:
            return
        _, message, _ = failure
        if _queue_retryable(failure) and attempt < _queue_max_attempts():
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
        if path == "/v1/stats":
            self._send_stats()
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
        # Name what is still waiting, in the same shape GET /v1/interrupted
        # uses, so a client that cleared one job can see the others.
        remaining = (
            [record.id for record in tracker.ledger.interrupted()]
            if tracker is not None
            else []
        )
        self._send_json(200, {"cleared": cleared, "remaining_recoverable": remaining})

    def _send_interrupted(self) -> None:
        tracker = self.server.work_tracker
        state = tracker.load_state() if tracker is not None else None
        if state is None:
            self._send_error(404, "not_found", "No interrupted work is recorded")
            return
        # Resume serves only the newest job, so say how many others are waiting
        # and name them; each is addressable by id via /v1/generations/{id}/fork.
        pending = tracker.ledger.interrupted()
        state["recoverable_count"] = len(pending)
        state["other_recoverable"] = [
            record.id for record in pending if record.id != state.get("generation_id")
        ]
        secret = _handoff_secret()
        newest = next((r for r in pending if r.id == state.get("generation_id")), None)
        if secret is not None and newest is not None:
            # Another bridge sharing the secret can resume this without the ledger.
            state["handoff_token"] = mint_handoff_token(newest, secret=secret)
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

    def _send_stats(self) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        self._send_json(200, ledger.stats())

    def _send_generation(self, path: str) -> None:
        ledger = self._ledger()
        if ledger is None:
            return
        generation_id = self._generation_id_from_path(path)
        record = ledger.load(generation_id) if generation_id else None
        if record is None:
            self._send_error(404, "not_found", "Generation not found")
            return
        payload = record.to_dict()
        secret = _handoff_secret()
        if secret is not None and record.status == "interrupted" and record.output:
            payload["handoff_token"] = mint_handoff_token(record, secret=secret)
        self._send_json(200, payload)

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
        if at_event is not None and (
            not isinstance(at_event, int) or isinstance(at_event, bool)
        ):
            # Silently ignoring a malformed sequence would label the fork with a
            # different step than the caller asked for.
            self._send_error(
                400, "invalid_request_error", "at_event must be an integer event sequence"
            )
            return
        if at_event is None:
            at_event = record.events[-1].seq if record.events else None
        elif not record.events or not 0 <= at_event < len(record.events):
            # Lineage points at a step that exists; a bogus sequence number would
            # record a fork of a generation state that never happened.
            last = len(record.events) - 1
            self._send_error(
                400,
                "invalid_request_error",
                f"at_event must be an event sequence between 0 and {last}",
            )
            return
        fork_body = GenerationLedger.fork_request(
            record,
            model=model,
            max_tokens=max_tokens,
            stream=bool(body.get("stream")),
        )
        child = ledger.create(
            fork_body, model, parent=record.id, fork_event_seq=at_event
        )
        # A fork of an interrupted run settles that run once it succeeds, the
        # same way a resume does. Forking a completed generation changes nothing.
        resolves = record if record.status == "interrupted" else None
        self._handle_messages(
            fork_body, model, route, upstream_model, record=child, resolves=resolves
        )

    def _resume_interrupted(self) -> None:
        """Re-issue saved interrupted work to a (possibly different) model.

        The work comes from this bridge's ledger, or, when the body carries a
        ``handoff_token`` minted by a bridge that shares this one's secret, from
        that token: no shared ledger is needed to move work between machines.
        """
        tracker = self.server.work_tracker
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
        token = body.get("handoff_token")
        state: dict[str, Any] | None
        if token is not None:
            secret = _handoff_secret()
            if secret is None:
                self._send_error(
                    400, "invalid_request_error",
                    "This bridge has no LLMBRIDGE_HANDOFF_SECRET (or LLMBRIDGE_API_KEY) "
                    "to verify handoff tokens with",
                )
                return
            try:
                state = _state_from_handoff(verify_handoff_token(token, secret=secret))
            except ValueError as error:
                self._send_error(400, "invalid_request_error", str(error))
                return
        else:
            state = tracker.load_state() if tracker is not None else None
            if state is None:
                self._send_error(404, "not_found", "No interrupted work is recorded")
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
        ledger = self.server.ledger
        handed_off = bool(state.get("handoff"))
        if ledger is None:
            # Without a generation ledger there is no lineage to attach the
            # second run to, so keep the earlier contract: replay the work and
            # drop the record it came from.
            if tracker is not None and not handed_off:
                tracker.clear_state()
            self._handle_messages(resume_body, model, route, upstream_model)
            return
        source = ledger.load(state["generation_id"]) if state.get("generation_id") else None
        child: GenerationRecord | None = None
        if source is not None:
            child = ledger.create(
                resume_body,
                model,
                parent=source.id,
                fork_event_seq=source.events[-1].seq if source.events else None,
            )
        elif handed_off:
            # The source lives on another bridge. Record the child here under the
            # original request's fingerprint and name the remote source in a
            # `handoff` event, so lineage is traceable across the two ledgers.
            child = ledger.create(resume_body, model, fingerprint=state.get("fingerprint") or None)
            try:
                ledger.handoff(
                    child,
                    source_generation=state.get("generation_id"),
                    source_model=state.get("model_name"),
                    resumed_chars=len(state.get("partial_output") or ""),
                )
            except Exception as error:  # pragma: no cover - defensive
                self.log_error("Could not record handoff: %s", error)
            self.log_message(
                "resuming handed-off work %s (%d chars) on %s",
                state.get("generation_id"), len(state.get("partial_output") or ""), model,
            )
        # Settling the interruption happens inside the run: deleting the source
        # would erase the only evidence that this answer needed a second try,
        # which is what keeps resume outcomes measurable.
        self._handle_messages(
            resume_body, model, route, upstream_model, record=child, resolves=source
        )

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
        if self._try_auto_resume(body, model, route, upstream_model):
            return
        self._handle_messages(body, model, route, upstream_model, coalesce=True)

    def _coalesce_or_lead(self, body: Any, fingerprint: str) -> tuple[bool, _Inflight | None]:
        """Join an identical in-flight request, or become its leader.

        Returns ``(served, entry)``. ``served`` is True when this request was
        answered from the leader's result and nothing more should be sent.
        ``entry`` is the in-flight slot this request now leads (to be finished
        with ``finish_inflight``), or None when coalescing does not apply.
        """
        stream = isinstance(body, dict) and bool(body.get("stream"))
        if not _coalescible(body, stream=stream):
            return False, None
        leader, entry = self.server.join_inflight(fingerprint)
        if leader:
            return False, entry
        if self._serve_coalesced(entry):
            return True, None
        return False, None

    def _try_auto_resume(
        self,
        body: Any,
        model: str,
        primary_route: ModelRoute,
        primary_model: str,
    ) -> bool:
        """Continue an interrupted run of this exact request instead of redoing it.

        Only when ``LLMBRIDGE_AUTO_RESUME`` is on. The match is the request
        fingerprint, so a byte-identical re-send (a client timeout retry, an
        agent loop, or a second bridge sharing the ledger) picks up where the
        earlier run stopped: the saved text is replayed to the client first,
        then the continuation is appended. The upstream is opened before any
        byte is sent, so when no route answers this returns False and the
        ordinary request path runs and reports the failure the usual way.
        """
        if not _auto_resume_enabled() or not isinstance(body, dict):
            return False
        ledger = self.server.ledger
        if ledger is None:
            return False
        fingerprint = _fingerprint(body)
        source = ledger.resumable(fingerprint)
        if source is None:
            return False
        partial_output = source.output
        resume_body = _build_heal_body(body, partial_output)
        if resume_body is None:
            return False
        tracker = self.server.work_tracker
        prompt, system_prompt = _extract_prompt_and_system(body)
        max_tokens = body.get("max_tokens")
        stream = bool(body.get("stream"))
        # Two identical re-sends arriving together must not both resume the
        # same source: the second attaches to the first exactly as an ordinary
        # duplicate would, and receives the resumed answer.
        served, inflight = self._coalesce_or_lead(body, fingerprint)
        if served:
            return True
        self._request_fingerprint = fingerprint
        self._last_response = None
        try:
            return self._auto_resume_inner(
                body, model, primary_route, primary_model, ledger, source,
                partial_output, resume_body, tracker, prompt, system_prompt,
                max_tokens, stream, fingerprint,
            )
        finally:
            if inflight is not None:
                self.server.finish_inflight(fingerprint, inflight, self._last_response)

    def _auto_resume_inner(
        self,
        body: dict[str, Any],
        model: str,
        primary_route: ModelRoute,
        primary_model: str,
        ledger: GenerationLedger,
        source: GenerationRecord,
        partial_output: str,
        resume_body: dict[str, Any],
        tracker: WorkTracker | None,
        prompt: str,
        system_prompt: str | None,
        max_tokens: Any,
        stream: bool,
        fingerprint: str,
    ) -> bool:
        opened: tuple[Any, str] | None = None
        for route, upstream_model in self._route_candidates(primary_route, primary_model):
            try:
                request = self._build_upstream_request(route, upstream_model, resume_body)
            except ValueError as error:
                self.log_error("Could not rebuild %s for auto-resume: %s", source.id, error)
                continue
            try:
                opened = (urlopen(request, timeout=_upstream_timeout_seconds()), upstream_model)
                break
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                if isinstance(error, HTTPError):
                    error.close()
                self.log_error("Auto-resume route %s unavailable: %s", upstream_model, error)
        if opened is None:
            return False
        upstream, upstream_model = opened

        child = ledger.create(
            resume_body,
            model,
            parent=source.id,
            fork_event_seq=source.events[-1].seq if source.events else None,
            fingerprint=fingerprint,
        )
        # The same handler state _handle_messages would have set, so the
        # completion settles ``source`` by id and any failure is recorded
        # against ``child`` under the original request's identity.
        self._resolves = source
        self._generation_id = child.id
        try:
            ledger.heal(
                child,
                from_model=source.model,
                to_model=upstream_model,
                resumed_chars=len(partial_output),
                reason="auto_resume",
                hinted=_heal_hint_enabled(),
            )
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not record auto-resume: %s", error)
        self.log_message(
            "auto-resuming %s as %s on %s after %d chars",
            source.id, child.id, upstream_model, len(partial_output),
        )
        partial_chunks = [partial_output]
        seam_at = len(partial_output)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-LLMBridge-Generation", child.id)
            self.send_header("X-LLMBridge-Resumed", source.id)
            self.end_headers()
            opening = [
                _sse_bytes(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": f"msg_{uuid.uuid4().hex}",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                ),
                _sse_bytes("ping", {"type": "ping"}),
                _sse_bytes(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                _sse_bytes(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": partial_output},
                    },
                ),
            ]
            try:
                for chunk in opening:
                    self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                upstream.close()
                self._interrupt_after_client_loss(
                    partial_chunks, source.output_tokens,
                    "Client disconnected during streaming response",
                    source.model, child, tracker, prompt, system_prompt, max_tokens, resume_body,
                )
                return True
            with upstream:
                self._relay_healed_stream(
                    upstream, upstream_model, partial_chunks, source.output_tokens,
                    source.model, child, tracker, prompt, system_prompt, max_tokens,
                    resume_body, seam_at,
                )
            return True

        response, failure = self._read_json_reply(upstream)
        anthropic_response: dict[str, Any] | None = None
        if failure is None:
            try:
                anthropic_response = openai_to_anthropic(response, model)
            except ValueError as error:
                self.log_error("Upstream returned an invalid response: %s", error)
                failure = ("invalid", f"upstream returned an invalid response: {error}", None)
        if failure is not None or anthropic_response is None:
            message = failure[1] if failure is not None else "Upstream returned an invalid response"
            self._finalize_generation(
                child, "interrupted", output=partial_output,
                output_tokens=source.output_tokens, message=message,
            )
            self._send_error(502, "upstream_error", message + _generation_recovery_hint(child))
            return True
        continuation = _anthropic_text(anthropic_response)
        partial_chunks.append(continuation)
        content = anthropic_response.get("content")
        if isinstance(content, list):
            first_text = next(
                (b for b in content if isinstance(b, dict) and b.get("type") == "text"), None
            )
            if first_text is not None:
                first_text["text"] = partial_output + str(first_text.get("text") or "")
            else:
                content.insert(0, {"type": "text", "text": partial_output})
        else:
            anthropic_response["content"] = [{"type": "text", "text": partial_output}]
        audit = self._audit_seam(child, partial_chunks, seam_at, source.model, upstream_model)
        extra_headers = {"X-LLMBridge-Resumed": source.id}
        if _seam_events_enabled() and audit is not None:
            extra_headers["X-LLMBridge-Seam"] = "suspect" if audit.get("suspect") else "clean"
        usage = anthropic_response.get("usage")
        output_tokens = _upstream_output_tokens(response)
        total_tokens = output_tokens
        if isinstance(output_tokens, int) and isinstance(source.output_tokens, int):
            total_tokens = output_tokens + source.output_tokens
        if isinstance(usage, dict) and isinstance(total_tokens, int):
            usage["output_tokens"] = total_tokens
        # The record carries the whole answer, so it carries the whole count,
        # matching what the streaming resume path writes.
        self._finalize_generation(
            child, "completed", output="".join(partial_chunks), output_tokens=total_tokens,
        )
        self._settle_interrupted_work(tracker, child, model)
        self._send_json(200, anthropic_response, headers=extra_headers)
        return True

    def _read_json_body(self) -> Any:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > 10 * 1024 * 1024:
            raise ProtocolError("Request body must be between 1 byte and 10 MiB")
        return json.loads(self.rfile.read(size))

    def _route_candidates(
        self, primary_route: ModelRoute, primary_model: str
    ) -> list[tuple[ModelRoute, str]]:
        """Primary route followed by its configured fallbacks, in order."""
        return _route_candidates(
            self.server.model_routes, primary_route, primary_model
        )

    def _handle_messages(
        self,
        body: Any,
        model: str,
        primary_route: ModelRoute,
        primary_model: str,
        record: GenerationRecord | None = None,
        resolves: GenerationRecord | None = None,
        coalesce: bool = False,
    ) -> None:
        """Translate, call upstream (following fallbacks), and answer the client.

        ``resolves`` is the interrupted record this request replays: when it
        completes, that record is settled by id instead of by request identity.
        ``coalesce`` lets a duplicate of this request share the answer instead
        of paying for a second upstream call.
        """
        tracker = self.server.work_tracker
        self._request_fingerprint = _fingerprint(body)
        self._resolves = resolves
        self._last_response = None
        prompt = ""
        system_prompt: str | None = None
        if tracker is not None and isinstance(body, dict):
            prompt, system_prompt = _extract_prompt_and_system(body)
        display_model = model or primary_model
        stream = isinstance(body, dict) and bool(body.get("stream"))
        entry: _Inflight | None = None
        if coalesce and _coalescible(body, stream=stream):
            leader, entry = self.server.join_inflight(self._request_fingerprint)
            if not leader:
                if self._serve_coalesced(entry):
                    return
                # The first request outlived the client timeout, so serve this
                # one independently rather than hang; entry stays unowned.
                entry = None
        try:
            if record is None:
                record = self._begin_generation(body, display_model)
            self._generation_id = record.id if record is not None else None
            candidates = self._route_candidates(primary_route, primary_model)
            max_tokens = body.get("max_tokens") if isinstance(body, dict) else None
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
        finally:
            if entry is not None:
                self.server.finish_inflight(
                    self._request_fingerprint, entry, self._last_response
                )

    def _serve_coalesced(self, entry: _Inflight) -> bool:
        """Replay the leader's response to a waiting duplicate.

        Returns False when the leader did not finish within the client timeout,
        leaving the caller to serve the request independently.
        """
        if not entry.done.wait(_client_timeout_seconds()):
            return False
        if entry.payload is None:
            self._send_error(
                502,
                "upstream_error",
                "The original request failed before producing a response",
            )
            return True
        headers = dict(entry.headers)
        headers["X-LLMBridge-Coalesced"] = "1"
        if entry.generation_id:
            headers["X-LLMBridge-Generation"] = entry.generation_id
        self.log_message(
            "coalesced a duplicate request into %s", entry.generation_id or "an in-flight run"
        )
        self._send_json(entry.status, entry.payload, headers=headers)
        return True

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
                    if self._wait_to_retry_http(failure, attempt):
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
                try:
                    anthropic_response = openai_to_anthropic(response, display_model)
                except ValueError as error:
                    # The upstream said 200 but the body is not a completion.
                    # Try the next candidate rather than dropping the connection.
                    self.log_error("Upstream returned an invalid response: %s", error)
                    failure = (
                        "invalid",
                        f"upstream returned an invalid response: {error}",
                        None,
                    )
                    break
                self._finalize_generation(
                    record,
                    "completed",
                    output=_anthropic_text(anthropic_response),
                    output_tokens=_upstream_output_tokens(response),
                )
                self._settle_interrupted_work(tracker, record, display_model)
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
        shadow_after = _shadow_after_seconds()
        if shadow_after > 0 and len(candidates) > 1:
            raced = self._race_shadow_stream(
                candidates, body, display_model, tracker, prompt, system_prompt,
                max_tokens, record, shadow_after,
            )
            if raced:
                return
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
                    if self._wait_to_retry_http(failure, attempt):
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

    def _wait_to_retry_http(self, failure: _Failure, attempt: int) -> bool:
        """Sleep and return True when an HTTP failure should retry this route.

        A ``Retry-After`` longer than ``_MAX_RETRY_AFTER_WAIT_SECONDS`` means
        another attempt here is pointless for now, so the route is skipped and
        the next candidate gets the request instead.
        """
        status = failure[2]
        if (
            status is None
            or not _is_retryable_status(status)
            or attempt >= _route_attempts()
        ):
            return False
        retry_after = self._upstream_retry_after
        if retry_after is not None and retry_after > _MAX_RETRY_AFTER_WAIT_SECONDS:
            self.log_message(
                "upstream asked to wait %.0fs; moving on instead of retrying",
                retry_after,
            )
            return False
        _retry_sleep(attempt, retry_after)
        return True

    def _race_shadow_stream(
        self,
        candidates: list[tuple[ModelRoute, str]],
        body: Any,
        display_model: str,
        tracker: WorkTracker | None,
        prompt: str,
        system_prompt: str | None,
        max_tokens: Any,
        record: GenerationRecord | None,
        shadow_after: float,
    ) -> bool:
        """Race the primary against its first fallback when the first token is slow.

        The primary is opened at once. If it has not produced a visible token
        within ``shadow_after`` seconds, the first fallback is opened too and
        whichever yields a token first carries the stream; the other is closed.
        A loser that already sent text is not kept: the client only ever sees
        one model's opening. Returns True when a stream was served (or the race
        failed in a way that was fully reported), False to fall back to the
        ordinary sequential path, which re-opens the primary.
        """
        primary_route, primary_model = candidates[0]
        shadow_route, shadow_model = candidates[1]
        try:
            primary_request = self._build_upstream_request(primary_route, primary_model, body)
            shadow_request = self._build_upstream_request(shadow_route, shadow_model, body)
        except ValueError:
            return False
        started = time.monotonic()
        primary = _Racer(primary_route, primary_model, primary_request).start()
        shadow: _Racer | None = None
        winner: _Racer | None = None
        reason = ""
        if primary.ready.wait(shadow_after) and primary.has_token:
            winner, reason = primary, "primary_first_token"
        elif primary.ready.is_set() and primary.error is not None and not primary.buffered:
            # Primary failed outright before the delay elapsed: no race to run,
            # let the sequential path report it and try the fallbacks in order.
            primary.close()
            return False
        else:
            shadow = _Racer(shadow_route, shadow_model, shadow_request).start()
            deadline = time.monotonic() + _upstream_timeout_seconds()
            while winner is None and time.monotonic() < deadline:
                if primary.ready.is_set() and primary.has_token:
                    winner, reason = primary, "primary_recovered"
                    break
                if shadow.ready.is_set() and shadow.has_token:
                    winner, reason = shadow, "shadow_first_token"
                    break
                if primary.ready.is_set() and not primary.has_token:
                    # The primary is dead. Nothing is left to race: the shadow
                    # is now the only candidate, so stream it the moment it
                    # opens instead of waiting for its first token here.
                    primary.close()
                    if shadow.ready.is_set():
                        break  # shadow failed too; sequential path reports it
                    shadow.ready.wait(max(0.0, deadline - time.monotonic()))
                    if shadow.has_token:
                        winner, reason = shadow, "primary_failed"
                    break
                time.sleep(0.01)
        for racer in (primary, shadow):
            if racer is not None and racer is not winner:
                racer.close()
        if winner is None:
            return False
        waited_ms = round((time.monotonic() - started) * 1000, 1)
        route_index = 0 if winner is primary else 1
        ledger = self.server.ledger
        if ledger is not None and record is not None:
            try:
                loser_racer = (shadow if winner is primary else primary) if shadow is not None else None
                ledger.shadow(
                    record,
                    winner=winner.upstream_model,
                    loser=loser_racer.upstream_model if loser_racer is not None else None,
                    reason=reason,
                    delay_seconds=shadow_after,
                    waited_ms=waited_ms,
                    # A loser whose request reached the provider was billed for
                    # whatever it generated before close(). It was sent if its
                    # response opened, or if it was still waiting for one when
                    # cancelled (urlopen sends before it blocks); only a failed
                    # connect means nothing was billed.
                    loser_sent=bool(
                        loser_racer is not None
                        and (loser_racer.upstream is not None or not loser_racer.ready.is_set())
                    ),
                )
            except Exception as error:  # pragma: no cover - defensive
                self.log_error("Could not record shadow race: %s", error)
        if shadow is not None:
            self.log_message(
                "shadow race: %s won after %.0f ms (%s)", winner.upstream_model, waited_ms, reason
            )
        upstream = _PrimedStream(
            winner.upstream, winner.buffered, getattr(winner, "_lines", None)
        )
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
                route_index=route_index,
            )
        return True

    def _record_http_failure(self, error: HTTPError) -> _Failure:
        status = error.code
        retry_after = None
        if error.headers is not None:
            retry_after = _parse_retry_after(error.headers.get("Retry-After"))
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
                fingerprint=getattr(self, "_request_fingerprint", "") or None,
            )
        except Exception as error:
            self.log_error("Could not record interrupted work: %s", error)

    def _work_ledger(self, tracker: WorkTracker | None) -> GenerationLedger | None:
        """The ledger interrupted work should be settled in."""
        ledger = self.server.ledger
        if ledger is None and tracker is not None:
            # A tracker always owns a ledger, even when generation tracking
            # was not configured separately.
            return tracker.ledger
        return ledger

    def _settle_interrupted_work(
        self,
        tracker: WorkTracker | None,
        record: GenerationRecord | None,
        model: str,
    ) -> None:
        """Close out the interrupted work this successful request answers for.

        A resume passes the record it replayed, so that interruption is settled
        by id. Any other successful request settles by its own request identity:
        a client that re-sends a dead request should stop being offered it. The
        match is a fingerprint, not prompt text, because two conversations can
        share a prompt and finishing one must not erase the other's work.
        """
        source = getattr(self, "_resolves", None)
        if source is not None:
            if record is None or record.status != "completed":
                return
            ledger = self._work_ledger(tracker)
            if ledger is None:
                return
            try:
                ledger.resume(source, by=record.id, model=model)
            except Exception as error:
                self.log_error("Could not mark interrupted work resumed: %s", error)
            return
        if tracker is None:
            return
        fingerprint = getattr(self, "_request_fingerprint", "")
        if not fingerprint:
            return
        try:
            tracker.resolve_matching(
                fingerprint, by=getattr(self, "_generation_id", None), model=model
            )
        except Exception as error:
            self.log_error("Could not mark interrupted work resumed: %s", error)

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
        block_state: dict[str, Any] = {"open": {}, "finished": False, "seen": set()}
        hold_tools = _heal_tool_calls_enabled()
        tool_state = _new_tool_state()
        while True:
            try:
                event = next(events)
            except StopIteration:
                self._finalize_generation(
                    record,
                    "completed",
                    output="".join(partial_chunks),
                    output_tokens=reported_tokens,
                )
                self._settle_interrupted_work(tracker, record, model)
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
                heal_index = _heal_text_index(block_state)
                if (
                    partial_output
                    and remaining
                    and heal_index is not None
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
                        block_index=heal_index,
                        open_indices=sorted(block_state["open"]),
                    ):
                        return
                if (
                    hold_tools
                    and tool_state["held"]
                    and remaining
                    and not block_state["finished"]
                    and _self_heal_enabled()
                ):
                    if self._heal_tool_stream(
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
                        block_state,
                        tool_state,
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
            _track_open_blocks(event, block_state)
            _collect_text_delta(event, partial_chunks)
            if reported_tokens is None:
                tokens = _event_output_tokens(event)
                if tokens is not None:
                    reported_tokens = tokens
            pending: list[bytes] = []
            if hold_tools:
                if _hold_tool_event(event, tool_state):
                    continue
                stop_data = _sse_event_data(event)
                if (
                    stop_data is not None
                    and stop_data.get("type") == "content_block_stop"
                    and stop_data.get("index") in tool_state["held"]
                ):
                    # The block is complete: show the client the whole call at
                    # once, then the stop that closes it.
                    pending = _release_tool_block(tool_state, stop_data["index"])
            try:
                for held_event in pending:
                    self.wfile.write(held_event)
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

    def _rank_remaining(
        self,
        remaining: list[tuple[ModelRoute, str]],
        model: str,
    ) -> tuple[list[tuple[ModelRoute, str]], list[dict[str, Any]] | None]:
        """Apply evidence-based ordering to heal candidates when it is enabled."""
        ledger = self.server.ledger
        if ledger is None or len(remaining) < 2 or not _adaptive_fallbacks_enabled():
            return remaining, None
        try:
            quality = ledger.fallback_quality(model, min_samples=_adaptive_min_samples())
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not read fallback quality: %s", error)
            return remaining, None
        if not quality:
            return remaining, None
        ranked, note = rank_heal_candidates(remaining, quality)
        if [m for _, m in ranked] != [m for _, m in remaining]:
            self.log_message(
                "adaptive fallbacks for %s: %s", model, " > ".join(m for _, m in ranked)
            )
        return ranked, note

    def _open_heal_candidate(
        self,
        remaining: list[tuple[ModelRoute, str]],
        heal_body: dict[str, Any],
        label: str,
    ) -> tuple[Any, str] | None:
        """Open the first candidate that accepts ``heal_body``; None if none does."""
        for route, upstream_model in remaining:
            try:
                request = self._build_upstream_request(route, upstream_model, heal_body)
            except ValueError as error:
                self.log_error("Could not rebuild the %s request for %s: %s", label, upstream_model, error)
                continue
            try:
                return urlopen(request, timeout=_upstream_timeout_seconds()), upstream_model
            except HTTPError as error:
                error.close()
                self.log_error("%s route %s unavailable: HTTP %s", label, upstream_model, error.code)
            except (URLError, TimeoutError, OSError) as error:
                self.log_error("%s route %s unavailable: %s", label, upstream_model, error)
        return None

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
        block_index: int = 0,
        open_indices: list[int] | None = None,
    ) -> bool:
        """Resume a broken stream on a fallback route, invisibly to the client.

        Tries each remaining candidate once; the first one that opens carries
        the continuation. ``block_index`` is the text block the caller still has
        open, and ``open_indices`` are the blocks the healed stream must close
        before its final ``message_delta``. Returns True when the outcome
        (success or a further failure) was fully handled, False when nothing
        was attempted and the caller should run the normal interrupt path.
        """
        partial_output = "".join(partial_chunks)
        heal_body = _build_heal_body(body, partial_output)
        if heal_body is None:
            return False
        ledger = self.server.ledger
        remaining, order_note = self._rank_remaining(remaining, model)
        opened = self._open_heal_candidate(remaining, heal_body, "heal")
        if opened is None:
            return False
        upstream, upstream_model = opened
        if ledger is not None and record is not None:
            try:
                ledger.heal(
                    record,
                    from_model=model,
                    to_model=upstream_model,
                    resumed_chars=len(partial_output),
                    order=order_note,
                    hinted=_heal_hint_enabled(),
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
                len(partial_output),
                block_index=block_index,
                open_indices=open_indices,
            )
        return True

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
        seam_at: int,
        block_index: int = 0,
        open_indices: list[int] | None = None,
    ) -> None:
        """Stream a continuation into the client's still-open response.

        Only text deltas are forwarded and they are re-indexed onto
        ``block_index``, the text block the caller left open. Envelope events
        from the continuation are swallowed; the bridge closes ``open_indices``
        itself before its own final ``message_delta``/``message_stop``, so the
        client never sees a message end with a block still open. ``seam_at`` is
        where the first model's text ends, so the join can be audited after.
        """
        if open_indices is None:
            open_indices = [block_index]
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
                        "content_block_stop",
                        {"type": "content_block_stop", "index": index},
                    )
                    for index in open_indices
                ]
                closing.append(
                    _sse_bytes(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": usage,
                        },
                    )
                )
                closing.append(_sse_bytes("message_stop", {"type": "message_stop"}))
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
                audit = self._audit_seam(record, partial_chunks, seam_at, model, upstream_model)
                self._emit_seam_event(record, audit, model, upstream_model, seam_at)
                self._finalize_generation(
                    record,
                    "completed",
                    output="".join(partial_chunks),
                    output_tokens=usage.get("output_tokens"),
                )
                self._settle_interrupted_work(tracker, record, model)
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
                self._audit_seam(record, partial_chunks, seam_at, model, upstream_model)
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
                    forwarded = _sse_bytes(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": delta,
                        },
                    )
                    try:
                        self.wfile.write(forwarded)
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

    def _heal_tool_stream(
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
        block_state: dict[str, Any],
        tool_state: dict[str, Any],
    ) -> bool:
        """Finish a turn that died inside a tool call on a fallback route.

        Held tool blocks whose arguments parse are released to the client first
        under their original ids: they are complete, and the fallback is told
        not to repeat them. The cut-off call was never shown to the client, so
        the fallback issues it afresh under its own id and the client sees one
        coherent block list. Returns True when the outcome was fully handled.
        """
        done, cut = _split_held_tools(tool_state)
        if not cut and not done:
            return False
        issued: list[dict[str, Any]] = []
        for index, entry in done:
            issued.append({"id": entry["id"], "name": entry["name"], "input": entry["json"].strip() or "{}"})
        cut_calls = [{"name": entry["name"], "json": entry["json"]} for _, entry in cut]
        partial_text = "".join(partial_chunks)
        heal_body = _build_tool_heal_body(body, partial_text, issued, cut_calls)
        if heal_body is None:
            return False
        ledger = self.server.ledger
        remaining, order_note = self._rank_remaining(remaining, model)
        opened = self._open_heal_candidate(remaining, heal_body, "tool-heal")
        if opened is None:
            return False
        upstream, upstream_model = opened
        # Release the complete calls now: the fallback has accepted the turn,
        # so the client will get a finished message either way.
        try:
            for index, _ in done:
                for held_event in _release_tool_block(tool_state, index):
                    self.wfile.write(held_event)
                self.wfile.write(
                    _sse_bytes("content_block_stop", {"type": "content_block_stop", "index": index})
                )
                block_state["open"].pop(index, None)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            upstream.close()
            self._interrupt_after_client_loss(
                partial_chunks, reported_tokens,
                "Client disconnected during streaming response",
                model, record, tracker, prompt, system, max_tokens, body,
            )
            return True
        for index, _ in cut:
            _release_tool_block(tool_state, index)
            block_state["open"].pop(index, None)
        if ledger is not None and record is not None:
            try:
                ledger.heal(
                    record,
                    from_model=model,
                    to_model=upstream_model,
                    resumed_chars=len(partial_text),
                    reason="tool_call",
                    order=order_note,
                    hinted=True,
                    tool={
                        "cut": [{"name": c["name"], "json_chars": len(c["json"])} for c in cut_calls],
                        "issued": [{"id": c["id"], "name": c["name"]} for c in issued],
                    },
                )
            except Exception as error:  # pragma: no cover - defensive
                self.log_error("Could not record tool heal: %s", error)
        self.log_message(
            "healing tool call %s onto %s after %d chars",
            ", ".join(c["name"] or "?" for c in cut_calls) or "(none)",
            upstream_model,
            len(partial_text),
        )
        with upstream:
            self._relay_healed_tool_stream(
                upstream, upstream_model, partial_chunks, reported_tokens, model,
                record, tracker, prompt, system, max_tokens, body, block_state,
                expected_names=[c["name"] for c in cut_calls],
            )
        return True

    def _relay_healed_tool_stream(
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
        block_state: dict[str, Any],
        expected_names: list[str | None],
    ) -> None:
        """Relay a fallback's tool-call continuation into the client's open message.

        Text goes onto the text block the client still has open, or a new one.
        Each tool block the fallback opens is re-indexed above every index the
        client has seen and forwarded with the fallback's own id. The bridge
        closes every open block, then ends the message with the fallback's own
        stop reason, so a turn that ended in a tool call still says ``tool_use``.
        """
        seam_at = len("".join(partial_chunks))
        seen: set[int] = set(block_state.get("seen") or set())
        next_index = (max(seen) + 1) if seen else 0
        open_text = [i for i, kind in block_state["open"].items() if kind == "text"]
        text_index: int | None = max(open_text) if open_text else None
        open_indices: list[int] = sorted(block_state["open"])
        index_map: dict[int, int] = {}
        issued_names: list[str] = []
        healed_tokens: int | None = None
        stop_reason = "end_turn"
        events = iter(stream_events(upstream, upstream_model))

        def write(chunk: bytes) -> bool:
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                self._interrupt_after_client_loss(
                    partial_chunks, healed_tokens or reported_tokens,
                    "Client disconnected during streaming response",
                    model, record, tracker, prompt, system, max_tokens, body,
                )
                return False

        while True:
            try:
                event = next(events)
            except StopIteration:
                usage: dict[str, Any] = {"output_tokens": (reported_tokens or 0) + (healed_tokens or 0)}
                closing = [
                    _sse_bytes("content_block_stop", {"type": "content_block_stop", "index": i})
                    for i in sorted(open_indices)
                ]
                closing.append(_sse_bytes("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage,
                }))
                closing.append(_sse_bytes("message_stop", {"type": "message_stop"}))
                if not write(b"".join(closing)):
                    return
                audit = self._audit_tool_seam(
                    record, partial_chunks, seam_at, model, upstream_model,
                    expected_names, issued_names,
                )
                self._emit_seam_event(record, audit, model, upstream_model, seam_at)
                self._finalize_generation(
                    record, "completed", output="".join(partial_chunks),
                    output_tokens=usage["output_tokens"],
                )
                self._settle_interrupted_work(tracker, record, model)
                return
            except (ProtocolError, TimeoutError, URLError, OSError, HTTPException) as error:
                message = str(error) if isinstance(error, ProtocolError) else (
                    "Upstream stream timed out" if isinstance(error, TimeoutError) else "Upstream stream failed"
                )
                self.log_error("Healed tool stream failed: %s", message)
                partial_output = "".join(partial_chunks)
                progress = _stream_progress(partial_output, healed_tokens or reported_tokens, max_tokens)
                self._save_interrupted(
                    tracker, prompt, model, system, message, progress=progress,
                    partial_output=partial_output, max_tokens=max_tokens, body=body, record=record,
                )
                self._finalize_generation(
                    record, "interrupted", output=partial_output,
                    output_tokens=healed_tokens or reported_tokens, message=message, progress=progress,
                )
                error_event = json.dumps({
                    "type": "error",
                    "error": {"type": "upstream_error", "message": message + _generation_recovery_hint(record)},
                })
                try:
                    self.wfile.write(f"event: error\ndata: {error_event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.log_error("Client disconnected during streaming response")
                return
            data = _sse_event_data(event)
            if data is None:
                continue
            kind = data.get("type")
            index = data.get("index")
            if kind == "content_block_start" and isinstance(index, int):
                block = data.get("content_block") or {}
                if block.get("type") == "text":
                    if text_index is not None:
                        index_map[index] = text_index  # append onto the open text block
                        continue
                    text_index = next_index
                else:
                    issued_names.append(str(block.get("name") or ""))
                index_map[index] = next_index
                open_indices.append(next_index)
                next_index += 1
                if not write(_sse_bytes("content_block_start", {**data, "index": index_map[index]})):
                    return
            elif kind == "content_block_delta" and isinstance(index, int):
                target = index_map.get(index)
                if target is None:
                    continue
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    _collect_text_delta(event, partial_chunks)
                if not write(_sse_bytes("content_block_delta", {**data, "index": target})):
                    return
            elif kind == "content_block_stop":
                # Closed in one place, at the end, so every index is stopped once.
                continue
            elif kind == "message_delta":
                tokens = _event_output_tokens(event)
                if tokens is not None:
                    healed_tokens = tokens
                delta = data.get("delta") or {}
                if isinstance(delta.get("stop_reason"), str):
                    stop_reason = delta["stop_reason"]
            # message_start / ping are swallowed: the client's message is still open.

    def _emit_seam_event(
        self,
        record: GenerationRecord | None,
        audit: dict[str, Any] | None,
        from_model: str,
        to_model: str,
        seam_at: int,
    ) -> None:
        """After message_stop, tell an opted-in client how the join was scored.

        Written last and never retried: the message is already complete, so a
        write failure here changes nothing the client has committed to.
        """
        if not _seam_events_enabled():
            return
        try:
            self.wfile.write(_seam_event_bytes(record, audit, from_model, to_model, seam_at))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _audit_tool_seam(
        self,
        record: GenerationRecord | None,
        partial_chunks: list[str],
        seam_at: int,
        from_model: str,
        to_model: str,
        expected_names: list[str | None],
        issued_names: list[str],
    ) -> dict[str, Any] | None:
        """Score a tool-call heal: did the fallback finish the call it was asked to?

        The text join (if any) is scored like every other seam. On top of that,
        a continuation that called a different tool than the one cut off, or no
        tool at all, is marked suspect: the client got a complete message, but
        not the turn the first model was making.
        """
        ledger = self.server.ledger
        if ledger is None or record is None:
            return None
        output = "".join(partial_chunks)
        if len(output) > seam_at and seam_at > 0:
            audit = score_seam(output[:seam_at], output[seam_at:])
        else:
            audit = {
                "continuation_chars": max(0, len(output) - seam_at),
                "overlap_chars": 0,
                "overlap_ratio": 0.0,
                "duplicated": False,
                "restarted": False,
                "suspect": False,
            }
        wanted = [name for name in expected_names if name]
        finished = all(name in issued_names for name in wanted) if wanted else bool(issued_names)
        audit.update({
            "hinted": True,
            "tool_call": True,
            "tool_expected": wanted,
            "tool_issued": issued_names,
            "tool_finished": finished,
            "suspect": bool(audit["suspect"]) or not finished,
        })
        try:
            ledger.heal_seam(record, from_model=from_model, to_model=to_model, seam_at=seam_at, audit=audit)
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not record tool heal seam: %s", error)
        return audit

    def _audit_seam(
        self,
        record: GenerationRecord | None,
        partial_chunks: list[str],
        seam_at: int,
        from_model: str,
        to_model: str,
    ) -> dict[str, Any] | None:
        """Flag a healed boundary the client cannot see. Never alters the response.

        Clean seams are recorded too: without them the suspect count has no
        denominator, and a heal rate cannot be read off the ledger. Returns the
        audit so the caller can also tell the client about it, if asked to.
        """
        ledger = self.server.ledger
        if ledger is None or record is None:
            return None
        output = "".join(partial_chunks)
        if len(output) <= seam_at:
            # The fallback contributed no text, so there is no joint to compare.
            # Scoring it would record a clean seam for a heal that never joined
            # anything, inflating the denominator with unearned passes.
            return None
        audit = score_seam(output[:seam_at], output[seam_at:])
        audit["hinted"] = _heal_hint_enabled()
        try:
            ledger.heal_seam(
                record,
                from_model=from_model,
                to_model=to_model,
                seam_at=seam_at,
                audit=audit,
            )
        except Exception as error:  # pragma: no cover - defensive
            self.log_error("Could not record heal seam: %s", error)
        return audit

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
        generation_id = getattr(self, "_generation_id", None)
        # Capture before writing: a duplicate request waiting on this one should
        # still receive the answer even if this client has already gone away.
        self._last_response = (status, value, dict(headers or {}), generation_id)
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, header_value in (headers or {}).items():
            self.send_header(key, header_value)
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


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:
    """Turn SIGTERM into the same path SIGINT takes, so ``stop()`` still runs.

    A plain SIGTERM kills the interpreter without unwinding ``main``'s
    ``finally``: running records would stay ``running`` and unconsolidated
    seam audits would be lost, which is exactly what graceful shutdown is
    documented to prevent.
    """
    raise KeyboardInterrupt


def install_shutdown_signals() -> None:
    """Route SIGTERM through the KeyboardInterrupt path; main thread only."""
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    except (ValueError, OSError):  # pragma: no cover - not the main thread
        pass


def main() -> None:  # pragma: no cover - process entrypoint, blocks in serve_forever
    install_shutdown_signals()
    settings = load_server_settings()
    if settings.ledger is not None:
        settled = settle_stale_generations(settings.ledger)
        if settled:
            print(
                f"Settled {len(settled)} generation(s) left running by a previous "
                "process as interrupted; resume them with `llmbridge fork <id>`"
            )
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
        # Flush seam audits before the records that hold them go out of reach;
        # nothing else is guaranteed to run once the process exits.
        server.stop()
        server.server_close()
