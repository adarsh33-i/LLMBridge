"""Durable Generations: an event-sourced, forkable ledger of model generations.

Every generation that flows through the bridge can be recorded as an
append-only sequence of events. Because a record keeps the original request
plus whatever assistant text was produced, a generation can later be:

* listed and inspected -- ``GET /v1/generations`` / ``GET /v1/generations/{id}``
* forked -- continued as a *new* generation from the point it reached,
  optionally on a different model (``POST /v1/generations/{id}/fork``)

This turns a one-shot, lossy request/response into durable, addressable work.
A generation is only ever appended to, so the record is a faithful history of
how the output was produced -- including the model that produced it and, for
forks, the parent generation it grew out of.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

if os.name == "nt":  # pragma: no cover - Windows-only lock implementation
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = ".llmbridge/generations"
_DEFAULT_FORK_MAX_TOKENS = 4096

# The ledger keeps only the newest records; older ones are pruned whenever a
# new generation is created. Interrupted work is surfaced from these records,
# so anything worth resuming must be among the retained few. Override with
# LLMBRIDGE_LEDGER_MAX_RECORDS.
MAX_TRACKED_INTERRUPTED: int = 10


def _env_max_records() -> int:
    """Read the retention cap from the environment, falling back on bad input."""
    raw = os.environ.get("LLMBRIDGE_LEDGER_MAX_RECORDS")
    if raw is None:
        return MAX_TRACKED_INTERRUPTED
    try:
        value = int(raw)
    except ValueError:
        return MAX_TRACKED_INTERRUPTED
    return value if value >= 1 else MAX_TRACKED_INTERRUPTED


def _lock_path_for(storage_path: Path) -> Path:
    """Return a stable lock-file path in the system temp directory.

    Keeping locks out of the working directory means a tracker never litters
    the project with ``.lock`` files. The name is derived from the resolved
    state path so every process targeting the same state file shares a lock.
    """
    key = hashlib.sha256(str(storage_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "llmbridge-locks" / f"{storage_path.name}.{key}.lock"


@contextlib.contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Serialize access to a ledger record across threads and processes.

    Uses an advisory file lock (fcntl on POSIX, msvcrt on Windows) so that
    concurrent writers sharing the same record cannot interleave reads,
    writes, or deletions. The lock is released automatically if a process
    exits, so a crash cannot leave the lock permanently held.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        if os.name == "nt":  # pragma: no cover - Windows-only lock implementation
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

# Event kinds. ``start`` opens a generation; ``complete`` / ``interrupt`` /
# ``fail`` close it; ``fork`` records that this generation descends from another;
# ``heal`` records a mid-stream failover onto a fallback model; ``queue``
# records that quota exhaustion parked the request for an automatic retry.
EVENT_START = "start"
EVENT_COMPLETE = "complete"
EVENT_INTERRUPT = "interrupt"
EVENT_FAIL = "fail"
EVENT_FORK = "fork"
EVENT_HEAL = "heal"
EVENT_QUEUED = "queue"

_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_INTERRUPTED = "interrupted"
_STATUS_FAILED = "failed"
_STATUS_QUEUED = "queued"

_ID_RE = re.compile(r"^gen_[A-Za-z0-9]+$")

# Scalar request parameters worth remembering for a fork. The message list and
# system prompt are stored separately so they can be rebuilt exactly.
_PARAM_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream")


def _validate_generation_id(generation_id: str) -> str:
    """Reject ids that could escape the ledger directory via a crafted name."""
    if not isinstance(generation_id, str) or not _ID_RE.match(generation_id):
        raise ValueError("invalid generation id")
    return generation_id


def _fingerprint(body: Any) -> str:
    """Content address for a request, so identical requests share a fingerprint."""
    try:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = str(body)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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


def _prompt_and_system(body: Any) -> tuple[str, Optional[str]]:
    """Pull the last user prompt and system text out of an Anthropic request."""
    if not isinstance(body, dict):
        return "", None
    system = body.get("system")
    if isinstance(system, str):
        system_text: Optional[str] = system or None
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


def _extract_params(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    return {key: body[key] for key in _PARAM_KEYS if key in body}


def request_replay(body: Any) -> dict[str, Any]:
    """Extract the parts of an Anthropic request needed to replay it later.

    A fork or resume that only keeps the last user prompt cannot faithfully
    continue a multi-turn or tool-using generation. This captures the message
    list, any tools/tool_choice, and the system prompt so the full conversation
    can be rebuilt rather than guessed from a single string.
    """
    if not isinstance(body, dict):
        return {}
    replay: dict[str, Any] = {}
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        replay["messages"] = messages
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        replay["tools"] = tools
    if body.get("tool_choice") is not None:
        replay["tool_choice"] = body["tool_choice"]
    if body.get("system") is not None:
        replay["system"] = body["system"]
    return replay


@dataclass
class GenerationEvent:
    """A single, immutable step in a generation's history."""

    seq: int
    type: str
    at: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.type, "at": self.at, "data": self.data}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationEvent":
        return cls(
            seq=int(value.get("seq", 0)),
            type=str(value.get("type", "")),
            at=float(value.get("at", 0.0)),
            data=value.get("data") if isinstance(value.get("data"), dict) else {},
        )


@dataclass
class GenerationRecord:
    """A durable, forkable record of one generation."""

    id: str
    fingerprint: str
    model: str
    prompt: str
    system: Optional[str]
    params: dict[str, Any]
    status: str
    created_at: float
    updated_at: float
    request: dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    fork_event_seq: Optional[int] = None
    output: str = ""
    output_tokens: Optional[int] = None
    events: list[GenerationEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "generation",
            "fingerprint": self.fingerprint,
            "model": self.model,
            "prompt": self.prompt,
            "system": self.system,
            "params": self.params,
            "request": self.request,
            "status": self.status,
            "parent": self.parent,
            "fork_event_seq": self.fork_event_seq,
            "output": self.output,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": [event.to_dict() for event in self.events],
        }

    def summary(self) -> dict[str, Any]:
        preview = self.prompt[:120] if isinstance(self.prompt, str) else ""
        return {
            "id": self.id,
            "object": "generation",
            "fingerprint": self.fingerprint,
            "model": self.model,
            "status": self.status,
            "parent": self.parent,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "prompt_preview": preview,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationRecord":
        events = value.get("events")
        return cls(
            id=str(value["id"]),
            fingerprint=str(value.get("fingerprint", "")),
            model=str(value.get("model", "")),
            prompt=value.get("prompt") if isinstance(value.get("prompt"), str) else "",
            system=value.get("system") if isinstance(value.get("system"), str) else None,
            params=value.get("params") if isinstance(value.get("params"), dict) else {},
            status=str(value.get("status", _STATUS_RUNNING)),
            created_at=float(value.get("created_at", 0.0)),
            updated_at=float(value.get("updated_at", 0.0)),
            request=value.get("request") if isinstance(value.get("request"), dict) else {},
            parent=value.get("parent") if isinstance(value.get("parent"), str) else None,
            fork_event_seq=(
                int(value["fork_event_seq"])
                if isinstance(value.get("fork_event_seq"), int)
                else None
            ),
            output=value.get("output") if isinstance(value.get("output"), str) else "",
            output_tokens=(
                int(value["output_tokens"])
                if isinstance(value.get("output_tokens"), int)
                else None
            ),
            events=[
                GenerationEvent.from_dict(event)
                for event in (events if isinstance(events, list) else [])
                if isinstance(event, dict)
            ],
        )


class GenerationLedger:
    """Append-only, content-addressable store of generations.

    Each generation is a single JSON document under ``root``. Writes are
    atomic and serialized with an advisory file lock, so concurrent requests
    (and multiple processes sharing a ledger) cannot corrupt a record.
    """

    def __init__(self, root: Optional[str] = None) -> None:
        base = root or os.environ.get("LLMBRIDGE_LEDGER_PATH") or _DEFAULT_ROOT
        self.root = Path(base)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_records = _env_max_records()
        self._lock = threading.RLock()

    def _path(self, generation_id: str) -> Path:
        return self.root / f"{_validate_generation_id(generation_id)}.json"

    def _lock_path(self, generation_id: str) -> Path:
        return _lock_path_for(self._path(generation_id))

    def _append(self, record: GenerationRecord, event_type: str, data: dict[str, Any]) -> None:
        record.events.append(
            GenerationEvent(
                seq=len(record.events),
                type=event_type,
                at=time.time(),
                data=data,
            )
        )
        record.updated_at = time.time()

    def _save(self, record: GenerationRecord) -> None:
        path = self._path(record.id)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                with _exclusive_lock(self._lock_path(record.id)):
                    json.dump(record.to_dict(), handle, indent=2)
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:  # pragma: no cover - platform dependent
                        logger.debug("fsync unavailable for %s; skipping", path)
                    os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def create(
        self,
        body: Any,
        model: str,
        *,
        parent: Optional[str] = None,
        fork_event_seq: Optional[int] = None,
    ) -> GenerationRecord:
        """Open a new generation. Forked children get a ``fork`` event instead of ``start``."""
        prompt, system = _prompt_and_system(body)
        now = time.time()
        record = GenerationRecord(
            id="gen_" + uuid.uuid4().hex[:12],
            fingerprint=_fingerprint(body),
            model=model,
            prompt=prompt,
            system=system,
            params=_extract_params(body),
            request=request_replay(body),
            status=_STATUS_RUNNING,
            created_at=now,
            updated_at=now,
            parent=parent,
            fork_event_seq=fork_event_seq,
        )
        if parent:
            self._append(
                record,
                EVENT_FORK,
                {"parent": parent, "fork_event_seq": fork_event_seq, "model": model},
            )
        else:
            self._append(
                record,
                EVENT_START,
                {"model": model, "fingerprint": record.fingerprint},
            )
        with self._lock:
            self._save(record)
            self._prune_locked()
        return record

    def complete(
        self,
        record: GenerationRecord,
        output: str,
        output_tokens: Optional[int] = None,
    ) -> GenerationRecord:
        record.output = output if isinstance(output, str) else ""
        record.output_tokens = output_tokens if isinstance(output_tokens, int) else None
        record.status = _STATUS_COMPLETED
        self._append(
            record,
            EVENT_COMPLETE,
            {
                "output": record.output,
                "output_tokens": record.output_tokens,
                "chars": len(record.output),
            },
        )
        with self._lock:
            self._save(record)
        return record

    def interrupt(
        self,
        record: GenerationRecord,
        message: str,
        *,
        output: str = "",
        output_tokens: Optional[int] = None,
        progress: float = 0.0,
    ) -> GenerationRecord:
        record.output = output if isinstance(output, str) else ""
        record.output_tokens = output_tokens if isinstance(output_tokens, int) else None
        record.status = _STATUS_INTERRUPTED
        self._append(
            record,
            EVENT_INTERRUPT,
            {
                "message": message,
                "output": record.output,
                "output_tokens": record.output_tokens,
                "progress_percent": progress,
            },
        )
        with self._lock:
            self._save(record)
        return record

    def fail(self, record: GenerationRecord, message: str) -> GenerationRecord:
        record.status = _STATUS_FAILED
        self._append(record, EVENT_FAIL, {"message": message})
        with self._lock:
            self._save(record)
        return record

    def load(self, generation_id: str) -> Optional[GenerationRecord]:
        try:
            path = self._path(generation_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            with _exclusive_lock(self._lock_path(generation_id)):
                with open(path, "r") as handle:
                    value = json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            logger.error("Failed to load generation %s: %s", generation_id, error)
            return None
        if not isinstance(value, dict) or "id" not in value:
            logger.error("Ignoring malformed generation record at %s", path)
            return None
        try:
            return GenerationRecord.from_dict(value)
        except (KeyError, TypeError, ValueError) as error:
            logger.error("Ignoring malformed generation record at %s: %s", path, error)
            return None

    def list(self, limit: int = 100) -> list[GenerationRecord]:
        records: list[GenerationRecord] = []
        try:
            entries = list(self.root.glob("gen_*.json"))
        except OSError:  # pragma: no cover - unreadable directory
            return []
        for path in entries:
            record = self.load(path.stem)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.created_at, reverse=True)
        if isinstance(limit, int) and limit >= 0:
            return records[:limit]
        return records

    def delete(self, generation_id: str) -> bool:
        try:
            path = self._path(generation_id)
        except ValueError:
            return False
        if not path.exists():
            return False
        try:
            with _exclusive_lock(self._lock_path(generation_id)):
                os.unlink(path)
        except OSError as error:
            logger.error("Failed to delete generation %s: %s", generation_id, error)
            return False
        return True

    def interrupted(self) -> list[GenerationRecord]:
        """Return interrupted generations, newest first."""
        return [
            record
            for record in self.list(limit=None)
            if record.status == _STATUS_INTERRUPTED
        ]

    def children(self, generation_id: str) -> list[GenerationRecord]:
        """Return forks whose parent is ``generation_id``, newest first."""
        return [
            record
            for record in self.list(limit=None)
            if record.parent == generation_id
        ]

    @staticmethod
    def _queue_event(record: GenerationRecord) -> Optional[GenerationEvent]:
        for event in reversed(record.events):
            if event.type == EVENT_QUEUED:
                return event
        return None

    @staticmethod
    def queue_attempt(record: GenerationRecord) -> int:
        """How many times this record has been queued for retry."""
        event = GenerationLedger._queue_event(record)
        if event is None:
            return 0
        attempt = event.data.get("attempt")
        if isinstance(attempt, int) and not isinstance(attempt, bool):
            return attempt
        return 0

    @staticmethod
    def queue_retry_at(record: GenerationRecord) -> float:
        """When the newest queue entry wants to be retried (epoch seconds)."""
        event = GenerationLedger._queue_event(record)
        if event is None:
            return 0.0
        retry_at = event.data.get("retry_at")
        if isinstance(retry_at, (int, float)) and not isinstance(retry_at, bool):
            return float(retry_at)
        return 0.0

    def queue(
        self,
        record: GenerationRecord,
        message: str,
        retry_at: float,
        attempt: int,
    ) -> GenerationRecord:
        """Park a generation for an automatic retry after a quota failure."""
        record.status = _STATUS_QUEUED
        self._append(
            record,
            EVENT_QUEUED,
            {"message": message, "retry_at": retry_at, "attempt": attempt},
        )
        with self._lock:
            self._save(record)
        return record

    def heal(
        self,
        record: GenerationRecord,
        *,
        from_model: str,
        to_model: str,
        resumed_chars: int,
    ) -> GenerationRecord:
        """Record that a broken stream was resumed onto a fallback model."""
        self._append(
            record,
            EVENT_HEAL,
            {
                "from_model": from_model,
                "to_model": to_model,
                "resumed_chars": resumed_chars,
            },
        )
        with self._lock:
            self._save(record)
        return record

    def queued(self) -> list[GenerationRecord]:
        """Return queued generations, earliest retry first."""
        return sorted(
            (
                record
                for record in self.list(limit=None)
                if record.status == _STATUS_QUEUED
            ),
            key=self.queue_retry_at,
        )

    def tree(self, generation_id: str, max_depth: int = 8) -> Optional[dict[str, Any]]:
        """Return a fork lineage: ancestor chain plus the nested subtree.

        The node at ``generation_id`` is expanded with its descendants (bounded
        by ``max_depth``); ``ancestors`` lists the parent chain root-first so a
        caller can render the full lineage without extra lookups.
        """
        record = self.load(generation_id)
        if record is None:
            return None

        def build(record_id: str, depth: int) -> Optional[dict[str, Any]]:
            current = self.load(record_id)
            if current is None:
                return None
            node: dict[str, Any] = {"record": current.summary(), "children": []}
            if depth < max_depth:
                for child in self.children(record_id):
                    nested = build(child.id, depth + 1)
                    if nested is not None:
                        node["children"].append(nested)
            return node

        ancestors: list[dict[str, Any]] = []
        seen = {record.id}
        cursor = record.parent
        while cursor is not None and cursor not in seen and len(ancestors) < 16:
            parent = self.load(cursor)
            if parent is None:
                break
            ancestors.append(parent.summary())
            seen.add(parent.id)
            cursor = parent.parent
        ancestors.reverse()
        return {"id": record.id, "ancestors": ancestors, "node": build(record.id, 0)}

    def prune(self, keep: Optional[int] = None) -> list[str]:
        """Delete every record except the newest ones.

        Args:
            keep: How many records to retain; defaults to the ledger's
                configured cap (``LLMBRIDGE_LEDGER_MAX_RECORDS``, else
                ``MAX_TRACKED_INTERRUPTED``).

        Returns:
            The ids of the removed records, oldest first.
        """
        with self._lock:
            return self._prune_locked(keep)

    def _prune_locked(self, keep: Optional[int] = None) -> list[str]:
        if keep is None:
            keep = self.max_records
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
            keep = 0
        removed: list[str] = []
        # self.list() is newest-first, so the stale tail is walked oldest-first.
        for record in reversed(self.list(limit=None)[keep:]):
            if self.delete(record.id):
                removed.append(record.id)
        if removed:
            logger.info(
                "Pruned %d generation record(s) beyond the newest %d", len(removed), keep
            )
        return removed

    @staticmethod
    def fork_request(
        record: GenerationRecord,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        include_output: bool = True,
    ) -> dict[str, Any]:
        """Build an Anthropic request that continues ``record`` where it stopped.

        When the record kept the original request (messages/tools/system), that
        full conversation is replayed so multi-turn and tool-using generations
        continue faithfully; otherwise it falls back to the recorded prompt. The
        recorded assistant text is replayed as an assistant turn so the upstream
        continues from it rather than starting over. Supplying a different
        ``model`` re-runs the remainder on a different model.
        """
        replay = record.request if isinstance(record.request, dict) else {}
        recorded_messages = replay.get("messages")
        if isinstance(recorded_messages, list) and recorded_messages:
            messages: list[dict[str, Any]] = [
                message for message in recorded_messages if isinstance(message, dict)
            ]
        else:
            messages = [{"role": "user", "content": record.prompt}]
        if include_output and record.output:
            messages.append({"role": "assistant", "content": record.output})
        resolved_max = max_tokens
        if not isinstance(resolved_max, int) or isinstance(resolved_max, bool) or resolved_max < 1:
            recorded = record.params.get("max_tokens")
            resolved_max = recorded if isinstance(recorded, int) and recorded > 0 else _DEFAULT_FORK_MAX_TOKENS
        request: dict[str, Any] = {
            "model": model or record.model,
            "max_tokens": resolved_max,
            "messages": messages,
            "stream": bool(stream),
        }
        system = record.system or replay.get("system")
        if isinstance(system, str) and system:
            request["system"] = system
        tools = replay.get("tools")
        if isinstance(tools, list) and tools:
            request["tools"] = tools
        if replay.get("tool_choice") is not None:
            request["tool_choice"] = replay["tool_choice"]
        return request
