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
from dataclasses import dataclass, field, replace
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
# new generation is created. Override with LLMBRIDGE_LEDGER_MAX_RECORDS.
MAX_TRACKED_INTERRUPTED: int = 10

# Recoverable work -- interrupted generations and requests parked on quota --
# outlives that window too, up to its own budget. Without this exemption the
# retention cap deletes a half-written answer after ten unrelated requests, so
# the resume endpoint reports no interrupted work for a job it could still
# finish. Newest first; ``0`` restores the plain window. Override with
# LLMBRIDGE_LEDGER_MAX_RECOVERABLE.
MAX_KEPT_RECOVERABLE: int = 10

# Suspect seam audits outlive the window above: a heal that joined badly is the
# evidence a later review needs, and ordinary traffic would otherwise erase it
# after ten unrelated requests. Forgetting therefore stays a stated policy with
# its own bound instead of a storage side effect. Override with
# LLMBRIDGE_AUDIT_MAX_SUSPECT.
MAX_KEPT_SUSPECT_AUDITS: int = 25


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


def _env_max_suspect_audits() -> int:
    """Read the suspect-audit exemption budget, which may be zero to disable it."""
    raw = os.environ.get("LLMBRIDGE_AUDIT_MAX_SUSPECT")
    if raw is None:
        return MAX_KEPT_SUSPECT_AUDITS
    try:
        value = int(raw)
    except ValueError:
        return MAX_KEPT_SUSPECT_AUDITS
    return value if value >= 0 else MAX_KEPT_SUSPECT_AUDITS


def _env_max_recoverable() -> int:
    """Read the recoverable-work exemption budget, which may be zero to disable it."""
    raw = os.environ.get("LLMBRIDGE_LEDGER_MAX_RECOVERABLE")
    if raw is None:
        return MAX_KEPT_RECOVERABLE
    try:
        value = int(raw)
    except ValueError:
        return MAX_KEPT_RECOVERABLE
    return value if value >= 0 else MAX_KEPT_RECOVERABLE


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile, or None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


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
# ``heal`` records a mid-stream failover onto a fallback model; ``heal_seam``
# audits how well that continuation joined on; ``queue`` records that quota
# exhaustion parked the request for an automatic retry; ``resume`` records that
# interrupted work was finished by a later generation.
EVENT_START = "start"
EVENT_COMPLETE = "complete"
EVENT_INTERRUPT = "interrupt"
EVENT_FAIL = "fail"
EVENT_FORK = "fork"
EVENT_HEAL = "heal"
EVENT_HEAL_SEAM = "heal_seam"
EVENT_QUEUED = "queue"
EVENT_RESUME = "resume"

_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_INTERRUPTED = "interrupted"
_STATUS_FAILED = "failed"
_STATUS_QUEUED = "queued"
# Set on an interrupted record once a later generation has finished that work.
# It leaves the recoverable set without erasing the interruption itself.
_STATUS_RESUMED = "resumed"

_ID_RE = re.compile(r"^gen_[A-Za-z0-9]+$")

# Scalar request parameters worth remembering for a fork. The message list and
# system prompt are stored separately so they can be rebuilt exactly.
_PARAM_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream")

# What a consolidated heal audit keeps. These are the seam numbers the server
# scores; the digest line is the durable form of a record's ``heal_seam`` event.
_AUDIT_NAME = "heal_audit.jsonl"
_AUDIT_FIELDS = (
    "from_model",
    "to_model",
    "seam_at",
    "continuation_chars",
    "overlap_chars",
    "overlap_ratio",
    "duplicated",
    "restarted",
    "suspect",
)


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


def append_assistant_turn(
    messages: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    """Append assistant text, merging into a trailing assistant turn.

    A conversation may already end with an assistant turn when the client used a
    prefill (the model is asked to continue that text). Appending another
    assistant message would put two in a row, which Anthropic-shaped upstreams
    reject, so the prefill and the new text are joined into one turn instead.
    """
    appended = [dict(message) for message in messages]
    if appended and appended[-1].get("role") == "assistant":
        last = appended[-1]
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = content + text
        elif isinstance(content, list):
            last["content"] = [*content, {"type": "text", "text": text}]
        else:
            last["content"] = text
        return appended
    appended.append({"role": "assistant", "content": text})
    return appended


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
        self.max_suspect_audits = _env_max_suspect_audits()
        self.max_recoverable = _env_max_recoverable()
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
        fingerprint: Optional[str] = None,
    ) -> GenerationRecord:
        """Open a new generation. Forked children get a ``fork`` event instead of ``start``.

        ``fingerprint`` overrides the identity of the request being captured.
        A caller that records an interruption from a rebuilt body (the resume
        path keeps only the replayable parts) passes the original request's
        fingerprint, so re-sending that request still resolves this record.
        """
        prompt, system = _prompt_and_system(body)
        now = time.time()
        record = GenerationRecord(
            id="gen_" + uuid.uuid4().hex[:12],
            fingerprint=(
                fingerprint
                if isinstance(fingerprint, str) and fingerprint
                else _fingerprint(body)
            ),
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

    def resume(
        self,
        record: GenerationRecord,
        *,
        by: Optional[str],
        model: str,
    ) -> GenerationRecord:
        """Mark interrupted work as finished by a later generation.

        The record is kept rather than deleted: erasing it would remove the only
        evidence that this answer was interrupted and had to be re-run, which is
        exactly what makes resume rates measurable. It leaves the ``interrupted``
        status, so it is no longer offered for resumption.
        """
        record.status = _STATUS_RESUMED
        self._append(record, EVENT_RESUME, {"resumed_by": by, "model": model})
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

    def settle_running(
        self,
        message: str,
        *,
        older_than: Optional[float] = None,
        now: Optional[float] = None,
    ) -> list[GenerationRecord]:
        """Interrupt ``running`` records that no live request can still finish.

        A record is only ``running`` while its request is in flight, and only the
        serving process ever finalizes it. If that process exits mid-stream the
        record keeps claiming the work is happening and never shows up as
        recoverable, so a restart has to settle it here.

        ``older_than`` skips records touched more recently than that many
        seconds, which keeps a startup sweep from claiming a request another
        process is still serving from the same ledger.
        """
        timestamp = time.time() if now is None else now
        settled: list[GenerationRecord] = []
        for record in self.list(limit=None):
            if record.status != _STATUS_RUNNING:
                continue
            if older_than is not None and timestamp - record.updated_at < older_than:
                continue
            settled.append(self.interrupt(record, message))
        return settled

    @staticmethod
    def is_recoverable(record: GenerationRecord) -> bool:
        """Whether this record holds work that can still be finished or retried."""
        return record.status in (_STATUS_INTERRUPTED, _STATUS_QUEUED)

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

    def heal_seam(
        self,
        record: GenerationRecord,
        *,
        from_model: str,
        to_model: str,
        seam_at: int,
        audit: dict[str, Any],
    ) -> GenerationRecord:
        """Record how well a healed continuation joined onto the truncated text.

        The audit flags the boundary; it does not gate it. Absence of this event
        means the continuation never finished, so there was no seam to score.
        """
        self._append(
            record,
            EVENT_HEAL_SEAM,
            {**audit, "from_model": from_model, "to_model": to_model, "seam_at": seam_at},
        )
        with self._lock:
            self._save(record)
        return record

    @staticmethod
    def _heal_seam_event(record: GenerationRecord) -> Optional[GenerationEvent]:
        for event in reversed(record.events):
            if event.type == EVENT_HEAL_SEAM:
                return event
        return None

    @staticmethod
    def heal_seam_audit(record: GenerationRecord) -> Optional[dict[str, Any]]:
        """The newest seam audit for this record, or None if it was never healed."""
        event = GenerationLedger._heal_seam_event(record)
        return None if event is None else dict(event.data)

    @staticmethod
    def heal_seam_suspect(record: GenerationRecord) -> bool:
        """Whether the last recorded heal joined badly enough to be worth reviewing."""
        audit = GenerationLedger.heal_seam_audit(record)
        return bool(audit and audit.get("suspect"))

    @property
    def audit_digest_path(self) -> Path:
        """Append-only file holding one consolidated line per seam audit."""
        return self.root / _AUDIT_NAME

    @staticmethod
    def read_audit_digest(path: Path) -> list[dict[str, Any]]:
        """Parse a heal digest, oldest line first, ignoring unreadable lines."""
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        entries: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _consolidate_records(self, records: list[GenerationRecord]) -> tuple[int, int]:
        """Copy these records' seam audits into the digest.

        Append-only and idempotent: a seam already keyed by
        ``(generation_id, seq)`` is not written twice. Returns
        ``(audits_found, entries_written)``.
        """
        path = self.audit_digest_path
        lock_path = _lock_path_for(path)
        found = 0
        fresh: list[dict[str, Any]] = []
        # Read, compare, and append under the digest lock so two processes
        # consolidating at once cannot both write the same seam.
        with self._lock, _exclusive_lock(lock_path):
            existing = self.read_audit_digest(path)
            seen = {
                (entry.get("generation_id"), entry.get("seq")) for entry in existing
            }
            for record in records:
                for event in record.events:
                    if event.type != EVENT_HEAL_SEAM:
                        continue
                    found += 1
                    if (record.id, event.seq) in seen:
                        continue
                    entry: dict[str, Any] = {
                        "generation_id": record.id,
                        "seq": event.seq,
                        "at": event.at,
                    }
                    entry.update({key: event.data.get(key) for key in _AUDIT_FIELDS})
                    fresh.append(entry)
            if fresh:
                with open(path, "a", encoding="utf-8") as handle:
                    for entry in sorted(
                        fresh, key=lambda entry: (entry["at"], entry["seq"])
                    ):
                        handle.write(json.dumps(entry, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return found, len(fresh)

    def consolidate(self) -> dict[str, Any]:
        """Copy seam audits out of records into the append-only digest.

        Records are episodic: the retention window holds only the newest ones, so
        a clean seam is erased by ordinary traffic and a suspect rate loses its
        denominator. The digest is the durable form - one line per audit, keyed by
        ``(generation_id, event seq)`` so re-running it never duplicates a seam,
        and it survives the record it came from. Pruning consolidates the
        records it is about to delete, and a graceful shutdown consolidates
        everything, so this method is the on-demand pass.

        Returns:
            Counts over the digest after this pass, including the suspect rate
            (``None`` when nothing has been audited, rather than a false zero).
        """
        path = self.audit_digest_path
        found, fresh = self._consolidate_records(self.list(limit=None))
        entries = self.read_audit_digest(path)
        total = len(entries)
        suspect = sum(1 for entry in entries if entry.get("suspect"))
        return {
            "digest_path": str(path),
            "audits_on_disk": found,
            "new_entries": fresh,
            "total_entries": total,
            "suspect_entries": suspect,
            "suspect_rate": round(suspect / total, 3) if total else None,
        }

    def stats(self) -> dict[str, Any]:
        """Summarise the ledger in the numbers an operator asks for.

        Rates are reported next to their denominators because the working set is
        small by design (records are pruned), and a suspect rate over three heals
        is not the same claim as one over three hundred. ``digest_entries`` is the
        durable count that outlives pruning; ``unconsolidated`` says how many
        seam audits are still only in records.
        """
        records = self.list(limit=None)
        by_status: dict[str, int] = {}
        tokens = 0
        token_counts = 0
        wall_times: list[float] = []
        healed = 0
        audit_events = 0
        audited = 0
        suspect = 0
        duplicated = 0
        restarted = 0
        for record in records:
            by_status[record.status] = by_status.get(record.status, 0) + 1
            if isinstance(record.output_tokens, int):
                tokens += record.output_tokens
                token_counts += 1
            if any(event.type == EVENT_HEAL for event in record.events):
                healed += 1
            audit_events += sum(1 for event in record.events if event.type == EVENT_HEAL_SEAM)
            audit = self.heal_seam_audit(record)
            if audit is not None:
                audited += 1
                if audit.get("suspect"):
                    suspect += 1
                duplicated += 1 if audit.get("duplicated") else 0
                restarted += 1 if audit.get("restarted") else 0
            complete = next(
                (event for event in record.events if event.type == EVENT_COMPLETE), None
            )
            if complete is not None and complete.at >= record.created_at:
                wall_times.append(round((complete.at - record.created_at) * 1000, 1))

        digest = self.read_audit_digest(self.audit_digest_path)
        digest_suspect = sum(1 for entry in digest if entry.get("suspect"))
        return {
            "object": "stats",
            "ledger": {"path": str(self.root), "records": len(records)},
            "generations": {
                "total": len(records),
                "by_status": by_status,
                "recoverable": sum(1 for record in records if self.is_recoverable(record)),
            },
            "tokens": {"output": tokens, "records_with_tokens": token_counts},
            "wall_time_ms": {
                "n": len(wall_times),
                "p50": _percentile(wall_times, 0.50),
                "p95": _percentile(wall_times, 0.95),
                "max": max(wall_times) if wall_times else None,
            },
            "heals": {
                "healed": healed,
                "audited": audited,
                "suspect": suspect,
                "clean": audited - suspect,
                "suspect_rate": round(suspect / audited, 3) if audited else None,
                "duplicated": duplicated,
                "restarted": restarted,
                "digest_entries": len(digest),
                "digest_suspect": digest_suspect,
                "digest_suspect_rate": (
                    round(digest_suspect / len(digest), 3) if digest else None
                ),
                "unconsolidated": max(0, audit_events - len(digest)),
            },
        }

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
                ``MAX_TRACKED_INTERRUPTED``). With no argument the retention
                policy also exempts recent suspect seam audits and recoverable
                work. Passing an explicit number is a manual hard window:
                nothing is exempted.

        Returns:
            The ids of the removed records, oldest first.
        """
        with self._lock:
            return self._prune_locked(keep)

    def _prune_locked(self, keep: Optional[int] = None) -> list[str]:
        policy_window = keep is None
        if keep is None:
            keep = self.max_records
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
            keep = 0
        # self.list() is newest-first, so the stale tail is walked oldest-first
        # for deletion and newest-first when spending the exemption budgets:
        # both should go to the freshest evidence and the freshest unfinished work.
        records = self.list(limit=None)[keep:]
        exempt: set[str] = set()
        suspects = 0
        recoverable = 0
        if policy_window:
            for record in records:
                if (
                    suspects < self.max_suspect_audits
                    and GenerationLedger.heal_seam_suspect(record)
                ):
                    suspects += 1
                    exempt.add(record.id)
                elif (
                    recoverable < self.max_recoverable
                    and GenerationLedger.is_recoverable(record)
                ):
                    recoverable += 1
                    exempt.add(record.id)
        removed: list[str] = []
        victims = [record for record in reversed(records) if record.id not in exempt]
        # Only records that actually carry a seam audit need the digest read and
        # write; the common victim has never healed anything.
        audited_victims = [
            record
            for record in victims
            if any(event.type == EVENT_HEAL_SEAM for event in record.events)
        ]
        if audited_victims:
            # Records are about to disappear; copy their seam audits into the
            # digest first so a clean seam is not lost with its record. This is
            # best-effort: never let a digest problem block retention.
            try:
                _, consolidated = self._consolidate_records(audited_victims)
            except Exception as error:  # pragma: no cover - defensive
                logger.error(
                    "Could not consolidate seam audits before pruning: %s", error
                )
            else:
                if consolidated:
                    logger.info(
                        "Consolidated %d seam audit(s) before pruning", consolidated
                    )
        for record in victims:
            if self.delete(record.id):
                removed.append(record.id)
        if removed:
            logger.info(
                "Pruned %d generation record(s) beyond the newest %d (%d suspect "
                "seam audit(s) and %d recoverable record(s) kept beyond the window)",
                len(removed),
                keep,
                suspects,
                recoverable,
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
            messages = append_assistant_turn(messages, record.output)
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

    @classmethod
    def dream_request(
        cls,
        record: GenerationRecord,
        *,
        at: int,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build the request a fallback would have seen if this answer had died at ``at``.

        The recorded output is complete, so no stream has to actually break for a
        heal to be tested: the answer is cut at ``at``, the cut text is replayed as
        an assistant turn, and the model is asked to continue. Everything after
        ``at`` is the ground truth the continuation can be scored against.
        """
        if not isinstance(record.output, str) or not record.output:
            raise ValueError("generation has no stored output to cut and replay")
        cut = max(0, min(int(at), len(record.output)))
        return cls.fork_request(
            replace(record, output=record.output[:cut]),
            model=model,
            max_tokens=max_tokens,
        )
