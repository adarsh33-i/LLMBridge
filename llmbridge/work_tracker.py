"""Track partially finished AI generation work for resumption with different models.

Design philosophy:
- Interrupted work is stored as durable generation records in the ledger
  (``.llmbridge/generations`` by default) — the same records the server
  already keeps for every request. There is no separate partial-work file.
- An interrupted generation is a record with status ``interrupted``: it holds
  the original prompt, the full replayable request, whatever partial output
  was produced, a progress estimate, and the error that stopped it.
- Only the newest records are retained (``MAX_TRACKED_INTERRUPTED``), so
  resumable work is always among the kept records.
- Saved state has no expiry — resume the next day if needed.
"""

import logging
import threading
import time
from typing import Any, Optional

from .ledger import (
    EVENT_INTERRUPT,
    GenerationLedger,
    GenerationRecord,
)

logger = logging.getLogger(__name__)

_UNSET: Any = object()


def _format_percent(value: Any) -> str:
    """Format a progress value as a percentage, tolerating missing/invalid data."""
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


class WorkTracker:
    """Tracks partially finished AI generation work so it can be resumed
    with a different model after quota exhaustion or errors.

    State lives in a :class:`GenerationLedger`: saving interrupted work opens
    a generation record and closes it with an ``interrupt`` event. Because
    each interruption is its own record, concurrent generations no longer
    overwrite one another's saved state; the newest interrupted record is the
    one surfaced for resumption.

    Key design:
    - State is ONLY saved when generation stops (not on every call)
    - The interrupted prompt, the partial output generated so far, and minimal
      metadata are stored in the ledger record
    - Saved state persists until resumed, cleared, or pruned with older
      records; recoverable records are exempt from the retention window up to
      their own budget
    - Marked ``resumed`` automatically when the same request completes, or
      deleted on demand
    - In-memory session state is guarded by a lock, so a single instance may
      be shared across threads
    """

    def __init__(self, ledger: Optional[GenerationLedger] = None):
        """Initialize the work tracker.

        Args:
            ledger: The generation ledger that stores interrupted work. If
                None, a ledger at the default root (``.llmbridge/generations``,
                overridable via ``LLMBRIDGE_LEDGER_PATH``) is used.
        """
        self.ledger = ledger if ledger is not None else GenerationLedger()
        self._session_lock = threading.RLock()
        self._session: Optional[dict[str, Any]] = None

    def start_tracking(
        self,
        *,
        prompt: str,
        model_name: str,
        system: Optional[str] = None,
        request: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mark the start of a generation session.

        This should be called when beginning a generation run. It does NOT
        save to the ledger immediately — it just sets internal state. State is
        saved only when generation stops (via save_interrupted()).

        Args:
            prompt: The original user prompt that started the generation.
            model_name: The model being used at the start.
            system: The system prompt/instructions (optional).
            request: The original request's replayable parts — ``messages``,
                ``tools``, ``tool_choice`` — so a resume can rebuild the full
                conversation instead of only the last prompt (optional).
        """
        with self._session_lock:
            self._session = {
                "prompt": prompt,
                "model_name": model_name,
                "system": system,
                "progress_percent": 0.0,
                "error": None,
                "messages": [],
                "partial_output": "",
                "max_tokens": None,
                "request": request,
            }

    def update_progress(
        self,
        *,
        progress_percent: float,
        error: Any = _UNSET,
        messages: Optional[list[dict[str, Any]]] = None,
        partial_output: Any = _UNSET,
    ) -> None:
        """Update the current generation progress (called during generation).

        This can be called periodically during a long generation. It updates
        internal state but does NOT save to the ledger. Only the final
        interrupted state will be saved.

        Args:
            progress_percent: Current progress percentage (0-100).
            error: Error message if generation stopped. Pass an empty string to
                clear a previously recorded error. Omit to leave it unchanged.
            messages: Conversation history so far (optional). Omit to leave it
                unchanged; pass an empty list to clear it.
            partial_output: Assistant text generated so far (optional). Omit to
                leave it unchanged; pass an empty string to clear it. Non-string
                values are treated as empty.
        """
        with self._session_lock:
            session = self._require_session()
            try:
                session["progress_percent"] = max(0.0, min(100.0, float(progress_percent)))
            except (TypeError, ValueError):
                session["progress_percent"] = 0.0
            if error is not _UNSET:
                session["error"] = error or None
            if messages is not None:
                session["messages"] = messages
            if partial_output is not _UNSET:
                session["partial_output"] = (
                    partial_output if isinstance(partial_output, str) else ""
                )

    def save_interrupted(self) -> dict[str, Any]:
        """Save the interrupted generation state to the ledger.

        Call this method when generation stops unexpectedly (quota error,
        rate limit, etc.) so the work can be resumed with a different model.

        The saved state becomes an interrupted generation record holding the
        current prompt, any partial output recorded via update_progress(),
        and minimal metadata.

        Returns:
            The saved state dictionary (same format as load_state() returns).
        """
        with self._session_lock:
            session = dict(self._require_session())
        return self.save(
            prompt=session["prompt"],
            model_name=session["model_name"],
            system=session["system"],
            progress_percent=session["progress_percent"],
            error=session["error"],
            messages=session["messages"],
            partial_output=session["partial_output"],
            max_tokens=session["max_tokens"],
            request=session["request"],
        )

    def save(
        self,
        *,
        prompt: str,
        model_name: str,
        system: Optional[str] = None,
        progress_percent: float = 0.0,
        error: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        partial_output: str = "",
        max_tokens: Optional[int] = None,
        request: Optional[dict[str, Any]] = None,
        fingerprint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Save an interrupted generation directly, without a session.

        Stateless counterpart to :meth:`save_interrupted` for callers that build
        the state per request instead of using start_tracking()/update_progress()
        (for example, a server handling concurrent requests). Each call records
        its own ledger entry, so two threads calling ``save()`` cannot
        attribute one request's prompt to another.

        Args:
            partial_output: Assistant text generated before the interruption
                (optional). Non-string values are stored as an empty string.
            max_tokens: The generation's token limit, kept so a resume can reuse
                it (optional).
            request: The original request's replayable parts — ``messages``,
                ``tools``, ``tool_choice`` — so a resume can rebuild the full
                conversation (optional).
            fingerprint: Identity of the original client request, used instead
                of hashing the rebuilt body (optional). Without it, a record
                saved from a replayed body could never be matched to the
                request that interrupted.

        Returns:
            The saved state dictionary (same format as ``load_state()``).
        """
        record = self.ledger.create(
            self._build_record_body(
                prompt=prompt,
                model_name=model_name,
                system=system,
                max_tokens=max_tokens,
                messages=messages,
                request=request,
            ),
            model_name,
            fingerprint=fingerprint,
        )
        self.ledger.interrupt(
            record,
            error if isinstance(error, str) else ("" if error is None else str(error)),
            output=partial_output if isinstance(partial_output, str) else "",
            progress=progress_percent if isinstance(progress_percent, (int, float)) else 0.0,
        )
        state = self._state_from_record(record)
        self._log_saved(state)
        return state

    @staticmethod
    def _build_record_body(
        *,
        prompt: str,
        model_name: str,
        system: Optional[str],
        max_tokens: Optional[int],
        messages: Optional[list[dict[str, Any]]],
        request: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build an Anthropic-style body that a ledger record can capture."""
        replay = request if isinstance(request, dict) else {}
        replay_messages = replay.get("messages")
        if isinstance(replay_messages, list) and replay_messages:
            body_messages = [item for item in replay_messages if isinstance(item, dict)]
        elif isinstance(messages, list) and messages:
            body_messages = [item for item in messages if isinstance(item, dict)]
        else:
            body_messages = [{"role": "user", "content": prompt if isinstance(prompt, str) else ""}]

        body: dict[str, Any] = {
            "model": model_name,
            "messages": body_messages,
            "stream": False,
        }
        resolved_system = system if isinstance(system, str) and system else replay.get("system")
        if isinstance(resolved_system, str) and resolved_system:
            body["system"] = resolved_system
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
            body["max_tokens"] = max_tokens
        tools = replay.get("tools")
        if isinstance(tools, list) and tools:
            body["tools"] = tools
        if replay.get("tool_choice") is not None:
            body["tool_choice"] = replay["tool_choice"]
        return body

    @staticmethod
    def _state_from_record(record: GenerationRecord) -> dict[str, Any]:
        """Map an interrupted ledger record to the partial-work state shape."""
        error: Optional[str] = None
        progress = 0.0
        for event in reversed(record.events):
            if event.type != EVENT_INTERRUPT:
                continue
            message = event.data.get("message")
            error = message if isinstance(message, str) and message else None
            try:
                progress = float(event.data.get("progress_percent") or 0.0)
            except (TypeError, ValueError):
                progress = 0.0
            break
        replay = record.request if isinstance(record.request, dict) else {}
        messages = replay.get("messages")
        params = record.params if isinstance(record.params, dict) else {}
        max_tokens = params.get("max_tokens")
        return {
            "saved_at": record.updated_at,
            "prompt": record.prompt if isinstance(record.prompt, str) else "",
            "model_name": record.model,
            "system": record.system,
            "progress_percent": progress,
            "error": error,
            "messages": messages if isinstance(messages, list) else [],
            "partial_output": record.output if isinstance(record.output, str) else "",
            "max_tokens": (
                max_tokens
                if isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
                else None
            ),
            "request": replay or None,
            "generation_id": record.id,
        }

    def _require_session(self) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError(
                "No generation session in progress. Call start_tracking() first."
            )
        return self._session

    def _log_saved(self, state: dict[str, Any]) -> None:
        prompt = state["prompt"]
        prompt_preview = prompt[:50] if isinstance(prompt, str) else ""
        logger.info(
            "Saved interrupted work as %s in %s (prompt=%r, progress=%s, error=%r)",
            state["generation_id"],
            self.ledger.root,
            prompt_preview,
            _format_percent(state["progress_percent"]),
            state["error"],
        )

    def load_state(self) -> Optional[dict[str, Any]]:
        """Load the most recently interrupted generation's state.

        Returns:
            The saved state dictionary, or None if no interrupted work
            is recorded in the ledger.
        """
        records = self.ledger.interrupted()
        if not records:
            return None
        state = self._state_from_record(records[0])
        prompt = state["prompt"]
        prompt_preview = prompt[:50] if isinstance(prompt, str) else ""
        logger.info(
            "Loaded interrupted work %s from %s (prompt=%r, progress=%s)",
            state["generation_id"],
            self.ledger.root,
            prompt_preview,
            _format_percent(state["progress_percent"]),
        )
        return state

    def clear_state(self) -> bool:
        """Clear the newest saved interrupted work record.

        Returns:
            True if a record was cleared, False if there was nothing to clear.
        """
        records = self.ledger.interrupted()
        if not records:
            logger.info("No interrupted work state to clear in %s", self.ledger.root)
            self._reset_session()
            return False
        cleared = self.ledger.delete(records[0].id)
        if cleared:
            logger.info(
                "Cleared interrupted work %s from %s", records[0].id, self.ledger.root
            )
        self._reset_session()
        return cleared

    def resolve_matching(
        self, fingerprint: str, *, by: Optional[str], model: str
    ) -> Optional[str]:
        """Mark the newest interrupted record for this request as resumed.

        Matched on the request's content fingerprint rather than its prompt text.
        Prompt is not an identity: two concurrent conversations can both ask for
        "continue", and finishing one would then erase the other's recoverable
        work. The record is transitioned to ``resumed`` instead of deleted, so
        the interruption stays measurable.

        Returns:
            The id of the resolved record, or None if nothing matched.
        """
        resolved: Optional[str] = None
        for record in self.ledger.interrupted():
            if record.fingerprint == fingerprint and record.id != by:
                self.ledger.resume(record, by=by, model=model)
                resolved = record.id
                break
        if resolved is not None:
            logger.info(
                "Marked interrupted work %s resumed in %s (by %s)",
                resolved,
                self.ledger.root,
                by,
            )
            # Only a resolution touches the in-memory session. An unrelated
            # request completing must not clear someone else's tracking run.
            self._reset_session()
        return resolved

    def _reset_session(self) -> None:
        """Forget any in-memory generation session."""
        with self._session_lock:
            self._session = None

    def get_progress_summary(self) -> str:
        """Get a human-readable summary of the current interrupted work state.

        Returns:
            A string summary, or "No interrupted work in progress" if nothing is saved.
        """
        state = self.load_state()
        if state is None:
            return "No interrupted work in progress"

        try:
            progress = float(state.get("progress_percent") or 0.0)
        except (TypeError, ValueError):
            progress = 0.0
        stage = "interrupted"  # Always "interrupted" since we only save on stop
        model = state.get("model_name") or "unknown"
        error = state.get("error") or None

        lines = [
            "Interrupted work in progress:",
            f"  Record: {state.get('generation_id') or 'unknown'}",
            f"  Model: {model}",
            f"  Stage: {stage}",
            f"  Progress: {progress:.1f}%",
        ]
        saved_at = state.get("saved_at")
        if isinstance(saved_at, (int, float)) and saved_at:
            lines.append(f"  Saved: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(saved_at))}")
        if error:
            lines.append(f"  Error: {error}")

        partial_output = state.get("partial_output")
        if isinstance(partial_output, str) and partial_output:
            collapsed = " ".join(partial_output.split())
            preview = collapsed[:60] + ("..." if len(collapsed) > 60 else "")
            lines.append(f"  Partial output: {preview}")

        # Also show prompt preview
        prompt = state.get("prompt")
        if isinstance(prompt, str) and prompt:
            lines.append(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

        return "\n".join(lines)


def track_generation(
    prompt: str,
    model_name: str,
    system: Optional[str] = None,
    ledger: Optional[GenerationLedger] = None,
) -> WorkTracker:
    """Quick-track a generation session for later resumption.

    This is a convenience function that sets up the tracker with the given
    prompt and model. You should call save_interrupted() when generation
    stops (quota error, etc.), then load_state() to retrieve it later.

    Args:
        prompt: The original user prompt.
        model_name: The model being used.
        system: System prompt/instructions (optional).
        ledger: The generation ledger that stores interrupted work (optional;
            defaults to ``.llmbridge/generations``).

    Returns:
        A WorkTracker instance with the session started.
    """
    tracker = WorkTracker(ledger)
    tracker.start_tracking(
        prompt=prompt,
        model_name=model_name,
        system=system,
    )
    return tracker
