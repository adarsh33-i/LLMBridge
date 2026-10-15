"""Track partially finished AI generation work for resumption with different models.

Design philosophy:
- Track ONLY the interrupted/prompt that was being processed when generation stopped
- No size limits during ongoing generation (just mark "in progress")
- When generation stops (quota error, etc.), save the interrupted prompt, any
  partial output already generated, and a progress estimate
- After resumption/completion, clear the state
- State is small (one prompt + partial output + metadata), so no arbitrary KB caps needed
- Saved state has no expiry — resume the next day if needed
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional, List

if os.name == "nt":  # pragma: no cover - Windows-only lock implementation
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _lock_path_for(storage_path: Path) -> Path:
    """Return a stable lock-file path in the system temp directory.

    Keeping locks out of the working directory means a tracker never litters
    the project with ``.lock`` files. The name is derived from the resolved
    state path so every process targeting the same state file shares a lock.
    """
    key = hashlib.sha256(str(storage_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "llmbridge-locks" / f"{storage_path.name}.{key}.lock"


def _validate_namespace(namespace: str) -> str:
    """Reject namespaces that could escape the working directory.

    A namespace is interpolated into a filename, so path separators and
    leading dots (``..``) must be refused to prevent path traversal.
    """
    if not isinstance(namespace, str) or not _NAMESPACE_RE.match(namespace):
        raise ValueError(
            "namespace must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return namespace


def _format_percent(value: Any) -> str:
    """Format a progress value as a percentage, tolerating missing/invalid data."""
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


@contextlib.contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Serialize access to the state file across threads and processes.

    Uses an advisory file lock (fcntl on POSIX, msvcrt on Windows) so that
    concurrent trackers sharing the same state file cannot interleave reads,
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


class WorkTracker:
    """Tracks partially finished AI generation work so it can be resumed
    with a different model after quota exhaustion or errors.

    Key design:
    - State is ONLY saved when generation stops (not on every call)
    - The interrupted prompt, the partial output generated so far, and minimal
      metadata are stored
    - No arbitrary KB caps — the state is naturally small (one prompt + the
      partial output of a single generation)
    - Saved state persists until resumed or explicitly cleared
    - Clear automatically after resumption or on demand
    - Access to the state file is serialized with an advisory file lock, so
      concurrent trackers sharing a path cannot corrupt or interleave writes
    - In-memory session state is guarded by a lock, so a single instance may be
      shared across threads

    Note: the default storage path is a single shared slot. If you expect
    multiple concurrent generations that must resume independently, give each
    one its own ``storage_path`` (or ``namespace``).
    """

    def __init__(self, storage_path: Optional[str] = None, *, namespace: Optional[str] = None):
        """Initialize the work tracker.

        Args:
            storage_path: Path to the JSON file for storing partial work.
                If None, uses ".partial_work.json" (optionally namespaced) in
                the session cwd.
            namespace: Optional name to scope the default state file, producing
                ".partial_work.<namespace>.json". Ignored when ``storage_path``
                is provided. Use one namespace per independent generation.
        """
        if storage_path is None:
            if namespace:
                self.storage_path = Path(f".partial_work.{_validate_namespace(namespace)}.json")
            else:
                self.storage_path = Path(".partial_work.json")
        else:
            self.storage_path = Path(storage_path)
        self._ensure_storage_dir()
        self._lock_path = _lock_path_for(self.storage_path)
        self._session_lock = threading.RLock()

    def _ensure_storage_dir(self) -> None:
        """Ensure the directory for the storage file exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def start_tracking(
        self,
        *,
        prompt: str,
        model_name: str,
        system: Optional[str] = None,
    ) -> None:
        """Mark the start of a generation session.

        This should be called when beginning a generation run. It does NOT
        save to disk immediately — it just sets internal state. State is saved
        only when generation stops (via save_interrupted()).

        Args:
            prompt: The original user prompt that started the generation.
            model_name: The model being used at the start.
            system: The system prompt/instructions (optional).
        """
        with self._session_lock:
            self._current_prompt = prompt
            self._current_model = model_name
            self._current_system = system
            self._progress_percent = 0.0
            self._error: Optional[str] = None
            self._messages: List[dict[str, Any]] = []
            self._partial_output = ""

        # Do NOT save to disk here — just set internal state
        # This avoids saving on every intermediate step

    def update_progress(
        self,
        *,
        progress_percent: float,
        error: Any = _UNSET,
        messages: Optional[List[dict[str, Any]]] = None,
        partial_output: Any = _UNSET,
    ) -> None:
        """Update the current generation progress (called during generation).

        This can be called periodically during a long generation. It updates
        internal state but does NOT save to disk. Only the final interrupted
        state will be saved.

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
            self._progress_percent = progress_percent
            if error is not _UNSET:
                self._error = error or None
            if messages is not None:
                self._messages = messages
            if partial_output is not _UNSET:
                self._partial_output = partial_output if isinstance(partial_output, str) else ""

    def save_interrupted(self) -> dict[str, Any]:
        """Save the interrupted generation state to disk.

        Call this method when generation stops unexpectedly (quota error,
        rate limit, etc.) so the work can be resumed with a different model.

        The saved state holds the current prompt, any partial output recorded
        via update_progress(), and minimal metadata — no intermediate progress
        steps. Message history is only included if it was passed to
        update_progress(). This keeps the state naturally small (typically
        500 bytes - a few KB) without needing arbitrary KB caps.

        The write is atomic: the state is written to a temporary file and then
        moved into place, so a crash mid-write cannot leave a corrupt state
        file behind.

        Returns:
            The saved state dictionary (same format as load_state() returns).
        """
        with self._session_lock:
            if not hasattr(self, "_current_prompt"):
                raise RuntimeError("No generation session in progress. Call start_tracking() first.")

            state = {
                "saved_at": time.time(),
                "prompt": self._current_prompt,
                "model_name": self._current_model,
                "system": self._current_system,
                "progress_percent": self._progress_percent,
                "error": self._error,
                "messages": self._messages,
                "partial_output": self._partial_output,
            }

        self._write_state(state)
        self._log_saved(state)
        return state

    def save(
        self,
        *,
        prompt: str,
        model_name: str,
        system: Optional[str] = None,
        progress_percent: float = 0.0,
        error: Optional[str] = None,
        messages: Optional[List[dict[str, Any]]] = None,
        partial_output: str = "",
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Save an interrupted generation directly, without a session.

        Stateless counterpart to :meth:`save_interrupted` for callers that build
        the state per request instead of using start_tracking()/update_progress()
        (for example, a server handling concurrent requests). Because it reads
        no shared session state, two threads calling ``save()`` cannot attribute
        one request's prompt to another; the on-disk write is still serialized
        by the advisory file lock.

        Args:
            partial_output: Assistant text generated before the interruption
                (optional). Non-string values are stored as an empty string.
            max_tokens: The generation's token limit, kept so a resume can reuse
                it (optional).

        Returns:
            The saved state dictionary (same format as ``load_state()``).
        """
        state = {
            "saved_at": time.time(),
            "prompt": prompt,
            "model_name": model_name,
            "system": system,
            "progress_percent": progress_percent,
            "error": error,
            "messages": list(messages) if messages is not None else [],
            "partial_output": partial_output if isinstance(partial_output, str) else "",
            "max_tokens": max_tokens,
        }
        self._write_state(state)
        self._log_saved(state)
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        # Write atomically — no size cap needed because this stores only:
        # - One prompt string (the original user prompt)
        # - Model name (short string)
        # - Progress percentage (number)
        # - Error message (optional, usually short)
        # - Partial output (the text streamed before the stop)
        # - Optional message history (only if provided)
        # Typical total: 500 bytes - a few KB
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.storage_path.parent),
            prefix=f".{self.storage_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                with _exclusive_lock(self._lock_path):
                    json.dump(state, f, indent=2)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        logger.debug(
                            "fsync unavailable for %s; skipping durability flush",
                            self.storage_path,
                        )
                    os.replace(tmp_path, self.storage_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _log_saved(self, state: dict[str, Any]) -> None:
        prompt = state["prompt"]
        prompt_preview = prompt[:50] if isinstance(prompt, str) else ""
        logger.info(
            "Saved interrupted work to %s (prompt=%r, progress=%s, error=%r)",
            self.storage_path,
            prompt_preview,
            _format_percent(state["progress_percent"]),
            state["error"],
        )

    def load_state(self) -> Optional[dict[str, Any]]:
        """Load the previously saved interrupted work state.

        Returns:
            The saved state dictionary, or None if no state is saved
            or the file doesn't exist.
        """
        if not self.storage_path.exists():
            return None

        try:
            with _exclusive_lock(self._lock_path):
                with open(self.storage_path, "r") as f:
                    state = json.load(f)

            if not isinstance(state, dict):
                logger.error(
                    "Ignoring interrupted work at %s: expected a JSON object, got %s",
                    self.storage_path,
                    type(state).__name__,
                )
                return None

            prompt = state.get("prompt")
            prompt_preview = prompt[:50] if isinstance(prompt, str) else ""
            logger.info(
                "Loaded interrupted work from %s (prompt=%r, progress=%s)",
                self.storage_path,
                prompt_preview,
                _format_percent(state.get("progress_percent")),
            )
            return state

        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load interrupted work from %s: %s", self.storage_path, e)
            return None

    def clear_state(self) -> bool:
        """Clear the saved interrupted work state.

        Returns:
            True if the state was cleared, False if there was nothing to clear.
        """
        if not self.storage_path.exists():
            logger.info("No interrupted work state to clear at %s", self.storage_path)
            self._reset_session()
            return False

        with _exclusive_lock(self._lock_path):
            try:
                self.storage_path.unlink()
            except FileNotFoundError:
                logger.info("No interrupted work state to clear at %s", self.storage_path)
                self._reset_session()
                return False
        logger.info("Cleared interrupted work state from %s", self.storage_path)
        self._reset_session()
        return True

    def _reset_session(self) -> None:
        """Forget any in-memory generation session."""
        with self._session_lock:
            for attribute in (
                "_current_prompt",
                "_current_model",
                "_current_system",
                "_progress_percent",
                "_error",
                "_messages",
                "_partial_output",
            ):
                if hasattr(self, attribute):
                    delattr(self, attribute)

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
    storage_path: Optional[str] = None,
    namespace: Optional[str] = None,
) -> WorkTracker:
    """Quick-track a generation session for later resumption.

    This is a convenience function that sets up the tracker with the given
    prompt and model. You should call save_interrupted() when generation
    stops (quota error, etc.), then load_state() to retrieve it later.

    Args:
        prompt: The original user prompt.
        model_name: The model being used.
        system: System prompt/instructions (optional).
        storage_path: Path to the JSON state file (optional; defaults to
            ".partial_work.json" in the current working directory).
        namespace: Optional name to scope the default state file so independent
            generations do not share one slot. Ignored when ``storage_path``
            is provided.

    Returns:
        A WorkTracker instance with the session started.
    """
    tracker = WorkTracker(storage_path=storage_path, namespace=namespace)
    tracker.start_tracking(
        prompt=prompt,
        model_name=model_name,
        system=system,
    )
    return tracker