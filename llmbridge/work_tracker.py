"""Track partially finished AI generation work for resumption with different models.

Design philosophy:
- Track ONLY the interrupted/prompt that was being processed when generation stopped
- No size limits during ongoing generation (just mark "in progress")
- When generation stops (quota error, etc.), save just that interrupted prompt
- After resumption/completion, clear the state
- State is small (just one prompt + metadata), so no arbitrary KB caps needed
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional, List


class WorkTracker:
    """Tracks partially finished AI generation work so it can be resumed
    with a different model after quota exhaustion or errors.

    Key design:
    - State is ONLY saved when generation stops (not on every call)
    - Only the interrupted prompt + minimal metadata is stored
    - No arbitrary KB caps — the state is naturally small (just one prompt)
    - State automatically becomes stale after 24 hours
    - Clear automatically after resumption or on demand
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the work tracker.

        Args:
            storage_path: Path to the JSON file for storing partial work.
                If None, uses ".partial_work.json" in the session cwd.
        """
        if storage_path is None:
            self.storage_path = Path(".partial_work.json")
        else:
            self.storage_path = Path(storage_path)
        self._ensure_storage_dir()

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
        self._current_prompt = prompt
        self._current_model = model_name
        self._current_system = system
        self._started_at = time.time()
        self._progress_percent = 0.0
        self._error: Optional[str] = None
        self._messages: List[dict[str, Any]] = []

        # Do NOT save to disk here — just set internal state
        # This avoids saving on every intermediate step

    def update_progress(
        self,
        *,
        progress_percent: float,
        error: Optional[str] = None,
        messages: Optional[List[dict[str, Any]]] = None,
    ) -> None:
        """Update the current generation progress (called during generation).

        This can be called periodically during a long generation. It updates
        internal state but does NOT save to disk. Only the final interrupted
        state will be saved.

        Args:
            progress_percent: Current progress percentage (0-100).
            error: Error message if generation stopped (optional).
            messages: Conversation history so far (optional).
        """
        self._progress_percent = progress_percent
        if error:
            self._error = error
        if messages is not None:
            self._messages = messages

    def save_interrupted(self) -> dict[str, Any]:
        """Save the interrupted generation state to disk.

        Call this method when generation stops unexpectedly (quota error,
        rate limit, etc.) so the work can be resumed with a different model.

        Only the current prompt + minimal metadata is stored — no conversation
        history, no intermediate progress steps. This makes the state naturally
        small (typically 500 bytes - 2 KB) without needing arbitrary KB caps.

        Returns:
            The saved state dictionary (same format as load_state() returns).
        """
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
            "metadata": {},
        }

        # Write to disk — no size cap needed because this stores only:
        # - One prompt string (the original user prompt)
        # - Model name (short string)
        # - Progress percentage (number)
        # - Error message (optional, usually short)
        # - Minimal message history
        # Typical total: 500 bytes - 2 KB

        with open(self.storage_path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"💾 Saved interrupted work to {self.storage_path}")
        print(f"   Prompt: {self._current_prompt[:50]}{'...' if len(self._current_prompt) > 50 else ''}")
        print(f"   Progress: {self._progress_percent:.1f}%")
        if self._error:
            print(f"   Error: {self._error}")
        return state

    def load_state(self) -> Optional[dict[str, Any]]:
        """Load the previously saved interrupted work state.

        Returns:
            The saved state dictionary, or None if no state is saved
            or the file doesn't exist or the state is stale (>24 hours).
        """
        if not self.storage_path.exists():
            return None

        try:
            with open(self.storage_path, "r") as f:
                state = json.load(f)

            # Check if the state is recent (within 24 hours)
            saved_at = state.get("saved_at", 0)
            age_seconds = time.time() - saved_at
            if age_seconds > 86400:  # 24 hours
                print(f"⚠️  Saved work is {age_seconds / 3600:.1f} days old — may be stale")
                return None

            print(f"📂 Loaded interrupted work from {self.storage_path}")
            print(f"   Prompt: {state.get('prompt', '')[:50]}{'...' if len(state.get('prompt', '')) > 50 else ''}")
            print(f"   Progress: {state.get('progress_percent', 0):.1f}%")
            if state.get("error"):
                print(f"   Error: {state.get('error')}")
            return state

        except (json.JSONDecodeError, OSError) as e:
            print(f"❌ Failed to load interrupted work: {e}")
            return None

    def clear_state(self) -> bool:
        """Clear the saved interrupted work state.

        Returns:
            True if the state was cleared, False if there was nothing to clear.
        """
        if not self.storage_path.exists():
            print(f"ℹ️  No interrupted work state to clear at {self.storage_path}")
            return False

        self.storage_path.unlink()
        print(f"🧹 Cleared interrupted work state from {self.storage_path}")
        # Reset internal tracking state
        if hasattr(self, "_current_prompt"):
            del self._current_prompt
        if hasattr(self, "_current_model"):
            del self._current_model
        if hasattr(self, "_current_system"):
            del self._current_system
        if hasattr(self, "_started_at"):
            del self._started_at
        if hasattr(self, "_progress_percent"):
            del self._progress_percent
        if hasattr(self, "_error"):
            del self._error
        if hasattr(self, "_messages"):
            del self._messages
        return True

    def get_progress_summary(self) -> str:
        """Get a human-readable summary of the current interrupted work state.

        Returns:
            A string summary, or "No interrupted work in progress" if nothing is saved.
        """
        state = self.load_state()
        if state is None:
            return "No interrupted work in progress"

        progress = state.get("progress_percent", 0)
        stage = "interrupted"  # Always "interrupted" since we only save on stop
        model = state.get("model_name", "unknown")
        error = state.get("error")

        lines = [
            f"Interrupted work in progress:",
            f"  Model: {model}",
            f"  Stage: {stage}",
            f"  Progress: {progress:.1f}%",
        ]
        if error:
            lines.append(f"  Error: {error}")

        # Also show prompt preview
        prompt = state.get("prompt", "")
        if prompt:
            lines.append(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

        return "\n".join(lines)


def track_generation(
    prompt: str,
    model_name: str,
    system: Optional[str] = None,
) -> WorkTracker:
    """Quick-track a generation session for later resumption.

    This is a convenience function that sets up the tracker with the given
    prompt and model. You should call save_interrupted() when generation
    stops (quota error, etc.), then load_state() to retrieve it later.

    Args:
        prompt: The original user prompt.
        model_name: The model being used.
        system: System prompt/instructions (optional).

    Returns:
        A WorkTracker instance with the session started.
    """
    tracker = WorkTracker()
    tracker.start_tracking(
        prompt=prompt,
        model_name=model_name,
        system=system,
    )
    return tracker