import contextlib
import io
import json
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from llmbridge.work_tracker import WorkTracker, _exclusive_lock, track_generation


class WorkTrackerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / ".partial_work.json"

    def _tracker(self):
        return WorkTracker(storage_path=str(self.path))

    def test_start_tracking_does_not_write_to_disk(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="hello", model_name="model-a")
        self.assertFalse(self.path.exists())

    def test_save_and_load_round_trip(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="do the thing", model_name="model-a", system="be terse")
        tracker.update_progress(progress_percent=42.5, error="quota exhausted")
        tracker.save_interrupted()

        self.assertTrue(self.path.exists())
        loaded = self._tracker().load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["prompt"], "do the thing")
        self.assertEqual(loaded["model_name"], "model-a")
        self.assertEqual(loaded["system"], "be terse")
        self.assertEqual(loaded["progress_percent"], 42.5)
        self.assertEqual(loaded["error"], "quota exhausted")

    def test_saved_state_is_valid_json(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="json check", model_name="model-a")
        tracker.save_interrupted()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["prompt"], "json check")

    def test_save_interrupted_without_session_raises(self):
        with self.assertRaises(RuntimeError):
            self._tracker().save_interrupted()

    def test_save_writes_state_without_a_session(self):
        tracker = self._tracker()
        saved = tracker.save(
            prompt="direct save",
            model_name="model-b",
            system="be brief",
            error="quota exhausted",
        )
        self.assertTrue(self.path.exists())
        self.assertEqual(saved["prompt"], "direct save")
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["prompt"], "direct save")
        self.assertEqual(loaded["model_name"], "model-b")
        self.assertEqual(loaded["system"], "be brief")
        self.assertEqual(loaded["progress_percent"], 0.0)
        self.assertEqual(loaded["error"], "quota exhausted")
        self.assertEqual(loaded["messages"], [])

    def test_save_does_not_disturb_an_active_session(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="session prompt", model_name="model-a")
        tracker.update_progress(progress_percent=10.0)

        tracker.save(prompt="unrelated", model_name="model-b")

        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["prompt"], "session prompt")
        self.assertEqual(loaded["model_name"], "model-a")
        self.assertEqual(loaded["progress_percent"], 10.0)

    def test_save_copies_messages(self):
        tracker = self._tracker()
        messages = [{"role": "user", "content": "hi"}]
        tracker.save(prompt="p", model_name="m", messages=messages)
        messages.append({"role": "assistant", "content": "changed"})
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["messages"], [{"role": "user", "content": "hi"}])

    def test_save_defaults_messages_to_empty_list(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m")
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["messages"], [])

    def test_load_state_returns_none_when_missing(self):
        self.assertIsNone(self._tracker().load_state())

    def test_load_state_returns_none_on_corrupt_file(self):
        self.path.write_text("{ not valid json", encoding="utf-8")
        self.assertIsNone(self._tracker().load_state())

    def test_state_has_no_expiry(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="old work", model_name="model-a")
        tracker.save_interrupted()

        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["saved_at"] = 0  # ancient timestamp
        self.path.write_text(json.dumps(data), encoding="utf-8")

        loaded = self._tracker().load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["prompt"], "old work")

    def test_update_progress_does_not_write_to_disk(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=90.0)
        self.assertFalse(self.path.exists())

    def test_messages_are_optional(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["messages"], [])

    def test_messages_are_persisted_when_provided(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=10.0, messages=[{"role": "user", "content": "hi"}])
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["messages"], [{"role": "user", "content": "hi"}])

    def test_clear_state_removes_file_and_resets_session(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()

        self.assertTrue(tracker.clear_state())
        self.assertFalse(self.path.exists())
        self.assertIsNone(tracker.load_state())
        # Clearing twice reports nothing to clear.
        self.assertFalse(tracker.clear_state())
        # Session is reset, so saving now fails.
        with self.assertRaises(RuntimeError):
            tracker.save_interrupted()

    def test_clear_state_without_file_returns_false(self):
        self.assertFalse(self._tracker().clear_state())

    def test_clear_state_without_file_resets_session(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        self.assertFalse(tracker.clear_state())
        with self.assertRaises(RuntimeError):
            tracker.save_interrupted()

    def test_load_state_returns_none_for_non_dict_json(self):
        for content in ("[1, 2, 3]", "42", '"a string"', "null"):
            self.path.write_text(content, encoding="utf-8")
            self.assertIsNone(self._tracker().load_state())

    def test_load_state_survives_null_prompt(self):
        self.path.write_text('{"prompt": null, "saved_at": 1}', encoding="utf-8")
        loaded = self._tracker().load_state()
        self.assertEqual(loaded, {"prompt": None, "saved_at": 1})

    def test_get_progress_summary_survives_invalid_field_types(self):
        self.path.write_text(
            '{"prompt": null, "progress_percent": null, "model_name": null, "saved_at": "x"}',
            encoding="utf-8",
        )
        summary = self._tracker().get_progress_summary()
        self.assertIn("0.0%", summary)
        self.assertIn("unknown", summary)

    def test_update_progress_can_clear_error(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=5.0, error="boom")
        tracker.update_progress(progress_percent=6.0, error="")
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertIsNone(loaded["error"])

    def test_update_progress_omitting_error_leaves_it_unchanged(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=5.0, error="boom")
        tracker.update_progress(progress_percent=6.0)
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["error"], "boom")

    def test_saved_state_omits_dead_metadata_field(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("metadata", data)
        self.assertFalse(hasattr(tracker, "_started_at"))

    def test_get_progress_summary_without_state(self):
        self.assertEqual(self._tracker().get_progress_summary(), "No interrupted work in progress")

    def test_get_progress_summary_with_state(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="summarize me", model_name="model-a")
        tracker.update_progress(progress_percent=33.0, error="rate limited")
        tracker.save_interrupted()

        summary = self._tracker().get_progress_summary()
        self.assertIn("model-a", summary)
        self.assertIn("33.0%", summary)
        self.assertIn("rate limited", summary)
        self.assertIn("summarize me", summary)
        # saved_at is surfaced rather than being dead metadata.
        self.assertIn("Saved:", summary)

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        leftovers = [
            p.name
            for p in self.path.parent.iterdir()
            if p != self.path and not p.name.endswith(".lock")
        ]
        self.assertEqual(leftovers, [])

    def test_namespace_scopes_default_state_file(self):
        with TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                tracker = WorkTracker(namespace="session-a")
                self.assertEqual(tracker.storage_path.name, ".partial_work.session-a.json")
                other = WorkTracker(namespace="session-b")
                self.assertNotEqual(tracker.storage_path, other.storage_path)
            finally:
                os.chdir(cwd)

    def test_namespace_ignored_when_storage_path_given(self):
        tracker = WorkTracker(storage_path=str(self.path), namespace="ignored")
        self.assertEqual(tracker.storage_path, self.path)

    def test_state_file_is_locked_during_save(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        self.assertTrue(tracker._lock_path.exists())

    def test_save_and_load_do_not_write_to_stdout(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            tracker.save_interrupted()
            self._tracker().load_state()
        self.assertEqual(buffer.getvalue(), "")

    def test_save_emits_log_record(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        with self.assertLogs("llmbridge.work_tracker", level="INFO") as captured:
            tracker.save_interrupted()
        self.assertTrue(any("Saved interrupted work" in line for line in captured.output))

    def test_track_generation_convenience_uses_custom_path(self):
        tracker = track_generation(prompt="p", model_name="m", storage_path=str(self.path))
        self.assertEqual(tracker.storage_path, self.path)
        tracker.save_interrupted()
        self.assertTrue(self.path.exists())

    def test_track_generation_namespace(self):
        with TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                tracker = track_generation(prompt="p", model_name="m", namespace="xyz")
                self.assertEqual(tracker.storage_path.name, ".partial_work.xyz.json")
            finally:
                os.chdir(cwd)

    def test_invalid_namespace_is_rejected(self):
        for namespace in ["../evil", "foo/../../escape", "a/b", "..", ".hidden", "a b", 123]:
            with self.subTest(namespace=namespace):
                with self.assertRaises(ValueError):
                    WorkTracker(namespace=namespace)

    def test_valid_namespace_is_accepted(self):
        for namespace in ["session-a", "abc123", "a.b_c-1", "A9"]:
            with self.subTest(namespace=namespace):
                tracker = WorkTracker(namespace=namespace)
                self.assertEqual(tracker.storage_path.name, f".partial_work.{namespace}.json")

    def test_namespaces_do_not_interfere(self):
        with TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                first = WorkTracker(namespace="a")
                first.start_tracking(prompt="A", model_name="ma")
                first.save_interrupted()

                second = WorkTracker(namespace="b")
                second.start_tracking(prompt="B", model_name="mb")
                second.save_interrupted()

                self.assertEqual(first.load_state()["prompt"], "A")
                self.assertEqual(second.load_state()["prompt"], "B")
            finally:
                os.chdir(cwd)

    def test_save_survives_non_string_prompt(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt=None, model_name="m")  # type: ignore[arg-type]
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertIsNone(loaded["prompt"])

    def test_summary_truncates_long_prompt(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="x" * 100, model_name="m")
        tracker.save_interrupted()
        self.assertIn("...", self._tracker().get_progress_summary())

    def test_failed_save_cleans_up_temp_file(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        with mock.patch("llmbridge.work_tracker.json.dump", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                tracker.save_interrupted()
        leftovers = [p.name for p in self.path.parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertFalse(self.path.exists())

    def test_load_state_returns_none_on_oserror(self):
        self.path.write_text('{"prompt": "x"}', encoding="utf-8")
        with mock.patch("llmbridge.work_tracker.json.load", side_effect=OSError("nope")):
            self.assertIsNone(self._tracker().load_state())

    def test_clear_state_handles_file_vanishing(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        with mock.patch.object(Path, "unlink", side_effect=FileNotFoundError):
            self.assertFalse(tracker.clear_state())

    def test_exclusive_lock_serializes_threads(self):
        lock_path = self.path.with_name(self.path.name + ".lock")
        guard = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def worker():
            with _exclusive_lock(lock_path):
                with guard:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                time.sleep(0.02)
                with guard:
                    state["active"] -= 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(state["max_active"], 1)

    def test_lock_file_is_not_created_next_to_state_file(self):
        import tempfile

        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        siblings = [p.name for p in self.path.parent.iterdir() if p.name.endswith(".lock")]
        self.assertEqual(siblings, [])
        self.assertTrue(str(tracker._lock_path).startswith(tempfile.gettempdir()))

    def test_fsync_failure_does_not_abort_save(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        with mock.patch("llmbridge.work_tracker.os.fsync", side_effect=OSError("unsupported")):
            state = tracker.save_interrupted()
        self.assertTrue(self.path.exists())
        self.assertEqual(state["prompt"], "p")

    def test_shared_instance_is_thread_safe(self):
        tracker = self._tracker()
        errors = []

        def worker(index):
            try:
                for _ in range(25):
                    tracker.start_tracking(prompt=f"p{index}", model_name=f"m{index}")
                    tracker.update_progress(progress_percent=float(index))
                    tracker.save_interrupted()
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        loaded = self._tracker().load_state()
        self.assertIsInstance(loaded, dict)
        self.assertIsInstance(loaded["prompt"], str)


if __name__ == "__main__":
    unittest.main()
