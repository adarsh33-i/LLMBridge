import contextlib
import io
import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llmbridge.ledger import EVENT_INTERRUPT, EVENT_RESUME, EVENT_START, GenerationLedger
from llmbridge.work_tracker import WorkTracker, track_generation


class WorkTrackerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _tracker(self):
        return WorkTracker(self.ledger)

    def _record_files(self):
        return list(self.ledger.root.glob("gen_*.json"))

    def test_start_tracking_does_not_write_to_disk(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="hello", model_name="model-a")
        self.assertEqual(self._record_files(), [])

    def test_save_interrupted_creates_interrupted_ledger_record(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="do the thing", model_name="model-a", system="be terse")
        tracker.update_progress(progress_percent=42.5, error="quota exhausted")
        state = tracker.save_interrupted()

        files = self._record_files()
        self.assertEqual(len(files), 1)
        data = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "interrupted")
        self.assertEqual(data["prompt"], "do the thing")
        self.assertEqual(
            [event["type"] for event in data["events"]], [EVENT_START, EVENT_INTERRUPT]
        )
        self.assertTrue(state["generation_id"].startswith("gen_"))
        self.assertEqual(state["generation_id"], data["id"])

    def test_save_and_load_round_trip(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="do the thing", model_name="model-a", system="be terse")
        tracker.update_progress(progress_percent=42.5, error="quota exhausted")
        tracker.save_interrupted()

        loaded = self._tracker().load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["prompt"], "do the thing")
        self.assertEqual(loaded["model_name"], "model-a")
        self.assertEqual(loaded["system"], "be terse")
        self.assertEqual(loaded["progress_percent"], 42.5)
        self.assertEqual(loaded["error"], "quota exhausted")

    def test_save_writes_state_without_a_session(self):
        tracker = self._tracker()
        saved = tracker.save(
            prompt="direct save",
            model_name="model-b",
            system="be brief",
            error="quota exhausted",
        )
        self.assertEqual(len(self._record_files()), 1)
        self.assertEqual(saved["prompt"], "direct save")
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["prompt"], "direct save")
        self.assertEqual(loaded["model_name"], "model-b")
        self.assertEqual(loaded["system"], "be brief")
        self.assertEqual(loaded["progress_percent"], 0.0)
        self.assertEqual(loaded["error"], "quota exhausted")
        self.assertEqual(
            loaded["messages"], [{"role": "user", "content": "direct save"}]
        )

    def test_save_interrupted_without_session_raises(self):
        with self.assertRaises(RuntimeError):
            self._tracker().save_interrupted()

    def test_update_progress_without_session_raises(self):
        with self.assertRaises(RuntimeError):
            self._tracker().update_progress(progress_percent=1.0)

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

    def test_concurrent_saves_do_not_overwrite_each_other(self):
        first = self._tracker()
        first.save(prompt="first", model_name="m", error="quota")
        second = self._tracker()
        second.save(prompt="second", model_name="m", error="rate limited")

        self.assertEqual(len(self._record_files()), 2)
        # The newest interrupted record is the one offered for resumption.
        self.assertEqual(self._tracker().load_state()["prompt"], "second")

    def test_save_copies_messages(self):
        tracker = self._tracker()
        messages = [{"role": "user", "content": "hi"}]
        tracker.save(prompt="p", model_name="m", messages=messages)
        messages.append({"role": "assistant", "content": "changed"})
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["messages"], [{"role": "user", "content": "hi"}])

    def test_save_defaults_messages_to_prompt_fallback(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m")
        loaded = self._tracker().load_state()
        # The record keeps the prompt as a replayable user turn.
        self.assertEqual(
            loaded["messages"], [{"role": "user", "content": "p"}]
        )
        record = self.ledger.load(loaded["generation_id"])
        self.assertEqual(record.prompt, "p")

    def test_save_persists_partial_output(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m", partial_output="half an answer")
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["partial_output"], "half an answer")

    def test_save_coerces_non_string_partial_output(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m", partial_output=123)  # type: ignore[arg-type]
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["partial_output"], "")

    def test_save_persists_max_tokens(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m", max_tokens=512)
        self.assertEqual(self._tracker().load_state()["max_tokens"], 512)

    def test_save_ignores_invalid_max_tokens(self):
        tracker = self._tracker()
        for bad in (0, -5, "big", True, None):
            state = tracker.save(prompt="p", model_name="m", max_tokens=bad)
            self.assertIsNone(state["max_tokens"])

    def test_update_progress_persists_partial_output(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=50.0, partial_output="so far")
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["partial_output"], "so far")

    def test_update_progress_omitting_partial_output_leaves_it_unchanged(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=50.0, partial_output="kept")
        tracker.update_progress(progress_percent=60.0)
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["partial_output"], "kept")

    def test_update_progress_can_clear_partial_output(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=50.0, partial_output="kept")
        tracker.update_progress(progress_percent=60.0, partial_output="")
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["partial_output"], "")

    def test_update_progress_coerces_non_string_partial_output(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=10.0, partial_output=42)  # type: ignore[arg-type]
        tracker.save_interrupted()
        self.assertEqual(self._tracker().load_state()["partial_output"], "")

    def test_load_state_returns_none_when_nothing_interrupted(self):
        self.assertIsNone(self._tracker().load_state())

    def test_load_state_ignores_completed_generations(self):
        record = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            "m",
        )
        self.ledger.complete(record, "done", 3)
        self.assertIsNone(self._tracker().load_state())

    def test_load_state_tolerates_corrupt_records(self):
        tracker = self._tracker()
        tracker.save(prompt="good", model_name="m")
        (self.ledger.root / "gen_broken.json").write_text("{ not valid json", encoding="utf-8")
        self.assertEqual(self._tracker().load_state()["prompt"], "good")

    def test_state_has_no_expiry(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="old work", model_name="model-a")
        tracker.save_interrupted()

        path = self._record_files()[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        data["updated_at"] = 0  # ancient timestamp
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = self._tracker().load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["prompt"], "old work")

    def test_update_progress_does_not_write_to_disk(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=90.0)
        self.assertEqual(self._record_files(), [])

    def test_messages_are_persisted_when_provided(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=10.0, messages=[{"role": "user", "content": "hi"}])
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["messages"], [{"role": "user", "content": "hi"}])

    def test_clear_state_removes_record_and_resets_session(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()

        self.assertTrue(tracker.clear_state())
        self.assertEqual(self._record_files(), [])
        self.assertIsNone(tracker.load_state())
        # Clearing twice reports nothing to clear.
        self.assertFalse(tracker.clear_state())
        # Session is reset, so saving now fails.
        with self.assertRaises(RuntimeError):
            tracker.save_interrupted()

    def test_clear_state_without_records_returns_false(self):
        self.assertFalse(self._tracker().clear_state())

    def test_clear_state_without_records_resets_session(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        self.assertFalse(tracker.clear_state())
        with self.assertRaises(RuntimeError):
            tracker.save_interrupted()

    def test_clear_state_only_removes_interrupted_records(self):
        tracker = self._tracker()
        completed = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "done"}]},
            "m",
        )
        self.ledger.complete(completed, "done", 3)
        tracker.save(prompt="interrupted work", model_name="m")

        self.assertTrue(tracker.clear_state())
        remaining = self.ledger.list(limit=None)
        self.assertEqual([record.id for record in remaining], [completed.id])
        self.assertEqual(remaining[0].status, "completed")

    def _fingerprint_of(self, generation_id):
        record = self.ledger.load(generation_id)
        self.assertIsNotNone(record)
        return record.fingerprint

    def test_resolve_matching_marks_the_record_resumed_and_keeps_it(self):
        tracker = self._tracker()
        state = tracker.save(prompt="Hello", model_name="m")
        resolved = tracker.resolve_matching(
            self._fingerprint_of(state["generation_id"]), by="gen_newer000000", model="m"
        )
        self.assertEqual(resolved, state["generation_id"])
        record = self.ledger.load(state["generation_id"])
        self.assertEqual(record.status, "resumed")
        self.assertEqual(record.events[-1].type, EVENT_RESUME)
        self.assertEqual(record.events[-1].data["resumed_by"], "gen_newer000000")
        # No longer offered, but still on disk as evidence of the interruption.
        self.assertIsNone(tracker.load_state())
        self.assertTrue(self.ledger._path(state["generation_id"]).exists())

    def test_resolve_matching_ignores_other_requests(self):
        tracker = self._tracker()
        state = tracker.save(prompt="Hello", model_name="m")
        self.assertIsNone(tracker.resolve_matching("some-other-fingerprint", by=None, model="m"))
        self.assertEqual(len(self._record_files()), 1)
        self.assertEqual(self._tracker().load_state()["prompt"], "Hello")

    def test_resolve_matching_without_records_returns_none(self):
        self.assertIsNone(self._tracker().resolve_matching("abc", by=None, model="m"))

    def test_resolve_matching_tolerates_corrupt_state(self):
        (self.ledger.root / "gen_broken.json").write_text("{ not json", encoding="utf-8")
        self.assertIsNone(self._tracker().resolve_matching("abc", by=None, model="m"))

    def test_resolve_matching_only_resolves_the_request_that_ran(self):
        tracker = self._tracker()
        # Two concurrent jobs that share the prompt text "continue" but are
        # different conversations: prompt text is not an identity.
        first = tracker.save(
            prompt="continue",
            model_name="m",
            messages=[{"role": "user", "content": "first job: continue"}],
        )
        second = tracker.save(
            prompt="continue",
            model_name="m",
            messages=[{"role": "user", "content": "second job: continue"}],
        )
        resolved = tracker.resolve_matching(
            self._fingerprint_of(first["generation_id"]), by="gen_newer000001", model="m"
        )
        self.assertEqual(resolved, first["generation_id"])
        pending = self.ledger.interrupted()
        self.assertEqual(
            [record.id for record in pending], [second["generation_id"]]
        )
        self.assertEqual(pending[0].status, "interrupted")
        self.assertEqual(
            self.ledger.load(first["generation_id"]).status, "resumed"
        )

    def test_resolve_matching_never_resolves_itself(self):
        tracker = self._tracker()
        state = tracker.save(prompt="Hello", model_name="m")
        mine = self._fingerprint_of(state["generation_id"])
        self.assertIsNone(
            tracker.resolve_matching(mine, by=state["generation_id"], model="m")
        )
        self.assertEqual(self.ledger.load(state["generation_id"]).status, "interrupted")

    def test_resolver_leaves_an_unrelated_session_alone(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="my own work", model_name="m")
        # A different request completing says nothing about this session.
        self.assertIsNone(
            tracker.resolve_matching("no-such-fingerprint", by=None, model="m")
        )
        self.assertEqual(tracker.save_interrupted()["prompt"], "my own work")

    def test_resolver_clears_the_session_it_resolved(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="my own work", model_name="m")
        state = tracker.save_interrupted()
        resolved = tracker.resolve_matching(
            self._fingerprint_of(state["generation_id"]), by="gen_newer000002", model="m"
        )
        self.assertEqual(resolved, state["generation_id"])
        with self.assertRaises(RuntimeError):
            tracker.save_interrupted()

    def test_save_persists_replayable_request(self):
        tracker = self._tracker()
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "search"}],
            "tool_choice": {"type": "auto"},
        }
        tracker.save(prompt="hi", model_name="m", request=request)
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["request"], request)

    def test_start_tracking_persists_replayable_request(self):
        tracker = self._tracker()
        request = {"messages": [{"role": "user", "content": "hi"}]}
        tracker.start_tracking(prompt="hi", model_name="m", request=request)
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["request"], request)

    def test_replayable_request_defaults_to_recorded_prompt_turn(self):
        tracker = self._tracker()
        tracker.save(prompt="hi", model_name="m")
        self.assertEqual(
            self._tracker().load_state()["request"],
            {"messages": [{"role": "user", "content": "hi"}]},
        )

    def test_state_survives_a_new_tracker_and_ledger_instance(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="durable", model_name="model-a")
        tracker.save_interrupted()

        reopened = WorkTracker(GenerationLedger(str(self.ledger.root)))
        self.assertEqual(reopened.load_state()["prompt"], "durable")

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
        self.assertIn("gen_", summary)
        # saved_at is surfaced rather than being dead metadata.
        self.assertIn("Saved:", summary)

    def test_get_progress_summary_survives_invalid_field_types(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m")
        path = self._record_files()[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        data["model"] = ""
        data["events"][-1]["data"]["progress_percent"] = None
        path.write_text(json.dumps(data), encoding="utf-8")

        summary = self._tracker().get_progress_summary()
        self.assertIn("0.0%", summary)
        self.assertIn("unknown", summary)

    def test_summary_truncates_long_prompt(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="x" * 100, model_name="m")
        tracker.save_interrupted()
        self.assertIn("...", self._tracker().get_progress_summary())

    def test_summary_includes_partial_output_preview(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.update_progress(progress_percent=10.0, partial_output="def solve():\n    return 42")
        tracker.save_interrupted()
        summary = self._tracker().get_progress_summary()
        self.assertIn("Partial output:", summary)
        self.assertIn("def solve()", summary)
        self.assertNotIn("\n    return", summary)

    def test_summary_omits_partial_output_when_absent(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        self.assertNotIn("Partial output:", self._tracker().get_progress_summary())

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt="p", model_name="m")
        tracker.save_interrupted()
        leftovers = [
            p.name
            for p in self.ledger.root.iterdir()
            if not p.name.endswith(".json")
        ]
        self.assertEqual(leftovers, [])

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

    def test_track_generation_convenience_uses_custom_ledger(self):
        tracker = track_generation(prompt="p", model_name="m", ledger=self.ledger)
        self.assertIs(tracker.ledger, self.ledger)
        tracker.save_interrupted()
        self.assertEqual(len(self._record_files()), 1)

    def test_track_generation_defaults_to_default_ledger(self):
        with TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                tracker = track_generation(prompt="p", model_name="m")
                self.assertEqual(tracker.ledger.root, Path(".llmbridge/generations"))
            finally:
                os.chdir(cwd)

    def test_save_survives_non_string_prompt(self):
        tracker = self._tracker()
        tracker.start_tracking(prompt=None, model_name="m")  # type: ignore[arg-type]
        tracker.save_interrupted()
        loaded = self._tracker().load_state()
        self.assertEqual(loaded["prompt"], "")

    def test_load_state_survives_missing_record_fields(self):
        tracker = self._tracker()
        tracker.save(prompt="p", model_name="m")
        path = self._record_files()[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("events", None)
        data.pop("request", None)
        data.pop("params", None)
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = self._tracker().load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["prompt"], "p")
        self.assertEqual(loaded["progress_percent"], 0.0)
        self.assertIsNone(loaded["max_tokens"])

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
