import itertools
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent

from llmbridge.ledger import (
    EVENT_COMPLETE,
    EVENT_FAIL,
    EVENT_FORK,
    EVENT_HEAL,
    EVENT_HEAL_SEAM,
    EVENT_INTERRUPT,
    EVENT_QUEUED,
    EVENT_START,
    MAX_TRACKED_INTERRUPTED,
    MAX_KEPT_RECOVERABLE,
    MAX_KEPT_SUSPECT_AUDITS,
    GenerationLedger,
    GenerationRecord,
    _exclusive_lock,
    _fingerprint,
    _lock_path_for,
    _prompt_and_system,
    _validate_generation_id,
    request_replay,
)


def _request(prompt="Hello", model="nemotron", **extra):
    body = {"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    return body


class GenerationLedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def test_create_records_metadata_and_start_event(self):
        record = self.ledger.create(_request(system=None), "nemotron")
        self.assertTrue(record.id.startswith("gen_"))
        self.assertEqual(record.status, "running")
        self.assertEqual(record.model, "nemotron")
        self.assertEqual(record.prompt, "Hello")
        self.assertEqual(record.params["max_tokens"], 100)
        self.assertEqual([event.type for event in record.events], [EVENT_START])
        self.assertTrue((self.ledger.root / f"{record.id}.json").exists())

    def test_create_captures_system_prompt(self):
        body = _request()
        body["system"] = [{"type": "text", "text": "be terse"}]
        record = self.ledger.create(body, "nemotron")
        self.assertEqual(record.system, "be terse")

    def test_fingerprint_is_stable_and_request_specific(self):
        first = self.ledger.create(_request(), "nemotron")
        second = self.ledger.create(_request(), "nemotron")
        third = self.ledger.create(_request(prompt="Different"), "nemotron")
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, third.fingerprint)
        self.assertNotEqual(first.id, second.id)

    def test_complete_sets_status_output_and_event(self):
        record = self.ledger.create(_request(), "nemotron")
        self.ledger.complete(record, "Hello world", 7)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.output, "Hello world")
        self.assertEqual(record.output_tokens, 7)
        self.assertEqual([event.type for event in record.events], [EVENT_START, EVENT_COMPLETE])

        reloaded = self.ledger.load(record.id)
        self.assertEqual(reloaded.output, "Hello world")
        self.assertEqual(reloaded.output_tokens, 7)
        self.assertEqual(reloaded.status, "completed")

    def test_interrupt_persists_partial_output(self):
        record = self.ledger.create(_request(), "nemotron")
        self.ledger.interrupt(record, "quota exhausted", output="Half", output_tokens=2, progress=10.0)
        self.assertEqual(record.status, "interrupted")
        reloaded = self.ledger.load(record.id)
        self.assertEqual(reloaded.output, "Half")
        self.assertEqual(reloaded.events[-1].type, EVENT_INTERRUPT)
        self.assertEqual(reloaded.events[-1].data["message"], "quota exhausted")
        self.assertEqual(reloaded.events[-1].data["progress_percent"], 10.0)

    def test_fail_marks_generation_failed(self):
        record = self.ledger.create(_request(), "nemotron")
        self.ledger.fail(record, "bad request")
        self.assertEqual(record.status, "failed")
        reloaded = self.ledger.load(record.id)
        self.assertEqual(reloaded.events[-1].type, EVENT_FAIL)

    def test_forked_child_records_parent_and_fork_event(self):
        parent = self.ledger.create(_request(), "nemotron")
        self.ledger.complete(parent, "partial", 3)
        child = self.ledger.create(
            _request(), "qwen-local", parent=parent.id, fork_event_seq=1
        )
        self.assertEqual(child.parent, parent.id)
        self.assertEqual(child.fork_event_seq, 1)
        self.assertEqual(child.events[0].type, EVENT_FORK)
        self.assertEqual(child.events[0].data["parent"], parent.id)

    def test_list_returns_newest_first_and_respects_limit(self):
        first = self.ledger.create(_request(prompt="one"), "nemotron")
        second = self.ledger.create(_request(prompt="two"), "nemotron")
        first.created_at = 1.0
        second.created_at = 2.0
        self.ledger._save(first)
        self.ledger._save(second)

        ordered = self.ledger.list()
        self.assertEqual([record.id for record in ordered], [second.id, first.id])
        self.assertEqual(len(self.ledger.list(limit=1)), 1)

    def test_delete_removes_record(self):
        record = self.ledger.create(_request(), "nemotron")
        self.assertTrue(self.ledger.delete(record.id))
        self.assertIsNone(self.ledger.load(record.id))
        self.assertFalse(self.ledger.delete(record.id))

    def test_load_unknown_or_invalid_id_returns_none(self):
        self.assertIsNone(self.ledger.load("gen_missing"))
        self.assertIsNone(self.ledger.load("../../etc/passwd"))

    def test_malformed_record_is_ignored(self):
        path = self.ledger.root / "gen_broken.json"
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.ledger.load("gen_broken"))
        # A well-formed JSON object missing its id is also ignored.
        path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
        self.assertIsNone(self.ledger.load("gen_broken"))
        # Listing skips it rather than failing.
        self.assertEqual(self.ledger.list(), [])

    def test_state_survives_a_new_ledger_instance(self):
        record = self.ledger.create(_request(), "nemotron")
        self.ledger.complete(record, "done", 4)
        reopened = GenerationLedger(str(self.ledger.root))
        reloaded = reopened.load(record.id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.output, "done")

    def test_create_records_replayable_request(self):
        body = {
            "model": "nemotron",
            "max_tokens": 100,
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
            "tools": [{"name": "search", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
        }
        record = self.ledger.create(body, "nemotron")
        self.assertEqual(record.prompt, "second")
        self.assertEqual(record.request["system"], "be terse")
        self.assertEqual(record.request["messages"], body["messages"])
        self.assertEqual(record.request["tools"], body["tools"])
        self.assertEqual(record.request["tool_choice"], {"type": "auto"})

    def test_replayable_request_survives_round_trip(self):
        body = {
            "model": "nemotron",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "search"}],
        }
        record = self.ledger.create(body, "nemotron")
        reloaded = GenerationLedger(str(self.ledger.root)).load(record.id)
        self.assertEqual(reloaded.request["messages"], body["messages"])
        self.assertEqual(reloaded.request["tools"], body["tools"])

    def test_request_replay_omits_empty_and_absent_fields(self):
        self.assertEqual(request_replay({}), {})
        self.assertEqual(
            request_replay({"messages": [], "tools": [], "tool_choice": None}),
            {},
        )
        self.assertNotIn("system", request_replay({"system": None, "messages": [{"a": 1}]}))


class ForkRequestTests(unittest.TestCase):
    def _record(self, **kwargs):
        defaults = dict(
            id="gen_abc",
            fingerprint="fp",
            model="nemotron",
            prompt="Hello",
            system=None,
            params={"max_tokens": 200},
            status="completed",
            created_at=0.0,
            updated_at=0.0,
            output="Half",
        )
        defaults.update(kwargs)
        return GenerationRecord(**defaults)

    def test_replays_output_and_uses_recorded_max_tokens(self):
        request = GenerationLedger.fork_request(self._record())
        self.assertEqual(request["messages"][0], {"role": "user", "content": "Hello"})
        self.assertEqual(request["messages"][1], {"role": "assistant", "content": "Half"})
        self.assertEqual(request["max_tokens"], 200)
        self.assertEqual(request["model"], "nemotron")
        self.assertFalse(request["stream"])

    def test_can_override_model_max_tokens_and_stream(self):
        request = GenerationLedger.fork_request(
            self._record(system="be terse"),
            model="qwen-local",
            max_tokens=64,
            stream=True,
        )
        self.assertEqual(request["model"], "qwen-local")
        self.assertEqual(request["max_tokens"], 64)
        self.assertTrue(request["stream"])
        self.assertEqual(request["system"], "be terse")

    def test_without_output_start_fresh_and_default_max_tokens(self):
        request = GenerationLedger.fork_request(
            self._record(output="", params={})
        )
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["max_tokens"], 4096)

    def test_include_output_false_omits_assistant_turn(self):
        request = GenerationLedger.fork_request(self._record(), include_output=False)
        self.assertEqual(len(request["messages"]), 1)

    def test_replays_full_recorded_conversation_and_tools(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        record = self._record(
            prompt="second",
            request={
                "messages": history,
                "tools": [{"name": "search"}],
                "tool_choice": {"type": "auto"},
                "system": "be terse",
            },
        )
        request = GenerationLedger.fork_request(record)
        # Full history is preserved, then the partial output continues it.
        self.assertEqual(request["messages"][:3], history)
        self.assertEqual(request["messages"][3], {"role": "assistant", "content": "Half"})
        self.assertEqual(request["tools"], [{"name": "search"}])
        self.assertEqual(request["tool_choice"], {"type": "auto"})
        self.assertEqual(request["system"], "be terse")

    def test_falls_back_to_prompt_when_no_request_recorded(self):
        request = GenerationLedger.fork_request(self._record())
        self.assertEqual(request["messages"][0], {"role": "user", "content": "Hello"})

    def test_merges_into_a_trailing_assistant_prefill(self):
        history = [
            {"role": "user", "content": "write the function"},
            {"role": "assistant", "content": "def f():"},
        ]
        request = GenerationLedger.fork_request(
            self._record(request={"messages": history})
        )
        # A prefill is joined, not duplicated: two assistant turns in a row
        # would be rejected upstream.
        self.assertEqual(len(request["messages"]), 2)
        self.assertEqual(request["messages"][-1]["content"], "def f():Half")
        self.assertEqual(history[-1]["content"], "def f():")

    def test_merges_into_a_trailing_assistant_block_list(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure"}]},
        ]
        request = GenerationLedger.fork_request(
            self._record(request={"messages": history})
        )
        self.assertEqual(
            request["messages"][-1]["content"],
            [{"type": "text", "text": "Sure"}, {"type": "text", "text": "Half"}],
        )


class LedgerHelperTests(unittest.TestCase):
    def test_validate_generation_id_rejects_traversal(self):
        with self.assertRaises(ValueError):
            _validate_generation_id("../evil")
        with self.assertRaises(ValueError):
            _validate_generation_id("nope")
        self.assertEqual(_validate_generation_id("gen_abc"), "gen_abc")

    def test_prompt_and_system_flattens_blocks(self):
        prompt, system = _prompt_and_system(
            {
                "system": [{"type": "text", "text": "sys"}],
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": [{"type": "text", "text": "second"}]},
                ],
            }
        )
        self.assertEqual(prompt, "second")
        self.assertEqual(system, "sys")

    def test_prompt_and_system_handles_non_dict(self):
        self.assertEqual(_prompt_and_system(None), ("", None))

    def test_fingerprint_tolerates_unserializable(self):
        first = _fingerprint({"value": object()})
        second = _fingerprint({"value": object()})
        self.assertEqual(len(first), 16)
        self.assertIsInstance(second, str)


class LedgerPruneTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _files(self):
        return sorted(self.ledger.root.glob("gen_*.json"))

    def test_create_prunes_to_the_newest_records(self):
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            for index in range(MAX_TRACKED_INTERRUPTED + 4):
                self.ledger.create(
                    {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{index}"}]},
                    "m",
                )
        records = self.ledger.list(limit=None)
        self.assertEqual(len(records), MAX_TRACKED_INTERRUPTED)
        # Newest first, with only the oldest four pruned.
        prompts = [record.prompt for record in records]
        self.assertEqual(
            prompts, [f"p{index}" for index in range(MAX_TRACKED_INTERRUPTED + 3, 3, -1)]
        )

    def test_prune_returns_removed_ids_oldest_first(self):
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            created = [
                self.ledger.create(
                    {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{i}"}]},
                    "m",
                )
                for i in range(4)
            ]
        removed = self.ledger.prune(keep=2)
        self.assertEqual(removed, [created[0].id, created[1].id])
        self.assertEqual([record.id for record in self.ledger.list(limit=None)], [created[3].id, created[2].id])

    def test_prune_below_cap_is_a_no_op(self):
        self.ledger.create({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "only"}]}, "m")
        self.assertEqual(self.ledger.prune(), [])
        self.assertEqual(len(self._files()), 1)

    def test_prune_deletes_files_from_disk(self):
        for index in range(MAX_TRACKED_INTERRUPTED + 2):
            self.ledger.create(
                {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{index}"}]},
                "m",
            )
        self.assertEqual(len(self._files()), MAX_TRACKED_INTERRUPTED)

    def _interrupted_run(self, ledger, prompt):
        record = ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]},
            "m",
        )
        ledger.interrupt(record, "quota", output="half", progress=10.0)
        return record

    def test_recoverable_work_survives_the_retention_window(self):
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            made = [
                self._interrupted_run(self.ledger, f"p{index}")
                for index in range(MAX_TRACKED_INTERRUPTED + 1)
            ]
        # The oldest of these sits outside the window, but it is still unfinished
        # work the bridge could offer to resume.
        interrupted = self.ledger.interrupted()
        self.assertEqual(
            [record.id for record in interrupted], [record.id for record in reversed(made)]
        )
        for record in made:
            self.assertTrue(self.ledger._path(record.id).exists())

    def test_queued_work_survives_the_retention_window(self):
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            first = self.ledger.create(
                {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "wait"}]},
                "m",
            )
            self.ledger.queue(first, "over quota", time.time() + 60, 1)
            for index in range(MAX_TRACKED_INTERRUPTED + 4):
                done = self.ledger.create(
                    {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{index}"}]},
                    "m",
                )
                self.ledger.complete(done, "fine", 1)
        kept = self.ledger.load(first.id)
        self.assertIsNotNone(kept)
        self.assertEqual(kept.status, "queued")
        self.assertEqual([record.id for record in self.ledger.queued()], [first.id])

    def test_recoverable_exemption_is_bounded_and_freshest_first(self):
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            made = [
                self._interrupted_run(self.ledger, f"p{index}")
                for index in range(MAX_TRACKED_INTERRUPTED + MAX_KEPT_RECOVERABLE + 2)
            ]
        retained = [record.id for record in self.ledger.interrupted()]
        self.assertEqual(
            retained,
            [
                record.id
                for record in reversed(
                    made[-(MAX_TRACKED_INTERRUPTED + MAX_KEPT_RECOVERABLE):]
                )
            ],
        )
        self.assertEqual(
            len(self.ledger.list(limit=None)),
            MAX_TRACKED_INTERRUPTED + MAX_KEPT_RECOVERABLE,
        )
        for record in made[:2]:
            self.assertIsNone(self.ledger.load(record.id))

    def test_zero_recoverable_budget_restores_the_plain_window(self):
        with mock.patch.dict(
            os.environ, {"LLMBRIDGE_LEDGER_MAX_RECOVERABLE": "0"}
        ):
            ledger = GenerationLedger(str(Path(self._tmp.name) / "no-recover"))
        self.assertEqual(ledger.max_recoverable, 0)
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            first = self._interrupted_run(ledger, "p0")
            for index in range(1, MAX_TRACKED_INTERRUPTED + 3):
                self._interrupted_run(ledger, f"p{index}")
        self.assertIsNone(ledger.load(first.id))
        self.assertEqual(len(ledger.list(limit=None)), MAX_TRACKED_INTERRUPTED)

    def test_bad_recoverable_env_value_falls_back_to_the_default(self):
        with mock.patch.dict(os.environ, {"LLMBRIDGE_LEDGER_MAX_RECOVERABLE": "many"}):
            ledger = GenerationLedger(str(Path(self._tmp.name) / "bad-recover"))
        self.assertEqual(ledger.max_recoverable, MAX_KEPT_RECOVERABLE)

    def test_explicit_prune_window_is_a_hard_cut(self):
        made = [self._interrupted_run(self.ledger, f"p{index}") for index in range(4)]
        removed = self.ledger.prune(keep=2)
        self.assertEqual(removed, [made[0].id, made[1].id])
        self.assertEqual(
            [record.id for record in self.ledger.interrupted()],
            [made[3].id, made[2].id],
        )

    def test_interrupted_lists_only_interrupted_records_newest_first(self):
        first = self.ledger.create({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "a"}]}, "m")
        self.ledger.interrupt(first, "quota")
        done = self.ledger.create({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "b"}]}, "m")
        self.ledger.complete(done, "ok", 1)
        last = self.ledger.create({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "c"}]}, "m")
        self.ledger.interrupt(last, "rate limited")

        interrupted = self.ledger.interrupted()
        self.assertEqual([record.id for record in interrupted], [last.id, first.id])


class LedgerLockTests(unittest.TestCase):
    def test_exclusive_lock_serializes_threads(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        lock_path = _lock_path_for(Path(self._tmp.name) / "gen_x.json")
        guard = threading.Lock()
        state = {"active": 0, "max_active": 0}
        workers = 5
        start = threading.Barrier(workers)

        def worker():
            start.wait()
            with _exclusive_lock(lock_path):
                with guard:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                time.sleep(0.02)
                with guard:
                    state["active"] -= 1

        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(state["max_active"], 1)

    def test_lock_file_lives_outside_the_ledger(self):
        import tempfile

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        record_path = Path(self._tmp.name) / "generations" / "gen_x.json"
        lock_path = _lock_path_for(record_path)
        self.assertTrue(str(lock_path).startswith(tempfile.gettempdir()))
        self.assertNotEqual(lock_path.parent, record_path.parent)


class LedgerRetentionConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patcher = mock.patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)
        os.environ.pop("LLMBRIDGE_LEDGER_MAX_RECORDS", None)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _make_records(self, count):
        for index in range(count):
            record = self.ledger.create(
                {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{index}"}]},
                "m",
            )
            self.ledger.complete(record, "ok", 1)

    def test_default_cap_is_max_tracked_interrupted(self):
        self.assertEqual(self.ledger.max_records, MAX_TRACKED_INTERRUPTED)

    def test_env_var_configures_the_cap(self):
        os.environ["LLMBRIDGE_LEDGER_MAX_RECORDS"] = "3"
        ledger = GenerationLedger(str(Path(self._tmp.name) / "capped"))
        self.assertEqual(ledger.max_records, 3)

        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            self._make_records_on(ledger, 5)
        self.assertEqual([record.prompt for record in ledger.list()], ["p4", "p3", "p2"])

    def _make_records_on(self, ledger, count):
        for index in range(count):
            record = ledger.create(
                {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{index}"}]},
                "m",
            )
            ledger.complete(record, "ok", 1)

    def test_invalid_env_values_fall_back_to_default(self):
        for raw in ["not-a-number", "0", "-5", ""]:
            with self.subTest(raw=raw):
                os.environ["LLMBRIDGE_LEDGER_MAX_RECORDS"] = raw
                ledger = GenerationLedger(str(Path(self._tmp.name) / f"invalid-{abs(hash(raw))}"))
                self.assertEqual(ledger.max_records, MAX_TRACKED_INTERRUPTED)

    def test_prune_without_keep_uses_configured_cap(self):
        os.environ["LLMBRIDGE_LEDGER_MAX_RECORDS"] = "2"
        ledger = GenerationLedger(str(Path(self._tmp.name) / "prune-config"))
        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            self._make_records_on(ledger, 4)
            ledger.prune()
        self.assertEqual([record.prompt for record in ledger.list()], ["p3", "p2"])


class LedgerCrossProcessTests(unittest.TestCase):
    WORKER = (
        "import json, sys, time\n"
        "from llmbridge.ledger import GenerationLedger\n"
        "root, tag = sys.argv[1], sys.argv[2]\n"
        "ledger = GenerationLedger(root)\n"
        "for index in range(20):\n"
        "    record = ledger.create(\n"
        "        {\"model\": \"m\", \"max_tokens\": 8,\n"
        "         \"messages\": [{\"role\": \"user\", \"content\": f\"{tag}-{index}\"}]},\n"
        "        \"m\",\n"
        "    )\n"
        "    ledger.complete(record, \"done\", 1)\n"
        "    time.sleep(0.002)\n"
    )

    def test_concurrent_processes_never_corrupt_records(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name) / "stress"
        workers = 3
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", self.WORKER, str(root), f"w{index}"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for index in range(workers)
        ]
        failures = []
        for proc in procs:
            _, stderr = proc.communicate(timeout=120)
            if proc.returncode != 0:
                failures.append(stderr.decode("utf-8", "replace"))
        self.assertEqual(failures, [], "worker processes crashed")

        # Concurrent creates can race a prune's list snapshot, so the final
        # count may slightly exceed the cap; corruption is the real risk.
        files = sorted(root.glob("gen_*.json"))
        self.assertLessEqual(len(files), MAX_TRACKED_INTERRUPTED + workers - 1)
        for path in files:
            record = GenerationLedger(str(root)).load(path.stem)
            self.assertIsNotNone(record, f"{path.name} failed to parse")
            self.assertEqual(record.id, path.stem)


class LedgerTreeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _family(self):
        root = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "seed"}]},
            "nemotron",
        )
        self.ledger.complete(root, "root answer", 2)
        child = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "seed"}]},
            "qwen-local",
            parent=root.id,
            fork_event_seq=1,
        )
        self.ledger.complete(child, "child answer", 3)
        grandchild = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "seed"}]},
            "m",
            parent=child.id,
            fork_event_seq=1,
        )
        return root, child, grandchild

    def test_children_lists_direct_forks_newest_first(self):
        root, child, grandchild = self._family()
        second = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "seed"}]},
            "m",
            parent=root.id,
            fork_event_seq=1,
        )
        self.assertEqual(
            [record.id for record in self.ledger.children(root.id)],
            [second.id, child.id],
        )
        self.assertEqual([record.id for record in self.ledger.children(child.id)], [grandchild.id])
        self.assertEqual(self.ledger.children(grandchild.id), [])

    def test_tree_nests_descendants_and_lists_ancestors_root_first(self):
        root, child, grandchild = self._family()
        tree = self.ledger.tree(child.id)
        self.assertIsNotNone(tree)
        self.assertEqual(tree["id"], child.id)
        self.assertEqual([node["id"] for node in tree["ancestors"]], [root.id])
        self.assertEqual(tree["node"]["record"]["id"], child.id)
        self.assertEqual(
            [node["record"]["id"] for node in tree["node"]["children"]],
            [grandchild.id],
        )

    def test_tree_of_root_has_no_ancestors(self):
        root, _, _ = self._family()
        tree = self.ledger.tree(root.id)
        self.assertEqual(tree["ancestors"], [])
        self.assertEqual(len(tree["node"]["children"]), 1)

    def test_tree_unknown_id_returns_none(self):
        self.assertIsNone(self.ledger.tree("gen_missing"))

    def test_tree_survives_parent_cycles(self):
        first = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "a"}]}, "m",
            parent="gen_cycle1", fork_event_seq=1,
        )
        second = self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "b"}]}, "m",
            parent=first.id, fork_event_seq=1,
        )
        # Forge a cycle: first's parent is rewritten to point at second.
        first.parent = second.id
        self.ledger._save(first)

        tree = self.ledger.tree(second.id)
        self.assertIsNotNone(tree)

    def test_tree_respects_max_depth(self):
        root, child, grandchild = self._family()
        tree = self.ledger.tree(root.id, max_depth=1)
        nested = tree["node"]["children"][0]
        self.assertEqual(nested["record"]["id"], child.id)
        self.assertEqual(nested["children"], [])


class LedgerQueueTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _record(self, prompt="p"):
        return self.ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]},
            "m",
        )

    def test_queue_parks_a_record_with_attempt_metadata(self):
        record = self._record()
        self.ledger.queue(record, "quota exhausted", retry_at=1000.0, attempt=2)

        self.assertEqual(record.status, "queued")
        self.assertEqual(record.events[-1].type, EVENT_QUEUED)
        self.assertEqual(record.events[-1].data["attempt"], 2)
        self.assertEqual(record.events[-1].data["retry_at"], 1000.0)

        reloaded = GenerationLedger(str(self.ledger.root)).load(record.id)
        self.assertEqual(reloaded.status, "queued")
        self.assertEqual(GenerationLedger.queue_attempt(reloaded), 2)
        self.assertEqual(GenerationLedger.queue_retry_at(reloaded), 1000.0)

    def test_queue_attempt_is_zero_before_first_queue_event(self):
        record = self._record()
        self.assertEqual(GenerationLedger.queue_attempt(record), 0)
        self.assertEqual(GenerationLedger.queue_retry_at(record), 0.0)

    def test_queued_lists_earliest_retry_first(self):
        late = self._record(prompt="late")
        early = self._record(prompt="early")
        self.ledger.queue(late, "quota", retry_at=200.0, attempt=1)
        self.ledger.queue(early, "quota", retry_at=100.0, attempt=1)
        self.assertEqual([record.prompt for record in self.ledger.queued()], ["early", "late"])

    def test_queued_ignores_other_statuses(self):
        record = self._record()
        self.ledger.queue(record, "quota", retry_at=1.0, attempt=1)
        self.ledger.complete(record, "done", 1)
        self.assertEqual(self.ledger.queued(), [])

    def test_failing_a_queued_record_cancels_it(self):
        record = self._record()
        self.ledger.queue(record, "quota", retry_at=1.0, attempt=1)
        self.ledger.fail(record, "cancelled from the queue")
        self.assertEqual(record.status, "failed")
        self.assertEqual(self.ledger.queued(), [])

    def test_heal_appends_event_without_changing_status(self):
        record = self._record()
        self.ledger.heal(
            record, from_model="nemotron", to_model="qwen-local", resumed_chars=42
        )
        self.assertEqual(record.status, "running")
        self.assertEqual(record.events[-1].type, EVENT_HEAL)
        self.assertEqual(record.events[-1].data["resumed_chars"], 42)

        reloaded = GenerationLedger(str(self.ledger.root)).load(record.id)
        self.assertEqual(reloaded.events[-1].type, EVENT_HEAL)

    def test_heal_seam_appends_audit_without_changing_status(self):
        record = self._record()
        self.ledger.heal_seam(
            record,
            from_model="nemotron",
            to_model="qwen-local",
            seam_at=120,
            audit={
                "continuation_chars": 64,
                "overlap_chars": 31,
                "overlap_ratio": 0.194,
                "duplicated": True,
                "restarted": False,
                "suspect": True,
            },
        )
        self.assertEqual(record.status, "running")
        event = record.events[-1]
        self.assertEqual(event.type, EVENT_HEAL_SEAM)
        self.assertEqual(event.data["seam_at"], 120)
        self.assertTrue(event.data["duplicated"])

        reloaded = GenerationLedger(str(self.ledger.root)).load(record.id)
        self.assertEqual(reloaded.events[-1].type, EVENT_HEAL_SEAM)
        self.assertTrue(GenerationLedger.heal_seam_suspect(reloaded))

    def test_seam_readers_report_the_newest_audit(self):
        record = self._record()
        self.assertIsNone(GenerationLedger.heal_seam_audit(record))
        self.assertFalse(GenerationLedger.heal_seam_suspect(record))

        clean = {"overlap_chars": 2, "duplicated": False, "restarted": False,
                 "suspect": False}
        self.ledger.heal_seam(
            record, from_model="a", to_model="b", seam_at=10, audit=clean
        )
        self.assertEqual(
            GenerationLedger.heal_seam_audit(record), {**clean, "seam_at": 10,
                                                       "from_model": "a", "to_model": "b"}
        )
        self.assertFalse(GenerationLedger.heal_seam_suspect(record))

        self.ledger.heal_seam(
            record,
            from_model="a",
            to_model="c",
            seam_at=30,
            audit={**clean, "suspect": True, "to_model": "ignored"},
        )
        newest = GenerationLedger.heal_seam_audit(record)
        self.assertEqual(newest["seam_at"], 30)
        self.assertTrue(GenerationLedger.heal_seam_suspect(record))


def _seam_audit(suspect, **overrides):
    audit = {
        "continuation_chars": 60,
        "overlap_chars": 31 if suspect else 3,
        "overlap_ratio": 0.4 if suspect else 0.05,
        "duplicated": bool(suspect),
        "restarted": False,
        "suspect": bool(suspect),
    }
    audit.update(overrides)
    return audit


class SuspectAuditRetentionTests(unittest.TestCase):
    """Forgetting a suspect seam has to be a decision, not a side effect."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        os.environ.pop("LLMBRIDGE_LEDGER_MAX_RECORDS", None)
        os.environ.pop("LLMBRIDGE_AUDIT_MAX_SUSPECT", None)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _plain(self, ledger, prompt):
        record = ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]},
            "m",
        )
        ledger.complete(record, "fine", 1)
        return record

    def _healed(self, ledger, prompt, suspect):
        record = self._plain(ledger, prompt)
        ledger.heal(
            record, from_model="primary", to_model="backup", resumed_chars=8
        )
        ledger.heal_seam(
            record,
            from_model="primary",
            to_model="backup",
            seam_at=8,
            audit=_seam_audit(suspect),
        )
        return record

    def test_suspect_seam_audit_survives_fifty_later_generations(self):
        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            suspect = self._healed(self.ledger, "bad join", suspect=True)
            clean = self._healed(self.ledger, "good join", suspect=False)
            for index in range(50):
                self._plain(self.ledger, f"p{index}")
        kept = self.ledger.load(suspect.id)
        self.assertIsNotNone(kept)
        self.assertTrue(GenerationLedger.heal_seam_suspect(kept))
        self.assertIsNone(self.ledger.load(clean.id))
        self.assertEqual(
            len(self.ledger.list(limit=None)), MAX_TRACKED_INTERRUPTED + 1
        )

    def test_exempted_suspects_are_bounded_and_freshest_first(self):
        os.environ["LLMBRIDGE_AUDIT_MAX_SUSPECT"] = "2"
        ledger = GenerationLedger(str(Path(self._tmp.name) / "bounded"))
        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            made = [self._healed(ledger, f"seam {i}", suspect=True) for i in range(5)]
            for index in range(20):
                self._plain(ledger, f"p{index}")
        retained = [
            record.id
            for record in ledger.list(limit=None)
            if GenerationLedger.heal_seam_suspect(record)
        ]
        self.assertEqual(retained, [made[4].id, made[3].id])
        self.assertEqual(len(ledger.list(limit=None)), MAX_TRACKED_INTERRUPTED + 2)

    def test_zero_budget_restores_the_plain_window(self):
        os.environ["LLMBRIDGE_AUDIT_MAX_SUSPECT"] = "0"
        ledger = GenerationLedger(str(Path(self._tmp.name) / "no-budget"))
        self.assertEqual(ledger.max_suspect_audits, 0)
        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            suspect = self._healed(ledger, "bad join", suspect=True)
            for index in range(20):
                self._plain(ledger, f"p{index}")
        self.assertIsNone(ledger.load(suspect.id))

    def test_invalid_budget_falls_back_to_the_default(self):
        for raw in ["not-a-number", "-1", ""]:
            with self.subTest(raw=raw):
                os.environ["LLMBRIDGE_AUDIT_MAX_SUSPECT"] = raw
                ledger = GenerationLedger(
                    str(Path(self._tmp.name) / f"bad-budget-{abs(hash(raw))}")
                )
                self.assertEqual(ledger.max_suspect_audits, MAX_KEPT_SUSPECT_AUDITS)

    def test_explicit_prune_window_is_a_hard_window(self):
        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            suspect = self._healed(self.ledger, "bad join", suspect=True)
            for index in range(3):
                self._plain(self.ledger, f"p{index}")
            removed = self.ledger.prune(keep=1)
        self.assertIn(suspect.id, removed)
        self.assertIsNone(self.ledger.load(suspect.id))


class AuditConsolidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _healed(self, ledger, prompt, suspect, seam_at=8):
        record = ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]},
            "primary",
        )
        ledger.heal(record, from_model="primary", to_model="backup", resumed_chars=seam_at)
        ledger.heal_seam(
            record,
            from_model="primary",
            to_model="backup",
            seam_at=seam_at,
            audit=_seam_audit(suspect),
        )
        ledger.complete(record, "stitched", 2)
        return record

    def _plain(self, ledger, prompt):
        record = ledger.create(
            {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]},
            "m",
        )
        ledger.complete(record, "fine", 1)
        return record

    def test_consolidate_writes_one_line_per_seam_audit(self):
        first = self._healed(self.ledger, "a", suspect=False, seam_at=12)
        second = self._healed(self.ledger, "b", suspect=True, seam_at=24)
        stats = self.ledger.consolidate()
        self.assertEqual(stats["audits_on_disk"], 2)
        self.assertEqual(stats["new_entries"], 2)
        self.assertEqual(stats["total_entries"], 2)
        self.assertEqual(stats["suspect_entries"], 1)
        self.assertEqual(stats["suspect_rate"], 0.5)
        entries = GenerationLedger.read_audit_digest(self.ledger.audit_digest_path)
        self.assertEqual(
            [(entry["generation_id"], entry["seam_at"]) for entry in entries],
            [(first.id, 12), (second.id, 24)],
        )
        self.assertEqual(entries[1]["to_model"], "backup")
        self.assertTrue(entries[1]["suspect"])

    def test_consolidate_is_idempotent(self):
        self._healed(self.ledger, "a", suspect=True)
        self.ledger.consolidate()
        stats = self.ledger.consolidate()
        self.assertEqual(stats["new_entries"], 0)
        self.assertEqual(stats["total_entries"], 1)
        self.assertEqual(
            len(self.ledger.audit_digest_path.read_text(encoding="utf-8").splitlines()),
            1,
        )

    def test_digest_keeps_the_rate_after_the_records_age_out(self):
        clock = itertools.count()
        with mock.patch(
            "llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))
        ):
            for index in range(20):
                self._healed(self.ledger, f"heal {index}", suspect=index < 5)
                # Consolidating while the records still exist is what keeps a
                # denominator; run late and the clean seams are already gone.
                self.ledger.consolidate()
            for index in range(40):
                self._plain(self.ledger, f"p{index}")
            stats = self.ledger.consolidate()
        # Only the five suspect audits are still on disk; the digest remembers all twenty.
        self.assertEqual(stats["audits_on_disk"], 5)
        self.assertEqual(stats["new_entries"], 0)
        self.assertEqual(stats["total_entries"], 20)
        self.assertEqual(stats["suspect_entries"], 5)
        self.assertEqual(stats["suspect_rate"], 0.25)

    def test_rate_is_unmeasured_without_any_audit(self):
        self._plain(self.ledger, "only")
        stats = self.ledger.consolidate()
        self.assertIsNone(stats["suspect_rate"])
        self.assertEqual(stats["total_entries"], 0)
        self.assertFalse(self.ledger.audit_digest_path.exists())

    def test_pruning_consolidates_a_clean_seam_before_it_is_deleted(self):
        # Never consolidated by hand: the clean seams must survive because the
        # prune that deletes their records copies them first. Otherwise the
        # suspect count keeps its numerator and loses its denominator. The
        # suspect exemption is switched off so every seam is actually pruned.
        with mock.patch.dict(os.environ, {"LLMBRIDGE_AUDIT_MAX_SUSPECT": "0"}):
            ledger = GenerationLedger(str(self.ledger.root))
            for index in range(20):
                self._healed(ledger, f"heal {index}", suspect=index < 5)
            for index in range(40):
                self._plain(ledger, f"p{index}")
        stats = self.ledger.consolidate()
        self.assertEqual(stats["audits_on_disk"], 0)
        self.assertEqual(stats["new_entries"], 0)
        self.assertEqual(stats["total_entries"], 20)
        self.assertEqual(stats["suspect_entries"], 5)

    def test_pruning_records_without_audits_writes_no_digest(self):
        for index in range(15):
            self._plain(self.ledger, f"p{index}")
        self.assertEqual(len(self.ledger.list(limit=None)), 10)
        self.assertFalse(self.ledger.audit_digest_path.exists())

    def test_reader_skips_a_partial_or_corrupt_line(self):
        self._healed(self.ledger, "a", suspect=True)
        self.ledger.consolidate()
        with open(self.ledger.audit_digest_path, "a", encoding="utf-8") as handle:
            handle.write('{"generation_id": "truncated\n')
        entries = GenerationLedger.read_audit_digest(self.ledger.audit_digest_path)
        self.assertEqual(len(entries), 1)


class DreamRequestTests(unittest.TestCase):
    ANSWER = "Hello world, this answer keeps going for a while."

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = GenerationLedger(str(Path(self._tmp.name) / "generations"))

    def _completed(self, model="primary", output=ANSWER):
        record = self.ledger.create(
            {
                "model": model,
                "max_tokens": 64,
                "system": "be brief",
                "messages": [
                    {"role": "user", "content": "say something"},
                    {"role": "assistant", "content": "earlier turn"},
                    {"role": "user", "content": "keep going"},
                ],
            },
            model,
        )
        self.ledger.complete(record, output, 8)
        return record

    def test_cut_output_becomes_the_assistant_turn_to_continue(self):
        record = self._completed()
        body = GenerationLedger.dream_request(record, at=len("Hello world,"))
        self.assertEqual(
            body["messages"],
            [
                {"role": "user", "content": "say something"},
                {"role": "assistant", "content": "earlier turn"},
                {"role": "user", "content": "keep going"},
                {"role": "assistant", "content": "Hello world,"},
            ],
        )
        self.assertEqual(body["model"], "primary")
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["system"], "be brief")
        self.assertFalse(body["stream"])

    def test_record_and_cut_are_not_mutated(self):
        record = self._completed()
        GenerationLedger.dream_request(record, at=5)
        self.assertEqual(record.output, DreamRequestTests.ANSWER)
        self.assertEqual(len(record.events), 2)

    def test_model_and_token_limit_can_be_overridden(self):
        record = self._completed()
        body = GenerationLedger.dream_request(record, at=10, model="backup", max_tokens=32)
        self.assertEqual(body["model"], "backup")
        self.assertEqual(body["max_tokens"], 32)

    def test_cuts_are_clamped_into_the_stored_output(self):
        record = self._completed()
        late = GenerationLedger.dream_request(record, at=len(self.ANSWER) + 50)
        self.assertEqual(
            late["messages"][-1]["content"], self.ANSWER
        )
        early = GenerationLedger.dream_request(record, at=-10)
        self.assertEqual(early["messages"][-1]["role"], "user")
        self.assertEqual(len(early["messages"]), 3)

    def test_needs_a_stored_output_to_cut(self):
        record = self.ledger.create(
            {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "p"}]},
            "m",
        )
        with self.assertRaises(ValueError):
            GenerationLedger.dream_request(record, at=4)


if __name__ == "__main__":
    unittest.main()
