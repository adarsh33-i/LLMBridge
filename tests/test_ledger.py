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
    EVENT_INTERRUPT,
    EVENT_QUEUED,
    EVENT_START,
    MAX_TRACKED_INTERRUPTED,
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

    def test_interrupted_records_among_pruned_are_gone(self):
        clock = itertools.count()
        with mock.patch("llmbridge.ledger.time.time", side_effect=lambda: float(next(clock))):
            for index in range(MAX_TRACKED_INTERRUPTED + 1):
                record = self.ledger.create(
                    {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": f"p{index}"}]},
                    "m",
                )
                self.ledger.interrupt(record, "quota", output="half", progress=10.0)
        interrupted = self.ledger.interrupted()
        self.assertEqual(len(interrupted), MAX_TRACKED_INTERRUPTED)
        self.assertEqual(interrupted[0].prompt, f"p{MAX_TRACKED_INTERRUPTED}")

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


if __name__ == "__main__":
    unittest.main()
