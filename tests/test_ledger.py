import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llmbridge.ledger import (
    EVENT_COMPLETE,
    EVENT_FAIL,
    EVENT_FORK,
    EVENT_INTERRUPT,
    EVENT_START,
    GenerationLedger,
    GenerationRecord,
    _fingerprint,
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


if __name__ == "__main__":
    unittest.main()
