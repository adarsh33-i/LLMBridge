import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from llmbridge import cli

_VALID_CONFIG = """
[models.primary]
base_url = "http://primary.test/v1"
model = "primary-model"
fallbacks = ["backup"]

[models.backup]
base_url = "http://backup.test/v1"
model = "backup-model"
"""


class CliTests(unittest.TestCase):
    def _write(self, directory: str, text: str) -> str:
        path = Path(directory) / "llmbridge.toml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_check_reports_success(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._write(directory, _VALID_CONFIG)
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["check", "--config", config])
        self.assertEqual(code, 0)
        self.assertIn("OK:", out.getvalue())

    def test_check_reports_configuration_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._write(directory, "[models.a]\nmodel = 'm'\n")
            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(["check", "--config", config])
        self.assertEqual(code, 1)
        self.assertIn("error:", err.getvalue())

    def test_models_lists_aliases_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._write(directory, _VALID_CONFIG)
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["models", "--config", config])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("primary", rendered)
        self.assertIn("fallbacks: backup", rendered)
        self.assertIn("http://backup.test/v1", rendered)

    def test_models_reports_configuration_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._write(directory, "not = = toml")
            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(["models", "--config", config])
        self.assertEqual(code, 1)
        self.assertIn("error:", err.getvalue())

    def test_init_writes_a_starter_config(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "llmbridge.toml"
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["init", "--path", str(target)])
            self.assertEqual(code, 0)
            self.assertTrue(target.exists())
            self.assertIn("base_url", target.read_text(encoding="utf-8"))
            self.assertIn("Wrote starter configuration", out.getvalue())

    def test_init_refuses_to_clobber_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "llmbridge.toml"
            target.write_text("keep me", encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(["init", "--path", str(target)])
            self.assertEqual(code, 1)
            self.assertIn("already exists", err.getvalue())
            self.assertEqual(target.read_text(encoding="utf-8"), "keep me")

    def test_init_force_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "llmbridge.toml"
            target.write_text("old", encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["init", "--path", str(target), "--force"])
            self.assertEqual(code, 0)
            self.assertIn("[models.", target.read_text(encoding="utf-8"))

    def test_serve_applies_options_before_starting(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("llmbridge.cli.server.main") as run:
                code = cli.main(
                    [
                        "serve",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9000",
                        "--config",
                        "/tmp/routes.toml",
                    ]
                )
            self.assertEqual(code, 0)
            run.assert_called_once()
            self.assertEqual(os.environ["LLMBRIDGE_HOST"], "0.0.0.0")
            self.assertEqual(os.environ["LLMBRIDGE_PORT"], "9000")
            self.assertEqual(os.environ["LLMBRIDGE_CONFIG"], "/tmp/routes.toml")

    def test_no_subcommand_serves(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("llmbridge.cli.server.main") as run:
                code = cli.main([])
            self.assertEqual(code, 0)
            run.assert_called_once()

    def test_generations_lists_recorded_generations(self):
        from llmbridge.ledger import GenerationLedger

        with tempfile.TemporaryDirectory() as directory:
            ledger = GenerationLedger(str(Path(directory) / "generations"))
            record = ledger.create(
                {"model": "m", "messages": [{"role": "user", "content": "Hello there"}]},
                "m",
            )
            ledger.complete(record, "Hi", 3)

            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["generations", "--ledger", str(ledger.root)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn(record.id, rendered)
        self.assertIn("completed", rendered)
        self.assertIn("Hello there", rendered)

    def test_generations_reports_when_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["generations", "--ledger", str(Path(directory) / "generations")])
        self.assertEqual(code, 0)
        self.assertIn("No generations", out.getvalue())

    def test_generations_json_output(self):
        from llmbridge.ledger import GenerationLedger

        with tempfile.TemporaryDirectory() as directory:
            ledger = GenerationLedger(str(Path(directory) / "generations"))
            ledger.create(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "m"
            )
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(
                    ["generations", "--ledger", str(ledger.root), "--json"]
                )
        self.assertEqual(code, 0)
        self.assertIn('"object": "generation"', out.getvalue())

    def test_status_summarises_ledger_and_recovery(self):
        from llmbridge.ledger import GenerationLedger

        with tempfile.TemporaryDirectory() as directory:
            ledger = GenerationLedger(str(Path(directory) / "generations"))
            done = ledger.create(
                {"model": "m", "messages": [{"role": "user", "content": "a"}]}, "m"
            )
            ledger.complete(done, "ok", 5)
            stopped = ledger.create(
                {"model": "m", "messages": [{"role": "user", "content": "b"}]}, "m"
            )
            ledger.interrupt(stopped, "quota", output="Half", output_tokens=2)

            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["status", "--ledger", str(ledger.root)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("Generations: 2", rendered)
        self.assertIn("completed: 1", rendered)
        self.assertIn("interrupted: 1", rendered)
        self.assertIn("Output tokens recorded: 7", rendered)
        self.assertIn(f"llmbridge fork {stopped.id}", rendered)

    def test_status_reports_empty_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["status", "--ledger", str(Path(directory) / "g")])
        self.assertEqual(code, 0)
        self.assertIn("No generations recorded yet", out.getvalue())

    def test_status_without_interrupted_work_omits_recovery_hint(self):
        from llmbridge.ledger import GenerationLedger

        with tempfile.TemporaryDirectory() as directory:
            ledger = GenerationLedger(str(Path(directory) / "generations"))
            record = ledger.create(
                {"model": "m", "messages": [{"role": "user", "content": "a"}]}, "m"
            )
            ledger.complete(record, "ok")  # no output token count
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["status", "--ledger", str(ledger.root)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("Output tokens recorded: 0", rendered)
        self.assertNotIn("Recoverable", rendered)

    def test_fork_posts_to_running_server(self):
        class _Response:
            def __init__(self):
                self.headers = {"X-LLMBridge-Generation": "gen_child"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b'{"id": "msg_1", "type": "message"}'

        out = io.StringIO()
        err = io.StringIO()
        with patch("llmbridge.cli.urlopen", return_value=_Response()) as call:
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(
                    [
                        "fork",
                        "gen_parent",
                        "--model",
                        "strong",
                        "--max-tokens",
                        "50",
                        "--base-url",
                        "http://bridge.test:9999",
                    ]
                )

        self.assertEqual(code, 0)
        request = call.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://bridge.test:9999/v1/generations/gen_parent/fork",
        )
        sent = json.loads(request.data)
        self.assertEqual(sent["model"], "strong")
        self.assertEqual(sent["max_tokens"], 50)
        self.assertIn("msg_1", out.getvalue())
        self.assertIn("gen_child", err.getvalue())

    def test_fork_uses_env_base_url_and_stream_without_generation_header(self):
        class _Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b"{}"

        err = io.StringIO()
        with patch.dict(os.environ, {"LLMBRIDGE_BASE_URL": "http://env-brand"}):
            with patch("llmbridge.cli.urlopen", return_value=_Response()) as call:
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    code = cli.main(["fork", "gen_x", "--stream"])
        self.assertEqual(code, 0)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "http://env-brand/v1/generations/gen_x/fork")
        self.assertEqual(json.loads(request.data), {"stream": True})
        self.assertNotIn("forked generation", err.getvalue())

    def test_fork_sends_api_key_when_configured(self):
        class _Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b"{}"

        with patch("llmbridge.cli.urlopen", return_value=_Response()) as call:
            with redirect_stdout(io.StringIO()):
                cli.main(["fork", "gen_x", "--api-key", "secret-key"])
        request = call.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")

    def test_fork_reports_http_error(self):
        error = HTTPError("http://b", 404, "Not Found", {}, io.BytesIO(b'{"type":"error"}'))
        err = io.StringIO()
        with patch("llmbridge.cli.urlopen", side_effect=error):
            with redirect_stderr(err):
                code = cli.main(["fork", "gen_missing", "--base-url", "http://b"])
        self.assertEqual(code, 1)
        self.assertIn("404", err.getvalue())

    def test_fork_reports_unreachable_server(self):
        err = io.StringIO()
        with patch("llmbridge.cli.urlopen", side_effect=URLError("refused")):
            with redirect_stderr(err):
                code = cli.main(["fork", "gen_x", "--base-url", "http://nope"])
        self.assertEqual(code, 1)
        self.assertIn("could not reach the bridge", err.getvalue())


class SetupCommandTests(unittest.TestCase):
    def test_writes_config_non_interactively(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llmbridge.toml"
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(
                    ["setup", "--path", str(path), "--preset", "ollama", "--no-key"]
                )
            self.assertEqual(code, 0)
            text = path.read_text(encoding="utf-8")
            self.assertIn("[models.local]", text)
            self.assertIn("http://127.0.0.1:11434/v1", text)
            self.assertNotIn("api_key_env", text)
            mode = path.stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)
            self.assertIn("Next steps:", out.getvalue())

    def test_includes_api_key_env_from_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llmbridge.toml"
            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    ["setup", "--path", str(path), "--preset", "openrouter"]
                )
            self.assertEqual(code, 0)
            self.assertIn('api_key_env = "OPENROUTER_API_KEY"', path.read_text())

    def test_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llmbridge.toml"
            path.write_text("existing", encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(["setup", "--path", str(path), "--preset", "ollama"])
            self.assertEqual(code, 1)
            self.assertIn("already exists", err.getvalue())
            with redirect_stdout(io.StringIO()):
                code = cli.main(
                    ["setup", "--path", str(path), "--preset", "ollama", "--force"]
                )
            self.assertEqual(code, 0)

    def test_rejects_invalid_base_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llmbridge.toml"
            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(
                    [
                        "setup",
                        "--path",
                        str(path),
                        "--alias",
                        "x",
                        "--base-url",
                        "ftp://nope",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("http(s)", err.getvalue())

    def test_interactive_defaults_and_custom_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llmbridge.toml"
            with patch("llmbridge.cli.sys.stdin") as stdin:
                stdin.isatty.return_value = True
                with patch(
                    "builtins.input", side_effect=["myalias", "", "", ""]
                ):
                    with redirect_stdout(io.StringIO()):
                        code = cli.main(
                            ["setup", "--path", str(path), "--preset", "ollama"]
                        )
            self.assertEqual(code, 0)
            self.assertIn("[models.myalias]", path.read_text())
            self.assertIn("http://127.0.0.1:11434/v1", path.read_text())


class DoctorCommandTests(unittest.TestCase):
    def test_reports_ok_for_valid_config(self):
        import socket

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "llmbridge.toml"
            config.write_text(_VALID_CONFIG, encoding="utf-8")
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                free_port = probe.getsockname()[1]
            env = {
                "LLMBRIDGE_CONFIG": str(config),
                "LLMBRIDGE_LEDGER_PATH": str(Path(directory) / "gen"),
            }
            out = io.StringIO()
            with patch.dict(os.environ, env, clear=True):
                with redirect_stdout(out):
                    code = cli.main(["doctor", "--port", str(free_port)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("OK   config", rendered)
        self.assertIn("All checks passed.", rendered)

    def test_fails_when_config_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"LLMBRIDGE_CONFIG": str(Path(directory) / "absent.toml")}
            out = io.StringIO()
            with patch.dict(os.environ, env, clear=True):
                with redirect_stdout(out):
                    code = cli.main(["doctor"])
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out.getvalue())
        self.assertIn("problem(s) found", out.getvalue())

    def test_warns_when_api_key_env_is_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "llmbridge.toml"
            config.write_text(
                "[models.x]\nbase_url = 'http://x.test/v1'\n"
                "api_key_env = 'LLMBRIDGE_DOCTOR_UNSET_KEY'\n",
                encoding="utf-8",
            )
            env = {
                "LLMBRIDGE_CONFIG": str(config),
                "LLMBRIDGE_LEDGER_PATH": str(Path(directory) / "gen"),
            }
            out = io.StringIO()
            with patch.dict(os.environ, env, clear=True):
                with redirect_stdout(out):
                    code = cli.main(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("WARN", out.getvalue())
        self.assertIn("$LLMBRIDGE_DOCTOR_UNSET_KEY", out.getvalue())


class ClaudeCommandTests(unittest.TestCase):
    def test_print_env_defaults(self):
        out = io.StringIO()
        with patch.dict(os.environ, {"LLMBRIDGE_API_KEY": "sk-abc"}, clear=True):
            with redirect_stdout(out):
                code = cli.main(["claude", "--print-env"])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("ANTHROPIC_BASE_URL=http://127.0.0.1:8787", rendered)
        self.assertIn("ANTHROPIC_AUTH_TOKEN=sk-abc", rendered)

    def test_builds_url_from_host_and_port(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(
                ["claude", "--host", "0.0.0.0", "--port", "9000", "--print-env"]
            )
        self.assertEqual(code, 0)
        self.assertIn("ANTHROPIC_BASE_URL=http://0.0.0.0:9000", out.getvalue())

    def test_forwards_args_after_double_dash(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["claude", "--print-env", "--", "--model", "local"])
        self.assertEqual(code, 0)
        self.assertIn("claude --model local", out.getvalue())

    def test_reports_missing_claude_cli(self):
        err = io.StringIO()
        with patch("llmbridge.cli.os.execvpe", side_effect=FileNotFoundError()):
            with redirect_stderr(err):
                code = cli.main(["claude", "--model", "local"])
        self.assertEqual(code, 1)
        self.assertIn("not found on PATH", err.getvalue())


def _replay_request(prompt="seed prompt", model="primary", **extra):
    body = {
        "model": model,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(extra)
    return body


class LedgerBackedCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = cli.GenerationLedger(str(Path(self._tmp.name) / "gens"))

    def _completed(self, prompt="seed prompt", output="shared answer", model="primary"):
        record = self.ledger.create(_replay_request(prompt=prompt), model)
        self.ledger.complete(record, output, 3)
        return record

    def _heal(self, record, seam_at, suspect):
        self.ledger.heal(
            record, from_model="primary", to_model="backup", resumed_chars=seam_at
        )
        self.ledger.heal_seam(
            record,
            from_model="primary",
            to_model="backup",
            seam_at=seam_at,
            audit={
                "continuation_chars": 30,
                "overlap_chars": 31 if suspect else 2,
                "overlap_ratio": 0.4 if suspect else 0.05,
                "duplicated": suspect,
                "restarted": False,
                "suspect": suspect,
            },
        )

    def _run(self, *argv):
        out = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(out), redirect_stderr(errors):
            code = cli.main([*argv, "--ledger", str(self.ledger.root)])
        return code, out.getvalue() + errors.getvalue()


class RecoverableStatusTests(LedgerBackedCommandTests):
    """Status must account for every piece of work the bridge still holds."""

    def _dead_run(self, prompt):
        record = self.ledger.create(_replay_request(prompt=prompt), "primary")
        self.ledger.interrupt(record, "quota", output="half", progress=5.0)
        return record

    def test_status_lists_the_other_recoverable_jobs(self):
        oldest = self._dead_run("oldest")
        middle = self._dead_run("middle")
        newest = self._dead_run("newest")
        code, rendered = self._run("status")
        self.assertEqual(code, 0)
        self.assertIn("Recoverable: 3", rendered)
        self.assertIn(f"llmbridge fork {newest.id}", rendered)
        self.assertIn(f"also waiting: {middle.id}, {oldest.id}", rendered)

    def test_status_hides_the_waiting_line_for_a_single_job(self):
        self._dead_run("only")
        code, rendered = self._run("status")
        self.assertEqual(code, 0)
        self.assertIn("Recoverable: 1", rendered)
        self.assertNotIn("also waiting", rendered)

    def test_status_counts_work_parked_on_quota(self):
        record = self.ledger.create(_replay_request(prompt="waiting"), "primary")
        self.ledger.queue(record, "over quota", 1_800_000_000.0, 2)
        code, rendered = self._run("status")
        self.assertEqual(code, 0)
        self.assertIn("Parked on quota: 1 (see GET /v1/queue)", rendered)


class TreeCommandTests(LedgerBackedCommandTests):
    def test_renders_ancestors_and_children(self):
        parent = self._completed(prompt="root", output="root answer")
        child = self.ledger.create(
            _replay_request(prompt="root"), "backup",
            parent=parent.id, fork_event_seq=1,
        )
        self.ledger.complete(child, "child answer", 4)

        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["tree", child.id, "--ledger", str(self.ledger.root)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn(parent.id, rendered)
        self.assertIn(child.id, rendered)
        self.assertIn("backup", rendered)

    def test_json_output_is_machine_readable(self):
        parent = self._completed()
        child = self.ledger.create(
            _replay_request(), "backup", parent=parent.id, fork_event_seq=1
        )
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(
                ["tree", child.id, "--ledger", str(self.ledger.root), "--json"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["id"], child.id)
        self.assertEqual(payload["ancestors"][0]["id"], parent.id)

    def test_unknown_id_fails(self):
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(["tree", "gen_missing", "--ledger", str(self.ledger.root)])
        self.assertEqual(code, 1)
        self.assertIn("not found", err.getvalue())


class DiffCommandTests(LedgerBackedCommandTests):
    def test_reports_models_and_similarity(self):
        first = self._completed(output="alpha beta gamma")
        second = self._completed(prompt="other", output="alpha delta gamma")

        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["diff", first.id, second.id, "--ledger", str(self.ledger.root)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("similarity", rendered)
        self.assertIn(first.id, rendered)
        self.assertIn(second.id, rendered)
        self.assertIn("--- " + first.id, rendered)
        self.assertIn("+++ " + second.id, rendered)

    def test_identical_outputs_skip_the_unified_diff(self):
        first = self._completed()
        second = self._completed(prompt="other")

        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["diff", first.id, second.id, "--ledger", str(self.ledger.root)])
        self.assertEqual(code, 0)
        rendered = out.getvalue()
        self.assertIn("similarity", rendered)
        self.assertNotIn("--- " + first.id, rendered)

    def test_missing_generation_fails(self):
        existing = self._completed()
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(
                ["diff", existing.id, "gen_missing", "--ledger", str(self.ledger.root)]
            )
        self.assertEqual(code, 1)
        self.assertIn("gen_missing", err.getvalue())


class ReplayCommandTests(LedgerBackedCommandTests):
    _CONFIG = """
[models.primary]
base_url = "http://primary.test/v1"
model = "primary-model"

[models.backup]
base_url = "http://backup.test/v1"
model = "backup-model"
"""

    def setUp(self):
        super().setUp()
        config = Path(self._tmp.name) / "llmbridge.toml"
        config.write_text(self._CONFIG, encoding="utf-8")
        self._env = patch.dict(
            os.environ, {"LLMBRIDGE_CONFIG": str(config)}, clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_passes_when_replayed_output_matches(self):
        record = self._completed(output="the answer")
        with patch.object(cli, "_replay_output", return_value=("the answer", None)):
            code, rendered = self._run("replay", record.id)
        self.assertEqual(code, 0)
        self.assertIn("PASS", rendered)
        self.assertIn("1/1 replay(s) matched", rendered)

    def test_fails_below_threshold(self):
        record = self._completed(output="the answer")
        with patch.object(cli, "_replay_output", return_value=("utterly different", None)):
            code, rendered = self._run("replay", record.id)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", rendered)

    def test_threshold_option_controls_the_verdict(self):
        record = self._completed(output="the answer")
        replayed = "the answer but longer now"
        with patch.object(cli, "_replay_output", return_value=(replayed, None)):
            code, _ = self._run("replay", record.id)
            self.assertEqual(code, 1)
            code, rendered = self._run(
                "replay", record.id, "--threshold", "0.2"
            )
        self.assertEqual(code, 0)
        self.assertIn("PASS", rendered)

    def test_upstream_errors_fail_without_raising(self):
        record = self._completed()
        with patch.object(cli, "_replay_output", return_value=(None, "upstream down")):
            code, rendered = self._run("replay", record.id)
        self.assertEqual(code, 1)
        self.assertIn("upstream down", rendered)

    def test_requires_a_target(self):
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(["replay", "--ledger", str(self.ledger.root)])
        self.assertEqual(code, 1)
        self.assertIn("generation id or --all", err.getvalue())

    def test_all_replays_only_completed_records(self):
        done = self._completed(output="done answer")
        stalled = self.ledger.create(_replay_request(prompt="stalled"), "primary")
        self.ledger.interrupt(stalled, "quota", output="half", progress=10.0)
        with patch.object(
            cli, "_replay_output", return_value=("done answer", None)
        ) as replay:
            code, rendered = self._run("replay", "--all")
        self.assertEqual(code, 0)
        self.assertEqual(replay.call_count, 1)
        body = replay.call_args[0][2]
        self.assertEqual(body["messages"][0]["content"], "seed prompt")
        self.assertIn("1/1", rendered)

    def test_unknown_model_alias_fails_cleanly(self):
        record = self.ledger.create(_replay_request(model="ghost"), "ghost")
        self.ledger.complete(record, "text", 1)
        code, rendered = self._run("replay", record.id)
        self.assertEqual(code, 1)
        self.assertIn("ghost", rendered)

    def test_untranslatable_upstream_reply_is_reported_not_raised(self):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": "x"},
                                "finish_reason": "made_up_reason",
                            }
                        ]
                    }
                ).encode("utf-8")

        routes = cli._load_routes(None)
        with patch("llmbridge.cli.urlopen", return_value=_Response()):
            text, error = cli._replay_output(routes, "primary", _replay_request())

        self.assertIsNone(text)
        self.assertIn("Unsupported upstream completion reason", error)


class HealSeamStatusTests(LedgerBackedCommandTests):
    def _status(self):
        return self._run("status")

    def test_status_counts_heals_and_lists_suspect_seams(self):
        clean = self._completed(prompt="clean prompt")
        self._heal(clean, 20, suspect=False)
        suspect = self._completed(prompt="bad prompt")
        self._heal(suspect, 40, suspect=True)

        code, rendered = self._status()
        self.assertEqual(code, 0)
        self.assertIn("Healed streams audited: 2 (1 suspect)", rendered)
        self.assertIn(suspect.id, rendered)
        self.assertIn("primary -> backup at char 40", rendered)
        self.assertIn("31 chars repeated (duplicated)", rendered)
        self.assertIn("llmbridge replay", rendered)
        self.assertNotIn(f"{clean.id}:", rendered)

    def test_status_stays_quiet_without_heals(self):
        self._completed()
        code, rendered = self._status()
        self.assertEqual(code, 0)
        self.assertNotIn("Healed streams", rendered)

    def test_status_tells_an_unconsolidated_audit_to_consolidate(self):
        record = self._completed()
        self._heal(record, 20, suspect=False)
        code, rendered = self._status()
        self.assertEqual(code, 0)
        self.assertIn("not yet consolidated: run `llmbridge audit`", rendered)


class AuditCommandTests(LedgerBackedCommandTests):
    def test_audit_writes_a_digest_and_reports_the_full_history_rate(self):
        clean = self._completed(prompt="clean prompt")
        self._heal(clean, 20, suspect=False)
        suspect = self._completed(prompt="bad prompt")
        self._heal(suspect, 40, suspect=True)

        code, rendered = self._run("audit")
        self.assertEqual(code, 0)
        self.assertIn("Seam audits on disk: 2 (2 newly consolidated)", rendered)
        self.assertIn("Full history: 2 audit(s), 1 suspect (50%)", rendered)
        self.assertIn(f"{suspect.id}: primary -> backup at char 40 (duplicated)", rendered)
        self.assertNotIn(f"{clean.id}:", rendered)
        lines = self.ledger.audit_digest_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["generation_id"], clean.id)

        code, rendered = self._run("status")
        self.assertEqual(code, 0)
        self.assertIn("Consolidated audits: 2 (1 suspect, 50%)", rendered)

    def test_audit_repeats_without_duplicating_a_seam(self):
        record = self._completed()
        self._heal(record, 20, suspect=True)
        self._run("audit")
        code, rendered = self._run("audit")
        self.assertEqual(code, 0)
        self.assertIn("(0 newly consolidated)", rendered)
        self.assertIn("Full history: 1 audit(s), 1 suspect (100%)", rendered)
        self.assertEqual(
            len(self.ledger.audit_digest_path.read_text(encoding="utf-8").splitlines()),
            1,
        )

    def test_audit_json_reports_the_digest_path(self):
        record = self._completed()
        self._heal(record, 12, suspect=False)
        code, rendered = self._run("audit", "--json")
        self.assertEqual(code, 0)
        stats = json.loads(rendered)
        self.assertEqual(stats["total_entries"], 1)
        self.assertEqual(stats["suspect_rate"], 0.0)
        self.assertEqual(Path(stats["digest_path"]).name, "heal_audit.jsonl")

    def test_audit_says_nothing_was_healed(self):
        self._completed()
        code, rendered = self._run("audit")
        self.assertEqual(code, 0)
        self.assertIn("No heal seam audits recorded yet.", rendered)
        self.assertFalse(self.ledger.audit_digest_path.exists())


class DreamCommandTests(ReplayCommandTests):
    OUTPUT = "Once upon a time there was a bridge, and it healed quietly."

    def test_dream_continues_from_the_cut_point(self):
        record = self._completed(output=self.OUTPUT)
        continuation = "Sure, here is the answer"
        cut = len("Once upon a time ")
        with patch.object(
            cli, "_replay_output", return_value=(continuation, None)
        ) as send:
            code, rendered = self._run("dream", record.id, "--at", str(cut))
        self.assertEqual(code, 0)
        self.assertEqual(send.call_args.args[1], "primary")
        body = send.call_args.args[2]
        self.assertEqual(
            body["messages"][-1], {"role": "assistant", "content": self.OUTPUT[:cut]}
        )
        self.assertIn(
            f"Dreaming {record.id} on primary: cut at char {cut} of {len(self.OUTPUT)}",
            rendered,
        )
        self.assertIn(
            f"continuation {len(continuation)} chars vs recorded remainder", rendered
        )
        self.assertIn("a heal at this seam would have been flagged by the audit", rendered)

    def test_dream_defaults_to_the_midpoint(self):
        record = self._completed(output=self.OUTPUT)
        mid = len(self.OUTPUT) // 2
        with patch.object(
            cli, "_replay_output", return_value=(self.OUTPUT[mid:], None)
        ):
            code, rendered = self._run("dream", record.id)
        self.assertEqual(code, 0)
        self.assertIn(f"cut at char {mid} of", rendered)
        self.assertIn("similarity 1.00", rendered)
        self.assertIn("clean join, and the continuation tracks the recorded answer", rendered)

    def test_dream_json_carries_the_seam_and_similarity(self):
        record = self._completed(output=self.OUTPUT)
        cut = len("Once upon a time there was a ")
        with patch.object(
            cli, "_replay_output", return_value=("there was a bridge", None)
        ):
            code, rendered = self._run("dream", record.id, "--at", str(cut), "--json")
        self.assertEqual(code, 0)
        result = json.loads(rendered)
        self.assertEqual(result["generation_id"], record.id)
        self.assertEqual(result["at"], cut)
        self.assertEqual(result["truth_chars"], len(self.OUTPUT) - cut)
        self.assertEqual(result["continuation_chars"], len("there was a bridge"))
        self.assertTrue(result["seam"]["duplicated"])
        self.assertTrue(result["seam"]["suspect"])

    def test_dream_needs_a_record_with_output(self):
        running = self.ledger.create(_replay_request(prompt="unfinished"), "primary")
        code, rendered = self._run("dream", running.id)
        self.assertEqual(code, 1)
        self.assertIn("no stored output", rendered)

    def test_dream_reports_an_upstream_failure(self):
        record = self._completed(output=self.OUTPUT)
        with patch.object(
            cli, "_replay_output", return_value=(None, "upstream returned HTTP 502")
        ):
            code, rendered = self._run("dream", record.id)
        self.assertEqual(code, 1)
        self.assertIn("dream replay failed: upstream returned HTTP 502", rendered)

    def test_dream_reports_an_empty_continuation(self):
        record = self._completed(output=self.OUTPUT)
        with patch.object(cli, "_replay_output", return_value=("", None)):
            code, rendered = self._run("dream", record.id)
        self.assertEqual(code, 1)
        self.assertIn("returned no text", rendered)


class FailoverCommandTests(LedgerBackedCommandTests):
    _CONFIG = """
[models.primary]
base_url = "http://primary.test/v1"
model = "primary-model"
fallbacks = ["steady", "erratic"]

[models.steady]
base_url = "http://steady.test/v1"
model = "steady-model"

[models.erratic]
base_url = "http://erratic.test/v1"
model = "erratic-model"

[models.solo]
base_url = "http://solo.test/v1"
model = "solo-model"
"""

    OUTPUT = "Once upon a time there was a bridge that healed quietly in the dark."
    SEAM_AT = 28

    def setUp(self):
        super().setUp()
        config = Path(self._tmp.name) / "llmbridge.toml"
        config.write_text(self._CONFIG, encoding="utf-8")
        self._env = patch.dict(
            os.environ, {"LLMBRIDGE_CONFIG": str(config)}, clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _suspect(self, prompt="healed prompt", output=None, model="primary"):
        record = self._completed(
            prompt=prompt, output=output or self.OUTPUT, model=model
        )
        self._heal(record, self.SEAM_AT, suspect=True)
        return record

    def _answers(self, **by_model):
        """Fake _replay_output that answers per alias; absent aliases fail."""

        def replay(routes, alias, body):
            if alias in by_model:
                return by_model[alias], None
            return None, f"no answer for {alias}"

        return replay

    @property
    def ANSWERS(self):
        return {
            "steady": self.OUTPUT[self.SEAM_AT :],
            "erratic": "time there was a bridge that healed loudly",
            "solo": self.OUTPUT[self.SEAM_AT :],
        }

    def test_ranks_the_faithful_fallback_first(self):
        self._suspect()
        with patch.object(
            cli,
            "_replay_output",
            side_effect=self._answers(
                steady=self.ANSWERS["steady"], erratic=self.ANSWERS["erratic"]
            ),
        ):
            code, rendered = self._run("failover")
        self.assertEqual(code, 0)
        self.assertLess(rendered.index("steady"), rendered.index("erratic"))
        self.assertIn("Best fallback on these seams: steady", rendered)
        self.assertIn("mean similarity 1.00, no flagged joins", rendered)
        self.assertIn("2 real upstream call(s)", rendered)

    def test_defaults_the_cut_to_the_recorded_seam(self):
        record = self._suspect()
        seen = []

        def replay(routes, alias, body):
            seen.append(body)
            return self.OUTPUT[self.SEAM_AT :], None

        with patch.object(cli, "_replay_output", side_effect=replay):
            code, _ = self._run("failover", record.id, "--models", "steady")
        self.assertEqual(code, 0)
        self.assertEqual(len(seen), 1)
        self.assertEqual(
            seen[0]["messages"][-1]["content"], self.OUTPUT[: self.SEAM_AT]
        )

    def test_samples_only_suspect_heals_by_default(self):
        clean = self._completed(prompt="clean prompt")
        self._heal(clean, self.SEAM_AT, suspect=False)
        older = self._suspect(prompt="older suspect")
        newest = self._suspect(prompt="newest suspect")
        with patch.object(
            cli, "_replay_output", side_effect=self._answers(solo=self.ANSWERS["solo"])
        ) as replay:
            code, rendered = self._run("failover", "--limit", "1", "--models", "solo")
        self.assertEqual(code, 0)
        self.assertEqual(replay.call_count, 1)
        self.assertIn("Sampling 1 suspect heal(s)", rendered)
        self.assertIn(newest.id, rendered)
        self.assertNotIn(older.id, rendered)

    def test_dry_run_sends_nothing(self):
        self._suspect()
        with patch.object(cli, "_replay_output") as replay:
            code, rendered = self._run("failover", "--dry-run")
        self.assertEqual(code, 0)
        replay.assert_not_called()
        self.assertIn("2 real upstream call(s)", rendered)
        self.assertIn("dry run: nothing was sent", rendered)
        self.assertIn("steady, erratic", rendered)

    def test_a_failed_model_still_ranks_the_other(self):
        self._suspect()
        with patch.object(
            cli,
            "_replay_output",
            side_effect=self._answers(steady=self.ANSWERS["steady"]),
        ):
            code, rendered = self._run("failover")
        self.assertEqual(code, 0)
        self.assertIn("no answer for erratic", rendered)
        self.assertIn("Best fallback on these seams: steady", rendered)

    def test_models_override_the_configured_fallbacks(self):
        self._suspect()
        with patch.object(
            cli, "_replay_output", side_effect=self._answers(solo=self.ANSWERS["solo"])
        ) as replay:
            code, rendered = self._run("failover", "--models", "solo")
        self.assertEqual(code, 0)
        self.assertEqual(replay.call_count, 1)
        self.assertIn("Best fallback on these seams: solo", rendered)

    def test_a_repeated_model_is_scored_once(self):
        self._suspect()
        with patch.object(
            cli, "_replay_output", side_effect=self._answers(solo=self.ANSWERS["solo"])
        ) as replay:
            code, rendered = self._run("failover", "--models", "solo, solo")
        self.assertEqual(code, 0)
        self.assertEqual(replay.call_count, 1)
        self.assertIn("1 real upstream call(s)", rendered)

    def test_a_model_without_fallbacks_asks_for_explicit_candidates(self):
        self._suspect(model="solo")
        code, rendered = self._run("failover")
        self.assertEqual(code, 1)
        self.assertIn("--models a,b", rendered)

    def test_json_carries_the_ranking_and_call_count(self):
        self._suspect()
        with patch.object(
            cli,
            "_replay_output",
            side_effect=self._answers(
                steady=self.ANSWERS["steady"], erratic=self.ANSWERS["erratic"]
            ),
        ):
            code, rendered = self._run("failover", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(rendered)
        self.assertEqual(payload["calls"], 2)
        self.assertEqual(payload["ranking"][0]["model"], "steady")
        self.assertEqual(payload["ranking"][0]["faithful"], 1)
        self.assertEqual(payload["ranking"][0]["flagged"], 0)

    def test_unknown_id_fails_cleanly(self):
        code, rendered = self._run("failover", "gen_deadbeef0000")
        self.assertEqual(code, 1)
        self.assertIn("gen_deadbeef0000 not found", rendered)


if __name__ == "__main__":
    unittest.main()
