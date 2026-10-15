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


if __name__ == "__main__":
    unittest.main()
