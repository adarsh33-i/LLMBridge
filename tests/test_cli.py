import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
