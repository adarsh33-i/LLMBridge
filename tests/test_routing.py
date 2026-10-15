import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmbridge.routing import RoutingError, load_model_routes, resolve_model_route


class RoutingTests(unittest.TestCase):
    def test_loads_multiple_aliases_from_toml(self):
        config = """
[models.nemotron-local]
base_url = "http://127.0.0.1:11434/v1"
model = "nemotron"

[models.hosted-free]
base_url = "https://provider.example/v1"
model = "provider/free-model"
api_key_env = "HOSTED_KEY"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            routes = load_model_routes(str(config_path))

        route, upstream_model = resolve_model_route("nemotron-local", routes)
        self.assertEqual(route.completions_url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(upstream_model, "nemotron")
        self.assertEqual(resolve_model_route("hosted-free", routes)[1], "provider/free-model")

    def test_requires_secret_env_var_only_when_route_is_used(self):
        config = """
[models.hosted-free]
base_url = "https://provider.example/v1"
api_key_env = "HOSTED_KEY"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            routes = load_model_routes(str(config_path))

        route, _ = resolve_model_route("hosted-free", routes)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RoutingError, "HOSTED_KEY"):
                route.api_key()

    def test_loads_configured_reasoning_effort(self):
        config = """
[models.reasoning-model]
base_url = "https://provider.example/v1"
reasoning_effort = "high"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            routes = load_model_routes(str(config_path))

        self.assertEqual(routes["reasoning-model"].reasoning_effort, "high")

    def test_rejects_unknown_reasoning_effort(self):
        config = """
[models.reasoning-model]
base_url = "https://provider.example/v1"
reasoning_effort = "maximum"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(RoutingError, "reasoning_effort"):
                load_model_routes(str(config_path))

    def test_loads_explicit_route_capabilities(self):
        config = """
[models.text-model]
base_url = "https://provider.example/v1"
supports_images = false
supports_tools = true
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            route = load_model_routes(str(config_path))["text-model"]

        self.assertFalse(route.supports_images)
        self.assertTrue(route.supports_tools)

    def test_rejects_non_boolean_route_capability(self):
        config = """
[models.text-model]
base_url = "https://provider.example/v1"
supports_images = "no"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(RoutingError, "supports_images"):
                load_model_routes(str(config_path))

    def test_loads_tokenizer_encoding(self):
        config = """
[models.tokenized]
base_url = "https://provider.example/v1"
tokenizer_encoding = "o200k_base"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            route = load_model_routes(str(config_path))["tokenized"]

        self.assertEqual(route.tokenizer_encoding, "o200k_base")

    def test_rejects_unknown_tokenizer_encoding(self):
        config = """
[models.tokenized]
base_url = "https://provider.example/v1"
tokenizer_encoding = "unknown"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(RoutingError, "tokenizer_encoding"):
                load_model_routes(str(config_path))

    def test_unknown_alias_is_rejected(self):
        config = """
[models.nemotron-local]
base_url = "http://127.0.0.1:11434/v1"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "llmbridge.toml"
            config_path.write_text(config, encoding="utf-8")
            routes = load_model_routes(str(config_path))

        with self.assertRaisesRegex(RoutingError, "Unknown model"):
            resolve_model_route("not-configured", routes)


class RoutingLegacyEnvTests(unittest.TestCase):
    def test_legacy_environment_variables_build_catch_all_route(self):
        env = {
            "LLMBRIDGE_CONFIG": "/nonexistent/llmbridge.toml",
            "LLMBRIDGE_UPSTREAM_BASE_URL": "http://127.0.0.1:11434/v1",
            "LLMBRIDGE_UPSTREAM_MODEL": "qwen",
            "LLMBRIDGE_UPSTREAM_API_KEY": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            routes = load_model_routes()
            route = routes["*"]
            self.assertEqual(route.api_key(), "secret")

        self.assertEqual(route.completions_url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(route.model, "qwen")
        self.assertEqual(resolve_model_route("anything", routes)[1], "qwen")

    def test_legacy_env_requires_base_url(self):
        with patch.dict(os.environ, {"LLMBRIDGE_CONFIG": "/nonexistent/x.toml"}, clear=True):
            with self.assertRaisesRegex(RoutingError, "LLMBRIDGE_UPSTREAM_BASE_URL"):
                load_model_routes()


class RoutingConfigErrorTests(unittest.TestCase):
    def _config(self, text):
        directory = tempfile.mkdtemp()
        path = Path(directory) / "llmbridge.toml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_invalid_toml_is_rejected(self):
        with self.assertRaisesRegex(RoutingError, "Could not read model routes"):
            load_model_routes(self._config("this is = = not toml"))

    def test_config_without_models_table_is_rejected(self):
        with self.assertRaisesRegex(RoutingError, "at least one"):
            load_model_routes(self._config("[server]\nport = 1\n"))

    def test_route_must_be_a_table(self):
        with self.assertRaisesRegex(RoutingError, "must be a TOML table"):
            load_model_routes(self._config("[models]\nbroken = 3\n"))

    def test_route_rejects_invalid_configurations(self):
        cases = {
            "needs a base_url": "[models.x]\nmodel = 'm'\n",
            "must be an HTTP": "[models.x]\nbase_url = 'ftp://host/v1'\n",
            "query or fragment": "[models.x]\nbase_url = 'http://host/v1?a=1'\n",
            "non-empty string": "[models.x]\nbase_url = 'http://host/v1'\nmodel = ''\n",
            "api_key_env must be a non-empty": "[models.x]\nbase_url = 'http://host/v1'\napi_key_env = ''\n",
        }
        for message, text in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(RoutingError, message):
                    load_model_routes(self._config(text))


if __name__ == "__main__":
    unittest.main()
