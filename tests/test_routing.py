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


if __name__ == "__main__":
    unittest.main()
