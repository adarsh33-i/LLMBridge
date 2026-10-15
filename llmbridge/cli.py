"""Command-line interface for LLMBridge.

``llmbridge`` with no subcommand starts the server (the historical behaviour).
``check``, ``models`` and ``init`` make configuring and inspecting routes
possible without booting the HTTP listener.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import server
from .routing import RoutingError, load_model_routes

_SAMPLE_CONFIG = """# LLMBridge routing configuration.
# Each [models.<alias>] table maps a client-facing model name to an
# OpenAI-compatible upstream.

[models.gpt-oss]
base_url = "http://127.0.0.1:11434/v1"
model = "gpt-oss:20b"
# fallbacks = ["gpt-oss-backup"]
# supports_images = true
# supports_tools = true
# cache_control = "strip"   # or "passthrough"

[models.gpt-oss-backup]
base_url = "https://api.example.com/v1"
model = "gpt-oss-120b"
api_key_env = "EXAMPLE_API_KEY"
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmbridge",
        description="A local Anthropic-to-OpenAI-compatible LLM gateway.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the bridge server (default)")
    serve.add_argument("--host", help="Address to bind (default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="Port to listen on (default 8787)")
    serve.add_argument("--config", help="Path to llmbridge.toml")

    check = subparsers.add_parser("check", help="Validate the model routing config")
    check.add_argument("--config", help="Path to llmbridge.toml")

    models = subparsers.add_parser("models", help="List configured model aliases")
    models.add_argument("--config", help="Path to llmbridge.toml")

    init = subparsers.add_parser("init", help="Write a starter llmbridge.toml")
    init.add_argument("--path", default="llmbridge.toml", help="Where to write the file")
    init.add_argument("--force", action="store_true", help="Overwrite an existing file")

    return parser


def _load_routes(config: str | None):
    return load_model_routes(config)


def _check(args: argparse.Namespace) -> int:
    try:
        routes = _load_routes(args.config)
    except RoutingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(routes)} model route(s) configured")
    return 0


def _models(args: argparse.Namespace) -> int:
    try:
        routes = _load_routes(args.config)
    except RoutingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for alias in sorted(routes):
        route = routes[alias]
        line = f"{alias}\t{route.model or '(client model)'}\t{route.base_url}"
        if route.fallbacks:
            line += f"\tfallbacks: {', '.join(route.fallbacks)}"
        print(line)
    return 0


def _init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(
            f"error: {path} already exists (use --force to overwrite)",
            file=sys.stderr,
        )
        return 1
    path.write_text(_SAMPLE_CONFIG, encoding="utf-8")
    print(f"Wrote starter configuration to {path}")
    return 0


def _apply_serve_options(args: argparse.Namespace) -> None:
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    config = getattr(args, "config", None)
    if host:
        os.environ["LLMBRIDGE_HOST"] = host
    if port is not None:
        os.environ["LLMBRIDGE_PORT"] = str(port)
    if config:
        os.environ["LLMBRIDGE_CONFIG"] = config


def _serve(args: argparse.Namespace) -> int:
    _apply_serve_options(args)
    server.main()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    handlers: dict[str, Any] = {
        "serve": _serve,
        "check": _check,
        "models": _models,
        "init": _init,
    }
    return handlers[command](args)


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
