"""Command-line interface for LLMBridge.

``llmbridge`` with no subcommand starts the server (the historical behaviour).
``setup``, ``doctor``, ``check``, ``models`` and ``init`` make configuring and
inspecting routes possible without booting the HTTP listener, and ``claude``
launches Claude Code pointed at the bridge.
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import server
from .ledger import GenerationLedger
from .routing import (
    RoutingError,
    config_search_paths,
    discover_config_path,
    load_model_routes,
    user_config_path,
)

_DEFAULT_BASE_URL = "http://127.0.0.1:8787"

_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "gpt-oss:20b",
        "alias": "local",
        "api_key_env": None,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "alias": "openai",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemma-3-27b-it:free",
        "alias": "free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "alias": "nemotron",
        "api_key_env": "NVIDIA_API_KEY",
    },
}

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

    generations = subparsers.add_parser(
        "generations", help="List durable generations recorded by the server"
    )
    generations.add_argument("--ledger", help="Path to the generation ledger directory")
    generations.add_argument(
        "--limit", type=int, default=20, help="Maximum generations to show (default 20)"
    )
    generations.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )

    status = subparsers.add_parser(
        "status", help="Summarise recorded generations and recoverable work"
    )
    status.add_argument("--ledger", help="Path to the generation ledger directory")

    fork = subparsers.add_parser(
        "fork", help="Continue a recorded generation (requires a running server)"
    )
    fork.add_argument("id", help="Generation id, e.g. gen_1a2b3c4d5e6f")
    fork.add_argument("--model", help="Alias to continue on (defaults to the source model)")
    fork.add_argument("--max-tokens", type=int, help="Token limit for the continuation")
    fork.add_argument("--stream", action="store_true", help="Stream the forked response")
    fork.add_argument("--base-url", help="Bridge base URL (default http://127.0.0.1:8787)")
    fork.add_argument("--api-key", help="Bridge API key (or LLMBRIDGE_API_KEY)")

    setup = subparsers.add_parser(
        "setup", help="Write a model routing config interactively"
    )
    setup.add_argument("--path", help="Where to write the config")
    setup.add_argument(
        "--preset",
        choices=sorted(_PROVIDER_PRESETS),
        help="Provider preset to start from",
    )
    setup.add_argument("--alias", help="Model alias clients will request")
    setup.add_argument("--base-url", help="OpenAI-compatible base URL")
    setup.add_argument("--model", help="Upstream model name")
    setup.add_argument("--api-key-env", help="Environment variable holding the API key")
    setup.add_argument(
        "--no-key", action="store_true", help="Configure a route that needs no API key"
    )
    setup.add_argument("--force", action="store_true", help="Overwrite an existing config")

    doctor = subparsers.add_parser(
        "doctor", help="Check the environment and configuration for problems"
    )
    doctor.add_argument("--config", help="Path to llmbridge.toml")
    doctor.add_argument("--port", type=int, help="Port to check for availability")

    claude = subparsers.add_parser(
        "claude", help="Launch the Claude CLI pointed at this bridge"
    )
    claude.add_argument("--base-url", help="Bridge base URL (default http://127.0.0.1:8787)")
    claude.add_argument("--host", help="Bridge host to point Claude at")
    claude.add_argument("--port", type=int, help="Bridge port to point Claude at")
    claude.add_argument(
        "--client-api-key", help="Client API key for the bridge (or LLMBRIDGE_API_KEY)"
    )
    claude.add_argument(
        "--print-env",
        action="store_true",
        help="Print the environment instead of launching Claude",
    )

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


def _generations(args: argparse.Namespace) -> int:
    ledger = GenerationLedger(getattr(args, "ledger", None))
    records = ledger.list(limit=getattr(args, "limit", 20))
    if getattr(args, "json", False):
        import json

        print(json.dumps([record.summary() for record in records], indent=2))
        return 0
    if not records:
        print("No generations recorded yet.")
        return 0
    print(f"{'ID':<20}{'STATUS':<13}{'MODEL':<18}{'PARENT':<20}PROMPT")
    for record in records:
        parent = record.parent or "-"
        preview = record.prompt[:60].replace("\n", " ")
        print(f"{record.id:<20}{record.status:<13}{record.model:<18}{parent:<20}{preview}")
    return 0


def _status(args: argparse.Namespace) -> int:
    ledger = GenerationLedger(getattr(args, "ledger", None))
    records = ledger.list(limit=100000)
    print(f"Ledger: {ledger.root}")
    print(f"Generations: {len(records)}")
    if not records:
        print("No generations recorded yet — make a request to /v1/messages.")
        return 0
    counts: dict[str, int] = {}
    tokens = 0
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
        if isinstance(record.output_tokens, int):
            tokens += record.output_tokens
    for status_name in sorted(counts):
        print(f"  {status_name}: {counts[status_name]}")
    print(f"Output tokens recorded: {tokens}")
    latest = records[0]
    print(f"Latest: {latest.id} ({latest.status}, {latest.model})")
    recoverable = [record for record in records if record.status == "interrupted"]
    if recoverable:
        print(f"Recoverable: {len(recoverable)}")
        print(f"  continue one with: llmbridge fork {recoverable[0].id} --model <alias>")
    return 0


def _fork(args: argparse.Namespace) -> int:
    base_url = (
        args.base_url or os.environ.get("LLMBRIDGE_BASE_URL") or _DEFAULT_BASE_URL
    ).rstrip("/")
    payload: dict[str, Any] = {}
    if args.model:
        payload["model"] = args.model
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.stream:
        payload["stream"] = True
    headers = {"Content-Type": "application/json"}
    api_key = args.api_key or os.environ.get("LLMBRIDGE_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(
        f"{base_url}/v1/generations/{args.id}/fork",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request) as response:
            generation_id = response.headers.get("X-LLMBridge-Generation")
            body = response.read().decode("utf-8", "replace")
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        print(f"error: fork failed (HTTP {error.code}): {detail}", file=sys.stderr)
        error.close()
        return 1
    except URLError as error:
        print(f"error: could not reach the bridge at {base_url}: {error.reason}", file=sys.stderr)
        print("hint: start it with `llmbridge serve`", file=sys.stderr)
        return 1
    print(body)
    if generation_id:
        print(f"# forked generation: {generation_id}", file=sys.stderr)
    return 0


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{label}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or (default or "")


def _render_config(
    alias: str, base_url: str, model: str | None, api_key_env: str | None
) -> str:
    lines = [
        "# LLMBridge routing configuration.",
        "# Generated by `llmbridge setup`.",
        "",
        f"[models.{alias}]",
        f'base_url = "{base_url}"',
    ]
    if model:
        lines.append(f'model = "{model}"')
    if api_key_env:
        lines.append(f'api_key_env = "{api_key_env}"')
    lines.append("")
    return "\n".join(lines)


def _setup(args: argparse.Namespace) -> int:
    preset_name = args.preset
    interactive = sys.stdin.isatty()
    if preset_name is None and interactive:
        names = ", ".join(sorted(_PROVIDER_PRESETS))
        preset_name = _prompt(f"Provider preset ({names})", "ollama")
    preset = _PROVIDER_PRESETS.get(preset_name or "", {})

    def choose(value: str | None, key: str, label: str, default: str = "") -> str:
        if value is not None:
            return value
        fallback = preset.get(key, default)
        if interactive:
            return _prompt(label, fallback) or fallback
        return fallback

    alias = choose(args.alias, "alias", "Model alias clients will request", "local")
    base_url = choose(args.base_url, "base_url", "Upstream base URL")
    model = choose(args.model, "model", "Upstream model name", preset.get("model", ""))

    if args.no_key:
        api_key_env = None
    elif args.api_key_env is not None:
        api_key_env = args.api_key_env
    else:
        api_key_env = preset.get("api_key_env")
        if api_key_env is None and interactive:
            api_key_env = (
                _prompt("API key environment variable (blank for none)", "") or None
            )

    if not alias.strip():
        print("error: a model alias is required", file=sys.stderr)
        return 1
    if not base_url.startswith(("http://", "https://")):
        print("error: base_url must be an http(s) URL", file=sys.stderr)
        return 1

    if args.path:
        destination = Path(args.path)
    elif os.environ.get("LLMBRIDGE_CONFIG"):
        destination = Path(os.environ["LLMBRIDGE_CONFIG"])
    elif Path("llmbridge.toml").is_file():
        destination = Path("llmbridge.toml")
    else:
        destination = user_config_path()

    if destination.exists() and not args.force:
        print(
            f"error: {destination} already exists (use --force to overwrite)",
            file=sys.stderr,
        )
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.chmod(0o700)
    except OSError:
        pass
    destination.write_text(
        _render_config(alias, base_url, model, api_key_env), encoding="utf-8"
    )
    try:
        destination.chmod(0o600)
    except OSError:
        pass

    print(f"Wrote {destination}")
    print()
    print("Next steps:")
    print("  1. llmbridge serve")
    print("  2. export ANTHROPIC_BASE_URL=http://127.0.0.1:8787")
    if api_key_env:
        print(f"     (make sure ${api_key_env} is set)")
    return 0


def _probe_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".llmbridge-write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _doctor(args: argparse.Namespace) -> int:
    problems = 0
    print("LLMBridge doctor")

    if sys.version_info >= (3, 11):
        print(f"  OK   Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"  FAIL Python {sys.version_info.major}.{sys.version_info.minor} (need >= 3.11)")
        problems += 1

    config_path = args.config or os.environ.get("LLMBRIDGE_CONFIG")
    if not config_path:
        discovered = discover_config_path()
        config_path = str(discovered) if discovered else None

    routes = None
    if config_path is None:
        print("  FAIL no llmbridge.toml found")
        print("       searched: " + ", ".join(str(p) for p in config_search_paths()))
        print("       fix: run `llmbridge setup`")
        problems += 1
    else:
        try:
            routes = load_model_routes(config_path)
            print(f"  OK   config {config_path} ({len(routes)} route(s))")
        except RoutingError as error:
            print(f"  FAIL config {config_path}: {error}")
            problems += 1

    if routes:
        for alias, route in sorted(routes.items()):
            if route.api_key_env:
                if os.environ.get(route.api_key_env):
                    print(f"  OK   {alias}: ${route.api_key_env} is set")
                else:
                    print(
                        f"  WARN {alias}: ${route.api_key_env} is not set "
                        f"(export it before serving)"
                    )

    ledger_path = os.environ.get("LLMBRIDGE_LEDGER_PATH")
    ledger_dir = Path(ledger_path) if ledger_path else Path(".llmbridge") / "generations"
    if _probe_writable(ledger_dir):
        print(f"  OK   generation ledger writable at {ledger_dir}")
    else:
        print(f"  FAIL generation ledger not writable at {ledger_dir}")
        problems += 1

    host = os.environ.get("LLMBRIDGE_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("LLMBRIDGE_PORT", "8787"))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        print(f"  OK   port {port} is available")
    except OSError:
        print(f"  WARN port {port} is in use (a bridge may already be running)")

    print()
    print("All checks passed." if problems == 0 else f"{problems} problem(s) found.")
    return 1 if problems else 0


def _claude(args: argparse.Namespace) -> int:
    if args.base_url:
        base_url = args.base_url
    elif args.host or args.port:
        host = args.host or "127.0.0.1"
        port = args.port or 8787
        base_url = f"http://{host}:{port}"
    else:
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or _DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")

    api_key = (
        args.client_api_key
        or os.environ.get("LLMBRIDGE_API_KEY")
        or "llmbridge-local"
    )
    forwarded = list(args.claude_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.print_env:
        print(f"ANTHROPIC_BASE_URL={base_url}")
        print(f"ANTHROPIC_AUTH_TOKEN={api_key}")
        if forwarded:
            print("claude " + " ".join(forwarded))
        return 0

    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = api_key
    try:
        os.execvpe("claude", ["claude", *forwarded], env)
    except FileNotFoundError:
        print(
            "error: the `claude` CLI was not found on PATH. Install Claude Code, "
            "or run `llmbridge claude --print-env` to see the values.",
            file=sys.stderr,
        )
        return 1
    return 0  # pragma: no cover - execvpe replaces the process


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
    args, unknown = parser.parse_known_args(argv)
    command = args.command or "serve"
    if command == "claude":
        args.claude_args = list(unknown)
    elif unknown:
        parser.error("unrecognized arguments: " + " ".join(unknown))
    handlers: dict[str, Any] = {
        "serve": _serve,
        "check": _check,
        "models": _models,
        "init": _init,
        "generations": _generations,
        "status": _status,
        "fork": _fork,
        "setup": _setup,
        "doctor": _doctor,
        "claude": _claude,
    }
    return handlers[command](args)


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
