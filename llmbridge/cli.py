"""Command-line interface for LLMBridge.

``llmbridge`` with no subcommand starts the server (the historical behaviour).
``setup``, ``doctor``, ``check``, ``models`` and ``init`` make configuring and
inspecting routes possible without booting the HTTP listener, and ``claude``
launches Claude Code pointed at the bridge.
"""

import argparse
import difflib
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

    tree = subparsers.add_parser(
        "tree", help="Show a generation's fork lineage (ancestors and children)"
    )
    tree.add_argument("id", help="Generation id, e.g. gen_1a2b3c4d5e6f")
    tree.add_argument("--ledger", help="Path to the generation ledger directory")
    tree.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )

    diff = subparsers.add_parser(
        "diff", help="Compare two recorded generations side by side"
    )
    diff.add_argument("a", help="First generation id")
    diff.add_argument("b", help="Second generation id")
    diff.add_argument("--ledger", help="Path to the generation ledger directory")
    diff.add_argument(
        "--context", type=int, default=2, help="Diff context lines (default 2)"
    )

    replay = subparsers.add_parser(
        "replay", help="Re-run recorded requests and compare against saved outputs"
    )
    replay.add_argument("id", nargs="?", help="Generation id to replay")
    replay.add_argument(
        "--all", action="store_true", help="Replay every completed generation"
    )
    replay.add_argument(
        "--limit", type=int, default=10, help="With --all: how many to replay (default 10)"
    )
    replay.add_argument(
        "--model", help="Route alias to replay on (defaults to each record's model)"
    )
    replay.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity ratio that counts as a pass (default 0.8)",
    )
    replay.add_argument("--config", help="Path to llmbridge.toml")
    replay.add_argument("--ledger", help="Path to the generation ledger directory")

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


def _describe(record) -> str:
    preview = (record.prompt or "")[:48].replace("\n", " ")
    return f"{record.id} ({record.status}, {record.model}) {preview!r}"


def _tree(args: argparse.Namespace) -> int:
    ledger = GenerationLedger(getattr(args, "ledger", None))
    tree = ledger.tree(args.id)
    if tree is None:
        print(f"error: generation {args.id} not found", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(tree, indent=2))
        return 0
    ancestors = tree.get("ancestors", [])
    if ancestors:
        print(" → ".join(record["id"] for record in ancestors))
        print("↓")
    node = tree.get("node") or {}

    def render(current: dict, prefix: str = "") -> None:
        record = current.get("record", {})
        marker = "└── " if prefix else ""
        print(f"{prefix}{marker}{_describe_from_summary(record)}")
        children = current.get("children", [])
        for index, child in enumerate(children):
            last = index == len(children) - 1
            render(child, prefix + ("    " if not prefix or last else "│   "))

    render(node)
    return 0


def _describe_from_summary(summary: dict) -> str:
    preview = (summary.get("prompt_preview") or "")[:48].replace("\n", " ")
    return (
        f"{summary.get('id')} ({summary.get('status')}, {summary.get('model')})"
        f" {preview!r}"
    )


def _load_generation(ledger: GenerationLedger, generation_id: str):
    try:
        return ledger.load(generation_id)
    except ValueError:
        return None


def _diff(args: argparse.Namespace) -> int:
    ledger = GenerationLedger(getattr(args, "ledger", None))
    first = _load_generation(ledger, args.a)
    second = _load_generation(ledger, args.b)
    if first is None or second is None:
        missing = args.a if first is None else args.b
        print(f"error: generation {missing} not found", file=sys.stderr)
        return 1
    print(f"{'':<12}{'A':<24}{'B':<24}")
    print(f"{'model':<12}{first.model:<24}{second.model:<24}")
    print(f"{'status':<12}{first.status:<24}{second.status:<24}")
    first_tokens = "-" if first.output_tokens is None else str(first.output_tokens)
    second_tokens = "-" if second.output_tokens is None else str(second.output_tokens)
    print(f"{'tokens':<12}{first_tokens:<24}{second_tokens:<24}")
    print(f"{'chars':<12}{len(first.output):<24}{len(second.output):<24}")
    ratio = difflib.SequenceMatcher(None, first.output, second.output).ratio()
    print(f"{'similarity':<12}{ratio:.2f}")
    if first.output != second.output:
        print()
        for line in difflib.unified_diff(
            first.output.splitlines(),
            second.output.splitlines(),
            fromfile=first.id,
            tofile=second.id,
            lineterm="",
            n=getattr(args, "context", 2),
        ):
            print(line)
    return 0


def _replay_output(routes: dict, alias: str, body: dict) -> tuple[str | None, str | None]:
    """Send one replay request to a route; return (text, error)."""
    route = routes.get(alias)
    if route is None:
        return None, f"no configured model alias {alias!r}"
    try:
        from .protocol import anthropic_to_openai, openai_to_anthropic

        upstream_model = route.model or alias
        payload = anthropic_to_openai(
            body,
            upstream_model,
            reasoning_effort=route.reasoning_effort,
            supports_images=route.supports_images,
            supports_tools=route.supports_tools,
            cache_control=route.cache_control,
        )
        headers = {"Content-Type": "application/json"}
        api_key = route.api_key()
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        request = Request(
            route.completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            upstream_reply = json.loads(response.read().decode("utf-8", "replace"))
    except RoutingError as error:
        return None, str(error)
    except HTTPError as error:
        error.close()
        return None, f"upstream returned HTTP {error.code}"
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        return None, f"upstream request failed: {error}"
    anthropic_reply = openai_to_anthropic(upstream_reply, alias)
    text = "".join(
        block.get("text", "")
        for block in anthropic_reply.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return text, None


def _replay(args: argparse.Namespace) -> int:
    ledger = GenerationLedger(getattr(args, "ledger", None))
    if getattr(args, "all", False):
        targets = [
            record
            for record in ledger.list(limit=None)
            if record.status == "completed"
        ][: max(args.limit, 0)]
    elif args.id:
        record = _load_generation(ledger, args.id)
        if record is None:
            print(f"error: generation {args.id} not found", file=sys.stderr)
            return 1
        targets = [record]
    else:
        print("error: give a generation id or --all", file=sys.stderr)
        return 1
    if not targets:
        print("No completed generations to replay.")
        return 0
    try:
        routes = _load_routes(getattr(args, "config", None))
    except RoutingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    failures = 0
    for record in targets:
        alias = getattr(args, "model", None) or record.model
        body = dict(record.request)
        body.setdefault("model", alias)
        if record.params.get("max_tokens"):
            body.setdefault("max_tokens", record.params["max_tokens"])
        text, error = _replay_output(routes, alias, body)
        if error is not None:
            failures += 1
            print(f"FAIL {record.id}: {error}")
            continue
        ratio = difflib.SequenceMatcher(None, record.output, text or "").ratio()
        passed = ratio >= args.threshold
        if not passed:
            failures += 1
        label = "PASS" if passed else "FAIL"
        print(
            f"{label} {record.id}: similarity {ratio:.2f} "
            f"(threshold {args.threshold:.2f}, saved {len(record.output)} chars, "
            f"replayed {len(text or '')} chars)"
        )
    print(f"{len(targets) - failures}/{len(targets)} replay(s) matched")
    return 0 if failures == 0 else 1


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
        "tree": _tree,
        "diff": _diff,
        "replay": _replay,
        "setup": _setup,
        "doctor": _doctor,
        "claude": _claude,
    }
    return handlers[command](args)


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
