import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 only
    # Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except ImportError:
        raise ImportError(
            "Cannot parse TOML config: install `tomli` or use Python 3.11+"
        )


class RoutingError(ValueError):
    pass


@dataclass(frozen=True)
class ModelRoute:
    base_url: str
    model: str | None = None
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    supports_images: bool | None = None
    supports_tools: bool | None = None
    tokenizer_encoding: str | None = None
    fallbacks: tuple[str, ...] = ()
    cache_control: str = "strip"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RoutingError(
                f"Environment variable {self.api_key_env} is required for this model. "
                f"Set it with `export {self.api_key_env}=...` or run `llmbridge setup`."
            )
        return value

    @property
    def preserves_cache_control(self) -> bool:
        return self.cache_control == "passthrough"


def _parse_route(alias: str, value: Any) -> ModelRoute:
    if not isinstance(value, dict):
        raise RoutingError(f"Route {alias!r} must be a TOML table")
    base_url = value.get("base_url")
    if not isinstance(base_url, str):
        raise RoutingError(f"Route {alias!r} needs a base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RoutingError(f"Route {alias!r} base_url must be an HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise RoutingError(f"Route {alias!r} base_url cannot contain a query or fragment")

    model = value.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise RoutingError(f"Route {alias!r} model must be a non-empty string")
    api_key_env = value.get("api_key_env")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env
    ):
        raise RoutingError(f"Route {alias!r} api_key_env must be a non-empty string")
    reasoning_effort = value.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in (
        "low",
        "medium",
        "high",
    ):
        raise RoutingError(
            f"Route {alias!r} reasoning_effort must be low, medium, or high"
        )
    capabilities: dict[str, bool | None] = {}
    for capability in ("supports_images", "supports_tools"):
        configured = value.get(capability)
        if configured is not None and not isinstance(configured, bool):
            raise RoutingError(f"Route {alias!r} {capability} must be true or false")
        capabilities[capability] = configured
    tokenizer_encoding = value.get("tokenizer_encoding")
    if tokenizer_encoding is not None and tokenizer_encoding not in (
        "cl100k_base",
        "o200k_base",
        "p50k_base",
        "r50k_base",
    ):
        raise RoutingError(
            f"Route {alias!r} tokenizer_encoding must be a supported tiktoken encoding"
        )
    fallbacks = _parse_fallbacks(alias, value.get("fallbacks"))
    cache_control = value.get("cache_control", "strip")
    if cache_control not in ("strip", "passthrough"):
        raise RoutingError(
            f"Route {alias!r} cache_control must be 'strip' or 'passthrough'"
        )
    return ModelRoute(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key_env=api_key_env,
        reasoning_effort=reasoning_effort,
        tokenizer_encoding=tokenizer_encoding,
        fallbacks=fallbacks,
        cache_control=cache_control,
        **capabilities,
    )


def _parse_fallbacks(alias: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise RoutingError(
            f"Route {alias!r} fallbacks must be a non-empty array of model aliases"
        )
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RoutingError(
                f"Route {alias!r} fallbacks must contain non-empty model aliases"
            )
        if item == alias:
            raise RoutingError(f"Route {alias!r} cannot fall back to itself")
        if item in names:
            raise RoutingError(f"Route {alias!r} lists duplicate fallback {item!r}")
        names.append(item)
    return tuple(names)


def _validate_fallback_targets(routes: dict[str, ModelRoute]) -> None:
    """Reject fallbacks that point at aliases the config never defines."""
    for alias, route in routes.items():
        for target in route.fallbacks:
            if target not in routes:
                raise RoutingError(
                    f"Route {alias!r} falls back to unknown model {target!r}; "
                    f"configured model aliases: {', '.join(sorted(routes))}"
                )


def user_config_path() -> Path:
    """The per-user config location, honouring ``XDG_CONFIG_HOME``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "llmbridge" / "llmbridge.toml"


def config_search_paths() -> list[Path]:
    """Ordered candidate config locations.

    An explicit path (argument or ``LLMBRIDGE_CONFIG``) is authoritative on its
    own. Otherwise the working directory wins over the per-user config, so a
    project-local file still takes precedence once the user has run
    ``llmbridge setup``.
    """
    env_path = os.environ.get("LLMBRIDGE_CONFIG")
    if env_path:
        return [Path(env_path)]
    return [Path("llmbridge.toml"), user_config_path()]


def discover_config_path() -> Path | None:
    """Return the first existing config file, or ``None``."""
    return next((path for path in config_search_paths() if path.is_file()), None)


def load_model_routes(config_path: str | None = None) -> dict[str, ModelRoute]:
    candidates = [Path(config_path)] if config_path else config_search_paths()
    found = next((path for path in candidates if path.is_file()), None)
    if found is not None:
        path = found
        try:
            with path.open("rb") as config_file:
                config = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RoutingError(f"Could not read model routes from {path}: {error}") from error
        models = config.get("models")
        if not isinstance(models, dict) or not models:
            raise RoutingError(f"{path} must define at least one [models.<alias>] table")
        routes: dict[str, ModelRoute] = {}
        for alias, value in models.items():
            if not isinstance(alias, str) or not alias.strip():
                raise RoutingError("Model route aliases must be non-empty strings")
            routes[alias] = _parse_route(alias, value)
        _validate_fallback_targets(routes)
        return routes

    base_url = os.environ.get("LLMBRIDGE_UPSTREAM_BASE_URL")
    if not base_url:
        searched = ", ".join(str(path) for path in candidates)
        raise RoutingError(
            f"No model routing config found (looked in: {searched}). "
            "Configure [models.<alias>] in llmbridge.toml or set "
            "LLMBRIDGE_UPSTREAM_BASE_URL. Run `llmbridge setup` for guided "
            "configuration, or `llmbridge init` to write a starter file."
        )
    upstream_model = os.environ.get("LLMBRIDGE_UPSTREAM_MODEL")
    api_key_env = (
        "LLMBRIDGE_UPSTREAM_API_KEY"
        if os.environ.get("LLMBRIDGE_UPSTREAM_API_KEY")
        else None
    )
    route = _parse_route(
        "*", {"base_url": base_url, "model": upstream_model, "api_key_env": api_key_env}
    )
    return {"*": route}


def resolve_model_route(
    client_model: str, routes: dict[str, ModelRoute]
) -> tuple[ModelRoute, str]:
    # Explicit alias takes priority; "*" is the catch-all fallback
    if client_model in routes:
        route = routes[client_model]
    elif "*" in routes:
        route = routes["*"]
    else:
        route = None
    if route is None:
        available = ", ".join(sorted(routes))
        raise RoutingError(
            f"Unknown model {client_model!r}. Configured model aliases: {available}"
        )
    return route, route.model or client_model
