import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class RoutingError(ValueError):
    pass


@dataclass(frozen=True)
class ModelRoute:
    base_url: str
    model: str | None = None
    api_key_env: str | None = None

    @property
    def completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RoutingError(
                f"Environment variable {self.api_key_env} is required for this model"
            )
        return value


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
    return ModelRoute(base_url=base_url.rstrip("/"), model=model, api_key_env=api_key_env)


def load_model_routes(config_path: str | None = None) -> dict[str, ModelRoute]:
    path = Path(config_path or os.environ.get("LLMBRIDGE_CONFIG", "llmbridge.toml"))
    if path.is_file():
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
        return routes

    base_url = os.environ.get("LLMBRIDGE_UPSTREAM_BASE_URL")
    if not base_url:
        raise RoutingError(
            "Configure [models.<alias>] in llmbridge.toml or set "
            "LLMBRIDGE_UPSTREAM_BASE_URL"
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
    route = routes.get(client_model) or routes.get("*")
    if route is None:
        available = ", ".join(sorted(routes))
        raise RoutingError(
            f"Unknown model {client_model!r}. Configured model aliases: {available}"
        )
    return route, route.model or client_model
