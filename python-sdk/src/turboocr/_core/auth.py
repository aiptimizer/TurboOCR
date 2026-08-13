from __future__ import annotations

from collections.abc import Mapping

from .env import USER_AGENT, AuthScheme


def resolve_auth_pair(scheme: AuthScheme, api_key: str) -> tuple[str, str]:
    """Decide the (header-name, value) pair for an auth scheme.

    Returns HTTP-canonical casing (`Authorization`, `X-API-Key`). The gRPC
    transport lowercases the name on its side — wire conventions differ but
    the scheme decision is the same.
    """
    if callable(scheme):
        return scheme(api_key)
    match scheme:
        case "bearer":
            return ("Authorization", f"Bearer {api_key}")
        case "x-api-key":
            return ("X-API-Key", api_key)


def build_headers(
    *,
    api_key: str | None,
    auth_scheme: AuthScheme,
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        name, value = resolve_auth_pair(auth_scheme, api_key)
        headers[name] = value
    if extra:
        headers.update(extra)
    return headers
