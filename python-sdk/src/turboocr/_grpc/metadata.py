from __future__ import annotations

from collections.abc import Mapping

from .._core.auth import resolve_auth_pair
from .._core.env import USER_AGENT, AuthScheme

type Metadata = list[tuple[str, str]]


def build_metadata(
    *,
    api_key: str | None,
    auth_scheme: AuthScheme,
    request_id: str,
    extra: Mapping[str, str] | None = None,
) -> Metadata:
    # gRPC requires all metadata keys to be lowercase ASCII; mixed-case keys
    # raise at send time. We normalize on entry so callers can pass headers
    # in any case (matching the HTTP transport's header conventions).
    md: Metadata = [
        ("user-agent", USER_AGENT),
        ("x-request-id", request_id),
    ]
    if api_key:
        name, value = resolve_auth_pair(auth_scheme, api_key)
        md.append((name.lower(), value))
    if extra:
        for k, v in extra.items():
            md.append((k.lower(), v))
    return md
