from __future__ import annotations

import base64
from typing import Any, Final

import httpx

from ..errors import APIConnectionError, NetworkError, ProtocolError, Timeout, raise_for_error


def classify_httpx_error(exc: httpx.HTTPError) -> APIConnectionError:
    msg = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, httpx.TimeoutException):
        return Timeout(msg)
    if isinstance(exc, httpx.RemoteProtocolError):
        return ProtocolError(msg)
    return NetworkError(msg)


_BODY_SNIPPET_MAX: Final[int] = 512


def extract_error(payload: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Pull (code, message) out of the server error envelope.

    Current servers nest it — `{"error": {"code": ..., "message": ...}}` —
    while pre-v3 servers used flat `{"error_code": ..., "error"/"message": ...}`.
    Handle both.
    """
    if not payload:
        return None, None
    err = payload.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        message = err.get("message")
        return (
            code if isinstance(code, str) else None,
            message if isinstance(message, str) else None,
        )
    code = payload.get("error_code")
    message = err if isinstance(err, str) else payload.get("message")
    return (
        code if isinstance(code, str) else None,
        message if isinstance(message, str) else None,
    )


def parse_text_response(response: httpx.Response) -> str:
    """Return a text (e.g. `text/markdown`) success body, raising the mapped
    error for JSON error envelopes."""
    if response.is_error:
        payload: dict[str, Any] | None = None
        try:
            decoded = response.json()
            payload = decoded if isinstance(decoded, dict) else None
        except ValueError:
            pass
        code, message = extract_error(payload)
        raise_for_error(
            status_code=response.status_code,
            code=code,
            message=message or response.text[:_BODY_SNIPPET_MAX] or f"HTTP {response.status_code}",
            payload=payload,
        )
    return response.text


def parse_response(response: httpx.Response) -> dict[str, Any]:
    raw_body = response.text
    try:
        decoded = response.json()
    except ValueError as exc:
        if response.is_error:
            raise_for_error(
                status_code=response.status_code,
                code=None,
                message=f"{raw_body[:_BODY_SNIPPET_MAX]} (HTTP {response.status_code})",
                payload=None,
            )
        raise ProtocolError(
            f"server returned non-JSON success body "
            f"({raw_body[:_BODY_SNIPPET_MAX]!r}) at HTTP {response.status_code}",
            status_code=response.status_code,
        ) from exc

    payload: dict[str, Any] | None = decoded if isinstance(decoded, dict) else None

    if response.is_error:
        code, message = extract_error(payload)
        raise_for_error(
            status_code=response.status_code,
            code=code,
            message=message or raw_body[:_BODY_SNIPPET_MAX] or f"HTTP {response.status_code}",
            payload=payload,
        )

    if payload is None:
        raise ProtocolError(
            f"server returned non-object JSON ({type(decoded).__name__}): "
            f"{raw_body[:_BODY_SNIPPET_MAX]!r}",
            status_code=response.status_code,
        )
    return payload


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
