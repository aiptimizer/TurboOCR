from __future__ import annotations

from typing import Any

from .._core.content import materialize, materialize_async
from .specs import RequestSpec


def _httpx_kwargs(
    spec: RequestSpec, *, request_id: str, timeout: float | None
) -> dict[str, Any]:
    return _build_kwargs(spec, request_id=request_id, timeout=timeout, async_body=False)


def _httpx_kwargs_async(
    spec: RequestSpec, *, request_id: str, timeout: float | None
) -> dict[str, Any]:
    return _build_kwargs(spec, request_id=request_id, timeout=timeout, async_body=True)


def _build_kwargs(
    spec: RequestSpec, *, request_id: str, timeout: float | None, async_body: bool
) -> dict[str, Any]:
    # Rebuilt fresh on every attempt so a body factory (e.g. a streaming file
    # reader for a Path input) is re-invoked on retry and replays cleanly.
    # httpx 0.28+ requires AsyncIterable[bytes] under AsyncClient; we adapt
    # sync iterables via materialize_async on that path.
    headers = dict(spec.headers)
    headers["X-Request-ID"] = request_id
    kwargs: dict[str, Any] = {"params": dict(spec.params), "headers": headers}
    if spec.content is not None:
        kwargs["content"] = (
            materialize_async(spec.content) if async_body else materialize(spec.content)
        )
    if spec.json_body is not None:
        kwargs["json"] = dict(spec.json_body)
    if spec.files is not None:
        field_name, file_tuple = spec.files
        kwargs["files"] = {field_name: file_tuple}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs
