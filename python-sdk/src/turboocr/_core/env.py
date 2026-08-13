from __future__ import annotations

import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution
from typing import Final, Literal

DEFAULT_BASE_URL: Final[str] = "http://localhost:8000"
DEFAULT_TIMEOUT: Final[float] = 30.0


def _package_version() -> str:
    try:
        return distribution("turboocr").version
    except PackageNotFoundError:
        return "dev"


USER_AGENT: Final[str] = f"turboocr/{_package_version()}"

type AuthSchemeName = Literal["bearer", "x-api-key"]
type AuthSchemeFn = Callable[[str], tuple[str, str]]
type AuthScheme = AuthSchemeName | AuthSchemeFn


def resolve_api_key(api_key: str | None) -> str | None:
    if api_key is not None:
        return api_key
    return os.environ.get("TURBO_OCR_API_KEY")


def resolve_base_url(base_url: str | None) -> str:
    if base_url is not None:
        return base_url
    return os.environ.get("TURBO_OCR_BASE_URL", DEFAULT_BASE_URL)
