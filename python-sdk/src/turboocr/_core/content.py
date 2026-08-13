from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, cast

# Accept anything reasonable: in-memory bytes, a path, an open binary file,
# or a pre-built byte-chunk iterable (e.g. from a generator).
type ImageInput = bytes | str | Path | IO[bytes] | Iterable[bytes]

# A `ContentProvider` is one of:
#   - bytes (in-memory, trivially replayable)
#   - Iterable[bytes] (one-shot stream; retries best-effort)
#   - Callable[[], Iterable[bytes]] (factory — invoked fresh per attempt, so
#     retries on streamed bodies are safe)
# RequestSpec stores this; the client materializes the callable on each send.
type ContentProvider = bytes | Iterable[bytes] | Callable[[], Iterable[bytes]]

_STREAM_CHUNK: int = 64 * 1024


def _stream_file(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while chunk := f.read(_STREAM_CHUNK):
            yield chunk


def _stream_reader(reader: IO[bytes]) -> Iterator[bytes]:
    # Caller owns the file handle; we do not close it.
    while chunk := reader.read(_STREAM_CHUNK):
        yield chunk


def _is_reader(image: object) -> bool:
    return hasattr(image, "read") and not isinstance(image, bytes | bytearray | memoryview)


def read_image_bytes(image: ImageInput) -> bytes:
    """Materialize to bytes. Used by endpoints that must base64-encode
    (e.g. /ocr and /ocr/batch) where streaming is not applicable."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, str | Path):
        return Path(image).read_bytes()
    if _is_reader(image):
        reader = cast(IO[bytes], image)
        return reader.read()
    return b"".join(image)


def streamable_content(image: ImageInput) -> ContentProvider:
    """Return a payload suitable for httpx's `content=`. bytes pass through
    unchanged. Paths become a *factory* that re-opens the file on each retry
    attempt — so big PDFs stream chunk-by-chunk yet retries replay cleanly.
    Open file handles and pre-built iterables are one-shot; retries on those
    may fail mid-upload (caller's trade-off when they own the lifecycle)."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, str | Path):
        path = Path(image)
        return lambda: _stream_file(path)
    if _is_reader(image):
        return _stream_reader(cast(IO[bytes], image))
    return image


def materialize(provider: ContentProvider) -> bytes | Iterable[bytes]:
    """Resolve a factory once. Bytes/iterables pass through unchanged."""
    if callable(provider):
        return provider()
    return provider


async def _aiter_from_sync(iterable: Iterable[bytes]) -> AsyncIterator[bytes]:
    for chunk in iterable:
        yield chunk


def materialize_async(provider: ContentProvider) -> bytes | AsyncIterator[bytes]:
    """Resolve a factory and adapt sync iterables for httpx.AsyncClient.

    httpx 0.28+ requires AsyncIterable[bytes] under AsyncClient and rejects
    sync iterables. This wrapper lets the same ContentProvider drive both
    sync and async clients without forking the body-building code.
    """
    if isinstance(provider, bytes):
        return provider
    sync_iter = provider() if callable(provider) else provider
    return _aiter_from_sync(sync_iter)
