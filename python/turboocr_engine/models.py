"""Model resolution + on-demand download.

Two ways to get the ONNX weights:

  * point at an existing directory (``models_dir=`` / ``TURBO_OCR_MODELS_DIR``
    / a ``./models`` folder in the CWD) — zero download, used as-is;
  * otherwise fetch just the tier you asked for from the pinned TurboOCR GitHub
    release, verify SHA256, and cache under ``~/.cache/turboocr``.

Only the assets a given model needs are downloaded, so ``tiny`` pulls ~6 MB,
not the whole 1.5 GB bundle.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional

# NOTE: urllib.request is imported lazily inside the two fetch helpers. It costs
# ~17 ms (it drags in http.client + ssl) and is only reachable on the download
# path — resolving models from a local dir or the cache never needs it.

from .catalog import ModelEntry

DEFAULT_RELEASE = "models-v3.0.0-ppocrv6"
RELEASE_BASE = (
    "https://github.com/aiptimizer/TurboOCR/releases/download/" + DEFAULT_RELEASE
)


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v else None


def user_cache_dir() -> str:
    override = _env("TURBO_OCR_CACHE_DIR")
    if override:
        return override
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "turboocr")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/turboocr")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "turboocr")


def _local_to_asset(rel: str) -> str:
    """Map a catalog-relative path to its release asset name."""
    rel = rel.replace("\\", "/")
    if rel.startswith("rec/") and rel.endswith("/rec.onnx"):
        lang = rel.split("/")[1]
        return f"rec-{lang}.onnx"
    if rel.startswith("rec/") and rel.endswith("/dict.txt"):
        lang = rel.split("/")[1]
        return f"dict-{lang}.txt"
    if rel.startswith("layout/"):
        return os.path.basename(rel)
    return os.path.basename(rel)


@dataclass
class ResolvedModel:
    det: str
    rec: str
    dict: str
    cls: Optional[str]
    name: str


class ModelStore:
    """Locates model files, downloading from the release when needed."""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        *,
        release_base: str = RELEASE_BASE,
        allow_download: bool = True,
    ) -> None:
        self.release_base = release_base.rstrip("/")
        self.allow_download = allow_download
        self._sha_cache: Optional[Dict[str, str]] = None

        self.local_dir = self._pick_local_dir(models_dir)
        self.cache_dir = os.path.join(user_cache_dir(), "models", DEFAULT_RELEASE)

    @staticmethod
    def _pick_local_dir(models_dir: Optional[str]) -> Optional[str]:
        candidates = [
            models_dir,
            _env("TURBO_OCR_MODELS_DIR"),
            os.path.join(os.getcwd(), "models"),
        ]
        for c in candidates:
            if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "det.onnx")):
                return os.path.abspath(c)
        # An explicit models_dir that exists but lacks det.onnx is still honored
        # (a tier-only dir); fall through to letting resolve() download into it.
        if models_dir and os.path.isdir(models_dir):
            return os.path.abspath(models_dir)
        return None

    # -- public ------------------------------------------------------------
    def resolve(self, entry: ModelEntry, *, want_cls: bool = False) -> ResolvedModel:
        det = self._ensure(entry.det_path())
        rec = self._ensure(entry.rec)
        dic = self._ensure(entry.dict)
        cls = self._ensure("cls.onnx") if want_cls else None
        return ResolvedModel(det=det, rec=rec, dict=dic, cls=cls, name=entry.name)

    def ensure_asset(self, rel: str) -> str:
        """Public single-asset resolver (used for layout/doc_ori/etc.)."""
        return self._ensure(rel)

    # -- internals ---------------------------------------------------------
    def _ensure(self, rel: str) -> str:
        # 1. Existing local dir wins.
        if self.local_dir:
            p = os.path.join(self.local_dir, rel)
            if os.path.exists(p):
                return p
        # 2. Cache dir.
        cached = os.path.join(self.cache_dir, rel)
        if os.path.exists(cached):
            return cached
        # 3. Download.
        if not self.allow_download:
            raise FileNotFoundError(
                f"model asset '{rel}' not found locally and downloads are disabled. "
                f"Set models_dir=... or allow downloads."
            )
        return self._download(rel, cached)

    def _sha_sums(self) -> Dict[str, str]:
        if self._sha_cache is not None:
            return self._sha_cache
        sums: Dict[str, str] = {}
        try:
            data = self._fetch_bytes(f"{self.release_base}/SHA256SUMS.txt")
            for line in data.decode("utf-8", "replace").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    sums[parts[1].lstrip("*")] = parts[0]
        except Exception:
            pass  # verification becomes best-effort if the sums file is gone
        self._sha_cache = sums
        return sums

    def _download(self, rel: str, dest: str) -> str:
        asset = _local_to_asset(rel)
        url = f"{self.release_base}/{asset}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        expected = self._sha_sums().get(asset)

        print(f"[turboocr] downloading {asset} -> {dest}", file=sys.stderr)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
        os.close(tmp_fd)
        try:
            self._fetch_to_file(url, tmp_path)
            if expected:
                actual = _sha256_file(tmp_path)
                if actual != expected:
                    raise RuntimeError(
                        f"SHA256 mismatch for {asset}: expected {expected}, got {actual}"
                    )
            os.replace(tmp_path, dest)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return dest

    @staticmethod
    def _fetch_bytes(url: str) -> bytes:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "turboocr-python"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            return r.read()

    @staticmethod
    def _fetch_to_file(url: str, path: str) -> None:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "turboocr-python"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:  # noqa: S310
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
