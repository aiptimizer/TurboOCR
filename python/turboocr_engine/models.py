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
import platform
import sys
import tarfile
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

# The Apple native-mode bundles live in their OWN release: the models release
# is immutable, and the bundles are derived artefacts with their own version
# (regenerating them = new release tag + new pins here, together). The SHA256
# of every asset is pinned IN CODE — stronger than a sums file, and the
# download refuses anything that does not match.
APPLE_NATIVE_RELEASE = "apple-native-v2"
APPLE_NATIVE_BASE = (
    "https://github.com/aiptimizer/TurboOCR/releases/download/" + APPLE_NATIVE_RELEASE
)
# v2 = the multi-canvas detection bundles (4 det canvases per tier, picked
# per page by the shared aspect policy). v1 (single 992x768 canvas) stays
# published and untouched — the 4.0.0a2 wheels have ITS url+hashes baked in.
APPLE_NATIVE_SHA256 = {
    "apple_native_tiny.tar.gz":
        "7967c9ecf3a8ae204922a58d9b36388b65a763c902e005d8ce8c41a04baecded",
    "apple_native_small.tar.gz":
        "178a5241c36db9f0d05faa5db65e6d7d5e292089d94a5f9bed61f99812fe76f6",
    "apple_native_medium.tar.gz":
        "76f3883d84509d83d591c2f047425465f3ed178308d5163a14e7d356c9abd6d8",
}


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
        # EXPLICIT intent wins outright when the directory exists — the
        # models_dir argument first, then TURBO_OCR_MODELS_DIR. Even a
        # partially-populated (tier-only) directory is honored: _ensure falls
        # back per-file to the cache/download for anything missing. These used
        # to be subject to the same det.onnx probe as the CWD heuristic below,
        # so OCR(models_dir=<tier-only dir>) run from a checkout with ./models
        # present silently resolved against the CWD copy instead — different
        # weights, and different Apple native exports, from the ones the
        # caller named.
        for c in (models_dir, _env("TURBO_OCR_MODELS_DIR")):
            if c and os.path.isdir(c):
                return os.path.abspath(c)
        # The CWD candidate is a heuristic, so it keeps the det.onnx probe:
        # it guards against latching onto an unrelated directory that merely
        # happens to be called "models".
        cwd = os.path.join(os.getcwd(), "models")
        if os.path.isdir(cwd) and os.path.exists(os.path.join(cwd, "det.onnx")):
            return os.path.abspath(cwd)
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

    def ensure_apple_native(self, entry: ModelEntry, resolved: "ResolvedModel") -> bool:
        """Best-effort: provision the Apple NATIVE-mode export bundle next to
        the resolved models, so ``backend="apple"`` runs Metal+MPSGraph with
        the ANE lane instead of the CoreML fallback.

        The engine's discovery is path-based (models/<stem>.onnx ->
        models/<stem>/graph.json, plus coreml/<tier>/ for the ANE packages —
        src/backends/apple/), so this only has to put the
        ``apple_native_<tier>.tar.gz`` release asset's contents into the same
        directory the models resolved from. Rules:

        * already provisioned (det export present) -> True, nothing done;
        * models resolved from a user-managed directory (models_dir /
          TURBO_OCR_MODELS_DIR) -> False: extracting into the cache would be
          invisible next to those models, and writing into a directory the
          user manages is not this store's call — generate exports there with
          tools/modelgen/apple/export_apple_native.py instead;
        * otherwise download + extract into the cache, once.

        Never raises: native mode is an upgrade, and every failure path
        (no such asset published, offline, bad archive) leaves the engine on
        its CoreML fallback exactly as before."""
        try:
            if sys.platform != "darwin" or platform.machine() != "arm64":
                return False
            if entry.name not in ("tiny", "small", "medium"):
                return False  # script models have no native bundles
            det_export = os.path.splitext(resolved.det)[0]
            if os.path.isfile(os.path.join(det_export, "graph.json")):
                return True
            cache = os.path.abspath(self.cache_dir)
            if os.path.abspath(os.path.dirname(resolved.det)) != cache:
                return False
            if not self.allow_download:
                return False
            rel = f"apple_native_{entry.name}.tar.gz"
            archive = os.path.join(self.cache_dir, rel)
            if not os.path.exists(archive):
                self._download_apple_bundle(rel, archive)
            with tarfile.open(archive) as tf:
                # 'data' filter (Python 3.12+): refuses absolute paths,
                # traversal and special files from the archive.
                tf.extractall(self.cache_dir, filter="data")
            return os.path.isfile(os.path.join(det_export, "graph.json"))
        except Exception as exc:  # pragma: no cover - network/broken archive
            print(f"[turboocr] apple native bundle unavailable ({exc}); "
                  "backend='apple' uses the CoreML fallback.", file=sys.stderr)
            return False

    def _download_apple_bundle(self, rel: str, dest: str) -> str:
        """Fetch one apple_native_* asset with its code-pinned SHA256.

        Unlike the models release's best-effort sums file, verification here
        is MANDATORY: an asset whose name is not pinned, or whose digest does
        not match, is refused."""
        expected = APPLE_NATIVE_SHA256.get(rel)
        if not expected:
            raise RuntimeError(f"no pinned SHA256 for {rel}")
        url = f"{APPLE_NATIVE_BASE}/{rel}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[turboocr] downloading {rel} -> {dest}", file=sys.stderr)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
        os.close(tmp_fd)
        try:
            self._fetch_to_file(url, tmp_path)
            actual = _sha256_file(tmp_path)
            if actual != expected:
                raise RuntimeError(
                    f"SHA256 mismatch for {rel}: expected {expected}, got {actual}")
            os.replace(tmp_path, dest)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return dest

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
