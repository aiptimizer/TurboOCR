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

import glob
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
import contextlib

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
# v2 = the dynamic-detection bundles: ONE det export per tier (det_c992x768),
# re-specialized per page shape by the engine at runtime. v1 (fixed-canvas
# mode) stays published and untouched — the 4.0.0a2 wheels have ITS
# url+hashes baked in.
APPLE_NATIVE_SHA256 = {
    # Single det export per tier (det_c992x768): the runtime specializes the
    # detector per page shape from it (fully-convolutional graph, shared
    # 128-grid snap, bounded canvas cache), so shipping more canvases would be
    # duplicate weights, not coverage.
    "apple_native_tiny.tar.gz":
        "2cf2c2e7a7d8250572ae6b1bce13e5480d82c67dc96d1410d52db334c8853c21",
    "apple_native_small.tar.gz":
        "32e1d3d119790e20d136ae2f768ae2afbb7b5eca9b9f94cd6b0e00d6ba8d2e62",
    "apple_native_medium.tar.gz":
        "3595e38c97388a019135baab2193da5bd2a179fa79aed95fc314e4c48c985218",
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
    """Map a catalog-relative path to its release asset name.

    The mapping must produce the EXACT published asset name: a miss is not a
    softer download, it is a 404 (formula/... mapped to a bare
    "inference_trt.onnx" for an asset published as "ppformulanet_s_trt.onnx",
    so OCR(formulas=True) with a cold cache could never provision itself), and
    a case miss silently skips SHA256 verification (GitHub serves asset URLs
    case-insensitively, so "SLANeXt_wired_encoder.onnx" downloaded the
    lowercase-published asset fine — but the sums file is keyed by the exact
    name, so `expected` came back None and the pin was never checked)."""
    rel = rel.replace("\\", "/")
    if rel.startswith("rec/") and rel.endswith("/rec.onnx"):
        lang = rel.split("/")[1]
        return f"rec-{lang}.onnx"
    if rel.startswith("rec/") and rel.endswith("/dict.txt"):
        lang = rel.split("/")[1]
        return f"dict-{lang}.txt"
    if rel == "formula/ppformulanet_s/inference_trt.onnx":
        return "ppformulanet_s_trt.onnx"
    if rel == "formula/ppformulanet_s/tokenizer.json":
        return "ppformulanet_s_tokenizer.json"
    if rel.startswith("table/slanext_encoder/"):
        # published lowercase; the local (cache/C++-expected) name keeps its case
        return os.path.basename(rel).lower()
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

    @staticmethod
    def _det_export_present(det_export: str) -> bool:
        """Is the MPSGraph det export already provisioned at ``det_export``?

        Two layouts count: the flat v1 form (``det_<tier>/graph.json``) and
        the v2 canvas form (``det_<tier>/det_c<H>x<W>/graph.json`` — the
        engine JIT-specializes from that one export). The probe MUST accept
        both, or a v2-provisioned cache re-extracts the whole bundle archive
        on every OCR() construction (measured 0.6 s for tiny, several seconds
        for medium) while reporting the bundle as absent."""
        if os.path.isfile(os.path.join(det_export, "graph.json")):
            return True
        # glob.escape: a cache path containing [, ? or * (a home dir like
        # /Users/user[1]/) silently never matched, so every construction
        # re-extracted the whole bundle.
        return bool(glob.glob(os.path.join(glob.escape(det_export),
                                           "det_c*", "graph.json")))

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
        # WHY it was unavailable, for the caller's error message. None means
        # "provisioned, or not applicable on this platform/model". Set on every
        # exit so a stale value from a previous call can never be reported.
        self.apple_native_reason: Optional[str] = None
        try:
            if sys.platform != "darwin" or platform.machine() != "arm64":
                return False  # not applicable: no reason to report
            if entry.name not in ("tiny", "small", "medium"):
                self.apple_native_reason = (
                    f"the '{entry.name}' model has no Apple native export "
                    "(only tiny/small/medium do)")
                return False
            det_export = os.path.splitext(resolved.det)[0]
            if self._det_export_present(det_export):
                return True
            cache = os.path.abspath(self.cache_dir)
            if os.path.abspath(os.path.dirname(resolved.det)) != cache:
                self.apple_native_reason = (
                    f"models_dir={os.path.dirname(resolved.det)!r} is a "
                    "user-managed directory, and the Apple native export "
                    f"({os.path.basename(det_export)}/graph.json) is not in "
                    "it. Nothing is ever downloaded into a directory you "
                    "manage")
                return False
            if not self.allow_download:
                self.apple_native_reason = (
                    "the Apple native export is not in the cache and "
                    "allow_download=False")
                return False
            rel = f"apple_native_{entry.name}.tar.gz"
            archive = os.path.join(self.cache_dir, rel)
            # Serialize competing provisioners (two processes constructing
            # OCR(backend="apple") at once used to both download and extract
            # straight into the cache, interleaving partial files). flock is
            # advisory but every provisioner comes through here; darwin-only
            # path, so fcntl is always available.
            import fcntl

            os.makedirs(self.cache_dir, exist_ok=True)
            lock_path = os.path.join(
                self.cache_dir, f".apple_native_{entry.name}.lock")
            with open(lock_path, "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    # A racer may have provisioned while we waited on the lock.
                    if self._det_export_present(det_export):
                        return True
                    if not os.path.exists(archive):
                        self._download_apple_bundle(rel, archive)
                    try:
                        self._extract_apple_bundle(archive)
                    except tarfile.TarError:
                        # A truncated/corrupt cached archive would otherwise
                        # fail identically on every construction forever —
                        # drop it so the next attempt re-downloads (the
                        # download path re-verifies the pinned SHA256).
                        # SCOPED to archive-shaped failures: an OSError
                        # (disk full, read-only cache) means the archive was
                        # fine and the re-download would fail too — deleting
                        # the SHA-verified copy would only make it worse.
                        with contextlib.suppress(OSError):
                            os.unlink(archive)
                        raise
                finally:
                    fcntl.flock(lk, fcntl.LOCK_UN)
            if self._det_export_present(det_export):
                return True
            # Download + extraction "succeeded" yet the export is not there
            # (an archive missing its graph.json, say). Without a reason the
            # constructor's refusal is skipped and the engine silently falls
            # back to CoreML — the exact trap the reason exists to close.
            self.apple_native_reason = (
                f"the downloaded bundle extracted without error but "
                f"{os.path.basename(det_export)}/graph.json is still missing "
                "from the cache — the archive appears incomplete. Delete "
                f"{archive!r} to force a re-download")
            return False
        except Exception as exc:  # pragma: no cover - network/broken archive
            # No stderr print: the CALLER decides what to do about it and owns
            # the message. A library must not narrate to the host application.
            self.apple_native_reason = (
                f"the Apple native bundle could not be provisioned "
                f"({type(exc).__name__}: {exc})")
            return False

    def _extract_apple_bundle(self, archive: str) -> None:
        """Extract ``archive`` via a private tempdir INSIDE the cache (same
        filesystem, so every move is an atomic ``os.replace``), then place
        files with the provisioned-probe files (``graph.json``) LAST: a
        reader that does not take the provision lock can never observe the
        bundle as provisioned while it is half-extracted. Files merge into
        existing directories (``coreml/`` is shared between tiers — never
        replace a directory wholesale)."""
        with tempfile.TemporaryDirectory(
            dir=self.cache_dir, prefix=".apple_native_tmp"
        ) as tmp:
            with tarfile.open(archive) as tf:
                try:
                    # 'data' filter: refuses absolute paths, traversal and
                    # special files from the archive.
                    tf.extractall(tmp, filter="data")
                except TypeError:
                    # PEP 706 filters landed in 3.12 and the 3.9.17/3.10.12/
                    # 3.11.4 backports; older micros raise TypeError. Apply
                    # the same safety checks by hand there.
                    members = []
                    for m in tf.getmembers():
                        name = m.name
                        # Both separators and drive-absolute Windows shapes:
                        # tarfile rewrites "/" to os.sep on extraction, so
                        # "C:/evil" and "a\\..\\b" escape on Windows even
                        # though tar itself always stores "/". Unreachable
                        # today (darwin-only caller, SHA-pinned archive) —
                        # defense in depth for the day either changes.
                        parts = name.replace("\\", "/").split("/")
                        if (name.startswith(("/", "\\"))
                                or ".." in parts
                                or (len(name) > 1 and name[1] == ":")
                                or not (m.isreg() or m.isdir())):
                            raise RuntimeError(
                                f"refusing unsafe archive member {name!r}"
                            ) from None
                        members.append(m)
                    tf.extractall(tmp, members=members)
            # Move files into place with the PROBE file — the DET export's
            # graph.json, what _det_export_present checks — strictly LAST.
            # "all graph.json last" was not enough: os.walk order put the det
            # one FIRST among them on some tiers (APFS name hash), so a
            # concurrent unlocked reader could see "provisioned" while every
            # recognizer export was still in the tempdir — and an interrupted
            # extraction left the cache permanently in that state (the probe
            # short-circuits every later repair).
            deferred = []
            det_probe = []
            for root, _dirs, files in os.walk(tmp):
                for fn in files:
                    src = os.path.join(root, fn)
                    rel_p = os.path.relpath(src, tmp)
                    dst = os.path.join(self.cache_dir, rel_p)
                    if fn == "graph.json":
                        top = rel_p.split(os.sep, 1)[0]
                        if top.startswith("det"):
                            det_probe.append((src, dst))
                        else:
                            deferred.append((src, dst))
                        continue
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(src, dst)
            for src, dst in deferred + det_probe:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(src, dst)

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
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    @staticmethod
    def _fetch_to_file(url: str, path: str) -> None:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "turboocr-python"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:
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
