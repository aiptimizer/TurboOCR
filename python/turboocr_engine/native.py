"""Loads the native `_turboocr` extension and maps a Python ``backend`` choice
onto the environment variables the C++ engine reads to select its ONNX Runtime
execution provider.

The C++ ``OrtEngine`` picks its EP from env (``ORT_EP``, ``OPENVINO_DEVICE``,
``DISABLE_COREML``, ...), so the Python layer's whole job here is to translate a
friendly ``backend=`` into those knobs *before* the pipeline is constructed.
"""

from __future__ import annotations

import contextlib
import os
import platform
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# Serializes (env mutation -> pipeline construction): configure_backend() sets
# process-global env that the C++ engine reads at construction, so two threads
# building OCR(...) with different backends must not interleave.
construct_lock = threading.Lock()


@contextlib.contextmanager
def quiet_stdout(enabled: bool = True) -> Iterator[None]:
    """Suppress C-level stdout (fd 1) for the duration of the block.

    The C++ engine prints a few ``[OrtEngine] ...`` banners to stdout at load
    time via raw std::cout, which Python's contextlib.redirect_stdout cannot
    catch. We redirect the OS file descriptor instead. stderr (errors) is left
    intact.

    KNOWN TRADE-OFF: fd 1 is process-global, so OTHER threads' stdout is also
    discarded for the duration (on the NVIDIA wheel that can span a minutes-
    long first-run TensorRT build inside warmup). The alternative — letting
    every construction spray C++ banners into user programs — was judged
    worse; a host app that logs to stdout from other threads during engine
    construction should construct with verbose=True (which skips this) or
    log to stderr."""
    if not enabled:
        yield
        return
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


# The exact LOG_LEVEL value this module last wrote, or None. Authorship is
# judged by VALUE, not a latch: a latch marked the variable "ours" forever,
# so a user exporting LOG_LEVEL=debug AFTER the first engine was clobbered
# by the next quiet one.
_log_level_written: Optional[str] = None


def set_log_level_default(verbose: bool) -> None:
    """Quiet the C++ structured logger by default (users can still set
    LOG_LEVEL explicitly to override).

    A value WE wrote is ours to update (so a later OCR(verbose=True) is not
    a silent no-op); any OTHER value — pre-existing or set by the user at
    any point — always wins."""
    global _log_level_written
    current = os.environ.get("LOG_LEVEL")
    if current is None or current == _log_level_written:
        _log_level_written = "info" if verbose else "warn"
        os.environ["LOG_LEVEL"] = _log_level_written


#: Native symbols this Python layer requires. A `_turboocr` older than these
#: is a STALE BUILD, not a missing one — see load_native().
_REQUIRED_NATIVE_ATTRS = ("Pipeline", "build_info")


# ---- vendor pip-package library preload -------------------------------------
# Two engine wheels have runtime libraries that live OUTSIDE the wheel:
#
#   * NVIDIA (cuda12/cuda13): the CUDA/TensorRT runtimes are deliberately not
#     vendored (excluded in the repair step, exactly as onnxruntime-gpu ships) —
#     `_turboocr` has DT_NEEDED on libnvinfer.so.10, libnvinfer_plugin.so.10,
#     libnvonnxparser.so.10, libcudart.so.1X and libnvjpeg.so.1X, supplied by a
#     system install or the tensorrt-cuXX-libs / nvidia-*-cuXX pip packages.
#   * openvino: the OpenVINO runtime is a REAL pip dependency of the wheel
#     (`openvino>=2026.2,<2026.3` — redistributable, unlike CUDA) — `_turboocr`
#     has DT_NEEDED on libopenvino.so.2621, shipped in site-packages/openvino/libs.
#
# Either way, pip puts the libraries in site-packages directories the dynamic
# loader never searches, so the plain import fails even though everything is
# installed. The fix is the trick torch and onnxruntime-gpu use: dlopen each
# library by ABSOLUTE path first — that puts its SONAME in the process link
# map, which satisfies the DT_NEEDED lookup when `_turboocr` is imported right
# after. The pip libraries carry their own $ORIGIN RPATHs for everything THEY
# load later (TensorRT's per-SM builder resources; OpenVINO's device plugins,
# discovered relative to libopenvino), so preloading the top-level libraries
# is sufficient.
#
# Ordered dependencies-first; prefixes not present on a machine are skipped, so
# this is a no-op unless the pip packages are actually installed.
_VENDOR_LIB_PRIORITY = (
    "libcudart.so",      # nvidia/cuda_runtime/lib — everything CUDA needs it
    "libcublasLt.so",    # nvidia/cublas/lib — before libcublas, which links it
    "libcublas.so",
    "libcudnn.so",       # nvidia/cudnn/lib — ORT CUDA EP (backend="cuda")
    "libcufft.so",
    "libcurand.so",
    "libnvjpeg.so",      # nvidia/nvjpeg/lib — the NVDEC image decoder
    "libnvinfer.so",     # tensorrt_libs — before plugin/onnxparser, which link it
    "libnvonnxparser.so",
    "libnvinfer_plugin.so",
    "libtbb.so",         # openvino/libs — libopenvino links it
    "libtbbmalloc.so",
    "libopenvino.so",    # the core; plugins resolve via its own $ORIGIN
)

#: Absolute paths successfully preloaded (for doctor/debugging).
_vendor_preloaded: List[str] = []
_vendor_preload_attempted = False


def _vendor_pip_lib_dirs(site_dir: Optional[str] = None) -> List[str]:
    """Directories the vendor pip packages install their libraries into, for
    the site-packages holding THIS package (pip installs the runtime packages
    next to the wheel that needs them). Empty when none exist."""
    if site_dir is None:
        site_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = []
    trt = os.path.join(site_dir, "tensorrt_libs")
    if os.path.isdir(trt):
        dirs.append(trt)
    nvidia = os.path.join(site_dir, "nvidia")
    if os.path.isdir(nvidia):
        for sub in sorted(os.listdir(nvidia)):
            lib = os.path.join(nvidia, sub, "lib")
            if os.path.isdir(lib):
                dirs.append(lib)
    ov = os.path.join(site_dir, "openvino", "libs")
    if os.path.isdir(ov):
        dirs.append(ov)
    return dirs


def _preload_vendor_pip_libs(site_dir: Optional[str] = None) -> int:
    """Best-effort dlopen of pip-provided vendor libraries, dependencies first.

    Returns how many loaded. Never raises: a library that fails to load (wrong
    arch, half-installed package, placeholder wheel) is skipped — the retried
    `_turboocr` import produces the real, actionable error."""
    global _vendor_preload_attempted
    _vendor_preload_attempted = True
    if platform.system() != "Linux":
        return 0
    import ctypes

    candidates: List[Tuple[int, str, str]] = []
    for d in _vendor_pip_lib_dirs(site_dir):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            for rank, prefix in enumerate(_VENDOR_LIB_PRIORITY):
                # Versioned real names only ("libcudart.so.12"): the bare ".so"
                # dev symlink may be absent and would double-load anyway.
                if name.startswith(prefix + "."):
                    candidates.append((rank, name, os.path.join(d, name)))
                    break
    loaded = 0
    for _rank, _name, path in sorted(candidates):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        _vendor_preloaded.append(path)
        loaded += 1
    return loaded


def _vendor_runtime_hint(err: str) -> str:
    """One actionable line for an ImportError naming a vendor-runtime soname."""
    if "libopenvino" in err:
        return (
            " This is the OpenVINO engine wheel and the OpenVINO runtime was "
            "not found. It normally arrives as this wheel's own pip dependency "
            "— `pip install 'openvino>=2026.2,<2026.3'` restores it (found "
            "automatically; no LD_LIBRARY_PATH needed)."
        )
    if not any(k in err for k in ("libnvinfer", "libnvonnxparser", "libcudart", "libnvjpeg")):
        return ""
    major = "13" if ".so.13" in err else ("12" if ".so.12" in err else "1X")
    return (
        " This is the NVIDIA engine wheel and a CUDA/TensorRT runtime library "
        "was not found. Install the matching pip packages — "
        f"pip install tensorrt-cu{major}-libs==10.15.1.29 "
        f"nvidia-cuda-runtime-cu{major} nvidia-nvjpeg-cu{major} "
        "— (they are found automatically), or install the system CUDA toolkit "
        "+ TensorRT."
    )


def load_native():
    """Import the compiled `_turboocr` extension, or raise with build guidance."""
    from .errors import NativeExtensionMissing

    try:
        from . import _turboocr  # type: ignore
    except ImportError as first_exc:
        # NVIDIA wheel + pip-provided runtimes: preload and retry ONCE. Only on
        # ImportError — anything else is not a missing-library problem.
        exc: Optional[BaseException] = first_exc
        retried = _preload_vendor_pip_libs() > 0
        if retried:
            try:
                from . import _turboocr  # type: ignore
            except Exception as second_exc:
                exc = second_exc
            else:
                exc = None
        if exc is not None:
            raise NativeExtensionMissing(
                "The native TurboOCR extension (_turboocr) is not built for this "
                "environment. Install a prebuilt wheel for your platform "
                "(`turboocr doctor` shows which), or build it from source with "
                "`cmake -B build -DUSE_CPU_ONLY=ON -DBUILD_PYTHON=ON` and copy the "
                "resulting _turboocr*.so into turboocr/."
                + _vendor_runtime_hint(str(exc))
            ) from exc
    except Exception as exc:  # pragma: no cover
        raise NativeExtensionMissing(
            "The native TurboOCR extension (_turboocr) is not built for this "
            "environment. Install a prebuilt wheel for your platform "
            "(`turboocr doctor` shows which), or build it from source with "
            "`cmake -B build -DUSE_CPU_ONLY=ON -DBUILD_PYTHON=ON` and copy the "
            "resulting _turboocr*.so into turboocr/."
        ) from exc

    # A STALE extension is the more likely failure in a source tree: the wheel
    # packages a PREBUILT .so (pyproject uses hatchling — `pip install ./python`
    # does NOT compile anything), so an .so left over from an older checkout
    # loads fine and then fails deep inside OCR() with a bare AttributeError.
    # Name the real problem here instead. (This is exactly what happened when
    # the native class was renamed CpuPipeline -> Pipeline.)
    missing = [a for a in _REQUIRED_NATIVE_ATTRS if not hasattr(_turboocr, a)]
    if missing:
        raise NativeExtensionMissing(
            f"The native TurboOCR extension at {getattr(_turboocr, '__file__', '?')} "
            f"is STALE — it is missing {', '.join(missing)}, so it predates this "
            "Python layer. Rebuild it (`cmake -B build -DUSE_CPU_ONLY=ON "
            "-DBUILD_PYTHON=ON && cmake --build build --target _turboocr`) and "
            "replace the .so in turboocr/, or reinstall a matching wheel."
        )
    return _turboocr


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


_BUILD_INFO: Optional[dict] = None


def build_info() -> dict:
    """Capabilities of the installed native build: {ort_version, providers}.
    Returns {} if the extension can't be loaded."""
    global _BUILD_INFO
    if _BUILD_INFO is None:
        try:
            _BUILD_INFO = dict(load_native().build_info())
        except Exception:
            _BUILD_INFO = {}
    return _BUILD_INFO


def native_providers() -> List[str]:
    """Execution providers compiled into THIS build's ONNX Runtime (the ground
    truth for what backends actually work), or [] if the extension is missing."""
    return list(build_info().get("providers", []))


def native_backends() -> List[str]:
    """Device backends registered in THIS build's backend seam (the C++
    ``backend::available_backends()``), or [] if the extension is missing.

    This is the *device* axis (which ``Backend`` implementation runs the
    stages) and is orthogonal to :func:`native_providers`, which is the ORT
    *execution provider* axis inside the ORT-based backends."""
    return list(build_info().get("backends", []))


# --------------------------------------------------------------------------- #
#  THE backend table — one row per selectable backend, one source of truth.    #
# --------------------------------------------------------------------------- #
# Two independent axes get selected by the single friendly ``backend=`` string:
#
#   * ``engine`` — which C++ Backend from the seam registry runs the stages
#     (``Pipeline.init(backend=...)`` -> ``backend::make_backend``). Only used
#     when that name is actually compiled into this build (native_backends()),
#     otherwise the ORT-based "cpu" backend runs it.
#   * ``ort_ep`` / ``provider`` — which ONNX Runtime execution provider the
#     ORT-based backend picks (via the ORT_EP env var the C++ engine reads).
#
# Keeping both on one row is what stops the alias tables from drifting apart:
# before this table, ``_NEEDS_PROVIDER``, ``_EP_ALIASES`` and
# ``providers.resolve_providers`` each carried their own copy, and had already
# diverged (``backend="tensorrt"`` resolved to TensorRT in one and to plain CPU
# in another).
@dataclass(frozen=True)
class BackendAlias:
    key: str  # canonical name (what configure_backend reports back)
    aliases: Tuple[str, ...] = ()
    #: ORT provider that must exist in the linked build for this to work.
    provider: str = ""
    #: value for the ORT_EP env var ("" => leave ORT_EP unset)
    ort_ep: str = ""
    #: seam registry backend name ("" => the ORT-based "cpu" backend)
    engine: str = ""
    summary: str = ""


_BACKEND_TABLE: Tuple[BackendAlias, ...] = (
    # -- ORT execution providers (run inside the "cpu" seam backend) ---------
    BackendAlias("cuda", ("gpu",), "CUDAExecutionProvider", "cuda",
                 summary="CUDA (NVIDIA)"),
    # Like "apple" below, this row is seam-first with an EP fallback: on the
    # turboocr-engine-openvino wheel the NATIVE intel backend is compiled in
    # (and measured faster than the ORT OpenVINO EP), so backend="openvino"
    # runs it; on a build without the seam backend it falls through to the
    # ORT EP exactly as before. Without engine="intel" the openvino WHEEL
    # rejected backend="openvino" outright — its vendored ORT is the plain
    # CPU build with no OpenVINO EP, so the EP check failed on the one wheel
    # whose whole point is OpenVINO.
    BackendAlias("openvino", ("ov",), "OpenVINOExecutionProvider", "openvino",
                 engine="intel", summary="OpenVINO (Intel CPU/iGPU/Arc/NPU)"),
    BackendAlias("directml", ("dml",), "DmlExecutionProvider", "dml",
                 summary="DirectML (any DX12 GPU)"),
    BackendAlias("migraphx", (), "MIGraphXExecutionProvider", "migraphx",
                 summary="MIGraphX (AMD)"),
    BackendAlias("rocm", (), "ROCMExecutionProvider", "rocm",
                 summary="ROCm (AMD)"),
    BackendAlias("coreml", (), "CoreMLExecutionProvider",
                 summary="CoreML (Apple GPU/ANE)"),
    BackendAlias("xnnpack", (), "", "xnnpack", summary="XNNPACK (CPU)"),
    BackendAlias("dnnl", (), "", "dnnl", summary="oneDNN (CPU)"),
    # -- device backends from the seam registry ------------------------------
    # These select a whole Backend implementation, not an ORT EP. They fall
    # back to the ORT path when the build doesn't have them linked in, so a
    # CPU-only wheel behaves exactly as it did before this table existed.
    BackendAlias("apple", ("metal", "mps"), "CoreMLExecutionProvider",
                 engine="apple", summary="Apple (Metal/MPSGraph)"),
    # "tensorrt" (the canonical user spelling — it names the actual engine,
    # consistent with "cuda"/"openvino" naming technologies), plus "trt" and
    # the legacy "turbo" — and, since 2026-08-12, the DEFAULT "auto" (via
    # _AUTO_SEAM_PREFERENCE) — reach the nvidia SEAM backend when this build
    # has it compiled in (resolve_engine checks native_backends()); on a
    # CPU-only wheel they fall through to the honest CPU-fallback branch in
    # configure_backend. Without these aliases, backend="tensorrt" could NEVER
    # reach the nvidia backend — even on a build that shipped it — and the
    # fallback message blamed a missing wheel the user might actually have.
    # "turbo" stays accepted forever: a rename must never break a caller.
    BackendAlias("nvidia", ("tensorrt", "trt", "turbo"), engine="nvidia",
                 summary="NVIDIA (TensorRT)"),
    BackendAlias("intel", (), engine="intel", summary="Intel (OpenVINO backend)"),
    BackendAlias("amd", (), engine="amd", summary="AMD"),
)


def _build_alias_index() -> Dict[str, BackendAlias]:
    idx: Dict[str, BackendAlias] = {}
    for b in _BACKEND_TABLE:
        for name in (b.key, *b.aliases):
            idx[name] = b
    return idx


_BY_ALIAS: Dict[str, BackendAlias] = _build_alias_index()

# Derived views, kept for the existing call sites (and so there is exactly one
# place a new backend has to be added).
_NEEDS_PROVIDER: Dict[str, str] = {
    name: b.provider for name, b in _BY_ALIAS.items() if b.provider
}
_EP_ALIASES: Dict[str, str] = {
    name: b.ort_ep for name, b in _BY_ALIAS.items() if b.ort_ep
}

# ORT execution provider -> the env var OrtEngine reads its device ORDINAL from
# (src/backends/onnx/cpu_engine.cpp). One table so a new EP is one row, not a new `if`
# that somebody forgets — which is exactly how CUDA and DirectML ended up
# silently ignoring device_id while ROCm honoured it.
_DEVICE_ID_ENV: Dict[str, str] = {
    "cuda": "CUDA_DEVICE_ID",
    "dml": "DML_DEVICE_ID",
    "rocm": "ROCM_DEVICE_ID",
    "migraphx": "ROCM_DEVICE_ID",
}

# Every env key configure_backend() may set or pop. OCR.__init__ snapshots
# these (plus its structure-model keys) around the construct_lock block and
# restores them on exit, so building one engine cannot leak EP configuration
# into the next build or into the caller's environment. Keep this in sync
# with configure_backend below — a key written there but missing here leaks.
CONSTRUCT_ENV_KEYS: Tuple[str, ...] = ("ORT_EP", "TURBO_EP_DEVICE", "OV_DEVICE", "OPENVINO_DEVICE", "DISABLE_COREML", "COREML_DEVICE", *tuple(sorted(set(_DEVICE_ID_ENV.values()))))


# Backend names that mean "you pick" — what OCR() uses when the caller names no
# backend at all. "fast"/"onnx" are deliberately NOT here: those explicitly ask
# for the no-build ONNX Runtime path, and must keep getting it.
_AUTO_NAMES: Tuple[str, ...] = ("auto", "default", "")

# Seam backends `auto` may resolve to, best first. ONLY nvidia: on the
# turboocr-engine-cuda12/13 wheels the native TensorRT engine is the default (user decision
# 2026-08-12) — it builds its engine once, caches it, and is then the fastest
# path by a wide margin. Every other vendor's default is unchanged: Apple, Intel
# and AMD keep landing on the ORT path from `auto`, so adding "apple" here would
# silently re-point the shipped macOS default. One list, so a future vendor is
# one row rather than a new `if` in two functions.
_AUTO_SEAM_PREFERENCE: Tuple[str, ...] = ("nvidia",)


def auto_engine() -> str:
    """The seam backend ``auto`` picks on THIS build, or ``""`` for the ORT path.

    Reads the build's own registry (``native_backends()``) rather than probing
    hardware: a wheel that has the nvidia backend compiled in IS the NVIDIA
    wheel, and one that doesn't cannot run it however many GPUs are attached."""
    have = native_backends()
    for name in _AUTO_SEAM_PREFERENCE:
        if name in have:
            return name
    return ""


def resolve_engine(backend: str) -> str:
    """The seam registry backend name to hand ``Pipeline.init(backend=...)``.

    Returns the vendor backend (``"apple"``, ``"nvidia"``, ``"intel"``, ...)
    when this build actually has it compiled in, else ``"cpu"`` — so a build
    without that backend keeps running the ORT path exactly as before.

    ``auto`` (the OCR() default) resolves through :func:`auto_engine`, so on the
    turboocr-engine-cuda12/13 wheels it lands on the native TensorRT backend."""
    key = (backend or "").strip().lower()
    if key in _AUTO_NAMES:
        return auto_engine() or "cpu"
    b = _BY_ALIAS.get(key)
    if b is None or not b.engine:
        return "cpu"
    have = native_backends()
    # An empty list means the extension is too old to report backends; be
    # conservative and stay on the ORT path rather than guessing.
    return b.engine if b.engine in have else "cpu"


def ensure_backend_supported(backend: str) -> None:
    """Raise a clear error if ``backend`` needs an EP this build doesn't have —
    e.g. backend='cuda' on the CPU/ONNX wheel — pointing at the right wheel."""
    backend = (backend or "auto").strip().lower()
    # A vendor backend from the seam registry brings its own inference engine,
    # so the ORT execution provider is irrelevant to it — only check the
    # provider when we'll actually be running the ORT path.
    if resolve_engine(backend) != "cpu":
        return
    # A seam-only name (no EP fallback in its row) on a build without that
    # seam has NOTHING to run: backend="intel" on the cpu wheel used to fall
    # into the generic ORT_EP switch and die with a ModelLoadError that
    # blamed backend 'cpu'; backend="turbo" silently OCR'd on CPU where the
    # docs promise a refusal. Raise the documented BackendUnavailable naming
    # the wheel instead. (openvino/apple carry an EP fallback in their rows
    # and keep their documented degrade.)
    row = _BY_ALIAS.get(backend)
    if row is not None and row.engine and not row.provider:
        try:
            seams = native_backends()
        except Exception:
            seams = []
        if seams and row.engine not in seams:
            from .errors import BackendUnavailable

            seam_wheel = {
                "nvidia": "turboocr-engine-cuda12 (driver R525+) or "
                          "turboocr-engine-cuda13 (R580+)",
                "intel": "turboocr-engine-openvino",
                "amd": "turboocr-engine-rocm",
            }.get(row.engine, f"a build with the '{row.engine}' backend")
            raise BackendUnavailable(
                f"backend='{backend}' needs the native {row.engine} engine, "
                f"which this build does not carry. Install {seam_wheel} — "
                "run `turboocr doctor` for the exact command."
            )
    need = _NEEDS_PROVIDER.get(backend)
    if not need:
        return  # cpu/auto/fast/turbo/xnnpack/dnnl always resolve to something valid
    have = native_providers()
    if have and need not in have:
        from .errors import BackendUnavailable

        # Only the four distributions that actually exist may be named here.
        # DirectML used to map to "turboocr-directml", a wheel nobody builds —
        # so the remedy for a missing DML EP was `pip install` a name that
        # resolves nowhere. An EP with no wheel gets the honest remedy instead.
        wheel: Optional[str] = {
            # NVIDIA is two distributions (CUDA 12 vs 13); the right one
            # depends on the host driver, so name both rather than
            # printing an install line that may not fit this machine.
            "CUDAExecutionProvider": "turboocr-engine-cuda12 (driver R525+) or turboocr-engine-cuda13 (R580+)",
            "OpenVINOExecutionProvider": "turboocr-engine-openvino",
            "MIGraphXExecutionProvider": "turboocr-engine-rocm",
            "ROCMExecutionProvider": "turboocr-engine-rocm",
        }.get(need)
        remedy = (
            f"Install {wheel} — run `turboocr doctor` for the exact command."
            if wheel
            else f"No turboocr wheel ships {need}; build the extension from "
            "source against an ONNX Runtime that carries it."
        )
        raise BackendUnavailable(
            f"backend='{backend}' needs {need}, which this build does not have "
            f"(it provides: {have}). {remedy}"
        )


def _trt_cache_dir() -> str:
    """Where the TensorRT engine cache lands, for MESSAGES only.

    Mirrors ``get_engine_cache_dir()`` in
    ``src/backends/nvidia/engine/trt_engine_cache.cpp`` — which owns the real
    behaviour, including creating the directory. Nothing here should act on
    this path; it exists so the note can name the directory the user can delete
    or point elsewhere."""
    override = os.environ.get("TRT_ENGINE_CACHE", "").strip()
    if override:
        return f"TRT_ENGINE_CACHE={override}"
    home = os.environ.get("HOME")
    return f"cached in {home}/.cache/turbo-ocr" if home else "cached in /tmp/turbo-ocr-engines"


_TRT_FIRST_RUN_NOTE = (
    "backend='auto' selected turbo (native TensorRT) — the default on this "
    "NVIDIA build. The FIRST run builds and caches a TensorRT engine, which "
    "takes minutes; every later run loads it from the cache "
    "(TRT_ENGINE_CACHE, default ~/.cache/turbo-ocr). Pass backend='cuda' for "
    "the instant-start ONNX Runtime CUDA path instead."
)

#: One note per process: configure_backend runs once per OCR() and a pool of
#: replicas would otherwise print it `replicas` times.
_trt_note_shown = False


def _note_trt_first_run(*, implicit: bool) -> None:
    """Say once that the TensorRT path builds an engine on its first run.

    Worth stderr rather than a debug log when ``implicit``: the caller passed no
    backend at all, and the very first ``read()`` then sits for minutes building
    an engine with no output — indistinguishable from a hang. Set
    ``TURBO_QUIET_TRT_NOTE=1`` to silence it."""
    global _trt_note_shown
    import logging

    logging.getLogger("turboocr").info(_TRT_FIRST_RUN_NOTE)
    if _trt_note_shown or not implicit or os.environ.get("TURBO_QUIET_TRT_NOTE"):
        return
    _trt_note_shown = True
    import sys

    print(f"[turboocr] {_TRT_FIRST_RUN_NOTE}", file=sys.stderr)


def configure_backend(
    backend: str = "auto",
    *,
    device: Optional[str] = None,
    device_id: int = 0,
) -> Tuple[str, str]:
    """Set the engine's EP env vars for ``backend`` and return
    (resolved_backend, human_summary).

    Must be called before constructing the native pipeline (the engine reads
    these at load time)."""
    backend = (backend or "auto").strip().lower()
    env = os.environ

    # A vendor backend from the seam registry runs its own engine, so none of
    # the ORT_EP plumbing below applies to it. Resolved first so the
    # Apple-silicon branch can't swallow e.g. backend="apple" into plain CPU.
    engine = resolve_engine(backend)
    if engine != "cpu":
        # `auto` has no row of its own — it resolved to a seam backend, so
        # describe THAT backend (_BY_ALIAS["nvidia"]) rather than KeyError-ing.
        spec = _BY_ALIAS.get(backend) or _BY_ALIAS[engine]
        # The vendor engine runs det/rec itself, but the AUX stages (cls,
        # layout, doc-ori, formula) still load OrtEngine sessions — and those
        # honour ORT_EP. A stale value in the caller's environment poisoned
        # them (measured: OCR(backend="apple", autorotate=True) with a
        # leftover ORT_EP=cuda failed to load the doc-ori model). Cleared, so
        # OrtEngine picks this build's best compiled-in provider; the
        # construct-lock env guard restores the caller's value afterwards.
        env.pop("ORT_EP", None)
        if device:
            # TURBO_EP_DEVICE is the name the engine reads
            # (src/service/server/unified/backend_stages.cpp); TURBO_DEVICE was read by nothing.
            env["TURBO_EP_DEVICE"] = device
            # TURBO_EP_DEVICE stops at the ONNX/fast path — the NATIVE intel
            # engine picks its device from OV_DEVICE, read by the backend
            # factory (backend_stages.cpp documents the split). Without this,
            # OCR(backend="openvino", device="NPU") on the openvino wheel
            # silently ran on the default device.
            if engine == "intel":
                env["OV_DEVICE"] = device
        # device=None deliberately leaves any pre-set TURBO_EP_DEVICE /
        # OV_DEVICE alone: configuration.md documents them as operator
        # knobs, so a value in the environment is an interface (same
        # precedence as the DET_* overrides), NOT staleness. Only ORT_EP is
        # derived state owned by backend= and therefore always written.
        summary = spec.summary or spec.key
        if engine == "nvidia":
            _note_trt_first_run(implicit=backend in _AUTO_NAMES)
            summary += (
                " — first run builds & caches the engine "
                f"(one-time; {_trt_cache_dir()})"
            )
        return spec.key, summary

    if is_apple_silicon():
        # On these SVTR/DBNet models the CoreML EP is slower than MLAS and can
        # fail on dynamic shapes, so auto/cpu force CPU by disabling CoreML;
        # coreml is opt-in. Every path here also states its ORT_EP explicitly:
        # this branch used to leave a pre-existing ORT_EP untouched, so a
        # stale value in the caller's environment (ORT_EP=cuda from a Linux
        # dotfile, say) poisoned the load with "Unknown ORT_EP" even though
        # the caller asked for plain cpu.
        if backend in ("coreml", "mps", "apple", "metal"):
            # "metal" is an apple alias everywhere else (resolve_engine);
            # leaving it out here sent one spelling to CPU and its synonyms
            # to CoreML on builds without the apple seam.
            env.pop("DISABLE_COREML", None)
            env["ORT_EP"] = "coreml"
            if device_id:
                env["COREML_DEVICE"] = str(device_id)
            return "coreml", "CoreML (Apple GPU/ANE)"
        env["DISABLE_COREML"] = "1"
        if backend in ("cpu",):
            env["ORT_EP"] = "cpu"
            return "cpu", "CPU (MLAS)"
        env.pop("ORT_EP", None)
        if backend in ("turbo", "tensorrt", "trt"):
            # Not swallowed into plain CPU silently: on a Mac wheel there is
            # no nvidia seam backend, so say what the caller actually got.
            return "tensorrt", "CPU (tensorrt needs a turboocr-engine-cuda12/13 wheel)"
        if backend in ("auto", "fast", "onnx", "default", ""):
            return "cpu", "CPU (MLAS)"
        # An EXPLICIT EP request (xnnpack, dnnl, an unknown name) falls
        # through to the generic ORT_EP switch below instead of being
        # silently swallowed into CPU — same spelling behaves the same on
        # every platform, and an unavailable EP fails loudly at load.

    # The generic ORT_EP switch (non-Apple, plus explicit EP requests on
    # Apple silicon falling through from above).
    if backend in ("cpu",):
        env["ORT_EP"] = "cpu"
        return "cpu", "CPU (MLAS)"
    if backend in ("auto", "fast", "onnx", "default", ""):
        # Reached only when this build has NO vendor seam backend for `auto` to
        # pick (resolve_engine returned "cpu" above) — i.e. the CPU/ONNX wheel.
        # On turboocr-engine-cuda12/13 the nvidia branch already returned turbo. Leaving
        # ORT_EP unset lets OrtEngine take its own best compiled-in provider.
        env.pop("ORT_EP", None)
        return "auto", "CPU (MLAS)"
    if backend in ("turbo", "tensorrt", "trt"):
        # Reached only when resolve_engine() above found NO nvidia seam
        # backend in this build (otherwise the engine branch returned) — so
        # the fallback message is now true by construction.
        env.pop("ORT_EP", None)
        return "tensorrt", "CPU (tensorrt needs a turboocr-engine-cuda12/13 wheel)"
    if backend in _EP_ALIASES:
        ep = _EP_ALIASES[backend]
        env["ORT_EP"] = ep
        if ep == "openvino" and device:
            env["OPENVINO_DEVICE"] = device
        # ONE table, not one `if` per EP: the engine reads a device ordinal for
        # CUDA (src/backends/onnx/cpu_engine.cpp CUDA_DEVICE_ID), DirectML
        # (DML_DEVICE_ID) and ROCm/MIGraphX (ROCM_DEVICE_ID). Only the ROCm arm
        # was written, so OCR(backend="cuda", device_id=1) ran on GPU 0.
        # Explicit device_id wins; device_id=0 (the default) defers to a
        # pre-set env ordinal — CUDA_DEVICE_ID etc. are documented operator
        # knobs (configuration.md), same precedence as the DET_* overrides.
        if device_id and ep in _DEVICE_ID_ENV:
            env[_DEVICE_ID_ENV[ep]] = str(device_id)
        return backend, ep
    # Unknown: pass through as a raw ORT_EP so new EPs work without a code
    # change — but SAY so: a typo used to surface later as a ModelLoadError
    # blaming the model ("Unknown ORT_EP='bogus'"), which reads as a
    # download problem, not a spelling problem.
    import warnings as _w

    _w.warn(
        f"backend={backend!r} is not a known backend name — passing it "
        "through as a raw ONNX Runtime EP. If this is a typo, valid names "
        "include: auto, cpu, turbo, apple, cuda, openvino, rocm, directml, "
        "coreml (see the docs' backend table).",
        stacklevel=3,
    )
    env["ORT_EP"] = backend
    return backend, backend
