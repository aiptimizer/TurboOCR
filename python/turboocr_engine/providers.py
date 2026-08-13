"""Backend / execution-provider resolution.

This is where "run full speed with Python on all backends" and "fast-setup by
default" live. TurboOCR-Python runs the same ONNX models the C++ engine uses,
but picks the inference backend through ONNX Runtime *execution providers*
(EPs) instead of a native build:

  * turbo (the DEFAULT where a vendor graph engine is compiled in — today the
    NVIDIA wheel): the native TensorRT engine with an on-disk cache. Peak
    throughput; the first run pays the engine build once, then every later run
    loads it from TRT_ENGINE_CACHE.

  * fast-setup (``backend="cuda"`` / ``"openvino"`` / ...): the best **no-build**
    EP for the detected hardware — CUDA on NVIDIA, ROCm/MIGraphX on AMD,
    OpenVINO on Intel, DirectML on Windows, MLAS-CPU otherwise. Nothing is
    compiled; you are running the second you have the wheel. This is also what
    ``auto`` lands on wherever no vendor graph engine is linked in (the CPU
    wheel), so nothing outside NVIDIA changed.

The install panel (:mod:`turboocr.doctor`) is driven by :data:`INSTALL_MATRIX`
below, so "which package do I install for my GPU" has a single source of truth.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

ProviderSpec = Tuple[str, Dict[str, object]]  # (ep_name, provider_options)


# --------------------------------------------------------------------------- #
#  Static install matrix — the "for which GPU, how you install it" panel data. #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackendInfo:
    key: str
    label: str
    hardware: str
    ep_names: Tuple[str, ...]
    pip: Tuple[str, ...]  # command(s) to install the right onnxruntime variant
    note: str
    build: bool = False  # True => first run compiles/caches an engine (slow once)
    #: False => NO wheel ships this backend; `pip` MUST then be empty and the
    #: panel prints "build from source" instead of an install command. There
    #: are exactly four engine distributions (turboocr-engine-cpu, -cuda,
    #: -openvino, -rocm), normally installed via the matching extra of the
    #: pure-Python `turboocr` umbrella; a row that names any other package
    #: sends the user to `pip install` a name that does not resolve anywhere.
    packaged: bool = True


INSTALL_MATRIX: Tuple[BackendInfo, ...] = (
    BackendInfo(
        key="cuda",
        label="NVIDIA CUDA (instant start)",
        hardware="NVIDIA GPU (Turing+), Linux/Windows",
        ep_names=("CUDAExecutionProvider",),
        pip=('pip install "turboocr[cuda12]"', 'pip install "turboocr[cuda13]"'),
        note="Runs ONNX directly on the GPU — no engine build, so it starts instantly. "
        "The fallback when you don't want to wait for the TensorRT engine: backend='cuda'.",
    ),
    BackendInfo(
        key="tensorrt",
        label="NVIDIA TensorRT (turbo)",
        hardware="NVIDIA GPU (Turing+), Linux/Windows",
        ep_names=("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        pip=('pip install "turboocr[cuda12]"', 'pip install "turboocr[cuda13]"'),
        # The DEFAULT on the turboocr-engine-cuda12/13 wheels, not an opt-in: that wheel compiles in
        # the nvidia seam backend, and native.resolve_engine sends backend="auto"
        # there. The engine build is a one-time cost the cache absorbs.
        note="Peak throughput, and the DEFAULT on turboocr-engine-cuda12/13 (backend='auto' picks it). "
        "The first run builds+caches a TensorRT engine — a one-time cost, persisted under "
        "TRT_ENGINE_CACHE (default ~/.cache/turbo-ocr); later runs load it. Use "
        "backend='cuda' to skip the build.",
        build=True,
    ),
    BackendInfo(
        key="rocm",
        label="AMD ROCm",
        hardware="AMD Instinct/Radeon, Linux + ROCm",
        ep_names=("MIGraphXExecutionProvider", "ROCMExecutionProvider"),
        pip=('pip install "turboocr[rocm]"',),
        # This note describes THE turboocr-engine-rocm WHEEL, not upstream's
        # onnxruntime-rocm: one Linux x86_64 cp312 abi3 wheel (so it also loads
        # on 3.13+), ROCm linked into _turboocr and taken from the host at run
        # time. It used to describe AMD's own index and their 3.10/3.12 pair,
        # which is a different artifact with different rules.
        note="Linux x86_64 only; one cp312 abi3 wheel (loads on 3.12+). The ROCm/MIGraphX "
        "runtime is NOT bundled — it comes from the host ROCm install, so match the host "
        "to the wheel's ROCm release. ROCMExecutionProvider was removed in onnxruntime "
        "1.23, so MIGraphX is the forward path. Compiles clean but has not yet run on AMD "
        "hardware — build it yourself (python/wheels/rocm) rather than expecting a release.",
    ),
    BackendInfo(
        key="openvino",
        label="Intel OpenVINO",
        hardware="Intel CPU / iGPU / Arc / NPU",
        ep_names=("OpenVINOExecutionProvider",),
        pip=('pip install "turboocr[openvino]"',),
        note="Set device via backend='openvino', device='GPU'|'NPU'|'CPU'|'AUTO'.",
    ),
    BackendInfo(
        key="directml",
        label="DirectML (any GPU, Windows)",
        hardware="Any DX12 GPU (NVIDIA/AMD/Intel), Windows",
        ep_names=("DmlExecutionProvider",),
        # NOT PACKAGED. There is no directml engine wheel — CI builds exactly
        # four (turboocr-engine-{cpu,cuda,openvino,rocm}) — so
        # this row must not print an install command. The EP is still listed
        # because a build linked against a DirectML ONNX Runtime exposes it and
        # backend="directml" reaches it; only the *install hint* was fiction.
        pip=(),
        packaged=False,
        note="Vendor-agnostic Windows path when you don't have CUDA/ROCm set up. "
        "NOT PACKAGED YET: no directml engine wheel exists — build the extension "
        "from source against an ONNX Runtime that carries the DirectML EP, then "
        "backend='directml' will find it.",
    ),
    BackendInfo(
        key="coreml",
        label="Apple CoreML (ANE/GPU)",
        hardware="Apple Silicon (M-series), macOS",
        ep_names=("CoreMLExecutionProvider",),
        pip=('pip install "turboocr[cpu]"',),
        note="CoreML EP ships in the macOS arm64 onnxruntime wheel. NOTE: for these "
        "SVTR/DBNet models the CoreML EP is often SLOWER than MLAS-CPU (per TurboOCR's "
        "Apple findings), so auto uses CPU on macOS — pass backend='coreml' to force it.",
    ),
    BackendInfo(
        key="cpu",
        label="CPU (MLAS / XNNPACK)",
        hardware="Any x86-64 or ARM64 CPU",
        ep_names=("CPUExecutionProvider",),
        pip=('pip install "turboocr[cpu]"',),
        note="Works everywhere, no accelerator needed. XNNPACK helps on ARM.",
    ),
)

_BACKEND_BY_KEY: Dict[str, BackendInfo] = {b.key: b for b in INSTALL_MATRIX}

# NOTE: _AUTO_PREFERENCE (an EP preference order for "auto") was deleted
# 2026-08-12 along with the rest of the dead resolver (see the tail of this
# file). Nothing read it, and its comment asserted the OPPOSITE of the shipped
# policy — "TensorRT is deliberately absent; only chosen for backend='turbo'" —
# which stopped being true when auto started resolving to the nvidia seam
# backend. `auto` is resolved in native.resolve_engine / configure_backend; a
# second, unread copy of that policy here is exactly how the EP tables drifted
# apart before.


# --------------------------------------------------------------------------- #
#  Hardware detection                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class HardwareInfo:
    os: str
    machine: str
    is_apple_silicon: bool = False
    has_nvidia: bool = False
    has_amd: bool = False
    has_intel_gpu: bool = False
    gpu_names: List[str] = field(default_factory=list)
    #: NVIDIA driver major, e.g. 580. None when nvidia-smi is absent or its
    #: output could not be parsed. This is what picks between the CUDA 12 and
    #: CUDA 13 engine wheels: a wheel links one CUDA major, and CUDA 13 needs
    #: a newer driver than CUDA 12 does. Nothing else depends on it, so an
    #: unparseable value degrades to "recommend the widely installable one".
    nvidia_driver_major: Optional[int] = None

    @property
    def vendor(self) -> str:
        if self.has_nvidia:
            return "nvidia"
        if self.has_amd:
            return "amd"
        if self.is_apple_silicon:
            return "apple"
        if self.has_intel_gpu:
            return "intel"
        return "cpu"


def _cmd_ok(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=4, check=False
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return ""


def detect_hardware() -> HardwareInfo:
    """Best-effort probe of the local accelerators. Never raises.

    Cached: the probe shells out to nvidia-smi / rocm-smi / clinfo (each with a
    4 s timeout), and the answer cannot change within a process. Call
    ``detect_hardware.cache_clear()`` to force a re-probe."""
    return _detect_hardware_cached()


@lru_cache(maxsize=1)
def _detect_hardware_cached() -> HardwareInfo:
    sysname = platform.system()
    machine = platform.machine()
    hw = HardwareInfo(os=sysname, machine=machine)

    hw.is_apple_silicon = sysname == "Darwin" and machine in ("arm64", "aarch64")

    # NVIDIA — nvidia-smi is the reliable signal on Linux/Windows.
    if _cmd_ok("nvidia-smi"):
        names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        gpus = [ln.strip() for ln in names.splitlines() if ln.strip()]
        if gpus:
            hw.has_nvidia = True
            hw.gpu_names.extend(gpus)
            # Same probe, one more field: the driver decides which CUDA major
            # this machine can run, and therefore which engine wheel to name.
            drv = _run(["nvidia-smi", "--query-gpu=driver_version",
                        "--format=csv,noheader"])
            for ln in drv.splitlines():
                ln = ln.strip()
                if ln and ln[0].isdigit():
                    try:
                        hw.nvidia_driver_major = int(ln.split(".")[0])
                    except ValueError:
                        pass
                    break

    # AMD — rocminfo / rocm-smi.
    if _cmd_ok("rocminfo") or _cmd_ok("rocm-smi"):
        hw.has_amd = True
        info = _run(["rocm-smi", "--showproductname"])
        for ln in info.splitlines():
            ln = ln.strip()
            if "Card series" in ln or "Card model" in ln:
                hw.gpu_names.append(ln.split(":")[-1].strip())

    # Intel GPU — clinfo / sycl-ls hints (best effort, non-fatal if absent).
    probe = _run(["clinfo", "-l"]) if _cmd_ok("clinfo") else ""
    if "Intel" in probe and ("Graphics" in probe or "Arc" in probe):
        hw.has_intel_gpu = True

    if hw.is_apple_silicon and not hw.gpu_names:
        hw.gpu_names.append(f"Apple {machine} GPU")
    return hw


# Expose the cache controls on the public name, so `detect_hardware.cache_clear()`
# works as documented above.
detect_hardware.cache_clear = _detect_hardware_cached.cache_clear  # type: ignore[attr-defined]
detect_hardware.cache_info = _detect_hardware_cached.cache_info  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  Runtime EP resolution                                                       #
# --------------------------------------------------------------------------- #
def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def available_providers() -> List[str]:
    """EPs the installed onnxruntime exposes, or [] if it isn't installed."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def onnxruntime_version() -> Optional[str]:
    try:
        import onnxruntime as ort

        return ort.__version__
    except Exception:
        return None


# NOTE: resolve_providers()/_trt_options()/_raise_missing() were deleted
# 2026-08-03. They were a fourth, already-drifted copy of the EP alias table:
# the native binding resolves execution providers in C++ (ORT_EP /
# TURBO_EP_PROVIDER — see native.py), and nothing in the package called them.
