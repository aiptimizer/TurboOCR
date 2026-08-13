"""The install panel — "for your GPU, here's how to install it".

``turboocr.doctor()`` (or ``turboocr doctor`` on the CLI) inspects the machine,
reports what ONNX Runtime backend is installed and available, recommends the
right one for the detected hardware, and prints the full install matrix. Uses
``rich`` for a pretty table when it's installed, and degrades to plain text
otherwise.

The four engine wheels are separate and MUTUALLY EXCLUSIVE — one distribution
per accelerator — so the whole point of the panel is to name exactly one of
them. Each is normally pulled in through the matching extra of the pure-Python
`turboocr` umbrella package:

    turboocr-engine-cpu        CPU (and Apple Silicon)    pip install "turboocr[cpu]"
    turboocr-engine-cuda       NVIDIA GPU                 pip install "turboocr[cuda]"
    turboocr-engine-openvino   Intel GPU / NPU            pip install "turboocr[openvino]"
    turboocr-engine-rocm       AMD GPU                    pip install "turboocr[rocm]"
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

from ._version import __version__
from .providers import (
    _BACKEND_BY_KEY,
    INSTALL_MATRIX,
    BackendInfo,
    HardwareInfo,
    available_providers,
    detect_hardware,
    onnxruntime_available,
    onnxruntime_version,
)

#: The pip distribution names. One per accelerator; installing two of them into
#: the same environment is unsupported (they provide the same import package).
PACKAGE_CPU = "turboocr-engine-cpu"
PACKAGE_CUDA = "turboocr-engine-cuda"
PACKAGE_OPENVINO = "turboocr-engine-openvino"
PACKAGE_ROCM = "turboocr-engine-rocm"

# hardware vendor -> (backend key, pip package, one-line reason).
# The package name here is authoritative for what doctor tells you to install;
# INSTALL_MATRIX[].pip stays the per-backend reference table.
_RECOMMEND: dict = {
    # The starred row is "tensorrt", not "cuda", because that IS what the wheel
    # runs by default: on turboocr-engine-cuda the nvidia seam backend is
    # compiled in, so backend="auto" resolves to turbo/TensorRT
    # (native.resolve_engine). Starring the no-build CUDA row described a
    # default the build no longer has.
    "nvidia": ("tensorrt", PACKAGE_CUDA, "NVIDIA GPU detected — turboocr-engine-cuda carries the native TensorRT engine (the default: backend='auto' picks it, and the first run builds+caches the engine) plus the CUDA execution provider as the instant-start backend='cuda' fallback."),
    "amd": ("rocm", PACKAGE_ROCM, "AMD GPU detected — turboocr-engine-rocm carries the ROCm/MIGraphX execution provider (Linux only)."),
    "intel": ("openvino", PACKAGE_OPENVINO, "Intel GPU/NPU detected — turboocr-engine-openvino carries the OpenVINO execution provider, the best acceleration on Intel silicon."),
    "apple": ("cpu", PACKAGE_CPU, "Apple Silicon — the CPU engine wheel is the right one; its macOS arm64 build carries the Apple backend, and there is no separate Apple wheel."),
    "cpu": ("cpu", PACKAGE_CPU, "No supported accelerator detected — the CPU engine wheel works everywhere."),
}

# Printed under the install command. Keep this honest: the per-backend wheels
# are not on PyPI yet, so don't imply a bare `pip install` will resolve them.
_INSTALL_NOTE = (
    "Pre-release: the accelerator wheels aren't on PyPI yet — point pip at the "
    "TestPyPI index, or build one with scripts/python/build_backend_wheel.sh. "
    "Re-run `turboocr doctor` afterwards to confirm the provider is live."
)

# Printed with the recommendation. The panel lists several pip lines (one per
# backend row), which reads like a menu you can combine — it isn't. All four
# engine distributions own the `turboocr_engine` import package and the
# `turboocr` console script, so a second one in the same site-packages is
# last-writer-wins.
_EXCLUSIVE_NOTE = (
    "Install exactly ONE turboocr-engine-* wheel per environment — the cpu, "
    "cuda, openvino and rocm engine wheels all provide the same "
    "`turboocr_engine` import package, so installing a second one overwrites "
    "the first. (The pure-Python `turboocr` umbrella is fine alongside any of "
    "them — its extras are how you normally pick one.)"
)

#: Shown in the install column for a backend no wheel ships (packaged=False).
_NOT_PACKAGED = "not packaged — build from source"

# Footer under the backend table. turbo is the DEFAULT on the NVIDIA wheel (the
# nvidia seam backend is compiled in, so backend="auto" resolves to it), which
# makes the one-time engine build the first thing an NVIDIA user meets — say so
# here rather than letting a multi-minute first run look like a hang.
_TURBO_LINE = (
    "NVIDIA (turboocr-engine-cuda): backend='auto' (the default) resolves to turbo — "
    "the native TensorRT engine. The FIRST run builds and caches it (one-time; "
    "TRT_ENGINE_CACHE, default ~/.cache/turbo-ocr); backend='cuda' is the "
    "instant-start ONNX Runtime path."
)


def _install_hint(b: BackendInfo) -> Tuple[str, ...]:
    """Install line(s) for one matrix row, or the honest 'no wheel' line.

    Only four distributions exist; a row without one (DirectML) has an empty
    ``pip`` by construction and must never render a ``pip install <name>`` that
    pip cannot resolve."""
    return b.pip if (b.packaged and b.pip) else (_NOT_PACKAGED,)


class Recommendation(NamedTuple):
    """What to install on this machine.

    ``backend`` is the row of :data:`~turboocr.providers.INSTALL_MATRIX` that
    will serve the hardware; ``package`` is the pip distribution that ships it;
    ``install`` is the literal command line(s) to run.
    """

    backend: BackendInfo
    package: str
    reason: str
    install: Tuple[str, ...]


def _backend_by_key(key: str) -> Optional[BackendInfo]:
    # providers already indexes INSTALL_MATRIX by key; don't re-scan it here.
    return _BACKEND_BY_KEY.get(key)


def install_commands(package: str) -> Tuple[str, ...]:
    """The exact command lines doctor prints for ``package``.

    The umbrella extra comes first — it is the front door, and also installs
    the `turboocr` client/CLI layer. The direct engine install is the second
    line, for environments that want only the engine distribution."""
    variant = package.removeprefix("turboocr-engine-")
    return (f'pip install "turboocr[{variant}]"', f"pip install {package}")


def recommend(hw: Optional[HardwareInfo] = None) -> Recommendation:
    """Pick the one wheel this machine should have. Never raises."""
    hw = hw or detect_hardware()
    key, package, reason = _RECOMMEND.get(hw.vendor, _RECOMMEND["cpu"])
    b = _backend_by_key(key)
    assert b is not None
    return Recommendation(b, package, reason, install_commands(package))


def effective_providers() -> Tuple[List[str], Optional[str], bool]:
    """``(providers, ort_version, from_native)`` for THIS install.

    The REAL capability is what the native extension's linked ONNX Runtime
    provides — not the pip ``onnxruntime`` package, which the native path never
    uses for inference. Prefer native; fall back to pip only if the extension
    isn't built yet. Single source of truth so the report and
    :func:`available_backends` can't disagree about what's installed."""
    from . import native

    native_bi = native.build_info()
    if native_bi:
        return list(native_bi.get("providers", [])), native_bi.get("ort_version"), True
    return available_providers(), onnxruntime_version(), False


def build_report(hw: Optional[HardwareInfo] = None) -> dict:
    hw = hw or detect_hardware()
    rec = recommend(hw)

    from . import native

    providers, ort_version, native_built = effective_providers()

    return {
        "version": __version__,
        "hardware": {
            "os": hw.os,
            "machine": hw.machine,
            "vendor": hw.vendor,
            "gpus": hw.gpu_names,
        },
        "native_extension": native_built,
        # Device backends compiled into the seam registry of this build — the
        # names OCR(backend=...) can actually reach (see native.resolve_engine).
        "native_backends": native.native_backends(),
        "onnxruntime": {
            "installed": native_built or onnxruntime_available(),
            "version": ort_version,
            "available_providers": providers,
            "source": "native extension" if native_built else "pip onnxruntime (extension not built)",
        },
        "pypdfium2": _pypdfium_installed(),
        "recommended": {
            "backend": rec.backend.key,
            "package": rec.package,
            "reason": rec.reason,
            "install": list(rec.install),
            "exclusive": _EXCLUSIVE_NOTE,
            "note": _INSTALL_NOTE,
        },
    }


def _pypdfium_installed() -> bool:
    try:
        import pypdfium2  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Rendering                                                                   #
# --------------------------------------------------------------------------- #
def doctor(hw: Optional[HardwareInfo] = None, *, plain: bool = False) -> dict:
    """Print the install panel and return the structured report."""
    report = build_report(hw)
    if plain or not _try_rich(report):
        _render_plain(report)
    return report


def _try_rich(report: dict) -> bool:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception:
        return False

    console = Console()
    hw = report["hardware"]
    ort = report["onnxruntime"]
    rec = report["recommended"]

    header = Text()
    header.append(f"TurboOCR {report['version']}\n", style="bold cyan")
    header.append(f"OS: {hw['os']} ({hw['machine']})   Vendor: ", style="dim")
    header.append(f"{hw['vendor']}", style="bold")
    if hw["gpus"]:
        header.append("\nGPU: " + ", ".join(hw["gpus"]), style="dim")
    ort_line = (
        f"onnxruntime {ort['version']}" if ort["installed"] else "onnxruntime NOT installed"
    )
    header.append(f"\nRuntime: {ort_line}", style="dim")
    if ort["available_providers"]:
        header.append("\nProviders: " + ", ".join(ort["available_providers"]), style="dim")
    header.append(
        f"\nPDF (pypdfium2): {'installed' if report['pypdfium2'] else 'not installed'}",
        style="dim",
    )
    console.print(Panel(header, title="doctor", border_style="cyan"))

    rec_txt = Text()
    rec_txt.append("→ Recommended package: ", style="bold green")
    rec_txt.append(f"{rec['package']}", style="bold")
    rec_txt.append(f"  (backend: {rec['backend']})\n", style="dim")
    rec_txt.append(rec["reason"] + "\n\n", style="dim")
    for cmd in rec["install"]:
        rec_txt.append("  " + cmd + "\n", style="bold yellow")
    rec_txt.append("\n" + rec["exclusive"] + "\n", style="bold")
    rec_txt.append("\n" + rec["note"], style="dim")
    console.print(Panel(rec_txt, title="install for your hardware", border_style="green"))

    table = Table(title="All backends", show_lines=False, header_style="bold")
    table.add_column("backend")
    table.add_column("hardware")
    table.add_column("install", style="yellow")
    table.add_column("EP")
    avail = set(ort["available_providers"])
    for b in INSTALL_MATRIX:
        ready = "✓" if any(ep in avail for ep in b.ep_names) else " "
        star = " ★" if b.key == rec["backend"] else ""
        table.add_row(
            f"{ready} {b.label}{star}",
            b.hardware,
            "\n".join(_install_hint(b)),
            "\n".join(b.ep_names),
        )
    console.print(table)
    console.print(
        "[dim]★ = recommended   ✓ = available in your onnxruntime\n"
        f"{_TURBO_LINE}[/dim]"
    )
    return True


def _render_plain(report: dict) -> None:
    hw = report["hardware"]
    ort = report["onnxruntime"]
    rec = report["recommended"]
    line = "=" * 68
    print(line)
    print(f" TurboOCR {report['version']} — doctor")
    print(line)
    print(f" OS:        {hw['os']} ({hw['machine']})")
    print(f" Vendor:    {hw['vendor']}")
    if hw["gpus"]:
        print(f" GPU:       {', '.join(hw['gpus'])}")
    if ort["installed"]:
        print(f" onnxruntime: {ort['version']}")
        print(f" providers:   {', '.join(ort['available_providers']) or '(none)'}")
    else:
        print(" onnxruntime: NOT installed")
    print(f" PDF (pypdfium2): {'installed' if report['pypdfium2'] else 'not installed'}")
    print(line)
    print(f" → Recommended package: {rec['package']}   (backend: {rec['backend']})")
    print(f"   {rec['reason']}")
    print()
    for cmd in rec["install"]:
        print(f"     {cmd}")
    print()
    print(f"   {rec['exclusive']}")
    print()
    print(f"   {rec['note']}")
    print(line)
    print(" All backends (✓ = available now, ★ = recommended):")
    avail = set(ort["available_providers"])
    for b in INSTALL_MATRIX:
        ready = "✓" if any(ep in avail for ep in b.ep_names) else " "
        star = " ★" if b.key == rec["backend"] else ""
        print(f"  [{ready}] {b.label}{star}")
        print(f"        hardware: {b.hardware}")
        if b.packaged and b.pip:
            for cmd in b.pip:
                print(f"        $ {cmd}")
        else:
            print(f"        {_NOT_PACKAGED}")
    print(line)
    print(_TURBO_LINE)
    print(line)


def available_backends() -> List[str]:
    """Backend keys whose EPs are present in the ONNX Runtime that will
    actually run inference (the native extension's, when it is built)."""
    avail = set(effective_providers()[0])
    return [b.key for b in INSTALL_MATRIX if any(ep in avail for ep in b.ep_names)]
