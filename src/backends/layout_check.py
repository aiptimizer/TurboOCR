#!/usr/bin/env python3
"""Enforce the src/backends/ layout convention. Exit 0 = clean, 1 = violations.

    python3 src/backends/layout_check.py

WHY THIS EXISTS: a convention that is only written down rots silently. Both of
the failure modes below had already happened in this tree before this file:

  * the Apple sources used bare sibling includes (`#import "metal_common.h"`),
    so moving a file into a subdirectory broke the build (check 3);
  * the Apple CMake target used a NON-recursive file(GLOB ... /apple/*.mm), so
    subdirectories would have silently dropped every Apple source and linked a
    backend with no stages — a broken binary, not a build error (check 5);
  * src/backends/amd/README.md documented a directory tree (`include/turbo_ocr/
    amd/`, `src/`, `build.sh`) that had not existed for months.

Pure stdlib, no build required. Safe to run on any checkout.
"""

from __future__ import annotations

import os
import re
import sys

# --- The convention -------------------------------------------------------
#
# One directory per seam concern. Each name below maps to an interface in
# include/turbo_ocr/backend/, so the mapping from "which interface" to "which
# directory" is mechanical rather than a matter of taste. `support` and `probes`
# are the two that do not, and they carry explicit membership rules instead.
CONCERNS = {
    "backend": "the Backend implementation + the one BackendRegistrar (backend.h, backend_registry.h)",
    "engine": "IEngine — how this vendor runs a model (engine.h, engine_mode.h)",
    "memory": "IDeviceAllocator + the device image/buffer types (backend.h, image_view.h)",
    "queue": "DeviceQueue / DeviceEvent — how this vendor orders work (device_queue.h)",
    "stages": "IDetector/IRecognizer/IClassifier/ILayout (+ table/formula) (stages.h)",
    "support": "used by >=2 of the above, implements no seam interface itself",
    "probes": "standalone executables that exercise ONE interface off the pipeline",
}
KERNELS_RE = re.compile(r"^kernels_[a-z0-9]+$")  # IKernels + one kernel toolchain

VENDORS = ("amd", "apple", "cpu", "intel", "nvidia")
ROOT_FILES_OK = (".md",)  # a vendor root holds prose only

# amd has no CMake target at all (the root CMakeLists FATAL_ERRORs on it — it
# needs ROCm hardware to validate), so check 5 cannot apply to it.
NO_CMAKE_TARGET = ("amd",)

SRC_EXT = (".h", ".hpp", ".cpp", ".cc", ".mm", ".hip", ".cu", ".metal")
INC_RE = re.compile(r'^\s*#\s*(?:include|import)\s*"([^"]+)"', re.M)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    bk = os.path.relpath(here, repo)
    os.chdir(repo)
    bad: list[str] = []

    # index every file under src/backends by "<vendor>/<subdir>/<name>"
    owned: dict[str, str] = {}
    for v in VENDORS:
        for dp, dn, fns in os.walk(os.path.join(bk, v)):
            for f in fns:
                p = os.path.join(dp, f)
                owned[os.path.relpath(p, bk)] = p

    for v in VENDORS:
        vdir = os.path.join(bk, v)
        if not os.path.isdir(vdir):
            bad.append(f"{vdir}: vendor directory is missing")
            continue

        # 1. a vendor root holds prose only — no flat source dumps
        for f in sorted(os.listdir(vdir)):
            p = os.path.join(vdir, f)
            if os.path.isfile(p) and not f.endswith(ROOT_FILES_OK):
                bad.append(f"{p}: source file at the vendor root — move it into a concern "
                           f"directory ({', '.join(sorted(CONCERNS))} or kernels_<toolchain>)")

        # 2. every subdirectory is a named concern
        for d in sorted(os.listdir(vdir)):
            if not os.path.isdir(os.path.join(vdir, d)):
                continue
            if d not in CONCERNS and not KERNELS_RE.match(d):
                bad.append(f"{vdir}/{d}/: not a known concern. Add it to CONCERNS in "
                           f"{__file__.split(os.sep)[-1]} with a one-line reason to exist, "
                           f"or fold it into an existing one")

        # 3. every vendor documents itself
        if not os.path.isfile(os.path.join(vdir, "README.md")):
            bad.append(f"{vdir}/README.md: missing — every vendor states what IS and is NOT "
                       f"verified on real hardware")

    # 4. cross-directory includes use the vendor-rooted form off -Isrc/backends
    for rel, p in sorted(owned.items()):
        if not p.endswith(SRC_EXT):
            continue
        vendor, sub = rel.split("/")[0], os.path.dirname(rel)
        try:
            text = open(p, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        for inc in INC_RE.findall(text):
            if inc.startswith("turbo_ocr/"):
                # absolute-from-include/. Verified rather than trusted: the
                # include/ tree is reorganised from time to time, and a vendor
                # arm no CI job compiles (amd, nvidia) would not notice a miss.
                if not os.path.isfile(os.path.join("include", inc)):
                    bad.append(f'{p}: #include "{inc}" — no such header under include/')
                continue
            if inc in owned:
                continue  # already vendor-rooted
            sibling = os.path.normpath(os.path.join(os.path.dirname(p), inc))
            if not os.path.exists(sibling):
                continue  # a system/third-party header, not ours
            # Device-kernel sources are compiled by a vendor toolchain (hipcc,
            # nvcc, metal) that need not carry -Isrc/backends, so a same-directory
            # sibling include is the only portable form there. Everywhere else a
            # bare include silently breaks the moment the file moves.
            if KERNELS_RE.match(os.path.basename(sub)) and \
               os.path.dirname(sibling) == os.path.dirname(p):
                continue
            want = os.path.relpath(sibling, bk)
            bad.append(f'{p}: #include "{inc}" — use the vendor-rooted form '
                       f'"{want}" (every backend target puts -I{bk} on the path)')

    # 5. no source is orphaned from the build, and no CMake path is dangling
    cmake = open("CMakeLists.txt", encoding="utf-8").read()
    for m in sorted(set(re.findall(r"src/backends/[A-Za-z0-9_./]+", cmake))):
        m = m.rstrip(".")
        if "." in os.path.basename(m) and not os.path.exists(m):
            bad.append(f"CMakeLists.txt: names {m}, which does not exist")
    for rel, p in sorted(owned.items()):
        v = rel.split("/")[0]
        # .cu / .metal are included deliberately: a device source that is not
        # named in CMakeLists.txt is compiled by nothing and the vendor silently
        # loses its kernels. That is invisible to every other gate here, because
        # nothing on a non-CUDA machine can configure the block that would name
        # it. .hip is covered by AMD being in NO_CMAKE_TARGET already.
        if v in NO_CMAKE_TARGET or not p.endswith((".cpp", ".mm", ".cu", ".metal")):
            continue
        if p not in cmake:
            bad.append(f"{p}: compiled by nothing — not named in CMakeLists.txt. A backend "
                       f"that silently drops a source links and then misbehaves at runtime")

    # 6. the cross-vendor type-check manifest still points at real files
    man = "tools/syntax_shims/sources.txt"
    if os.path.isfile(man):
        for i, line in enumerate(open(man, encoding="utf-8"), 1):
            s = line.strip()
            if s and not s.startswith("#") and not os.path.exists(s):
                bad.append(f"{man}:{i}: {s} does not exist — the shim check silently stopped "
                           f"covering this file")

    # 7. every relative link in a README still points at something. These READMEs
    # cross-link the guide, each other, and src/backends/nvidia/kernels_cuda/README.md
    # (which holds the ruling on why the CUDA .cu files are NOT under nvidia/)
    # — and the rest of the repo moves independently of this directory.
    for rel, p in sorted(owned.items()):
        if not p.endswith(".md"):
            continue
        for link in re.findall(r"\]\((\.\.?/[^)#\s]+)\)", open(p, encoding="utf-8").read()):
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(p), link))):
                bad.append(f"{p}: broken relative link -> {link}")

    # 8. and the other direction: any doc ANYWHERE that links INTO src/backends/.
    # Deliberately keyed on the link naming `backends/` rather than widening to
    # all of src/**/*.md: a gate called layout_check that fails over an unrelated
    # broken link in some other directory is a gate people learn to ignore. Every
    # failure here is about this directory being pointed at.
    for dp, dn, fns in os.walk("."):
        dn[:] = [d for d in dn if d not in (".git", "third_party", "models", "node_modules")
                 and not d.startswith("build")]
        for f in fns:
            if not f.endswith(".md") or dp.startswith(os.path.join(".", bk)):
                continue  # inside src/backends: already covered by check 7
            p = os.path.join(dp, f)
            try:
                text = open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for link in re.findall(r"\]\((\.\.?/[^)#\s]+)\)", text):
                if "backends/" not in link:
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(dp, link))):
                    bad.append(f"{p}: broken link into {bk}/ -> {link}")

    for b in bad:
        print("FAIL " + b)
    n_dirs = sum(len([d for d in os.listdir(os.path.join(bk, v))
                      if os.path.isdir(os.path.join(bk, v, d))]) for v in VENDORS)
    print(f"{'FAILED' if bad else 'ok'}: {len(bad)} violation(s); "
          f"{len(VENDORS)} vendors, {n_dirs} concern directories, {len(owned)} files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
