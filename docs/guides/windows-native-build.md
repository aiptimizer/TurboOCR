# Building TurboOCR natively on Windows

**Status: THE SERVER BUILDS AND SERVES.** As of 2026-08-10 `turboocr-server.exe`
builds under MSVC and answers real OCR requests natively — no WSL, no container:

```
POST /ocr/raw  tests/fixtures/images/png/receipt.png   ->  36 boxes
  Berqhotel / Grosse Scheidegg / 3818 Grindelwald / Rech.Nr. 4572 / ...
```

`turbo_ocr_tests.exe` passes all but two of the suite (570 of 572 as of 2026-08-12).
The two failures are in `test_font_style.cpp`, and they are now triaged — see
"Known: bold detection on Windows" below.

### Known: bold detection on Windows

Windows PDFium quantizes stroke width differently, and the absolute bold arm
cannot survive it. Running the repo's own `[.fontcal]` sweep on Windows gives:

| face | pt/scale | measured weight |
|---|---|---|
| Helvetica (regular) | 9 / 1.5 | **0.286** |
| Helvetica-Bold | 14 / 2.0 | 0.267 |
| Helvetica-Bold | 15 / 2.0 | **0.250** |

`kBoldAbsolute` is 0.258, chosen as the midpoint of the macOS/Linux gap where
regular tops out at 0.250 and bold bottoms out at 0.267. On Windows that gap does
not exist: regular reaches 0.286, *above* where bold bottoms out at 0.250. The
two classes overlap completely, so **no value of that constant separates them
here** — retuning it would only trade these two failures for different ones, and
would corrupt the calibration on the two platforms where it does hold.

The mechanism is quantization, exactly as the constant's own comment predicts:
`weight` is stem pixels over x-height pixels, so at these x-heights it lands on
coarse steps (3/12 = 0.250, 4/15 = 0.267, 2/7 = 0.286) and one pixel of
rasterization difference moves a face a whole step. `document_of` renders at
13/14/15pt, and it is only the 15pt line that falls through.

Scope: this affects the *absolute* arm, which exists solely for a document that
is bold **throughout** — where the median is itself bold and nothing stands out
against it. Ordinary mixed documents are decided by the relative arm
(`kBoldRelative` against the document's own median), which is unaffected because
it is scale- and platform-free by construction. Practically: bold-vs-regular in a
normal page is fine on Windows; a uniformly-bold page may report some lines as
regular. It changes styling in the editable text layer, never the recognized
text.

Stage 3 (the server) is no longer optional-and-untried: Drogon 1.9.13 and gRPC
1.81.1 install from vcpkg in ~19 minutes and the rest follows.

**PDF rasterization works too**, as of 2026-08-10:

```
unified server: PDF fully ENABLED (render + text + searchable)

POST /ocr/pdf?mode=ocr  academic_paper.pdf  ->  200, 15 pages, 104 words on page 1
```

This page previously said Windows could not rasterize a page and that those
routes answered 501. That was never a platform limit — it was a CMake gate. The
daemon renderer genuinely is Linux-bound (`pdf_daemon` uses
sigtimedwait/pipe2/inotify), but the *in-process* renderer beside it needs none
of that, and it was gated on `elseif(APPLE)` rather than on "no inotify". §5
below had already concluded "reuse as-is" by inspection; the two calls that
inspection missed, `mkdtemp` and `unlink`, are now `std::filesystem`, and the
branch is `else()`. macOS and Windows share the same renderer.

The port needed **no architectural change**: small platform fixes, listed in §3.
macOS (569/569) and Linux (620/620) pass with all of them in.

Originally written because the obvious route — build under WSL2 — cannot reach
the NPU. That part still holds; see §1.

!!! warning "The GPU is a separate axis"
    This page is about *building* on Windows. If you are pointing the build at a
    GPU, read
    [GPU providers fail loudly](../reference/configuration.md#gpu-providers-fail-loudly)
    first — CUDA, onnxruntime and cuDNN must agree on a version, and a mismatch
    runs on the CPU **silently** rather than failing.

---

## 1. Why WSL2 is a dead end for the NPU

Measured on the target box:

```
$ wsl -e bash -lc "ls /dev/accel"
ls: cannot access '/dev/accel': No such file or directory
```

`/dev/accel` is the device node the `intel_vpu` kernel driver creates and the one
OpenVINO's NPU plugin binds to. WSL2 does not pass the NPU through, so there is
nothing for the plugin to open. This is a platform limitation, not a
configuration problem — no driver install inside WSL2 changes it.

**The tree already knew this.** `src/backends/intel/SETUP.md:251` documents
`OV_DEVICE` as `CPU / GPU / NPU` with the note *"NPU unavailable under WSL2"*.
The measurement above is a confirmation, not a discovery — check SETUP.md before
re-deriving it.

The NPU hardware itself is present and healthy:

```
$ (Get-CimInstance Win32_Processor).Name      → Intel(R) Core(TM) Ultra 9 285K
$ Get-PnpDevice -FriendlyName '*AI Boost*'    → OK   Intel(R) AI Boost
```

So the NPU is reachable **only** from native Windows. Hence this document.

---

## 2. How big is the port, really

The instinct is "TurboOCR is Linux-only, this is a porting project." The code does
not support that. The entire non-vendor POSIX surface is **six files**, and CMake
already has the platform switch built.

### 2.1 CMake already degrades by feature, not by OS

`CMakeLists.txt:606-624` does not test for Linux. It tests for a *capability*:

```cmake
check_include_file_cxx("sys/inotify.h" TURBO_HAVE_INOTIFY)
if(TURBO_HAVE_INOTIFY)
    # Linux: fan pages out to fastpdf2png daemons, collect via inotify
    target_sources(turbo_ocr_cpu PRIVATE
        src/pdf/render/pdf_renderer.cpp
        src/pdf/render/pdf_daemon.cpp)
else()
    # every platform without inotify: in-process PDFium, no daemon pool,
    # none of the POSIX calls macOS and Windows lack
    target_sources(turbo_ocr_cpu PRIVATE
        src/pdf/render/pdf_renderer_inprocess.cpp)
```

There is already a working `else()` that produces a usable build with no page
renderer. The file's own comment (`CMakeLists.txt:601-605`) states the cost:

> "Nothing in the OCR pipeline, the text TUs, or page export depends on it — so
> its absence costs rasterizing scanned PDFs and nothing else."

**Historical:** a Windows build used to fall into that `else()` and lose
scanned-PDF rasterizing. It no longer does — the branch is now keyed on inotify
alone, so Windows compiles the in-process renderer that macOS uses.

### 2.2 The actual POSIX inventory

Counted with `grep -oE '\b(fork|execv|waitpid|inotify_\w+|mmap|pipe2?|poll|dlopen|pthread_\w+)\b'`:

| file | lines | POSIX calls | verdict |
|---|---|---|---|
| `src/pdf/render/pdf_renderer_inprocess.cpp` | 206 | `mkdtemp`, `unlink` (both now `std::filesystem`) | **reused** — this is the Windows renderer |
| `src/pdf/render/pdf_daemon.cpp` | 319 | fork ×13, pipe ×13, waitpid ×7, poll ×5, `_exit` ×5 | **skip** — Linux-only path |
| `src/pdf/render/pdf_renderer.cpp` | 343 | poll ×6, inotify ×8, mmap ×2 | **skip** — Linux-only path |
| `src/pdf/render/pdf_ppm.cpp` | 214 | mmap ×5, munmap ×2 | **port** — small |
| `src/analysis/vlm/crop_pool.cpp` | 208 | pipe ×3 | **port or exclude** |
| `src/analysis/vlm/crop_pool_transport.cpp` | 366 | pipe ×1 | **port or exclude** |

Two clarifications that matter, both easy to get wrong:

- **The in-process renderer had two POSIX calls, not zero** — this table said
  "none", and `mkdtemp`/`unlink` are why "reuse as-is" stalled for so long. Both
  are `std::filesystem` now. A grep for `pipe2` hits it,
  but the hit is in a comment at line 4 explaining that darwin *lacks* pipe2. The
  file is PDFium plus the C++ standard library. It was named `_darwin` and gated
  on `APPLE`, which is precisely how it stayed unavailable to Windows for so
  long; it is now `_inprocess`, because that is what it is.
- The vendor arms (`apple/`, `amd/`, `nvidia/`) also use POSIX headers, but none
  are compiled in a `cpu;intel` build. They are irrelevant here.

So the work is: **1 file reused, 2 files skipped, 3 small ports** — and the ports
are optional for a first build.

---

## 3. What the port actually needed

Fourteen fixes, all mechanical, none architectural. Recorded so the next
platform port knows the shape of the work — and so nobody re-derives them.

**Build flags (CMake, `if(MSVC)`)**

1. `-O3 -march= -Wall -Wextra -Wpedantic` → `/O2 /arch:AVX2 /W3 /permissive- /EHsc /utf-8 /wd4100`.
   MSVC rejects the GCC spellings outright: `D8021 invalid numeric argument`.
2. **`/Zc:preprocessor`** — load-bearing. MSVC's *legacy* preprocessor
   mis-tokenizes a raw string literal passed as a macro argument, so every
   `CHECK(j == R"(...\"...\"...)")` in the Catch2 suite failed with
   `invalid literal suffix 'oder'`. Not a code bug; a conformance flag.
3. **`NOMINMAX` + `WIN32_LEAN_AND_MEAN`** — `<windows.h>` arrives transitively
   (ORT, OpenCV, `<process.h>`) and its `min`/`max` *macros* eat every
   `std::numeric_limits<T>::min()`, Catch2 included.
4. `pthread` linked by literal name → `Threads::Threads`. There is no
   `pthread.lib`; threading is in the CRT.

**POSIX → Win32**

5. `gmtime_r` → `gmtime_s`, which takes its arguments in the **opposite order**
   and returns `errno_t`. Wrapped in `turbo_gmtime` (`base/log/logger.h`) — it
   cannot be a macro alias.
6. `setenv`/`unsetenv` in **9** test files → one force-included shim
   (`/FI tests/cpp/support/win_posix_shim.h`), so none of the 9 changed.
7. `std::aligned_alloc` → `_aligned_malloc`, which **must** pair with
   `_aligned_free`. The free path changed with it; the two cannot diverge.
8. `pid_t` in `pdf_renderer.h` → `intptr_t` on Windows. The daemon path is not
   compiled there, but the header is included everywhere, so the *member* still
   needs a type that exists.
9. `__restrict__` → `__restrict` via `TURBO_RESTRICT` (one non-vendor file).
10. `mmap` in `pdf_ppm.cpp` → read-into-buffer. The mapping is a zero-copy
    optimization whose second purpose — freeing `/dev/shm` the instant a page is
    claimed — has no Windows analogue, so a read loses nothing there.
11. The VLM **self-pipe** (`crop_pool.cpp`) — replaced with
    **`curl_multi_wakeup()`** on *every* platform, not `#ifdef`-ed around. The
    pipe was registered as an extra `curl_waitfd`, which needs
    `pipe()`/`fcntl()`/`read()` and, on Windows, a SOCKET rather than a pipe
    handle. libcurl has shipped a thread-safe wakeup for a blocked
    `curl_multi_poll` since 7.68, so the portability problem was self-inflicted.
    The fix **deletes** two fd members, the non-blocking setup, the drain
    lambda, the `curl_waitfd` plumbing and the close-on-shutdown — no platform
    branch remains in that path.

    The first attempt here was to skip the pipe on Windows and let the 200 ms
    poll timeout cover it. That compiled and passed, but it left the wake path
    different per platform and needed file-scope `write`/`close`/`read` shims
    shadowing POSIX. Deleting the mechanism beat conditionalising it.

**ONNX Runtime**

12. `ORTCHAR_T` is `wchar_t` on Windows, so all 6 `Ort::Session` sites needed
    UTF-8 → UTF-16 conversion (`include/turbo_ocr/onnx/ort_path.h`). UTF-8
    specifically: a model path under a directory with an umlaut would otherwise
    become "file not found".

**CMake gates that excluded a source but left its caller**

13. `BUILD_SERVER` (new, default ON) — `find_package(Drogon REQUIRED)` was a
    hard top-level requirement for something only `unified_server.cmake` links,
    and the proto codegen was unconditional in the CPU block.
14. The same bug shape twice: excluding the VLM sources left `vlm_factory.cpp`
    with unresolved symbols, and excluding `page_image_encoder.cpp` (no
    turbojpeg) left `region_extract.cpp` with one. **Gating a source is not the
    same as gating its callers** — check both.

## 3b. Staged plan (as executed)

Do **not** port the whole server first. The server drags in Drogon, gRPC and
protobuf; none of them are needed to answer "is the NPU worth targeting?"

### Stage 1 — `turbo_ocr_tests`, CPU only

Target: `turbo_ocr_tests` (`CMakeLists.txt:331`). It links no Drogon and no gRPC,
and `turbo_link_backends` (`CMakeLists.txt:1622`) links the backend registrars
into it, so the seam is exercised.

```
cmake -S . -B build-win -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON -DNATIVE_ARCH=OFF -DFETCH_MODELS=OFF
cmake --build build-win --target turbo_ocr_tests
ctest --test-dir build-win --output-on-failure
```

Expected friction, in the order you will hit it:
1. The renderer falls into the `else()` branch — expected, not an error.
2. `crop_pool*.cpp` fail on `<unistd.h>` / `pipe()`. They reach the test binary via
   `CMakeLists.txt:1624-1627`. Fastest unblock: drop those three `target_sources`
   lines for the first build. The VLM path is a remote-endpoint feature, unrelated
   to NPU measurement.
3. `pdf_ppm.cpp` fails on `<sys/mman.h>` if turbojpeg is found. It is already
   conditional on turbojpeg (`CMakeLists.txt:586-592`) — simply don't install
   turbojpeg and it is excluded.

### Stage 2 — add the Intel backend

```
cmake -S . -B build-win -G Ninja -DTURBO_BACKENDS="cpu;intel" -DOpenVINO_DIR=<ov>/runtime/cmake
```

`src/backends/intel/**` is toolchain-agnostic C++ over the seam types — the header
of `intel_stages.cpp` says so, and `tools/syntax_shims/check.sh` type-checks the
whole arm on a host with no OpenVINO at all. Compilation risk here is low; the
risk is in *linking* OpenVINO and in the NPU plugin finding the device.

Then measure with `OV_DEVICE=NPU` vs `GPU` vs `CPU` — read at
`src/backends/intel/engine/openvino_engine.cpp:60`, case-insensitive, anything
unrecognized falls back (`openvino_engine.h:75`). Keep `OV_PERF_HINT` at its
default `latency` (`openvino_engine.cpp:234`); SETUP.md records that `throughput`
starved the synchronous engine, 2.4 vs 5.5 img/s. The full runtime knob list is
`src/backends/intel/SETUP.md:249-256`.

### Stage 3 — the full server (optional)

Only if stages 1–2 justify it. Adds Drogon (`unified_server.cmake:98`) and
protobuf (`:100`). Drogon supports Windows via vcpkg but needs jsoncpp, zlib and
optionally c-ares. This is the step that genuinely deserves the phrase "porting
project"; stages 1–2 do not.

---

## 4. What must be installed

Verified present on the box (`Get-Command`):

| tool | state |
|---|---|
| `git` | `C:\Program Files\Git\cmd\git.exe` |
| `python` | `C:\Python314\python.exe` (also 3.11 present) |
| `ninja` | in the 3.11 Scripts dir |

Missing and required:

| item | why | how |
|---|---|---|
| **CMake** ≥ 3.24 | build system | `winget install Kitware.CMake` |
| **MSVC Build Tools 2022** | no `cl` on the box | `winget install Microsoft.VisualStudio.2022.BuildTools`, then add the *Desktop development with C++* workload. Build from a **x64 Native Tools Command Prompt** — otherwise `cl` is not on PATH. |
| **OpenVINO Runtime** (Windows) | the point of the exercise; ships the NPU plugin | archive from Intel, or `pip install openvino` for the Python side. Pass `-DOpenVINO_DIR=<root>/runtime/cmake`. |
| **ONNX Runtime** (Windows x64) | CPU inference path | `onnxruntime-win-x64-<ver>.zip`. Match the version the tree pins — `CMakeLists.txt:465` is `1.28.0`. |
| **OpenCV** (Windows x64) | image handling throughout | prebuilt release, or `vcpkg install opencv4:x64-windows`. |
| **PDFium** (Windows x64) | PDF text layer | `pdfium-binaries` release. Only needed for PDF input; skip for a first image-only build. |

Not needed for stages 1–2: Drogon, gRPC, protobuf, libturbojpeg.

**Intel NPU driver**: already present — Device Manager reports `Intel(R) AI Boost`
as `OK`. If OpenVINO later reports no NPU device, update it from Intel's NPU
driver page; the OpenVINO plugin requires a reasonably recent one.

---

## 5. Honest risk list

- **Nothing here is compile-verified on Windows.** The POSIX inventory and the
  CMake branch structure are verified; that the rest compiles under MSVC is not.
  MSVC is stricter than clang/gcc about two-phase lookup in templates, and this
  tree has never seen it.
- **`/W4` vs `-Wall`** — the tree's warning flags are gcc/clang spellings. MSVC
  will reject the flag strings themselves; expect to gate them on
  `if(NOT MSVC)`.
- **`__attribute__`, `#pragma GCC`, designated initializers, VLA-isms** — not
  audited. Any of these will surface as MSVC errors.
- **The 5090 is irrelevant to this document.** CUDA on native Windows is a
  separate arm (`TURBO_BACKENDS=nvidia`) with its own dependency set.
- **`.wslconfig` has an unknown key** (`wsl2.autoMemoryReclaim`, line 10) — WSL
  warns on every invocation. Harmless, but it pollutes any parsed output.

---

## 6. Recommendation

Stage 1 first, and treat it as a spike: the goal is to learn what MSVC rejects,
not to get a green build. If stage 1 completes in a session or two, stage 2 is
straightforward and the NPU numbers are within reach.

If the aim is only *"is the NPU worth targeting for OCR?"* — running the ONNX
models through OpenVINO Python on native Windows answers that in an afternoon
with no C++ build at all, and gives the same per-device comparison. That is the
cheaper experiment and it de-risks this one.
