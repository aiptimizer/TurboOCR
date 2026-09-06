# Vendored third-party libraries

Pinned versions of the libraries vendored under `third_party/`. Update this file
whenever a vendored library is bumped, and re-record the upstream URL and the
exact version/commit you pulled.

| Library | Vendored as | Pinned version / commit | Upstream |
|---|---|---|---|
| nlohmann/json | `nlohmann/json.hpp` (single header) | 3.12.0 | https://github.com/nlohmann/json/releases/tag/v3.12.0 |
| CLI11 | `cli11/CLI11.hpp` (single header) | 2.4.2 | https://github.com/CLIUtils/CLI11/releases/tag/v2.4.2 |
| ONNX Runtime | fetched into `onnxruntime/` (prebuilt) | 1.22.0 | https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0 |
| Catch2 | `catch2/catch_amalgamated.{hpp,cpp}` | 3.8.0 | https://github.com/catchorg/Catch2/releases/tag/v3.8.0 |
| Clipper | `clipper/clipper.{hpp,cpp}` | 6.4.2 | https://sourceforge.net/projects/polyclipping/ |
| Wuffs | `wuffs/wuffs-v0.4.c` (single file) | 0.4.0-alpha.9+3837.20240914 (rev `a14745aa458fd2b2785034efa04eab3c7b5b91e0`, 2024-09-14) | https://github.com/google/wuffs |
| simdutf | `simdutf/simdutf.{h,cpp}` (amalgamated) | 6.4.0 | https://github.com/simdutf/simdutf/releases/tag/v6.4.0 |
| PDFium (bblanchon binaries) | vendored x86-64 SDK in `pdfium/`; CMake (`cmake/TurboPdfium.cmake`) fetches the target's build into `build/_deps/` when the vendored one is another architecture; `scripts/install_pdfium.sh` does the same in place for Docker | `chromium/7857` (pinned in both; SHA256-verified per arch — x64 `2ad1fd42…b80541`, arm64 `0e24373e…068e35`) | https://github.com/bblanchon/pdfium-binaries |
| fastpdf2png (PDF page renderer) | built from source by CMake (`cmake/TurboFastpdf2png.cmake`) into `build/fastpdf2png`; `scripts/install_fastpdf2png.sh` builds it into `bin/` for Docker; the x86-64 copies committed in `bin/` are a fallback for x86-64 only | v2.0.10, commit `8358bdc1` (pinned in both) | https://github.com/aiptimizer/fastpdf2png |

## ONNX Runtime checksum pinning

The prebuilt ONNX Runtime tarball is fetched by `CMakeLists.txt` (CPU-only build)
when no system/vendored copy is found.

- **x86_64** (`onnxruntime-linux-x64-1.22.0.tgz`): pinned by default —
  SHA256 `8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3`.
- **aarch64** (`onnxruntime-linux-aarch64-1.22.0.tgz`): no verified default hash
  is shipped. The build fails with a hard error unless you pin the hash with
  `-DORT_SHA256=<sha256>`. To pin it: download the release tarball, run
  `sha256sum` on it, record the value here, and pass it via `-DORT_SHA256=`.
  `-DALLOW_UNVERIFIED_ORT=ON` bypasses the check for CI/dev only and emits a
  warning; never use it for release builds.
