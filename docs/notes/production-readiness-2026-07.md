# TurboOCR — Production Readiness Review

*Reviewed 2026-07-31, on branch `arch-plan-readonly`, after the multi-backend rebuild.
Read-only review: no git operations, no remote machines touched.*

> **Tree state when this was verified.** The repository was reorganised
> (directory layout only, no logic changes) immediately after this review was
> written; paths below reflect the new layout. The reorganisation was verified
> behaviour-neutral, not assumed to be:
>
> | gate | result |
> |---|---|
> | CMake configure (`USE_CPU_ONLY=ON`) | clean — `Backends: cpu;apple` |
> | Build | `[184/184]` link OK, exit 0 |
> | Catch2 unit suite | **473 cases / 3387 assertions**, all pass — case count unchanged, so no test source was silently dropped |
> | ctest | **12/12**, incl. `backend_conformance` + the three `golden_apple_*` per-stage diffs |
> | FUNSD `cpu_tiny` | **0.857863** (was 0.8579) |
> | FUNSD `cpu_medium` | **0.925579** (was 0.9256) |
> | FUNSD `apple_tiny` | **0.857174** (was 0.8572) |
>
> Every finding in this document was established **before** the move and
> re-checked against the new paths afterwards. None of them are artifacts of it.
>
> Note the coverage limit this inherits: these gates exercise the **CPU/Apple
> unified** configure only. The NVIDIA configure (`USE_CPU_ONLY=OFF`) was not
> compiled — no machine here can — which is finding §2.4, and it applies to the
> reorganisation exactly as it applies to the rebuild.

---

## Verdict

| Deployment | Verdict |
|---|---|
| **CPU image** (`docker/Dockerfile (--target cpu)`) | **Ship-ready** behind a trusted gateway. Built + smoke-tested in CI, unit suite green. |
| **NVIDIA image** (`docker/Dockerfile (--target nvidia)`) | **Ship-ready but ungated** — this is the CUDA-native server (`src/cuda/pipeline/` + `src/cuda/server/`), unchanged and proven, but *nothing in CI compiles it* and none of the rebuild's test gates run against it. |
| **Apple (native)** | Ship-ready as a native binary. No container is possible (macOS); that is correct, but it is nowhere documented. |
| **Intel / OpenVINO** | **Not deployable.** Builds only from source on a box with OpenVINO installed. No image, no CI, no docs. |
| **AMD / ROCm** | **Not deployable, and cannot be built at all.** 2 337 LOC exist; CMake hard-fails if you ask for it. |

The single most load-bearing fact in this review: **the multi-backend rebuild does not ship on NVIDIA.**
`docker/Dockerfile (--target nvidia)` configures without `USE_CPU_ONLY`, which takes the branch at
`CMakeLists.txt:834` — `turboocr-server` built from `src/cuda/server/gpu_server_main.cpp` +
`stages_gpu.cpp` + the `*_gpu.cpp` route set, with `TURBO_BACKENDS` **empty**. The unified
server (`src/service/server/unified/server_main.cpp`, the backend seam, the capability registry,
`unified_ocr_pipeline.cpp`) is only reached through `src/service/server/unified/unified_server.cmake`, which is
included **only** when `TARGET turbo_ocr_cpu` exists — i.e. the CPU/Apple configure.

Two `add_executable(turboocr-server ...)` definitions, mutually exclusive by `USE_CPU_ONLY`.
Everything verified this month — 473 test cases, the FUNSD gates, the conformance/golden
harness — exercised the *left* one. NVIDIA production runs the *right* one.

---

## 1. Backend × deployment matrix

| Backend | CMake target | Dockerfile | Built in CI | Server binary it produces | Status |
|---|---|---|---|---|---|
| **cpu** | `turbo_ocr_backend_cpu` | `Dockerfile.cpu` | yes (build + unit + smoke + image) | unified `server_main.cpp` | ✅ |
| **nvidia (CUDA-native)** | `turbo_ocr_gpu` | `Dockerfile.nvidia` | **no** | `src/cuda/server/gpu_server_main.cpp` | ⚠️ ungated |
| **nvidia (seam backend)** | `turbo_ocr_backend_nvidia` | **none** | **no** | unified — but never co-built with the GPU image | ⚠️ CMake emits `WARNING: UNVERIFIED — never configured or compiled on CUDA hardware` |
| **apple** | `turbo_ocr_backend_apple` | n/a (macOS) | **no** (no macOS runner) | unified | ✅ locally verified, ungated |
| **intel** | `turbo_ocr_backend_intel` | **none** | **no** | unified | ⚠️ source-build only |
| **amd** | **none** | **none** | **no** | — | ❌ `FATAL_ERROR: Backend 'amd' has no CMake target yet` (`CMakeLists.txt:1041`) |

---

## 2. Docker — what is missing

### 2.1 Three of five backends have no image
Only `Dockerfile.cpu` and `Dockerfile.nvidia` exist. Missing, in order of cost-to-add:

- **ROCm/AMD** — blocked upstream: there is no CMake target, so an image cannot be written
  yet. `src/backends/amd/` is complete (HIP queue/allocator/kernels + MIGraphX engine) but
  compiled by nothing. Base would be `rocm/dev-ubuntu-24.04`, needs `hipcc` for
  `kernels_hip/*.hip` and MIGraphX; configure `-DTURBO_BACKENDS=cpu;amd`.
- **Intel/OpenVINO** — unblocked: the target exists and `find_package(OpenVINO REQUIRED)`
  works. Base `openvino/ubuntu24_runtime` + the dev headers; configure
  `-DUSE_CPU_ONLY=ON -DTURBO_BACKENDS=cpu;intel`; runtime `OV_DEVICE=CPU|GPU|NPU`. This is
  the cheapest missing image and the one with a real hardware story today.
- **Apple** — genuinely impossible to containerize (Metal/MPSGraph need macOS). The gap is
  documentation: nothing tells an operator that the Apple backend is native-only.

### 2.2 Both images are single-stage
`grep -c '^FROM'` = 1 for each. The production image therefore ships `g++`, `cmake`, `git`,
`wget`, every `-dev` package, the Drogon source checkout, the ONNX Runtime tarball tree, the
full source tree **and** the entire build tree. Consequences:

- Attack surface: a compiler and a package manager inside the runtime container.
- Size: `Dockerfile.cpu`'s own header claims *"Image size: ~500MB (vs ~10GB for GPU image)"*.
  That number is not achievable for a single-stage image containing `libopencv-dev` +
  `build-essential` + Drogon-from-source + `build_cpu/`. The comment is stale documentation
  on the file that would have to change to make it true.
- A `builder` → `runtime` split is the standard fix and would also let the runtime stage drop
  to a `-runtime` CUDA base for the GPU image.

### 2.3 The image builds and ships the test suite
`turbo_ocr_tests` is declared *"(Catch2, always built)"* at `CMakeLists.txt:301` — not
`EXCLUDE_FROM_ALL`. Both Dockerfiles run a bare `make -j$(nproc)`, so every image build
compiles all 473 Catch2 cases and bakes the binary in. The CPU configure additionally builds
`turbo_bench`, `turbo_conformance`, `turbo_golden` and `turbo_backend_probe`, none of which
are `EXCLUDE_FROM_ALL` either. The dev drivers (`plusm_selftest`, `grpc_bench`, …) *are*
correctly excluded, with a comment explaining exactly why — the same reasoning was never
applied to the test targets.

Related brittleness: the Dockerfiles `COPY tests/ tests/` while `.dockerignore` excludes
`tests/fixtures/`, `tests/integration/`, `tests/benchmark/`, `tests/regression/`. It works
today only because the excluded parts are data and Python.

### 2.4 The GPU image is never built by CI
`ci.yml` builds `Dockerfile.cpu` on PRs and tags. The GPU image is *"published from a
maintainer machine — too large/slow for the standard runner"*. That is a defensible call for
publishing, but it means **no automated check ever compiles the CUDA configure** — not the
image, not even a configure-only job. The mitigation that exists locally
(`tools/syntax_shims/check.sh`, which type-checks every source in `tools/syntax_shims/sources.txt` against stub SDK headers
on macOS) is **not wired into CI either**. After a refactor of this size, the NVIDIA
production path has no compile gate at all.

### 2.5 Container starts as root
Neither Dockerfile has a `USER` directive. `entrypoint.sh` runs as root (it needs to: `chown`
the TRT cache, start nginx) and drops to `ocr` via `gosu` for the server itself. The server
process is correctly unprivileged, but:

- A Kubernetes `securityContext.runAsNonRoot: true` — standard in hardened clusters, and the
  default under the `restricted` Pod Security Standard — will **refuse to start the pod**.
- nginx's master process stays root for the container's lifetime.

Either document the required `securityContext`, or add a non-root path (pre-chown at build
time, nginx as `ocr`, drop `gosu`) and set `USER ocr`.

### 2.6 Compose
`docker/compose.yaml` wires **only** the GPU image (7 replicas + a gRPC profile). There is no
CPU compose file. No `deploy.resources.limits` (memory/CPU) on any service — only GPU
*reservations*. The healthcheck uses `/health`, which is unconditional liveness;
`/health/ready` exists and actually probes the pipeline, and nothing in compose or the docs
points an orchestrator at it.

### 2.7 Model supply chain
`fetch_release_models.sh` verifies every asset against `SHA256SUMS.txt` — but that sums file
is fetched from the **same** release URL over the same channel, so the integrity chain roots
in "whatever `MODELS_RELEASE_URL` serves", not in the source tree. Two models
(`cls_x1_0.onnx`, `doc_ori.onnx`) do have hardcoded in-repo SHA256 pins; that is the right
pattern and it is applied to 2 of ~20 assets. Pinning the digest of `SHA256SUMS.txt` itself
in the script would close it for all of them.

---

## 3. Dead code

| What | Size | Evidence |
|---|---|---|
| `src/backends/amd/**` | **2 337 LOC** | Named in no CMake list; `TURBO_BACKENDS=amd` is a hard `FATAL_ERROR`. It compiles nowhere on any platform. |
| `tools/` Apple/Metal research probes (`mps_*.mm`, `mtl_*.mm`, `ane_probe.mm`, `*.metal`) | **~2 979 LOC** | 21 files, referenced by no CMake target and no script. Genuine one-off research drivers from the Apple bring-up. |
| `DeviceReadbackFn`'s `bool` failure contract | 1 branch | `src/pipeline/unified/vlm_factory.cpp:95` returns unconditional `true`; the `if (!rb.copy(...)) return {};` guard at `:153` is unreachable. |
| `--backend cpu` CoreML branch on macOS | 1 branch | `src/onnx/cpu_engine.cpp:222`'s "preserved for the env ctor" path is unreachable from the server — `set_ep_config()` always runs first. |
| markdown tag-safety guard, safe→unsafe arm | 1 branch | `src/document/markdown/markdown_export.cpp:455` — proven unreachable by a 1.3M-case fuzz; `clean_tag` does all the work. |

**Not dead, but duplicated:** the CUDA-native NVIDIA path (`src/cuda/pipeline/`, renamed from
`legacy/` on 2026-08-01 because that name was false — it is what ships) is a second, parallel implementation of
the same product — ~1 885 LOC of GPU-only server + routes, ~2 152 LOC of GPU-only pipeline,
against ~1 618 LOC for the unified equivalents. Both are live and both must be maintained
until the nvidia seam backend is brought up on real CUDA hardware. That is the correct
sequencing, but it is ~4 000 LOC of carrying cost with a deadline attached, and today only
one of the two halves is tested.

`src/server/compat/` is referenced as deleted by `unified_server.cmake` and is indeed gone —
the comment is accurate.

### 3.1 7.8 MB of committed native binaries that `.gitignore` believes it excludes

`bin/` holds five **tracked** entries totalling **7.8 MB**:

| file | size in git |
|---|---|
| `bin/libpdfium.so` | 7 551 304 B |
| `bin/libfastpdf2png.so.2.0.0` | 215 592 B |
| `bin/fastpdf2png` | 32 216 B |
| `bin/libfastpdf2png.so`, `…so.2` | symlinks |

`.gitignore:27-29` names **exactly these paths**, with the reason spelled out: *"build
artifacts produced by `scripts/setup/install_fastpdf2png.sh` at build time (Docker + native). Never
commit: they bake the build machine's absolute path into compiled strings."*

The rules are inert. `.gitignore` only affects **untracked** files; these were committed
before the rules were added, so git keeps tracking them and the ignore entries do nothing.
The repo states the policy and violates it, in the same file.

The stated harm is real, and asymmetric between build paths: both Dockerfiles run
`install_fastpdf2png.sh`, which **overwrites** these, so in Docker they are pure dead weight
in every clone. A **native** build that doesn't re-run the installer silently links a
`libfastpdf2png.so` compiled on somebody else's machine, with that machine's absolute paths
baked into it — which is precisely the failure mode the comment warns about.

This is the same defect class as the `.gitignore` `python/` entry fixed earlier this month
(an ignore rule whose effect did not match its author's intent), in the opposite direction.

**Remedy (requires a git write, so not performed here):**
`git rm --cached bin/fastpdf2png bin/libfastpdf2png.so bin/libfastpdf2png.so.2 bin/libfastpdf2png.so.2.0.0 bin/libpdfium.so`
— the working-tree files stay, the ignore rules then take effect, and the installer keeps
producing them. History rewriting is not warranted; stopping the bleeding is.

### 3.2 Loose files at the repo root

Everything else at root is either legitimately tracked or correctly ignored — `.DS_Store`,
`.ocr-demo` (20 MB), `Testing/`, `uploads/`, `models/` (2.4 GB) and all six `build-*` trees
(189 MB) are untracked **and** ignored, so they cost nothing. Two exceptions:

- ~~`makefillable-geometry.png` and `ocr-editable-viewer.png` at root~~ — RESOLVED: moved to
  `docs/assets/`. (They had since become *tracked*, which was worse than when this was
  written.)
- ~~`custom_models/` at root~~ — RESOLVED: moved to `python/custom_models/`, so `python/` is
  again the only Python tree.
- ~~`bin/` — 5 tracked binaries (7.4 MB)~~ — RESOLVED: untracked. `.gitignore:27-29` already
  named those exact files, and `scripts/setup/install_fastpdf2png.sh` installs them; ignore
  rules do not apply to already-tracked files, which is why the declaration had no effect.

---

## 4. Correctness risks carried into production

From the round-1 critical review (single-agent, read-only, verified against the tree). Ranked
by production impact:

1. **`run_pipelined` has no exception path** (`src/pipeline/unified/unified_ocr_pipeline.cpp:624-693`).
   An in-flight detection future is abandoned on any throw, and `/ocr/batch`'s per-image retry
   then re-enters the same single-slot device buffers. Live by default on Apple. Needs an RAII
   drain around the loop.
2. **The *shared* ONNX recognizer has no dropped-crop accounting** (`CpuPaddleRec`). Every
   per-vendor recognizer counts and logs dropped crops; the one implementation shared by all
   five vendors on `--engine-mode onnx` silently returns empty text. This inverts the
   project's own rule that generic policy lives in the shared implementation.
3. **`/ocr/batch` silently ignores JSON-body flags** that `/ocr` now honours — no
   `X-Ignored-Params`, no strict-mode 400. One argument to fix, in two files.
4. **`flag_text_degraded` overwrites the under-return warning**, replacing a correct diagnosis
   ("recognizer under-returned") with a false one ("all crops decoded blank"). Present in both
   pipelines, so it must be fixed in both.
5. `DetectionBatcher::detect` leaves a dangling stack `Slot*` in the queue on unwind — inert
   by default (no backend overrides `max_batch_size()`), live under `TURBO_DET_BATCH`.
6. `emit_covers` violates the build-before-cover rule: a refused patch leaves new type printed
   over un-erased scan and counts it as covered.

Full detail and fixes are in the round-1 report.

---

## 5. Operational surface

**Solid.** This part of the system reads as production-grade:

- Graceful shutdown with a configurable grace period, on both servers, draining HTTP then gRPC;
  `entrypoint.sh` relays SIGTERM/SIGINT to the server and quits nginx *after* the backend is
  down, preserving the real exit status for Kubernetes.
- `/health`, `/health/live`, `/health/ready` — and `/health/ready` correctly answers **200
  when the pool is merely full**, because a busy pod is not an unready pod. That is a subtle
  flap that most services get wrong.
- Dependency-free Prometheus exporter on `/metrics`, per-route histograms, atomics only.
- `X-Request-Id` propagated from nginx through a pre/post-handling advice.
- nginx slow-loris timeouts, body cap validated identically in the entrypoint and the binary
  so the proxy and the server cannot disagree.
- Entrypoint preflight fails fast and *actionably* on an unwritable TRT cache — probing **as
  the `ocr` user**, not as root.
- Pool exhaustion answers 503 `SERVER_BUSY` rather than queueing unboundedly.

**Gaps:**

- **No authentication or authorization anywhere — deliberately — but the decision is
  undocumented.** This is *not* an oversight. Commit `6d23a311` removed the in-server
  `API_AUTH_TOKEN` bearer mechanism and the refuse-to-boot public-bind gate on an explicit
  rationale: *"This server is vLLM-class — auth/TLS/exposure/versioning belong to the fronting
  gateway."* That posture is defensible and matches vLLM, TGI and Triton. The gap is that the
  rationale exists **only in a git commit message** and in one comment inside
  `nginx.conf.template`. `docs/reference/http.md`, `docs/getting-started/docker.md`, `docs/reference/monitoring.md`
  and both READMEs say nothing about the trust boundary the operator is required to supply —
  while `BIND_HOST` defaults to `0.0.0.0` and `/ocr`, `/ocr/pdf`, `/profile` and `/metrics`
  are all open. The fix is a documented deployment contract, **not** re-adding in-server auth.
- **The docs describe an nginx behaviour the config does not have.** `docs/getting-started/docker.md:142`,
  `docs/reference/http.md:766-769` and `docs/reference/monitoring.md:101` all state that the bundled
  `nginx.conf.template` *"remaps upstream **502 → 503**"*, citing `:36-47`. It does not, and
  never did at `HEAD`: lines 45-53 are `error_page 502 =502 @error_502;` →
  `return 502 '{"error":"Backend unavailable"}'` — 502 is **preserved**. The cited line range
  doesn't even contain the handlers. This is not cosmetic: 502 vs 503 drives retry policy. A
  client SDK or upstream load balancer configured from these docs to retry-with-backoff on 503
  will not retry the 502 the server actually emits. Either implement the remap or correct all
  three documents.
- **31 environment variables are read by the code and absent from `docs/reference/configuration.md`** —
  including **`TURBO_BACKEND`**, the knob that selects which vendor backend runs. Also all
  `TURBO_APPLE_*` (14), all `OV_*` (7), `TURBO_DET_BATCH*`, `TURBO_INTEL_DEBUG`,
  `REC_IMAGE_H`, `FFDETR_COREML`. The entire multi-backend feature is operationally
  undocumented.
- **`TURBO_POOL_SIZE` vs `PIPELINE_POOL_SIZE`**: the Intel backend invented its own pool-size
  variable (`src/backends/intel/backend/intel_backend.cpp:126`) alongside the existing,
  documented, compose-wired `PIPELINE_POOL_SIZE`. Two names for one concept, one of them
  undocumented.
- **Endpoint parity**: `/ocr/markdown`, `/infer` and `/ocr/stream` are registered only in the
  GPU build. `/capabilities` advertises them conditionally (correct — verified), but an
  operator migrating from the NVIDIA image to CPU/Apple/Intel loses three endpoints with no
  migration note. Conversely `/profile` and `/capabilities/backend` exist only on the unified
  server, and `/capabilities/backend` is missing from the advertised `endpoints` array.

---

## 6. CI gaps

`ci.yml` runs: CPU build + unit suite, a CPU server endpoint smoke test (gated on a repo
variable), report-only cppcheck, the CPU Docker image, and nightly ASAN/TSAN. What it does
**not** run:

- Any CUDA configure — not even configure-only. The NVIDIA production path is uncompiled by CI.
- Any Intel/OpenVINO configure.
- `tools/syntax_shims/check.sh` — the cross-vendor type-check that exists precisely to catch
  vendor-source breakage from a Mac. It has caught real breakage repeatedly this month and is
  invoked only by hand.
- The FUNSD accuracy gates — they register only when `-DTURBO_FUNSD_CACHE=<dir>` is set, which
  CI never sets. Accuracy is verified only on a developer machine.
- `turbo_conformance` / `turbo_golden` — same gate. (`backend_probe` *does* run, via `ctest`.)
- The pytest integration suite (27 files) and the language matrix.
- No macOS runner, so the Apple backend has no automated verification.

---

## 7. Recommended order of work

> **Status update 2026-08-01 — items 1 and part of 2 are DONE.**
>
> - **DONE — CUDA compile gate.** `.github/workflows/ci.yml` now has
>   `cuda-compile-gate` (nvcr TensorRT container, configures `USE_CPU_ONLY=OFF`
>   and links `turboocr-server`; no GPU needed to compile) plus
>   `vendor-syntax-check` running `tools/syntax_shims/check.sh` on every push.
>   Not hypothetical: it exists because a real link failure had already reached
>   the tree — `src/analysis/forms/field_model.cpp` was listed only in
>   `turbo_ocr_cpu` while `pdf_json.cpp` (in `TURBO_HTTP_COMMON_SRCS`, compiled
>   into *both* servers) called `forms::FieldModel::load/run` unguarded, so the
>   NVIDIA server could not link. Fixed at `CMakeLists.txt` in the
>   `turbo_ocr_gpu` source list.
> - **DONE — round-1 finding #5** (`flag_text_degraded` overwriting a correct
>   diagnosis with a false one). `set_stage_degraded` was forked three ways —
>   unified's private copy, `src/cuda/pipeline/ocr/ocr_pipeline_dispatch.cpp`,
>   and absent from the shared header — and both copies **assigned**. Now one
>   implementation in `include/turbo_ocr/pipeline/ocr_pipeline_detail.h`, and
>   every writer of `text_warning` **appends**. Pinned by 4 new tests.
> - **DONE — the seam is now a foundation.** `include/turbo_ocr/core/` holds the
>   transport-free vocabulary that was misfiled under `service/`. Upward edges
>   into `service/` from below: **14 → 2**, and both survivors are in
>   `src/backends/nvidia/backend/cuda_backend.cpp` (CUDA-only, deferred).
>   `include/turbo_ocr/backend/` now depends on `backend`, `common`, `core` only.
> - **STILL OPEN:** round-1 findings #1–#4 (the `run_pipelined` exception path,
>   the shared ONNX recognizer's missing drop accounting, `/ocr/batch` dropping
>   JSON-body flags), and the deployment-contract doc (item 3 below).


**Before shipping the rebuild to NVIDIA users**
1. Add a CUDA **configure-and-compile** job to CI (a container job on the TensorRT base needs
   no GPU to compile), and wire `tools/syntax_shims/check.sh` in as a cheap always-on gate.
2. Fix round-1 findings #1–#4 (the exception path, the shared recognizer's drop accounting,
   `/ocr/batch` body flags, the warning overwrite).
3. Publish the deployment contract that commit `6d23a311` decided but never wrote down: the
   service is gateway-fronted by design, ships no auth/TLS, and binds `0.0.0.0` by default.
   One section in `docs/getting-started/docker.md` + a line in `docs/reference/http.md`.
4. Correct the 502→503 claim in the three documents that make it (or implement the remap) —
   it is a retry-policy contract, and clients are entitled to rely on it.

**Before calling multi-backend a shipped feature**
5. Document `TURBO_BACKEND` and the 30 other undocumented variables; reconcile
   `TURBO_POOL_SIZE` with `PIPELINE_POOL_SIZE`.
6. Add `Dockerfile.intel` (OpenVINO) — the cheapest missing image with real hardware behind it.
7. Give AMD a CMake target, or move `src/backends/amd/` somewhere that names it as
   pending-hardware. 2 337 LOC that compile nowhere will rot.

**Image hygiene**
8. Multi-stage both Dockerfiles; mark `turbo_ocr_tests` and the four backend test binaries
   `EXCLUDE_FROM_ALL` (or build the image with an explicit `--target turboocr-server`); correct
   or delete the "~500MB" claim.
9. Set `USER ocr` (or document the required `securityContext`); add resource limits and a CPU
   compose file; point the orchestrator readiness probe at `/health/ready`.
10. Pin the digest of `SHA256SUMS.txt` in `fetch_release_models.sh`.
