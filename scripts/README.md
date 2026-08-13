# scripts/

Every directory here has one reason to exist. If a new script doesn't fit one
of these sentences, the sentence is wrong or the script belongs elsewhere —
don't widen a directory into a junk drawer.

| Directory | Reason to exist |
|---|---|
| `setup/` | One-time bootstrap that makes a fresh clone or image able to **build** turbo-ocr. |
| `models/fetch/` | Pull pre-built model assets from the GitHub Release. **The only model dir the build and the running container depend on.** |
| `models/onnx/` | Produce and rewrite the ONNX we ship: framework export, graph surgery, fp16/simplify passes. |
| `models/trt/` | Produce the NVIDIA deployment artifacts: quantise → build the TensorRT engine → gate it on latency. |
| `models/train/` | Models we train in-house, each as a full pipeline (synthesize data → train → ONNX → smoke-test). |
| `bench/` | Measure **speed**. |
| `eval/` | Measure **accuracy** (OmniDocBench scoring, dataset validation). |
| `experiments/` | A/B harness: run the pipeline under one changed env var and diff OmniDocBench scores against a stored baseline. |
| `dev/` | Tools a developer runs by hand against a local checkout. Nothing in the build, image, or CI references them. |
| `git-hooks/` | The `core.hooksPath` target. |

## Things that are deliberately not where you'd guess

- **`entrypoint.sh` sits at the top level, not in a directory.** Both
  Dockerfiles declare `ENTRYPOINT ["/app/scripts/entrypoint.sh"]` and
  `docs/getting-started/docker.md` documents that path for overriding, so it is the
  published image's public ABI. Moving it would silently break anyone whose
  compose/k8s config overrides the entrypoint. It is the container runtime's
  only script; a directory for it would be a directory of one.
- **`git-hooks/` keeps its name.** `install_hooks.sh` writes
  `core.hooksPath = scripts/git-hooks` into each clone's git config. Renaming
  the directory would leave every existing clone pointing at a path that no
  longer exists, and git runs *no* hooks in that case — the pre-commit gate
  would disappear silently rather than fail loudly.
- **`install_hooks.sh` is in `setup/`, not `git-hooks/`.** Git treats every
  file under `core.hooksPath` as a candidate hook, so that directory must
  contain hooks and nothing else.
- **`quantize_onnx_*.py` is in `models/trt/`, not `models/onnx/`,** even though
  it emits an ONNX file. It uses `nvidia-modelopt` and the QDQ graph it writes
  exists only to be fed to TensorRT.
- **`permute_script_id_onnx.py` is in `models/train/`, not `models/onnx/`.**
  It fixes the class ordering of the model we just trained, so it is a stage of
  the script-id pipeline; `models/onnx/patch_*.py` by contrast is surgery on
  third-party models we downloaded.
- **Filenames keep their `bench_`/`exp_`/`install_` prefixes** even though the
  directory now says the same thing. Renaming them would break documented
  user-facing commands and every reference outside this repo for a purely
  cosmetic gain.

## Scripts with no caller

`experiments/` has no caller outside itself: `exp_runner.sh` is the harness and
the eight `exp_*.sh` are its drivers, but nothing in the build, CI, or docs
invokes any of them. Most of `models/trt/` and `models/onnx/` is likewise
one-shot release tooling rather than anything the build runs. Both are retained
deliberately — `models/onnx/export_*.py` in particular is the provenance recipe
for assets that `models/fetch/` downloads, cited by comment from
`models/fetch/fetch_release_models.sh`.

## Paths are load-bearing

`docker/Dockerfile.{cpu,nvidia}`, root `CMakeLists.txt`, `.github/workflows/ci.yml`
and `entrypoint.sh` all reference scripts here by exact path, and most failures
show up at image-build or container-start time rather than at compile time. A
multi-source Docker `COPY` also *flattens* its sources into the destination
directory, so the `COPY` destination must name the same subdirectory as the
source. Move a script here and you must move its references in the same change.
