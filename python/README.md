# TurboOCR engine for Python

The native engine bindings for [TurboOCR](https://github.com/aiptimizer/TurboOCR)
— the fast C++/CUDA document parser. This package (import name
`turboocr_engine`) is a thin [nanobind](https://nanobind.readthedocs.io)
wrapper over the **C++ engine**: detection, recognition, CTC decoding, warping,
and layout all run in native code (CUDA / TensorRT / Metal / AVX2), driven from
Python. It does **not** reimplement the pipeline in NumPy — you get the engine's
real speed.

Most people install it through the pure-Python
[`turboocr`](https://pypi.org/project/turboocr/) umbrella package, whose
`[cpu]`/`[apple]`/`[cuda]`/`[openvino]`/`[rocm]` extras pin the right engine wheel and
re-export the same API — `import turboocr` and `import turboocr_engine` then
both work.

- **Fast setup by default.** The default backend runs the ONNX graph on your GPU
  with **no TensorRT engine build** (CUDA on NVIDIA, CoreML/CPU on Apple,
  OpenVINO on Intel, DirectML on Windows, MLAS on CPU). TensorRT (peak throughput,
  slow first build) is one opt-in flag away: `backend="turbo"`.
- **One install panel.** `turboocr doctor` inspects your machine and tells you
  exactly which wheel to install for your GPU.
- **Layout, tables, PDF.** Detected layout regions, PDFium-rendered PDFs, and
  Markdown/JSON/TSV/hOCR exports.

## Install

One engine wheel per hardware target — **install exactly one**, they are
mutually exclusive (each bakes a different ONNX Runtime execution provider into
the same native extension). The umbrella extra on the left is the normal way
in; the engine name on the right is what it pins:

| Hardware | Install | Engine wheel |
|---|---|---|
| CPU — any x86-64 / ARM64 (**the default**) | `pip install "turboocr[cpu]"` | `turboocr-engine-cpu` |
| Apple Silicon — Metal + Neural Engine | `pip install "turboocr[apple]"` | `turboocr-engine-cpu` (its macOS arm64 build carries the full Apple backend) |
| NVIDIA GPU, driver R525+ | `pip install "turboocr[cuda12]"` | `turboocr-engine-cuda12` |
| NVIDIA GPU, driver R580+ | `pip install "turboocr[cuda13]"` | `turboocr-engine-cuda13` |
| Intel CPU / iGPU / Arc / NPU | `pip install "turboocr[openvino]"` | `turboocr-engine-openvino` |
| AMD GPU (ROCm) | `pip install "turboocr[rocm]"` | `turboocr-engine-rocm` |

There is no separate Apple package: the macOS arm64 `turboocr-engine-cpu`
wheel is built with the Apple backend and bundles the Metal shader library.

PDF support (read PDFs, write searchable PDFs) is **built in** — no extra
needed. Feature extras work on any engine wheel (and combine with the
umbrella's backend extras):

```bash
pip install "turboocr[cuda,rich]"       # extras combine with any backend
pip install "turboocr-engine-cpu[all]"  # engine-only: rich + pandas
```

`[rich]` prettifies the `turboocr doctor` panel, `[pandas]` enables
`PageResult.to_pandas()`, `[all]` is both. (`[pdf]` still resolves for
backwards compatibility — it is now a subset of the base install.)
`turboocr doctor` inspects the machine and prints the right install line for
it.

### Where to get them today

**On PyPI: cpu and openvino.** `turboocr-engine-cpu` (Linux, macOS with the
Apple backend, Windows), `turboocr-engine-openvino` and the `turboocr`
umbrella are published; because `4.0.0a6` is a pre-release, pip needs `--pre`
or an exact pin to select it:

```bash
pip install --pre "turboocr[cpu]"       # or [apple] | [openvino]
pip install "turboocr[cpu]==4.0.0a6"    # equivalent, explicit
```

**Not on PyPI yet:** the NVIDIA wheels (`-cuda12` / `-cuda13`) — built and
verified, awaiting PyPI's file-size approval — and `-rocm` (deliberately
unpublished until hardware-validated). For those, build from source (always
current, matches this host exactly). Use the helper script — it builds *and*
repairs the wheel:

```bash
# <variant> is one of: cpu | cuda12 | cuda13 | openvino | rocm
# (cpu builds turboocr-engine-cpu — also the Apple wheel on macOS arm64)
scripts/python/build_backend_wheel.sh cpu
pip install build-wheels/cpu/fixed/*.whl
```

A bare `pip wheel python/` is **not** installable anywhere but the machine that
built it — see [Building from source](#building-from-source) for why, and for
the manual `delocate` / `auditwheel` repair if you'd rather drive it by hand.

After installing, `turboocr doctor` reports the version and the provider it
actually selected.

## Usage

```python
import turboocr_engine as turboocr   # or just `import turboocr` with the umbrella installed

ocr = turboocr.OCR()                       # tiny model, backend="auto" (fast-setup)
page = ocr.read("document.png")
print(page.text)                           # reading-order text
for line in page:                          # per-line detail
    print(line.confidence, line.text, line.box)

page.to_json(indent=2)                     # structured output
page.save_overlay("boxes.png")             # draw detected boxes
page.to_tsv(); page.to_hocr()              # exports
page.filter(min_confidence=0.9)            # keep confident lines
```

Language and accuracy tier:

```python
turboocr.OCR(lang="en", tier="medium")     # Latin/CJK, most accurate tier
turboocr.OCR(lang="ko")                    # Korean script recognizer
turboocr.OCR(tier="small")                 # tiny | small | medium
turboocr.OCR("arabic")                     # explicit model always wins
```

Backend selection:

```python
turboocr.OCR(backend="cuda")               # NVIDIA, ONNX on GPU (no build)
turboocr.OCR(backend="turbo")              # NVIDIA TensorRT (opt-in, cached)
turboocr.OCR(backend="cpu")                # force CPU
turboocr.OCR(backend="openvino", device="NPU")
```

**NVIDIA (`turboocr-engine-cuda12` / `-cuda13`) — what the first run costs.** The wheel needs an
NVIDIA **driver** at runtime. The CUDA, cuDNN and TensorRT runtimes are **not**
bundled — the repair step excludes those sonames — so they come from the host
toolkit, or from the matching pip packages, which the wheel **finds
automatically** (`pip install tensorrt-cu12-libs==10.15.1.29
nvidia-cuda-runtime-cu12 nvidia-nvjpeg-cu12`, or the `-cu13` equivalents; no
`LD_LIBRARY_PATH` needed). They are not declared as dependencies of the engine
wheel. It carries two NVIDIA paths:

- `backend="cuda"` runs the ONNX graph on the CUDA execution provider.
  **Nothing is compiled — start-up is instant**, and steady-state speed is
  good. Pick it when you can't pay a one-time engine build.
- `backend="turbo"` (aliases `"tensorrt"`, `"trt"`) — **what `backend="auto"`,
  the default, resolves to on these wheels** — uses TensorRT for peak
  throughput. The **first** run builds an engine specialised to your GPU,
  driver and model — roughly ~90 s on an RTX 5090, up to an hour on older cards
  (`TRT_OPT_LEVEL=3` cuts it 3–5x). That cost is paid **once**: the engine is
  written to `TRT_ENGINE_CACHE` (default `~/.cache/turbo-ocr`) and later runs
  load it in a fraction of a second. Keep that directory persistent — mount it
  as a volume in containers. Changing GPU, driver, TensorRT or model correctly
  invalidates the cache and triggers one more build.

Both the **server**'s native NVIDIA arm and these Python wheels default to
TensorRT: the nvidia backend is compiled in, so `backend="auto"` resolves to
`"turbo"` and the first run pays the engine build. Pass `backend="cuda"` for an
instant first start on the CUDA execution provider.

Layout regions:

```python
ocr = turboocr.OCR(layout=True)            # loads PP-DocLayoutV3
page = ocr.read("paper.png", layout=True)
for region in page.layout:
    print(region.label, region.confidence, region.box)   # text / table / figure / ...
```

PDF — built in, no extra needed:

```python
doc = ocr.read_pdf("paper.pdf", dpi=150)
print(doc.to_markdown())
for page in doc:
    print(page.page, len(page.lines))
```

Pages fan out across the replica pool (`OCR(replicas=N)`) — on accelerator
backends a multi-page document reads ~2.4× faster at `replicas=3`, with
byte-identical output. Stream pages instead of waiting for the whole
document, or go async — every read method has an `async` twin:

```python
for page in ocr.read_pdf_stream("paper.pdf"):      # each page as soon as it's ready
    print(page.page, page.text[:40])               # ordered=False -> completion order

page = await ocr.aread("scan.png")                 # aread / aread_batch / aread_pdf
async for page in ocr.aread_pdf_stream("paper.pdf"):
    ...
```

Inputs: paths, raw bytes, NumPy arrays (BGR), and PIL images.

## CLI

```bash
turboocr doctor                          # install panel for your hardware
turboocr models                          # list models
turboocr ocr image.png --lang en --tier medium
turboocr ocr *.png -f tsv -o out.tsv     # globs, formats, output file
turboocr ocr scan.png --overlay boxes.png --layout
turboocr pdf doc.pdf -f markdown -o doc.md
```

Formats: `text` (default), `json`, `markdown`, `tsv`, `hocr`.

## Models

| name | tier / script |
|---|---|
| `tiny` (default) / `small` / `medium` | PP-OCRv6 — Latin + Chinese (Japanese needs `small`/`medium` — tiny omits kana) |
| `arabic` `eslav` `korean` `thai` `greek` | retained PP-OCRv5 script recognizers |

Weights resolve from an explicit `models_dir=`, `TURBO_OCR_MODELS_DIR`, a
`./models` folder, or an on-demand SHA256-verified download of just that tier
from the pinned TurboOCR GitHub release (cached under `~/.cache/turboocr`).

## Backends

`auto` (the wheel's default — resolves to `turbo` on the NVIDIA wheels,
first run builds a cached engine; CPU elsewhere) · `turbo` (TensorRT,
NVIDIA) · `apple` (native Metal/MPSGraph — the ~5x fast path on Apple
silicon) · `cpu` ·
`cuda` · `openvino` · `directml` · `rocm`/`migraphx` · `coreml`.

On Apple Silicon, `auto` uses **CPU** on purpose — for these SVTR/DBNet
models the CoreML EP is typically slower than MLAS (and stumbles on dynamic
shapes). The fast path is `backend="apple"`: the native Metal/MPSGraph
backend with the ANE lane (its export bundle downloads automatically), the
measured ~5x configuration. `backend="coreml"` forces the (slower) CoreML
EP if you specifically want it.

## Building from source

For a **development install** — builds in place, only ever used on this machine:

```bash
cmake -S . -B build -DBUILD_PYTHON=ON -DUSE_CPU_ONLY=ON \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
cmake --build build --target _turboocr
pip install ./python
```

For a **distributable wheel**, use the helper — it builds and then repairs:

```bash
scripts/python/build_backend_wheel.sh <cpu|cuda|openvino|rocm>
pip install build-wheels/<variant>/fixed/*.whl
```

### Why the repair step is mandatory

`pip wheel python/` alone produces a wheel that works **only on the machine that
built it**: it bundles zero shared libraries, and its RPATH points into your dev
checkout, so it breaks the moment you move it elsewhere or delete `build/`.
Making it self-contained is a separate *repair* pass that vendors the ~114
dylibs/shared objects it links (OpenCV, ONNX Runtime, PDFium, …). Never hand
someone an unrepaired wheel.

The helper script does this for you. By hand it is:

```bash
# macOS
pip wheel python/ --no-deps -w dist/
pip install delocate && delocate-wheel -w dist/fixed -v dist/turboocr_engine_*.whl
pip install dist/fixed/turboocr_engine_*.whl

# Linux
pip wheel python/ --no-deps -w dist/
pip install auditwheel && auditwheel repair -w dist/fixed dist/turboocr_engine_*.whl
pip install dist/fixed/turboocr_engine_*.whl
```

The `cuda` and `rocm` variants need more than the plain command above: the
driver/toolkit sonames (`libcuda.so.1`, `libcudart`, `libcudnn`, `libnvinfer`,
`libamdhip64`, `libmigraphx`, …) must be **excluded** so they resolve from the
host, as `onnxruntime-gpu` does; and CUDA needs a second pass, because ORT
`dlopen`s its provider libraries and `auditwheel` — which only follows
`DT_NEEDED` — drops them silently. Both are already encoded in
`scripts/python/build_backend_wheel.sh`; prefer it over reproducing them.

## License

MIT (same as TurboOCR). The wheel bundles ONNX Runtime, OpenCV and PDFium under
their respective licenses.
