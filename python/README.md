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
`[cpu]`/`[cuda]`/`[openvino]`/`[rocm]` extras pin the right engine wheel and
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
| Apple Silicon — Metal + Neural Engine | `pip install "turboocr[cpu]"` | `turboocr-engine-cpu` (its macOS arm64 build) |
| NVIDIA GPU, driver R525+ | `pip install "turboocr[cuda12]"` | `turboocr-engine-cuda12` |
| NVIDIA GPU, driver R580+ | `pip install "turboocr[cuda13]"` | `turboocr-engine-cuda13` |
| Intel CPU / iGPU / Arc / NPU | `pip install "turboocr[openvino]"` | `turboocr-engine-openvino` |
| AMD GPU (ROCm) | `pip install "turboocr[rocm]"` | `turboocr-engine-rocm` |

There is no separate Apple package: the macOS arm64 `turboocr-engine-cpu`
wheel is built with the Apple backend and bundles the Metal shader library.

Feature extras work on any engine wheel (and combine with the umbrella's
backend extras):

```bash
pip install "turboocr[cpu,pdf]"         # pypdfium2 + reportlab — read PDFs, write searchable PDFs
pip install "turboocr[cuda,pdf]"        # extras combine with any backend
pip install "turboocr-engine-cpu[all]"  # engine-only: pdf + rich + pandas
```

`[pdf]` reads PDFs and writes searchable ones, `[rich]` prettifies the
`turboocr doctor` panel, `[pandas]` enables `PageResult.to_pandas()`, `[all]`
is all three. `turboocr doctor` inspects the machine and prints the right
install line for it.

### Where to get them today

**The engine wheels are not on PyPI yet** — the `turboocr-engine-*` names are
unregistered there (`pip install turboocr` currently resolves to the published
0.3.0 client SDK, which this repo's `python-sdk/` continues). Two working
paths right now:

**Build from source** (always current, matches this host exactly). Use the
helper script — it builds *and* repairs the wheel:

```bash
# <variant> is one of: cpu | cuda | openvino | rocm
# (cpu builds turboocr-engine-cpu — also the Apple wheel on macOS arm64)
scripts/python/build_backend_wheel.sh cpu
pip install build-wheels/cpu/fixed/*.whl
```

A bare `pip wheel python/` is **not** installable anywhere but the machine that
built it — see [Building from source](#building-from-source) for why, and for
the manual `delocate` / `auditwheel` repair if you'd rather drive it by hand.

**Not published yet.** The `turboocr-engine-*` distributions have no PyPI
release until the first release run of `.github/workflows/wheels.yml`. Once
they land, the umbrella extras are the install path
(`pip install "turboocr[cpu]"`); because `4.0.0a1` is a pre-release, pip needs
`--pre` or an exact pin to select it:

```bash
pip install --pre "turboocr[cpu]"
pip install "turboocr[cpu]==4.0.0a1"    # equivalent, explicit
```

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
bundled — the repair step excludes those sonames — so they must come from the
host toolkit or the matching pip packages (`nvidia-cuda-runtime-cu12`,
`nvidia-cudnn-cu12`, `tensorrt-cu12==10.15.1.29`, and the `-cu13` equivalents).
They are not declared as dependencies of the engine wheel. It carries two
NVIDIA paths:

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
    print(region.label, region.score, region.box)   # text / table / figure / ...
```

PDF (needs the `pdf` extra — e.g. `pip install "turboocr[cpu,pdf]"`):

```python
doc = ocr.read_pdf("paper.pdf", dpi=150)
print(doc.to_markdown())
for page in doc:
    print(page.page, len(page.lines))
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
| `tiny` (default) / `small` / `medium` | PP-OCRv6 — Latin + Chinese + Japanese |
| `arabic` `eslav` `korean` `thai` `greek` | retained PP-OCRv5 script recognizers |

Weights resolve from an explicit `models_dir=`, `TURBO_OCR_MODELS_DIR`, a
`./models` folder, or an on-demand SHA256-verified download of just that tier
from the pinned TurboOCR GitHub release (cached under `~/.cache/turboocr`).

## Backends

`auto` (default, best no-build EP) · `turbo` (TensorRT, NVIDIA) · `cpu` ·
`cuda` · `openvino` · `directml` · `rocm`/`migraphx` · `coreml`.

On Apple Silicon, `auto` uses **CPU** on purpose — for these SVTR/DBNet models
the CoreML EP is typically slower than MLAS (and stumbles on dynamic shapes).
Pass `backend="coreml"` to force it.

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
