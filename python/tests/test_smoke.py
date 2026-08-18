"""Smoke tests for the TurboOCR Python bindings (native C++ engine).

Integration tests run only when the native `_turboocr` extension AND the model
weights are available; otherwise they skip, so this file is safe to collect
anywhere.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import turboocr_engine as turboocr
from turboocr_engine import catalog, native


def _models_dir():
    for c in (os.environ.get("TURBO_OCR_MODELS_DIR"), os.path.join(os.getcwd(), "models")):
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "det_tiny.onnx")):
            return c
    return None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = _models_dir() or (
    os.path.join(REPO_ROOT, "models")
    if os.path.exists(os.path.join(REPO_ROOT, "models", "det_tiny.onnx"))
    else None
)
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "images", "png", "receipt.png")


def _has_native() -> bool:
    try:
        native.load_native()
        return True
    except Exception:
        return False


needs_native = pytest.mark.skipif(not _has_native(), reason="_turboocr extension not built")
needs_models = pytest.mark.skipif(MODELS is None, reason="no local models")
needs_fixture = pytest.mark.skipif(not os.path.exists(FIXTURE), reason="no fixture image")


# -- pure-python units (no native / models needed) ------------------------
def test_catalog_resolves_aliases():
    assert catalog.resolve_model("tiny").name == "tiny"
    assert catalog.resolve_model("fast").name == "tiny"
    assert catalog.resolve_model("best").name == "medium"
    assert catalog.resolve_model("ru").name == "eslav"
    with pytest.raises(ValueError):
        catalog.resolve_model("nope")


def test_configure_backend_apple_auto_disables_coreml(monkeypatch):
    monkeypatch.setattr(native, "is_apple_silicon", lambda: True)
    monkeypatch.delenv("DISABLE_COREML", raising=False)
    resolved, summary = native.configure_backend("auto")
    assert resolved == "cpu"
    assert os.environ.get("DISABLE_COREML") == "1"
    resolved2, _ = native.configure_backend("coreml")
    assert resolved2 == "coreml"
    assert "DISABLE_COREML" not in os.environ


def test_configure_backend_non_apple_ort_ep(monkeypatch):
    monkeypatch.setattr(native, "is_apple_silicon", lambda: False)
    monkeypatch.delenv("ORT_EP", raising=False)
    _, _ = native.configure_backend("openvino", device="NPU")
    assert os.environ.get("ORT_EP") == "openvino"
    assert os.environ.get("OPENVINO_DEVICE") == "NPU"
    native.configure_backend("cpu")
    assert os.environ.get("ORT_EP") == "cpu"


def test_auto_resolves_to_turbo_on_the_nvidia_build(monkeypatch, capsys):
    """The turboocr-engine-cuda12/13 default. That wheel compiles in the nvidia seam
    backend, and the user decision (2026-08-12) is that the DEFAULT there is the
    native TensorRT engine — `auto` must reach it without anyone passing
    backend='turbo'. Faked at native_backends() because that IS the signal
    resolve_engine reads: a build carrying the nvidia backend is the NVIDIA
    wheel."""
    monkeypatch.setattr(native, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(native, "native_backends", lambda: ["cpu", "nvidia"])
    monkeypatch.setattr(native, "_trt_note_shown", False)
    monkeypatch.delenv("ORT_EP", raising=False)

    assert native.resolve_engine("auto") == "nvidia"
    assert native.resolve_engine("") == "nvidia"  # OCR(backend=None) path
    resolved, summary = native.configure_backend("auto")
    assert resolved == "nvidia"
    # The one-time engine build is stated, not discovered by waiting.
    assert "first run builds" in summary and "one-time" in summary
    # ...and warned about on stderr, because the caller never asked for TRT.
    assert "TensorRT" in capsys.readouterr().err

    # backend='cuda' stays the instant-start fallback: still the ORT path.
    assert native.resolve_engine("cuda") == "cpu"
    assert native.configure_backend("cuda")[0] == "cuda"
    assert os.environ.get("ORT_EP") == "cuda"


def test_auto_stays_on_the_ort_path_without_a_vendor_seam(monkeypatch):
    """The CPU wheel's default must not move: no nvidia backend compiled in
    means `auto` is still plain CPU, and 'fast'/'onnx' explicitly ask for the
    no-build path even where a seam backend exists."""
    monkeypatch.setattr(native, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(native, "native_backends", lambda: ["cpu"])
    assert native.resolve_engine("auto") == "cpu"
    assert native.configure_backend("auto")[0] == "auto"

    monkeypatch.setattr(native, "native_backends", lambda: ["cpu", "nvidia"])
    for name in ("fast", "onnx"):
        assert native.resolve_engine(name) == "cpu", name
        assert native.configure_backend(name)[0] == "auto", name


def test_no_phantom_directml_wheel_is_advertised():
    """There are exactly FOUR distributions. doctor printed
    `pip install turboocr-directml` on every run for a wheel CI never builds."""
    from turboocr_engine.providers import INSTALL_MATRIX

    for b in INSTALL_MATRIX:
        for cmd in b.pip:
            assert "directml" not in cmd, b.key
            assert _named_engine_package(cmd.split()[-1]) in _PACKAGES, (b.key, cmd)
        # A row with no wheel must say so rather than carry a silent empty pip.
        assert bool(b.pip) == b.packaged, b.key
        if not b.packaged:
            assert "NOT PACKAGED" in b.note, b.key


def test_doctor_never_prints_an_uninstallable_package(capsys):
    from turboocr_engine.doctor import doctor

    doctor(_hw("cpu"), plain=True)
    out = capsys.readouterr().out
    assert "directml" not in out.replace("DirectML", "").replace("DmlExecutionProvider", "")
    assert "not packaged — build from source" in out  # the DirectML row
    # Every printed `pip install X` names one of the four real engine wheels
    # (directly, or through the matching `turboocr[<variant>]` umbrella extra).
    for ln in out.splitlines():
        if "pip install" in ln:
            assert _named_engine_package(ln.strip().split()[-1]) in _PACKAGES, ln


def test_doctor_says_to_install_exactly_one_wheel(capsys):
    """Four pip lines in one panel read like a menu you can combine; they all
    provide the same import package, so a second install overwrites the first."""
    from turboocr_engine.doctor import doctor

    report = doctor(_hw("nvidia"), plain=True)
    out = capsys.readouterr().out
    assert "exactly ONE turboocr-engine-* wheel" in out
    assert "exactly ONE turboocr-engine-* wheel" in report["recommended"]["exclusive"]


def test_rocm_note_describes_the_turboocr_wheel_not_upstream():
    """The note used to describe AMD's own onnxruntime-rocm wheels (their index,
    py3.10/3.12) while sitting on the turboocr-engine-rocm row — a different artifact
    with different rules. CI builds ONE cp312 abi3 wheel."""
    from turboocr_engine.providers import _BACKEND_BY_KEY

    note = _BACKEND_BY_KEY["rocm"].note
    assert "cp312" in note and "abi3" in note
    assert "3.10" not in note


def test_doctor_report_builds():
    rep = turboocr.build_report()
    assert "recommended" in rep and rep["recommended"]["install"]
    rec = rep["recommended"]
    assert rec["package"] in _PACKAGES
    variant = rec["package"].removeprefix("turboocr-engine-")
    assert rec["install"] == [f'pip install "turboocr[{variant}]"', f"pip install {rec['package']}"]


# -- the wheel doctor tells you to install --------------------------------
# Four separate, mutually-exclusive distributions. Getting this mapping wrong
# sends an NVIDIA user to the CPU wheel and they never find out why it's slow,
# so pin the vendor -> package NAME directly (pure logic: no native module, no
# network, no hardware).
_PACKAGES = {
    "turboocr-engine-cpu",
    "turboocr-engine-cuda12",
    "turboocr-engine-cuda13",
    "turboocr-engine-openvino",
    "turboocr-engine-rocm",
}

def _named_engine_package(spec: str) -> str:
    """The engine distribution a printed install spec resolves to.

    Accepts both spellings the panel now emits: the direct
    `turboocr-engine-<variant>` name, and the umbrella-extra form
    `"turboocr[<variant>]"` (with or without the shell quotes)."""
    spec = spec.strip().strip('"')
    if spec.startswith("turboocr[") and spec.endswith("]"):
        return "turboocr-engine-" + spec[len("turboocr[") : -1]
    return spec


_EXPECTED_PACKAGE = {
    # doctor picks the CUDA major from the driver; with no driver info
    # detected in a test environment it names the safe cuda12 wheel.
    "nvidia": "turboocr-engine-cuda12",
    "amd": "turboocr-engine-rocm",
    "intel": "turboocr-engine-openvino",
    "apple": "turboocr-engine-cpu",
    "cpu": "turboocr-engine-cpu",
}


def _hw(vendor: str):
    """A HardwareInfo whose .vendor property is `vendor` (it is derived from
    the flags, so set the flags rather than faking the property)."""
    from turboocr_engine.providers import HardwareInfo

    return HardwareInfo(
        os="Linux",
        machine="x86_64",
        has_nvidia=vendor == "nvidia",
        has_amd=vendor == "amd",
        has_intel_gpu=vendor == "intel",
        is_apple_silicon=vendor == "apple",
    )


@pytest.mark.parametrize("vendor,package", sorted(_EXPECTED_PACKAGE.items()))
def test_recommend_maps_vendor_to_package(vendor, package):
    from turboocr_engine.doctor import recommend

    hw = _hw(vendor)
    assert hw.vendor == vendor  # the fixture really is that vendor
    rec = recommend(hw)
    assert rec.package == package
    variant = package.removeprefix("turboocr-engine-")
    assert rec.install[0] == f'pip install "turboocr[{variant}]"'
    assert rec.install[1] == f"pip install {package}"
    assert rec.reason  # every branch explains itself


def test_recommend_unknown_vendor_falls_back_to_cpu_wheel():
    from turboocr_engine.doctor import recommend
    from turboocr_engine.providers import HardwareInfo

    class _Odd(HardwareInfo):
        @property
        def vendor(self) -> str:
            return "s390x-mainframe"

    rec = recommend(_Odd(os="Linux", machine="s390x"))
    assert rec.package == "turboocr-engine-cpu" and rec.backend.key == "cpu"


def test_recommended_backend_row_matches_the_package():
    """The starred row of the install matrix and the named package must agree —
    they are rendered next to each other in the panel."""
    from turboocr_engine.doctor import recommend

    for vendor, package in _EXPECTED_PACKAGE.items():
        rec = recommend(_hw(vendor))
        variant = package.removeprefix("turboocr-engine-")
        assert f"turboocr[{variant}]" in rec.backend.pip[0], (
            vendor,
            rec.backend.key,
            rec.backend.pip,
        )


def test_doctor_plain_output_names_the_package(capsys):
    from turboocr_engine.doctor import doctor

    report = doctor(_hw("nvidia"), plain=True)
    out = capsys.readouterr().out
    assert 'pip install "turboocr[cuda12]"' in out
    assert "pip install turboocr-engine-cuda12" in out
    assert "turboocr doctor" in out  # tells you how to re-check after installing
    assert report["recommended"]["package"] == "turboocr-engine-cuda12"


# -- integration (need native + models) -----------------------------------
@needs_native
@needs_models
@needs_fixture
def test_ocr_reads_receipt():
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS)
    res = ocr.read(FIXTURE)
    assert len(res.lines) > 5
    joined = res.text.lower()
    assert "scheidegg" in joined or "grindelwald" in joined
    d = res.to_dict()
    assert d["results"][0]["confidence"] >= 0.5
    assert len(d["results"][0]["box"]) == 4


@needs_native
@needs_models
def test_ocr_handles_blank_image():
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS)
    res = ocr.read(np.full((200, 300, 3), 255, np.uint8))
    assert res.lines == []


@needs_native
@needs_fixture
@needs_models
def test_result_ergonomics_and_export():
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS)
    res = ocr.read(FIXTURE)
    assert res[0] is res.lines[0]
    assert res.results is res.lines
    assert res.image is not None
    overlay = res.draw()
    assert overlay.shape == res.image.shape
    assert res.lines[0].crop(res.image).ndim == 3
    assert res.to_tsv().splitlines()[0].startswith("index")
    assert "ocr_line" in res.to_hocr()
    hi = res.filter(min_confidence=0.99)
    assert all(l.confidence >= 0.99 for l in hi.lines)


@needs_native
@needs_models
def test_layout_regions():
    layout_onnx = os.path.join(MODELS, "layout", "layout.onnx")
    if not os.path.exists(layout_onnx):
        pytest.skip("no layout model")
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS, layout=True)
    res = ocr.read(FIXTURE, layout=True)
    assert isinstance(res.layout, list)
    if res.layout:
        assert res.layout[0].label and len(res.layout[0].box) == 4


# -- the shared request-option gate reaches Python ------------------------
# Python is a transport client like HTTP and gRPC, and runs the SAME
# parse_options_core (include/turbo_ocr/service/validation/options_core.h) via
# the python_options.h adapter. Before that adapter the binding took four plain
# bools straight into the pipeline: an unloaded capability was silently skipped
# and an impossible combination silently produced a full OCR result. These pin
# the rejections, message text included, since the whole point is that the
# strings are the server's and not a Python-side copy.
@needs_native
@needs_models
def test_layout_only_request_is_rejected_with_the_server_message():
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS)
    with pytest.raises(ValueError) as e:
        ocr.read(np.full((200, 300, 3), 255, np.uint8), text=False)
    # Verbatim from options_core.h — the same body an HTTP client gets back.
    assert "text=0" in str(e.value)


@needs_native
@needs_models
@needs_fixture
def test_layout_only_run_returns_regions_and_no_text():
    # text=False + layout=True is the LAYOUT-ONLY run: det/cls/rec are skipped
    # (RunFlags.text=false short-circuits the unified pipeline), the result
    # carries layout regions and zero recognized lines. This briefly raised
    # "not implemented" while the unified pipeline lacked the path.
    import cv2

    layout_onnx = os.path.join(MODELS, "layout", "layout.onnx")
    if not os.path.exists(layout_onnx):
        pytest.skip("no layout model")
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS, layout=True)
    res = ocr.read(cv2.imread(FIXTURE), layout=True, text=False)
    assert len(res.layout) > 0  # a structured page has regions
    assert len(res.lines) == 0  # and text was genuinely skipped


@needs_native
@needs_models
def test_native_run_with_layout_rejects_an_unloaded_capability():
    # Straight at the binding: a pipeline with no layout model asked for layout
    # must RAISE, naming the capability and the operator-facing remedy — not
    # return an empty layout list.
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS)
    if ocr.has_layout:
        pytest.skip("layout is loaded on this build")
    img = np.full((200, 300, 3), 255, np.uint8)
    with pytest.raises(ValueError) as e:
        ocr._pipe.run_with_layout(img, layout=True)
    assert "layout is required for this request" in str(e.value)


@needs_native
@needs_models
def test_ocr_read_explicit_unloaded_capability_raises_not_warns():
    # OCR.read(layout=True) on an OCR constructed WITHOUT layout used to
    # warn-and-ignore — the caller asked for layout and silently got a result
    # without it. The explicit flag now flows through to the shared gate, which
    # raises the SAME server message the binding test above pins. Behavior
    # change from warn to raise, deliberate (2026-08-02).
    ocr = turboocr.OCR("tiny", backend="auto", models_dir=MODELS)
    if ocr.has_layout:
        pytest.skip("layout is loaded on this build")
    img = np.full((200, 300, 3), 255, np.uint8)
    with pytest.raises(ValueError) as e:
        ocr.read(img, layout=True)
    assert "layout is required for this request" in str(e.value)


@needs_native
@needs_fixture
@needs_models
def test_autorotate_corrects_rotation():
    import cv2

    doc_ori = os.path.join(MODELS, "doc_ori.onnx")
    if not os.path.exists(doc_ori):
        pytest.skip("no doc_ori model")
    ocr = turboocr.OCR("tiny", models_dir=MODELS, autorotate=True)
    img = cv2.imread(FIXTURE)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    res = ocr.read(rotated, autorotate=True)
    assert res.orientation in (90, 270)  # detected a quarter turn
    assert len(res.lines) > 5  # and still read text after correcting


@needs_native
@needs_models
def test_progress_callback():
    seen = []
    ocr = turboocr.OCR("tiny", models_dir=MODELS)
    ocr.read_batch(
        [np.full((60, 90, 3), 255, np.uint8)] * 3,
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_result_roundtrip():
    from turboocr_engine import DocumentResult, PageResult

    page = PageResult(width=100, height=50)
    from turboocr_engine import TextLine

    page.lines.append(TextLine(text="hi", confidence=0.9, box=((0, 0), (10, 0), (10, 8), (0, 8))))
    rp = PageResult.from_json(page.to_json())
    assert rp.text == "hi" and len(rp.lines[0].box) == 4 and rp.width == 100

    doc = DocumentResult(pages=[page])
    rd = DocumentResult.from_dict(doc.to_dict())
    assert len(rd.pages) == 1 and rd.pages[0].text == "hi"


def test_document_exports_shape():
    from turboocr_engine import DocumentResult, PageResult, TextLine

    p = PageResult(width=100, height=50, page=1)
    p.lines.append(TextLine(text="a", confidence=0.9, box=((0, 0), (9, 0), (9, 8), (0, 8))))
    doc = DocumentResult(pages=[p])
    assert doc.to_tsv().splitlines()[0].startswith("page\t")
    assert doc.to_hocr().startswith("<!DOCTYPE html>") and "ocr-system" in doc.to_hocr()
    assert p.to_hocr(full=True).startswith("<!DOCTYPE")


@needs_native
@needs_models
def test_read_batch_returns_document():
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS)
    img = np.full((60, 90, 3), 255, np.uint8)
    doc = ocr.read_batch([img, img])
    from turboocr import DocumentResult

    assert isinstance(doc, DocumentResult) and len(doc) == 2


@needs_native
@needs_fixture
@needs_models
def test_searchable_pdf_text_layer(tmp_path):
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    pdfium = pytest.importorskip("pypdfium2")
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS)
    page = ocr.read(FIXTURE)
    from turboocr import DocumentResult

    out = str(tmp_path / "searchable.pdf")
    DocumentResult(pages=[page]).save_searchable_pdf(out)
    assert os.path.getsize(out) > 1000

    # The OCR text must be extractable from the generated PDF's text layer.
    pdf = pdfium.PdfDocument(out)
    extracted = pdf[0].get_textpage().get_text_range()
    hits = sum(1 for ln in page.lines if ln.text.split() and ln.text.split()[0] in extracted)
    assert hits >= max(1, len(page.lines) // 2)


@needs_native
@needs_models
def test_pdf_to_searchable_streaming(tmp_path):
    pytest.importorskip("reportlab")
    pdfium = pytest.importorskip("pypdfium2")
    pdf_fix = os.path.join(REPO_ROOT, "tests", "fixtures", "pdf", "simple_letter.pdf")
    if not os.path.exists(pdf_fix):
        pytest.skip("no pdf fixture")
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS)
    out = str(tmp_path / "out.pdf")
    ocr.pdf_to_searchable(pdf_fix, out, max_pages=1)
    doc = pdfium.PdfDocument(out)
    assert len(doc) == 1
    assert len(doc[0].get_textpage().get_text_range()) > 20


@needs_native
@needs_models
def test_tables_to_html():
    enc = os.path.join(MODELS, "table", "slanext_encoder", "SLANeXt_wired_encoder.onnx")
    tbl_img = os.path.join(REPO_ROOT, "tests", "fixtures", "images", "png", "table.png")
    if not (os.path.exists(enc) and os.path.exists(tbl_img)):
        pytest.skip("no table model/fixture")
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS, tables=True)
    if not ocr.has_tables:
        pytest.skip("table backend did not load")
    res = ocr.read(tbl_img, tables=True)
    assert res.tables, "expected at least one table"
    assert "<table" in res.tables[0].html
    d = res.to_dict()
    assert "tables" in d and d["tables"][0]["html"]


def test_searchable_pdf_multiscript_and_dpi():
    """Synthetic (no OCR models): pins the CID-font Unicode coverage and the
    DPI-correct media box permanently, cheaply, and always-runs."""
    pytest.importorskip("reportlab")
    pdfium = pytest.importorskip("pypdfium2")
    import io as _io

    from turboocr import DocumentResult, PageResult, TextLine

    img = np.full((80, 400, 3), 255, np.uint8)
    page = PageResult(width=400, height=80, image=img, page=1, dpi=200)
    scripts = ["你好世界", "안녕하세요", "Ελληνικά", "ไทย", "Привет"]
    for i, t in enumerate(scripts):
        y = 10 + i * 12
        page.lines.append(TextLine(text=t, confidence=0.9, box=((10, y), (380, y), (380, y + 10), (10, y + 10))))
    data = DocumentResult(pages=[page]).to_pdf_bytes()
    doc = pdfium.PdfDocument(_io.BytesIO(data))
    ext = doc[0].get_textpage().get_text_range()
    for t in scripts:
        assert t in ext, f"{t!r} missing from searchable text layer"
    # DPI-correct: 400x80 px at 200 DPI => 144x28.8 pt (not 400x80).
    w_pt, h_pt = doc[0].get_size()
    assert abs(w_pt - 400 * 72 / 200) < 1


@needs_native
def test_cuda_backend_rejected_on_cpu_build():
    # On a build whose native ORT lacks CUDA, backend='cuda' must fail clearly.
    # Needs the extension: without it the constructor raises
    # NativeExtensionMissing long before any EP check, which proves nothing.
    if "CUDAExecutionProvider" in native.native_providers():
        pytest.skip("this build has CUDA")
    with pytest.raises(turboocr.BackendUnavailable, match="CUDAExecutionProvider|turboocr-engine-cuda1"):
        turboocr.OCR("tiny", backend="cuda", models_dir=MODELS)


@needs_native
@needs_fixture
@needs_models
def test_concurrent_reads_are_safe_and_correct():
    """The per-instance run lock must let many threads hit ONE OCR without the
    shared-scratch data race corrupting results (architect finding #3)."""
    import concurrent.futures as cf

    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS)
    golden = ocr.read(FIXTURE).text

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        texts = list(ex.map(lambda _: ocr.read(FIXTURE).text, range(24)))

    # Every concurrent read must equal the single-threaded golden output.
    assert all(t == golden for t in texts)


@needs_native
@needs_fixture
@needs_models
def test_keep_image_false_frees_memory():
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS, keep_image=False)
    res = ocr.read(FIXTURE)
    assert res.image is None
    # draw() without a stored image must error clearly, not crash.
    with pytest.raises(ValueError):
        res.draw()


@needs_native
@needs_models
def test_concurrent_instance_construction():
    """construct_lock must let different-backend OCRs build concurrently without
    the env-mutation race (architect finding #2)."""
    import concurrent.futures as cf

    def build(backend):
        return turboocr.OCR("tiny", backend=backend, models_dir=MODELS).backend

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        backends = list(ex.map(build, ["cpu", "auto", "cpu", "auto"]))
    assert len(backends) == 4

def test_doctor_picks_the_cuda_wheel_from_the_driver():
    """NVIDIA is two distributions and the DRIVER decides which one loads, so
    this is the one piece of new decision logic the split turns on. It had no
    coverage: recommend() was only ever exercised with nvidia_driver_major
    unset, which takes the fallback branch and hides the comparison entirely.

    The unknown-driver case must resolve to cuda12, not cuda13 — naming a wheel
    the machine cannot import is worse than naming the conservative one."""
    from turboocr_engine.doctor import (
        CUDA12_MIN_DRIVER, CUDA13_MIN_DRIVER, recommend)
    from turboocr_engine.providers import HardwareInfo

    def pick(drv):
        hw = HardwareInfo(os="Linux", machine="x86_64", has_nvidia=True,
                          gpu_names=["NVIDIA GPU"], nvidia_driver_major=drv)
        return recommend(hw)

    # Unknown driver -> the widely installable wheel.
    assert pick(None).package == "turboocr-engine-cuda12"
    # Exactly at and above the CUDA 13 floor -> cuda13.
    assert pick(CUDA13_MIN_DRIVER).package == "turboocr-engine-cuda13"
    assert pick(CUDA13_MIN_DRIVER + 15).package == "turboocr-engine-cuda13"
    # One below the floor must NOT recommend cuda13.
    assert pick(CUDA13_MIN_DRIVER - 1).package == "turboocr-engine-cuda12"
    # Below both floors: still no cuda13, and the reason must say the driver is
    # the problem rather than quietly naming a wheel that will not load.
    too_old = pick(CUDA12_MIN_DRIVER - 50)
    assert too_old.package != "turboocr-engine-cuda13"
    assert "driver" in too_old.reason.lower()
    # Never recommends a distribution that is not one of the two real ones.
    for drv in (None, 470, 525, 570, 580, 600):
        assert pick(drv).package in {
            "turboocr-engine-cuda12", "turboocr-engine-cuda13"}


def test_driver_probe_survives_hostile_nvidia_smi_output(monkeypatch):
    """detect_hardware() is documented "Never raises". The driver parse reads
    free-form nvidia-smi text, which on a broken host is an error string rather
    than a version, so the parse must not throw or invent a driver."""
    from turboocr_engine import providers

    for payload in ("", "\n", "Insufficient Permissions", "N/A",
                    "Failed to initialize NVML: Driver/library version mismatch",
                    "not-a-number", "580.65.06\n580.65.06\n"):
        monkeypatch.setattr(providers, "_cmd_ok", lambda _c: True)
        monkeypatch.setattr(providers, "_run",
                            lambda _a, _p=payload: "NVIDIA GPU" if "name" in " ".join(_a) else _p)
        providers.detect_hardware.cache_clear()
        hw = providers.detect_hardware()  # must not raise
        drv = hw.nvidia_driver_major
        assert drv is None or isinstance(drv, int)
    providers.detect_hardware.cache_clear()


def test_nvidia_pip_lib_discovery_orders_dependencies_first(tmp_path):
    """The preload must dlopen dependencies before their dependents (cudart
    before nvinfer, nvinfer before its plugin/parser), across BOTH layouts the
    pip packages use (tensorrt_libs/ flat, nvidia/<pkg>/lib/ nested). Ordering
    is the part a refactor would silently break — a wrong order still 'works'
    on machines where the loader finds the dep elsewhere, and only fails on the
    pip-only machine the preload exists for."""
    from turboocr_engine import native as n

    site = tmp_path
    (site / "tensorrt_libs").mkdir()
    (site / "tensorrt_libs" / "libnvinfer.so.10").write_bytes(b"")
    (site / "tensorrt_libs" / "libnvinfer_plugin.so.10").write_bytes(b"")
    (site / "tensorrt_libs" / "libnvonnxparser.so.10").write_bytes(b"")
    # Must be ignored: dev symlink name and the builder resources (nvinfer's
    # own $ORIGIN RPATH handles those — preloading them is pure waste).
    (site / "tensorrt_libs" / "libnvinfer.so").write_bytes(b"")
    (site / "tensorrt_libs" / "libnvinfer_builder_resource_sm90.so.10.15.1").write_bytes(b"")
    (site / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    (site / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12").write_bytes(b"")
    (site / "nvidia" / "nvjpeg" / "lib").mkdir(parents=True)
    (site / "nvidia" / "nvjpeg" / "lib" / "libnvjpeg.so.12").write_bytes(b"")
    # The openvino wheel's runtime layout (pip `openvino` package): TBB must
    # load before libopenvino, which links it; plugins/frontends are ignored
    # (libopenvino dlopens them itself via $ORIGIN).
    (site / "openvino" / "libs").mkdir(parents=True)
    (site / "openvino" / "libs" / "libopenvino.so.2621").write_bytes(b"")
    (site / "openvino" / "libs" / "libtbb.so.12").write_bytes(b"")
    (site / "openvino" / "libs" / "libopenvino_intel_gpu_plugin.so").write_bytes(b"")

    dirs = n._vendor_pip_lib_dirs(str(site))
    assert str(site / "tensorrt_libs") in dirs
    assert str(site / "nvidia" / "cuda_runtime" / "lib") in dirs
    assert str(site / "openvino" / "libs") in dirs

    # Reconstruct the exact load order the preload would use.
    order = []
    for d in dirs:
        import os as _os
        for name in _os.listdir(d):
            for rank, prefix in enumerate(n._VENDOR_LIB_PRIORITY):
                if name.startswith(prefix + "."):
                    order.append((rank, name))
                    break
    order = [name for _r, name in sorted(order)]
    assert order == [
        "libcudart.so.12",
        "libnvjpeg.so.12",
        "libnvinfer.so.10",
        "libnvonnxparser.so.10",
        "libnvinfer_plugin.so.10",
        "libtbb.so.12",
        "libopenvino.so.2621",
    ]


def test_nvidia_preload_never_raises(tmp_path):
    """detect-and-retry must be safe on every machine: fake unloadable files
    (empty, not ELF) must be skipped, not crash — and a site dir with nothing
    NVIDIA in it must be a clean no-op returning 0."""
    from turboocr_engine import native as n

    empty = tmp_path / "empty-site"
    empty.mkdir()
    assert n._preload_vendor_pip_libs(str(empty)) == 0

    site = tmp_path / "fake-site"
    (site / "tensorrt_libs").mkdir(parents=True)
    (site / "tensorrt_libs" / "libnvinfer.so.10").write_bytes(b"not an ELF")
    # Every candidate fails CDLL -> loaded == 0, no exception.
    assert n._preload_vendor_pip_libs(str(site)) == 0


def test_vendor_runtime_hint_names_the_matching_major():
    """The remedy line must follow the SONAME in the error: a .so.12 miss names
    the cu12 pip packages, .so.13 names cu13, and a non-NVIDIA import error
    gets no NVIDIA advice at all."""
    from turboocr_engine.native import _vendor_runtime_hint

    assert "tensorrt-cu12-libs" in _vendor_runtime_hint(
        "libcudart.so.12: cannot open shared object file")
    assert "tensorrt-cu13-libs" in _vendor_runtime_hint(
        "libnvjpeg.so.13: cannot open shared object file")
    assert "tensorrt-cu" in _vendor_runtime_hint(
        "libnvinfer.so.10: cannot open shared object file")
    assert _vendor_runtime_hint("No module named '_turboocr'") == ""


def test_openvino_backend_routes_to_the_native_intel_engine(monkeypatch):
    """On the turboocr-engine-openvino wheel, backend="openvino" must run the
    NATIVE intel backend (compiled in, measured faster) — the wheel's vendored
    ORT is the plain CPU build with NO OpenVINO EP, so the previous EP-only
    routing rejected backend="openvino" on the one wheel built for it. The
    clean-machine wheel smoke caught this; this pins the routing.

    The device= knob must land in OV_DEVICE: the native engine reads it there
    (TURBO_EP_DEVICE stops at the ONNX path — backend_stages.cpp)."""
    monkeypatch.setattr(native, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(native, "native_backends", lambda: ["cpu", "intel"])
    monkeypatch.delenv("OV_DEVICE", raising=False)

    assert native.resolve_engine("openvino") == "intel"
    assert native.resolve_engine("intel") == "intel"
    # ensure_backend_supported must NOT demand the ORT EP the wheel lacks.
    native.ensure_backend_supported("openvino")

    resolved, summary = native.configure_backend("openvino", device="NPU")
    assert resolved == "openvino"
    assert "OpenVINO" in summary
    assert os.environ.get("OV_DEVICE") == "NPU"

    # Without the seam backend (CPU wheel), the old behaviour is unchanged:
    # the ORT-EP path, and the EP check rejects it with the wheel remedy.
    monkeypatch.setattr(native, "native_backends", lambda: ["cpu"])
    assert native.resolve_engine("openvino") == "cpu"
    monkeypatch.setattr(native, "native_providers", lambda: ["CPUExecutionProvider"])
    import pytest as _pytest
    from turboocr_engine.errors import BackendUnavailable
    with _pytest.raises(BackendUnavailable, match="turboocr-engine-openvino"):
        native.ensure_backend_supported("openvino")


def test_vendor_runtime_hint_names_openvino():
    """A libopenvino miss must point at the wheel's own pip dependency, not at
    CUDA packages."""
    from turboocr_engine.native import _vendor_runtime_hint

    hint = _vendor_runtime_hint("libopenvino.so.2621: cannot open shared object file")
    assert "openvino>=2026.2" in hint
    assert "tensorrt" not in hint


def test_explicit_models_dir_beats_cwd_models(tmp_path, monkeypatch):
    """OCR(models_dir=...) must resolve against THAT directory, even when it is
    tier-only (no det.onnx) and the CWD has a fully-populated ./models. The old
    probe demanded det.onnx (the MEDIUM det) of every candidate, so an explicit
    tier-only dir silently lost to the checkout's ./models — different weights,
    and different Apple native exports, from the ones the caller named."""
    from turboocr_engine.models import ModelStore

    explicit = tmp_path / "mine"
    explicit.mkdir()
    (explicit / "det_small.onnx").write_bytes(b"x")  # tier-only: no det.onnx

    cwd = tmp_path / "checkout"
    (cwd / "models").mkdir(parents=True)
    (cwd / "models" / "det.onnx").write_bytes(b"x")
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("TURBO_OCR_MODELS_DIR", raising=False)

    assert ModelStore(str(explicit)).local_dir == str(explicit)
    # The CWD heuristic still works when nothing explicit is given...
    assert ModelStore(None).local_dir == str(cwd / "models")
    # ...but not for a directory that merely happens to be called "models".
    (cwd / "models" / "det.onnx").unlink()
    assert ModelStore(None).local_dir is None


def test_apple_native_bundle_provisioning(tmp_path, monkeypatch):
    """ensure_apple_native must (a) no-op when the export already exists,
    (b) refuse to touch a user-managed models dir, (c) download+extract the
    tar into the cache exactly once, and (d) never raise on failure — the
    engine's CoreML fallback depends on that contract."""
    import io
    import tarfile as tf

    from turboocr_engine import models as m
    from turboocr_engine.catalog import find_model

    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.platform, "machine", lambda: "arm64")
    entry = find_model("small")

    cache = tmp_path / "cache" / "models" / m.DEFAULT_RELEASE
    cache.mkdir(parents=True)
    (cache / "det_small.onnx").write_bytes(b"x")
    store = m.ModelStore(None, allow_download=True)
    store.local_dir = None
    store.cache_dir = str(cache)
    resolved = m.ResolvedModel(det=str(cache / "det_small.onnx"),
                               rec=str(cache / "rec_small.onnx"),
                               dict=str(cache / "keys.txt"), cls=None,
                               name="small")

    # (c) download+extract: fake the release asset with an in-memory tar.
    calls = []

    def fake_download(rel, dest):
        calls.append(rel)
        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as t:
            info = tf.TarInfo("det_small/graph.json")
            info.size = 2
            t.addfile(info, io.BytesIO(b"{}"))
        (cache / rel).write_bytes(buf.getvalue())
        return str(cache / rel)

    monkeypatch.setattr(store, "_download_apple_bundle", fake_download)
    assert store.ensure_apple_native(entry, resolved) is True
    assert calls == [f"apple_native_small.tar.gz"]
    assert (cache / "det_small" / "graph.json").is_file()

    # (a) second call: already provisioned, no new download.
    assert store.ensure_apple_native(entry, resolved) is True
    assert len(calls) == 1

    # (b) user-managed dir: never write there, never download.
    managed = tmp_path / "mine"
    managed.mkdir()
    (managed / "det_small.onnx").write_bytes(b"x")
    resolved2 = m.ResolvedModel(det=str(managed / "det_small.onnx"),
                                rec=str(managed / "rec_small.onnx"),
                                dict=str(managed / "keys.txt"), cls=None,
                                name="small")
    assert store.ensure_apple_native(entry, resolved2) is False
    assert len(calls) == 1

    # (d) a broken download must not raise, only report False.
    (cache / "det_small" / "graph.json").unlink()
    (cache / "apple_native_small.tar.gz").unlink()

    def broken_download(rel, dest):
        raise RuntimeError("release asset missing")

    monkeypatch.setattr(store, "_download_apple_bundle", broken_download)
    assert store.ensure_apple_native(entry, resolved) is False

    # Not macOS -> clean False before any path logic.
    monkeypatch.setattr(m.sys, "platform", "linux")
    assert store.ensure_apple_native(entry, resolved) is False
