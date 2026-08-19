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
    resolved, _summary = native.configure_backend("auto")
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
    hocr = res.to_hocr()
    assert "ocr_line" in hocr
    # Honest granularity: the engine recognizes LINES; there is no word-level
    # output, so the hOCR must not fabricate ocrx_word spans.
    assert "ocrx_word" not in hocr
    assert "ocrx_word" not in res.to_hocr(full=True)
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
    try:
        ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS, tables=True)
    except turboocr.ModelLoadError as exc:
        # tables=True now RAISES when the backend cannot honour it (silent
        # degrade removed); in this env that just means skip.
        pytest.skip(f"table backend did not load: {exc}")
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
    assert abs(h_pt - 80 * 72 / 200) < 1  # BOTH dims, as the comment claims


@needs_native
def test_cuda_backend_rejected_on_cpu_build():
    # On a build whose native ORT lacks CUDA, backend='cuda' must fail clearly.
    # Needs the extension: without it the constructor raises
    # NativeExtensionMissing long before any EP check, which proves nothing.
    if "CUDAExecutionProvider" in native.native_providers():
        pytest.skip("this build has CUDA")
    with pytest.raises(turboocr.BackendUnavailable, match=r"CUDAExecutionProvider|turboocr-engine-cuda1"):
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

    # (c) download+extract: fake the release asset with an in-memory tar in
    # the REAL v2 layout — det export as a canvas subdir, no flat graph.json.
    # (The old fake used the flat v1 layout, which is exactly how the
    # re-extract-on-every-construction regression slipped past this test:
    # the presence probe knew only the flat form.)
    calls = []

    def fake_download(rel, dest):
        calls.append(rel)
        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as t:
            info = tf.TarInfo("det_small/det_c992x768/graph.json")
            info.size = 2
            t.addfile(info, io.BytesIO(b"{}"))
        (cache / rel).write_bytes(buf.getvalue())
        return str(cache / rel)

    monkeypatch.setattr(store, "_download_apple_bundle", fake_download)
    assert store.ensure_apple_native(entry, resolved) is True
    assert calls == ["apple_native_small.tar.gz"]
    assert (cache / "det_small" / "det_c992x768" / "graph.json").is_file()

    # (a) second call: already provisioned (canvas layout!), no new download —
    # and no re-extract: a poisoned archive would throw if extraction re-ran.
    (cache / "apple_native_small.tar.gz").write_bytes(b"not a tar")
    assert store.ensure_apple_native(entry, resolved) is True
    assert len(calls) == 1

    # (a') flat v1 layout is still recognized as provisioned.
    import shutil
    shutil.rmtree(cache / "det_small")
    (cache / "det_small").mkdir()
    (cache / "det_small" / "graph.json").write_bytes(b"{}")
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


def test_async_wrappers_delegate_and_are_coroutines():
    """aread/aread_batch/aread_pdf are asyncio sugar over the sync methods via
    asyncio.to_thread — this pins (a) they are real coroutine functions, (b)
    they pass arguments through unchanged and return the sync result, without
    needing the native extension (the instance is constructed bare)."""
    import asyncio
    import inspect

    from turboocr_engine.pipeline import OCR

    for name in ("aread", "aread_batch", "aread_pdf"):
        assert inspect.iscoroutinefunction(getattr(OCR, name)), name

    o = object.__new__(OCR)  # no native init — the wrappers touch only self.<sync>
    calls = []
    o.read = lambda image, **kw: calls.append(("read", image, kw)) or "P"
    o.read_batch = lambda images, **kw: calls.append(("batch", images, kw)) or "D"
    o.read_pdf = lambda pdf, **kw: calls.append(("pdf", pdf, kw)) or "F"

    async def drive():
        assert await o.aread("img.png", layout=True) == "P"
        assert await o.aread_batch(["a", "b"], batch_size=4) == "D"
        assert await o.aread_pdf("doc.pdf", dpi=200) == "F"

    asyncio.run(drive())
    assert calls == [
        ("read", "img.png", {"layout": True}),
        ("batch", ["a", "b"], {"batch_size": 4}),
        ("pdf", "doc.pdf", {"dpi": 200}),
    ]


def test_parallel_map_ordered_bounded_and_contained():
    """_parallel_map (ordered mode) is the page fan-out under read_pdf/pdf_to_searchable:
    (a) results come back in INPUT order even when later items finish first,
    (b) at most `workers` calls run concurrently and at most workers+lookahead
    items are in flight (this is what bounds retained page rasters),
    (c) workers<=1 is a plain inline loop (no threads at all),
    (d) a failing item re-raises at its position and queued work is cancelled.
    """
    import threading
    import time

    from turboocr_engine.pipeline import _parallel_map

    # (a)+(b): earlier items sleep LONGER, so completion order is reversed —
    # the yield order must still be input order.
    running = 0
    peak = 0
    mu = threading.Lock()

    def slow_inverse(i):
        nonlocal running, peak
        with mu:
            running += 1
            peak = max(peak, running)
        time.sleep(0.03 - i * 0.002)
        with mu:
            running -= 1
        return i * 10

    out = list(_parallel_map(range(8), slow_inverse, workers=3))
    assert out == [i * 10 for i in range(8)]
    assert peak <= 3

    # (b) in-flight window: the producer must not run ahead of the consumer by
    # more than workers+lookahead items.
    produced = []

    def counting_items():
        for i in range(20):
            produced.append(i)
            yield i

    consumed = 0
    for _ in _parallel_map(counting_items(), lambda i: i, workers=2, lookahead=1):
        consumed += 1
        assert len(produced) - consumed <= 2 + 1

    # (c) inline degenerate path: no thread may be created.
    before = threading.active_count()
    assert list(_parallel_map(range(4), lambda i: i + 1, workers=1)) == [1, 2, 3, 4]
    assert threading.active_count() == before

    # (d) exception propagation at the failing item's position.
    def boom(i):
        if i == 2:
            raise RuntimeError("page 2 failed")
        return i

    got = []
    try:
        for v in _parallel_map(range(6), boom, workers=3):
            got.append(v)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "page 2 failed" in str(e)
    assert got == [0, 1]  # everything before the failure, in order


def test_read_pdf_uses_parallel_map(monkeypatch):
    """read_pdf must route pages through _parallel_map with the replica
    count — pin the wiring so a refactor cannot silently fall back to the
    sequential loop."""
    from turboocr_engine import pipeline as P

    calls = {}

    def fake_map(items, fn, workers, ordered=True, lookahead=1, executor=None):
        calls["workers"] = workers
        calls["ordered"] = ordered
        for item in items:
            yield fn(item)

    monkeypatch.setattr(P, "_parallel_map", fake_map)

    ocr = object.__new__(P.OCR)  # no native engine needed
    ocr.replicas = 3
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()  # eager validation probes for a live pipeline
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False
    pages_read = []

    def fake_read_array(arr, *, drop_score, page, keep_image, rotate=0):
        pages_read.append(page)
        return P.PageResult(width=1, height=1, page=page)

    ocr._read_array = fake_read_array

    import sys
    import types

    fake_pdf_mod = types.SimpleNamespace(
        pdf_page_count=lambda p, password=None: 3,
        iter_pdf_pages=lambda p, dpi, pages, max_pages, mode, password=None, text_with_raster=False, on_error="raise": iter(
            [("img", 1, "a1"), ("img", 2, "a2"), ("img", 3, "a3")]
        ),
        extract_pdf_text=lambda p, dpi, pages, max_pages, password=None: iter(()),
    )
    monkeypatch.setitem(sys.modules, "turboocr_engine.pdf", fake_pdf_mod)

    doc = P.OCR.read_pdf(ocr, "fake.pdf")
    assert calls["workers"] == 3 and calls["ordered"] is True
    assert pages_read == [1, 2, 3]
    assert [p.page for p in doc.pages] == [1, 2, 3]


def test_parallel_map_completion_mode_bounded_and_complete():
    """_parallel_map ordered=False (read_pdf_stream's completion mode): yields the
    full result set in completion order, never exceeds the worker bound, and
    a failure propagates. The inline workers=1 path stays ordered."""
    import threading
    import time

    from turboocr_engine.pipeline import _parallel_map

    running = 0
    peak = 0
    mu = threading.Lock()

    def slow_inverse(i):
        nonlocal running, peak
        with mu:
            running += 1
            peak = max(peak, running)
        time.sleep(0.03 - i * 0.002)
        with mu:
            running -= 1
        return i

    out = list(_parallel_map(range(8), slow_inverse, workers=3, ordered=False))
    assert sorted(out) == list(range(8))  # complete...
    assert out != list(range(8))          # ...and genuinely completion-ordered
    assert peak <= 3

    assert list(_parallel_map(range(4), lambda i: i, workers=1, ordered=False)) == [0, 1, 2, 3]

    def boom(i):
        if i == 1:
            raise RuntimeError("bad page")
        return i

    with pytest.raises(RuntimeError, match="bad page"):
        list(_parallel_map(range(5), boom, workers=2, ordered=False))


def _fake_pdf_ocr(monkeypatch, n_pages=4):
    """An OCR shell whose read_pdf_stream renders n fake pages and 'OCRs' them
    without the native engine."""
    import sys
    import types

    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 3
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()  # eager validation probes for a live pipeline
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False

    def fake_read_array(arr, *, drop_score, page, keep_image, rotate=0):
        return P.PageResult(width=1, height=1, page=page)

    ocr._read_array = fake_read_array
    fake_pdf_mod = types.SimpleNamespace(
        pdf_page_count=lambda p, password=None: n_pages,
        iter_pdf_pages=lambda p, dpi, pages, max_pages, mode, password=None, text_with_raster=False, on_error="raise": iter(
            ("img", i, f"arr{i}") for i in range(1, n_pages + 1)
        ),
        extract_pdf_text=lambda p, dpi, pages, max_pages, password=None: iter(()),
    )
    monkeypatch.setitem(sys.modules, "turboocr_engine.pdf", fake_pdf_mod)
    return P, ocr


def test_read_pdf_stream_modes_and_early_close(monkeypatch):
    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=6)

    # ordered: page order, streaming.
    got = [pr.page for pr in P.OCR.read_pdf_stream(ocr, "f.pdf")]
    assert got == [1, 2, 3, 4, 5, 6]

    # unordered: full set, each result self-identifies via .page.
    got = sorted(pr.page for pr in P.OCR.read_pdf_stream(ocr, "f.pdf", ordered=False))
    assert got == [1, 2, 3, 4, 5, 6]

    # early close: no PER-STREAM threads may leak. The engine's SHARED
    # page pool (<= replicas workers) persists by design — so the baseline
    # is measured after the pool exists, and early-closing more streams
    # must not grow past it.
    import threading
    import time

    baseline = threading.active_count()  # pool already built by the runs above
    for _ in range(4):
        gen = P.OCR.read_pdf_stream(ocr, "f.pdf")
        assert next(gen).page == 1
        gen.close()
    deadline = time.time() + 5
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= baseline


def test_aread_pdf_stream_iterates_and_cleans_up(monkeypatch):
    import asyncio

    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=5)

    async def consume_all():
        return [pr.page async for pr in P.OCR.aread_pdf_stream(ocr, "f.pdf")]

    assert asyncio.run(consume_all()) == [1, 2, 3, 4, 5]

    async def consume_two():
        out = []
        async for pr in P.OCR.aread_pdf_stream(ocr, "f.pdf"):
            out.append(pr.page)
            if len(out) == 2:
                break  # must trigger clean shutdown of the sync generator
        return out

    assert asyncio.run(consume_two()) == [1, 2]


def test_read_pdf_on_error_skip_contains_page_failures(monkeypatch):
    """A corrupt page mid-document must not cost the other pages: with
    on_error='skip' it becomes an empty PageResult that says why it is empty
    (page_failed warning, correct .page); the default still raises."""
    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=4)

    def failing_read_array(arr, *, drop_score, page, keep_image, rotate=0):
        if page == 2:
            raise RuntimeError("decoder exploded")
        return P.PageResult(width=1, height=1, page=page)

    ocr._read_array = failing_read_array

    doc = P.OCR.read_pdf(ocr, "f.pdf", on_error="skip")
    assert [pr.page for pr in doc.pages] == [1, 2, 3, 4]
    failed = doc.pages[1]
    assert failed.lines == []
    assert len(failed.warnings) == 1
    assert failed.warnings[0].startswith("page_failed: RuntimeError: decoder")
    assert all(not pr.warnings for i, pr in enumerate(doc.pages) if i != 1)

    with pytest.raises(RuntimeError, match="decoder exploded"):
        P.OCR.read_pdf(ocr, "f.pdf")  # default on_error='raise'

    with pytest.raises(ValueError, match="on_error"):
        P.OCR.read_pdf(ocr, "f.pdf", on_error="ignore")


def test_read_pdf_autorotate_rotates_pages(monkeypatch):
    """autorotate on the PDF path: each rendered page is orientation-detected,
    rotated upright BEFORE OCR, and the angle lands in PageResult.orientation.
    Used to be silently ignored — OCR(autorotate=True).read_pdf() OCR'd the
    sideways pages as-is."""
    import contextlib
    import sys
    import types

    import numpy as np

    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()  # eager validation probes for a live pipeline
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False

    a1 = np.zeros((4, 6, 3), np.uint8)   # upright
    a2 = np.zeros((6, 4, 3), np.uint8)   # "sideways" — taller than wide
    fake_pdf_mod = types.SimpleNamespace(
        pdf_page_count=lambda p, password=None: 2,
        iter_pdf_pages=lambda p, dpi, pages, max_pages, mode, password=None, text_with_raster=False, on_error="raise": iter(
            [("img", 1, a1), ("img", 2, a2)]
        ),
        extract_pdf_text=lambda p, dpi, pages, max_pages, password=None: iter(()),
    )
    monkeypatch.setitem(sys.modules, "turboocr_engine.pdf", fake_pdf_mod)

    class FakePipe:
        def __init__(self, has_ori):
            self._has = has_ori

        def has_doc_ori(self):
            return self._has

        def detect_orientation(self, arr):
            return 90 if arr.shape[0] > arr.shape[1] else 0

    ocr._pipe = FakePipe(True)
    ocr._checkout = lambda: contextlib.nullcontext(ocr._pipe)

    seen = []

    def fake_read_array(arr, *, drop_score, page, keep_image, rotate=0):
        seen.append((page, rotate, arr.shape[:2]))
        return P.PageResult(width=arr.shape[1], height=arr.shape[0],
                            page=page, orientation=rotate)

    ocr._read_array = fake_read_array

    doc = P.OCR.read_pdf(ocr, "f.pdf", autorotate=True)
    # page 2 reached OCR already rotated upright, and carries its angle.
    assert seen == [(1, 0, (4, 6)), (2, 90, (4, 6))]
    assert [p.orientation for p in doc.pages] == [0, 90]

    # Engine-level OCR(autorotate=True) applies without the per-call flag.
    seen.clear()
    ocr.autorotate = True
    P.OCR.read_pdf(ocr, "f.pdf")
    assert [s[1] for s in seen] == [0, 90]

    # Explicit True without the model refuses instead of silently no-opping.
    ocr.autorotate = False
    ocr._pipe = FakePipe(False)
    with pytest.raises(ValueError, match="document-orientation"):
        list(P.OCR.read_pdf_stream(ocr, "f.pdf", autorotate=True))


def _batch_shell(replicas: int):
    """An OCR shell that takes read_batch's per-image (non-batchable) path."""
    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = replicas
    ocr.keep_image = None
    ocr._closed = False
    ocr._pipe = object()
    ocr._can_batch = lambda *, layout, autorotate: False
    return P, ocr


def test_read_batch_on_error_raise_stops_queueing():
    """on_error='raise' propagates the failure AND cancels queued images —
    the executor teardown uses cancel_futures, so a 64-image batch that dies
    on image one must not grind through the remaining 63 first."""
    import threading

    P, ocr = _batch_shell(replicas=2)
    calls = []
    mu = threading.Lock()

    def fake_read(im, **kw):
        with mu:
            calls.append(im)
        raise RuntimeError("poison image")

    ocr.read = fake_read
    with pytest.raises(RuntimeError, match="poison image"):
        P.OCR.read_batch(ocr, list(range(64)))
    assert len(calls) < 64  # queued work was cancelled, not drained


def test_read_batch_on_error_skip_keeps_other_images():
    P, ocr = _batch_shell(replicas=2)

    def fake_read(im, **kw):
        if im == "bad":
            raise ValueError("unreadable")
        return P.PageResult(width=1, height=1)

    ocr.read = fake_read
    doc = P.OCR.read_batch(ocr, ["a", "bad", "c"], on_error="skip")
    assert len(doc.pages) == 3
    assert doc.pages[1].warnings == ["page_failed: ValueError: unreadable"]
    assert not doc.pages[0].warnings and not doc.pages[2].warnings

    # The sequential (replicas=1) path contains identically.
    ocr.replicas = 1
    doc = P.OCR.read_batch(ocr, ["a", "bad"], on_error="skip")
    assert len(doc.pages) == 2 and doc.pages[1].warnings[0].startswith("page_failed:")

    with pytest.raises(ValueError, match="on_error"):
        P.OCR.read_batch(ocr, ["a"], on_error="nope")


def test_read_batch_native_chunk_failure_falls_back_per_image():
    """When the NATIVE whole-batch call dies, skip mode re-runs that chunk one
    image at a time so the failure lands on the culprit image only."""
    import contextlib

    import numpy as np

    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.keep_image = True
    ocr._closed = False
    ocr._pipe = object()
    ocr._can_batch = lambda *, layout, autorotate: True

    class FakePipe:
        def run_batch(self, arrays):
            raise RuntimeError("native chunk kaboom")

    ocr._checkout = lambda: contextlib.nullcontext(FakePipe())

    good = np.zeros((4, 4, 3), np.uint8)
    bad = np.zeros((5, 5, 3), np.uint8)

    def fake_read_array(arr, *, drop_score, keep_image, want_layout=None):
        assert want_layout is False  # the rescue run must not resurrect layout
        if arr.shape[0] == 5:
            raise RuntimeError("this raster is the poison")
        return P.PageResult(width=arr.shape[1], height=arr.shape[0])

    ocr._read_array = fake_read_array

    doc = P.OCR.read_batch(ocr, [good, bad], on_error="skip")
    assert len(doc.pages) == 2
    assert not doc.pages[0].warnings
    assert doc.pages[1].warnings[0].startswith("page_failed: RuntimeError: this raster")
    assert doc.pages[1].width == 5  # dims survive containment

    with pytest.raises(RuntimeError, match="native chunk kaboom"):
        P.OCR.read_batch(ocr, [good, bad])  # default raises


def test_keep_image_defaults_per_path(monkeypatch):
    """read() keeps the raster by default; read_pdf/read_batch DROP it (a
    raster is ~6 MB — long documents silently retained GBs). Per-call beats
    engine-level beats the per-path default."""
    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=1)
    seen = {}

    def rec(arr, *, drop_score, page, keep_image, rotate=0):
        seen[page] = keep_image
        return P.PageResult(width=1, height=1, page=page)

    ocr._read_array = rec
    P.OCR.read_pdf(ocr, "f.pdf")
    assert seen[1] is False                       # PDF default: drop
    P.OCR.read_pdf(ocr, "f.pdf", keep_image=True)
    assert seen[1] is True                        # per-call wins
    ocr.keep_image = True                         # engine-level OCR(keep_image=True)
    P.OCR.read_pdf(ocr, "f.pdf")
    assert seen[1] is True

    Pb, b = _batch_shell(replicas=1)
    got = {}

    def fake_read(im, **kw):
        got.update(kw)
        return Pb.PageResult(width=1, height=1)

    b.read = fake_read
    Pb.OCR.read_batch(b, ["x"])
    assert got["keep_image"] is False             # batch default: drop
    Pb.OCR.read_batch(b, ["x"], keep_image=True)
    assert got["keep_image"] is True


def test_searchable_pdf_refuses_all_rasterless_pages():
    """A searchable PDF from pages that ALL lost their raster (the new
    keep_image=False default) must raise an error naming keep_image — while a
    MIXED document (one failed page, the on_error='skip' shape) still writes."""
    pytest.importorskip("reportlab")
    import warnings as W

    from turboocr import DocumentResult, PageResult, TextLine

    bare = PageResult(width=100, height=50, page=1)
    bare.lines.append(TextLine(text="hi", confidence=0.9,
                               box=((1, 1), (50, 1), (50, 10), (1, 10))))
    with W.catch_warnings():
        W.simplefilter("ignore")
        with pytest.raises(ValueError, match="keep_image=True"):
            DocumentResult(pages=[bare]).to_pdf_bytes()

    img = np.full((50, 100, 3), 255, np.uint8)
    with_img = PageResult(width=100, height=50, page=2, image=img, dpi=72)
    with W.catch_warnings():
        W.simplefilter("ignore")
        data = DocumentResult(pages=[bare, with_img]).to_pdf_bytes()
    assert data[:4] == b"%PDF"


def test_read_rejects_pdfs_with_a_pointer_to_read_pdf(tmp_path):
    """A PDF handed to the IMAGE reader must say 'use read_pdf', not the bare
    'could not decode' that sent users debugging the wrong thing. Sniffed by
    content (%PDF- magic), so a PDF misnamed .png gets the same pointer; plain
    garbage keeps the generic message."""
    from turboocr_engine.imaging import load_image

    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.7\nfake body")
    with pytest.raises(ValueError, match="read_pdf"):
        load_image(str(p))

    with pytest.raises(ValueError, match="read_pdf"):
        load_image(b"%PDF-1.4 fake bytes")

    sneaky = tmp_path / "sneaky.png"  # misnamed: content wins over extension
    sneaky.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(ValueError, match="read_pdf"):
        load_image(str(sneaky))

    with pytest.raises(ValueError, match="could not decode"):
        load_image(b"definitely not an image")


def test_read_pdf_rejects_non_pdfs_with_a_pointer_to_read(tmp_path):
    """The mirror: an image handed to the PDF reader names read()/read_batch()
    instead of PDFium's opaque parse error; a missing path is a plain
    FileNotFoundError."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine.pdf import _open_document

    with pytest.raises(ValueError, match=r"read\(\) / read_batch\(\)"):
        _open_document(b"\x89PNG not a pdf")

    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG fake image bytes")
    with pytest.raises(ValueError, match=r"read\(\) / read_batch\(\)"):
        _open_document(str(img))

    with pytest.raises(FileNotFoundError):
        _open_document(str(tmp_path / "missing.pdf"))


def test_table_region_to_pandas_and_aggregators():
    """Tables get real DataFrames — the tabular view is tables_to_pandas() /
    TableRegion.to_pandas(), while to_pandas() stays the text-lines frame.
    Covers a SLANet-shaped table (thead header, colspan merge), provenance in
    DataFrame.attrs, and the page/document aggregators."""
    pytest.importorskip("pandas")
    pytest.importorskip("lxml")
    from turboocr_engine.result import DocumentResult, PageResult, TableRegion

    box = ((0, 0), (100, 0), (100, 50), (0, 50))
    html = (
        "<table><thead><tr><td>Item</td><td>Qty</td><td>Price</td></tr></thead>"
        "<tbody><tr><td>Apples</td><td>3</td><td>2.40</td></tr>"
        "<tr><td colspan=\"2\">Total</td><td>2.40</td></tr></tbody></table>"
    )
    region = TableRegion(html=html, score=0.93, box=box)

    df = region.to_pandas()
    assert list(df.columns) == ["Item", "Qty", "Price"]
    assert df.shape == (2, 3)
    row = df.iloc[0]
    assert row["Item"] == "Apples" and str(row["Qty"]) == "3"
    assert float(row["Price"]) == pytest.approx(2.40)
    # colspan expands the way read_html expands it: the value repeats.
    assert df.iloc[1]["Item"] == "Total" and df.iloc[1]["Qty"] == "Total"
    assert df.attrs["score"] == pytest.approx(0.93)
    assert df.attrs["box"] == [[0, 0], [100, 0], [100, 50], [0, 50]]

    page = PageResult(width=100, height=50, page=7, tables=[region, region])
    frames = page.tables_to_pandas()
    assert len(frames) == 2
    assert all(f.attrs["page"] == 7 for f in frames)

    doc = DocumentResult(pages=[page, PageResult(width=1, height=1, page=8,
                                                 tables=[region])])
    all_frames = doc.tables_to_pandas()
    assert [f.attrs["page"] for f in all_frames] == [7, 7, 8]

    # The lines frame is a DIFFERENT view and stays line-shaped.
    assert list(page.to_pandas().columns[:2]) == ["text", "confidence"] or page.to_pandas().empty


def _digital_pdf(tmp_path, texts_per_page):
    """A born-digital PDF: real vector text on each page (None = image-only)."""
    rl = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "digital.pdf"
    c = rl.Canvas(str(path), pagesize=(400, 300))
    for content in texts_per_page:
        if content is not None:
            c.setFont("Helvetica", 14)
            for k, line in enumerate(content):
                c.drawString(40, 250 - 24 * k, line)
        else:
            c.rect(60, 60, 200, 100, fill=1)  # marks only, no text layer
        c.showPage()
    c.save()
    return str(path)


def _ocr_shell(fake_read_array):
    """An OCR without the native engine: enough attrs for read_pdf_stream."""
    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 2
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()  # eager validation probes for a live pipeline
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False
    ocr._read_array = fake_read_array
    return P, ocr


def test_pdf_text_mode_reads_without_any_ocr(tmp_path):
    """mode='text' serves the embedded text layer: exact strings, source='pdf',
    confidence 1.0 — and the OCR path is provably never entered (the fake
    raises if touched). The parallel extractor keeps page order."""
    pytest.importorskip("pypdfium2")
    pdf = _digital_pdf(tmp_path, [["hello page one", "second line"],
                                  ["page two text"]])

    def boom(*a, **k):
        raise AssertionError("mode='text' must never OCR")

    P, ocr = _ocr_shell(boom)
    pages = list(P.OCR.read_pdf_stream(ocr, pdf, mode="text"))
    assert [p.page for p in pages] == [1, 2]
    assert [ln.text for ln in pages[0].lines] == ["hello page one", "second line"]
    assert all(ln.source == "pdf" and ln.confidence == 1.0
               for p in pages for ln in p.lines)
    # geometry is in rendered-pixel space: line 1 sits above line 2.
    y_first = pages[0].lines[0].box[0][1]
    y_second = pages[0].lines[1].box[0][1]
    assert 0 < y_first < y_second < pages[0].height

    # pdfium is globally thread-hostile; the process-wide lock must make
    # CONCURRENT extraction from different documents safe (it used to crash
    # the interpreter) and keep every result correct and ordered.
    import threading

    from turboocr_engine.pdf import extract_pdf_text

    many = _digital_pdf(tmp_path / "..", [[f"p{i}"] for i in range(1, 13)])
    results = {}

    def pull(name, src):
        results[name] = list(extract_pdf_text(src))

    threads = [threading.Thread(target=pull, args=("a", many)),
               threading.Thread(target=pull, args=("b", pdf))]
    for t in threads: t.start()
    for t in threads: t.join()
    assert [o[0] for o in results["a"]] == list(range(1, 13))
    assert [o[3][0][0] for o in results["a"]] == [f"p{i}" for i in range(1, 13)]
    assert results["b"][0][3][0][0] == "hello page one"


def test_pdf_auto_mode_ocrs_only_textless_pages(tmp_path):
    """mode='auto': the digital page comes from the text layer; only the
    image-only page reaches OCR."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine.result import PageResult

    # Page 1 carries enough text to clear the auto-mode trust gate (the
    # gate refuses thin layers — a Bates stamp must not hijack a scan).
    pdf = _digital_pdf(tmp_path, [["digital words on a real page of text",
                                   "with a second line to be trustworthy"],
                                  None])
    ocr_pages = []

    def fake_read_array(arr, *, drop_score, page, keep_image, rotate=0):
        ocr_pages.append(page)
        return PageResult(width=arr.shape[1], height=arr.shape[0], page=page)

    P, ocr = _ocr_shell(fake_read_array)
    pages = list(P.OCR.read_pdf_stream(ocr, pdf, mode="auto"))
    assert [p.page for p in pages] == [1, 2]
    assert pages[0].lines[0].text.startswith("digital words")
    assert pages[0].lines[0].source == "pdf"
    assert ocr_pages == [2]  # ONLY the scanned page was OCR'd

    with pytest.raises(ValueError, match="mode"):
        list(P.OCR.read_pdf_stream(ocr, pdf, mode="nope"))


def test_draw_layout_overlay_is_stable():
    """draw(layout=True) paints layout regions in stable per-label colors;
    lines=False gives a layout-only overlay; base image is untouched."""
    from turboocr_engine.result import LayoutBox, PageResult

    base = np.zeros((80, 120, 3), dtype=np.uint8)
    page = PageResult(
        width=120, height=80, image=base,
        layout=[LayoutBox("table", 0.9, ((10, 10), (60, 10), (60, 40), (10, 40))),
                LayoutBox("text", 0.8, ((70, 45), (110, 45), (110, 70), (70, 70)))],
    )
    out1 = page.draw(layout=True, lines=False)
    out2 = page.draw(layout=True, lines=False)
    assert out1.any() and (out1 == out2).all()   # drew something, deterministically
    assert not base.any()                        # base untouched


def test_concurrent_read_pdf_render_path_is_pdfium_safe(tmp_path, monkeypatch):
    """The OCR-path pdf iterator (iter_pdf_pages — under read_pdf/aread_pdf)
    must hold the process-wide pdfium lock: pdfium is GLOBALLY thread-hostile
    and two concurrent documents on the render path used to be able to crash
    the interpreter. Two threads render two documents concurrently through
    the real pdfium; results must be complete and correct."""
    pytest.importorskip("pypdfium2")
    import threading

    from turboocr_engine.pdf import iter_pdf_pages

    a = _digital_pdf(
        tmp_path,
        [["alpha page with plenty of embedded text",
          "so the auto-mode trust gate accepts it"]] * 5)
    b = _digital_pdf(tmp_path / "..", [None] * 5)  # image-only pages: renders
    out = {}

    def pump(name, src, mode):
        out[name] = list(iter_pdf_pages(src, mode=mode))

    ts = [threading.Thread(target=pump, args=("a", a, "auto")),
          threading.Thread(target=pump, args=("b", b, "ocr"))]
    for t in ts: t.start()
    for t in ts: t.join()
    assert [k for k, *_ in out["a"]] == ["text"] * 5
    assert [k for k, *_ in out["b"]] == ["img"] * 5 and out["b"][0][2].ndim == 3


def test_closed_engine_raises_instead_of_hanging():
    """read() after close() used to block forever on the drained replica pool."""
    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr._closed = True
    with pytest.raises(RuntimeError, match="closed"), P.OCR._checkout(ocr):
        pass


def test_out_of_range_pages_raise(tmp_path):
    """An explicit pages= selection matching nothing must raise, not return an
    empty document indistinguishable from a blank one."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine.pdf import extract_pdf_text

    pdf = _digital_pdf(tmp_path, [["only page"]])
    with pytest.raises(ValueError, match="1-based"):
        list(extract_pdf_text(pdf, pages=[7, 8]))
    # in-range-plus-out-of-range still works (partial selections are normal)
    assert [o[0] for o in extract_pdf_text(pdf, pages=[1, 7])] == [1]


def test_document_to_markdown_accepts_structured():
    """docs promise DocumentResult.to_markdown(structured=...) — it raised
    TypeError before."""
    from turboocr_engine.result import DocumentResult, PageResult

    doc = DocumentResult(pages=[PageResult(width=1, height=1, page=1)])
    assert isinstance(doc.to_markdown(structured=False), str)


def test_cli_structure_flags_and_info(monkeypatch, capsys):
    """CLI parity: --tables/--formulas/--autorotate parse on ocr AND pdf and
    reach the OCR constructor; `turboocr info` prints the engine's resolved
    configuration as JSON."""
    import json
    import types

    from turboocr_engine import cli, pipeline

    p = cli.build_parser()
    a = p.parse_args(["ocr", "x.png", "--tables", "--formulas", "--autorotate",
                      "--replicas", "2"])
    assert a.tables and a.formulas and a.autorotate and a.replicas == 2
    a = p.parse_args(["pdf", "d.pdf", "--mode", "auto", "--tables"])
    assert a.mode == "auto" and a.tables and not a.formulas

    captured = {}

    class FakeOCR:
        def __init__(self, *args, **kw):
            captured.update(kw)
            self.provider_summary = "fake"
            self.model_name = "tiny"

    monkeypatch.setattr(pipeline, "OCR", FakeOCR)
    cli._build_ocr(p.parse_args(["ocr", "x.png", "--tables", "--formulas",
                                 "--autorotate"]))
    assert captured["tables"] and captured["formulas"] and captured["autorotate"]

    fake = types.SimpleNamespace(info=lambda: {"model": "tiny", "backend": "cpu"})
    seen = {}

    def fake_build(args):
        seen["tables"] = args.tables
        return fake

    monkeypatch.setattr(cli, "_build_ocr", fake_build)
    assert cli.main(["info", "--tables"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "tiny"
    assert seen["tables"] is True


@needs_native
@needs_models
def test_construction_restores_env(monkeypatch):
    """The construct block's env mutations (EP selection, structure-model
    paths) must not leak: after OCR() returns, every key is back to its
    pre-construction value — including keys the build overwrote — so a second
    engine (or the host app) sees the environment it started with."""
    # backend="cpu" overwrites ORT_EP on every platform (the apple branch
    # used to leave it alone — see the stale-ORT_EP regression test); the
    # guard must put the caller's value back afterwards.
    monkeypatch.setenv("ORT_EP", "sentinel-ep")
    monkeypatch.delenv("TABLE_SLANEXT_ENCODER_ONNX", raising=False)
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS)
    assert os.environ.get("ORT_EP") == "sentinel-ep"           # overwrite undone
    assert "TABLE_SLANEXT_ENCODER_ONNX" not in os.environ      # set undone
    # And the engine built under the guard still works after restore.
    res = ocr.read(np.full((40, 60, 3), 255, np.uint8))
    assert res.lines == []


def test_apple_bundle_extraction_atomicity(tmp_path, monkeypatch):
    """The bundle extracts into a private tempdir and moves files into place
    with the PROBE file (graph.json) last, so a concurrent reader can never
    see the bundle as provisioned while half-extracted; a lockfile in the
    cache serializes competing provisioners; no tempdir residue remains."""
    import io
    import tarfile as tf

    from turboocr_engine import models as m
    from turboocr_engine.catalog import find_model

    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.platform, "machine", lambda: "arm64")
    entry = find_model("small")
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "det_small.onnx").write_bytes(b"x")
    store = m.ModelStore(None, allow_download=True)
    store.local_dir = None
    store.cache_dir = str(cache)
    resolved = m.ResolvedModel(det=str(cache / "det_small.onnx"), rec="r",
                               dict="d", cls=None, name="small")

    def fake_download(rel, dest):
        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as t:
            for name, data in [
                ("det_small/det_c992x768/graph.json", b"{}"),
                ("det_small/det_c992x768/weights.bin", b"wwww"),
                ("coreml/small/rec.mlmodelc/model.bin", b"mmmm"),
            ]:
                info = tf.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        (cache / rel).write_bytes(buf.getvalue())
        return str(cache / rel)

    monkeypatch.setattr(store, "_download_apple_bundle", fake_download)

    order = []
    real_replace = os.replace

    def spy_replace(src, dst):
        order.append(os.path.basename(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(m.os, "replace", spy_replace)

    assert store.ensure_apple_native(entry, resolved) is True
    assert (cache / "det_small" / "det_c992x768" / "weights.bin").is_file()
    assert (cache / "coreml" / "small" / "rec.mlmodelc" / "model.bin").is_file()
    # graph.json — what _det_export_present probes — moved into place LAST.
    assert order[-1] == "graph.json" and order.count("graph.json") == 1
    # No tempdir residue; the flock target exists.
    assert not [p for p in os.listdir(cache) if p.startswith(".apple_native_tmp")]
    assert (cache / ".apple_native_small.lock").exists()


def test_multipage_tiff_rejected_single_page_ok(tmp_path):
    """cv2 silently decodes only page 1 of a multi-page TIFF — silent data
    loss. load_image must refuse with the page count and point at
    read_batch; a single-page TIFF keeps decoding as before."""
    import struct

    from turboocr_engine.imaging import _tiff_page_count, load_image

    def ifd(endian, nxt):
        return (struct.pack(endian + "H", 1)
                + struct.pack(endian + "HHII", 256, 3, 1, 1)
                + struct.pack(endian + "I", nxt))

    two_le = struct.pack("<2sHI", b"II", 42, 8) + ifd("<", 26) + ifd("<", 0)
    two_be = struct.pack(">2sHI", b"MM", 42, 8) + ifd(">", 26) + ifd(">", 0)
    assert _tiff_page_count(two_le) == 2
    assert _tiff_page_count(two_be) == 2

    with pytest.raises(ValueError, match=r"multi-page TIFF \(2 pages\)"):
        load_image(two_le)
    p = tmp_path / "two.tif"
    p.write_bytes(two_be)
    with pytest.raises(ValueError, match="read_batch"):
        load_image(str(p))

    # A real single-page TIFF still decodes.
    import cv2

    ok, enc = cv2.imencode(".tif", np.full((5, 7, 3), 200, np.uint8))
    assert ok
    one = enc.tobytes()
    assert _tiff_page_count(one) == 1
    assert load_image(one).shape == (5, 7, 3)

    # Non-TIFF data that happens to start with II/MM must not be mistaken.
    assert _tiff_page_count(b"MM plain text, not a tiff at all") == 0
    assert _tiff_page_count(b"II" + b"\x00" * 3) == 0


def test_image_size_ceiling_sniffed_before_decode(tmp_path, monkeypatch):
    """A PNG/JPEG header claiming absurd dimensions is refused in
    microseconds — BEFORE the decoder allocates — with the env knob named;
    other formats hit the post-decode fallback; TURBO_MAX_IMAGE_MP raises or
    disables the ceiling."""
    import struct
    import time

    from turboocr_engine.imaging import _sniff_dims, load_image

    def png(w, h):
        return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
                + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00junk")

    assert _sniff_dims(png(100, 50)) == (100, 50)

    big = png(200000, 200000)  # 40 000 000 MP claimed in 37 bytes
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="TURBO_MAX_IMAGE_MP"):
        load_image(big)
    assert time.perf_counter() - t0 < 0.5  # rejected by sniff, not by OOM

    jpg = (b"\xff\xd8"
           + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
           + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08"
           + struct.pack(">HH", 60000, 60000) + b"\x03" + b"\x00" * 9)
    assert _sniff_dims(jpg) == (60000, 60000)
    with pytest.raises(ValueError, match="megapixels"):
        load_image(jpg)
    p = tmp_path / "huge.png"
    p.write_bytes(big)
    with pytest.raises(ValueError, match=r"huge\.png"):
        load_image(str(p))

    # Raising the ceiling lets the sniff pass — the junk body then fails with
    # the ordinary decode error, proving the sniff no longer intercepts.
    monkeypatch.setenv("TURBO_MAX_IMAGE_MP", "99999999")
    with pytest.raises(ValueError, match="could not decode"):
        load_image(big)
    monkeypatch.setenv("TURBO_MAX_IMAGE_MP", "0")  # 0 disables
    with pytest.raises(ValueError, match="could not decode"):
        load_image(big)
    monkeypatch.delenv("TURBO_MAX_IMAGE_MP")

    # Normal images pass untouched.
    import cv2

    ok, enc = cv2.imencode(".png", np.zeros((4, 6, 3), np.uint8))
    assert ok and load_image(enc.tobytes()).shape == (4, 6, 3)


def test_requested_stage_load_failure_raises():
    """OCR(layout/tables/formulas/autorotate=True) whose stage model did NOT
    load must raise ModelLoadError at construction — the silent degrade used
    to return zero regions/tables forever with no signal."""
    from turboocr_engine import pipeline as P
    from turboocr_engine.errors import ModelLoadError

    ocr = object.__new__(P.OCR)
    ocr.engine = "cpu"

    class Pipe:
        def __init__(self, **have):
            self.have = have

        def has_layout(self):
            return self.have.get("layout", False)

        def has_table_backend(self):
            return self.have.get("tables", False)

        def has_formula_backend(self):
            return self.have.get("formulas", False)

        def has_doc_ori(self):
            return self.have.get("ori", False)

    def bind(pipe, **req):
        kw = {"layout": False, "tables": False, "formulas": False, "autorotate": False}
        kw.update(req)
        P.OCR._bind_stages(ocr, pipe, **kw)

    with pytest.raises(ModelLoadError, match="layout stage"):
        bind(Pipe(), layout=True)
    with pytest.raises(ModelLoadError, match="layout stage"):
        bind(Pipe(), tables=True)  # tables imply layout; layout is the culprit
    with pytest.raises(ModelLoadError, match="table backend"):
        bind(Pipe(layout=True), tables=True)
    with pytest.raises(ModelLoadError, match="formula backend"):
        bind(Pipe(layout=True), formulas=True)
    with pytest.raises(ModelLoadError, match="orientation"):
        bind(Pipe(), autorotate=True)

    # Everything requested and loaded -> flags bound true.
    bind(Pipe(layout=True, tables=True, formulas=True, ori=True),
         layout=True, tables=True, formulas=True, autorotate=True)
    assert ocr.has_layout and ocr.has_tables and ocr.has_formulas and ocr.autorotate

    # Nothing requested -> nothing raises, everything off (a stage the engine
    # carries but the caller did not ask for stays unavailable).
    bind(Pipe(layout=True, tables=True, formulas=True, ori=True))
    assert not (ocr.has_layout or ocr.has_tables or ocr.has_formulas or ocr.autorotate)


def test_roundtrip_carries_dpi_and_parent_id():
    """to_dict/from_dict fidelity: dpi survives (a restored page must build a
    correctly-sized searchable PDF) and LayoutBox.parent_id survives (region
    nesting); an unset parent_id stays out of the JSON (server-shape additive
    only)."""
    from turboocr import DocumentResult, PageResult, TextLine
    from turboocr_engine.result import LayoutBox

    pg = PageResult(width=400, height=80, page=3, dpi=200)
    pg.lines.append(TextLine(text="t", confidence=0.9,
                             box=((1, 1), (9, 1), (9, 9), (1, 9))))
    pg.layout.append(LayoutBox(label="table", confidence=0.8,
                               box=((0, 0), (10, 0), (10, 10), (0, 10)),
                               id=2, parent_id=1))
    back = PageResult.from_json(pg.to_json())
    assert back.dpi == 200 and back.page == 3
    assert back.layout[0].parent_id == 1 and back.layout[0].id == 2

    d = LayoutBox(label="x", confidence=0.5,
                  box=((0, 0), (1, 0), (1, 1), (0, 1))).to_dict()
    assert "parent_id" not in d

    # Searchable sizing from the RESTORED page: 400 px @ 200 DPI -> 144 pt.
    pytest.importorskip("reportlab")
    pdfium = pytest.importorskip("pypdfium2")
    import io as _io

    back.image = np.full((80, 400, 3), 255, np.uint8)
    data = DocumentResult(pages=[back]).to_pdf_bytes()
    doc = pdfium.PdfDocument(_io.BytesIO(data))
    w_pt, _h = doc[0].get_size()
    assert abs(w_pt - 400 * 72 / 200) < 1


def test_table_formula_confidence_aliases_score():
    """TextLine/LayoutBox say `confidence`, Table/FormulaRegion said only
    `score` — both spellings now work; `score` stays canonical in to_dict."""
    from turboocr_engine.result import FormulaRegion, TableRegion

    box = ((0, 0), (1, 0), (1, 1), (0, 1))
    t = TableRegion(html="<table></table>", score=0.75, box=box)
    f = FormulaRegion(latex="x^2", score=0.5, box=box)
    assert t.confidence == t.score == 0.75
    assert f.confidence == f.score == 0.5
    assert "score" in t.to_dict() and "confidence" not in t.to_dict()
    assert "score" in f.to_dict() and "confidence" not in f.to_dict()


def test_default_engine_cache_keys_and_race(monkeypatch):
    """The module-level default engine is a small keyed LRU behind a lock:
    alternating keys reuse their engines instead of rebuilding (the old
    single slot rebuilt on every alternation), and a same-key thread race
    constructs exactly once."""
    import threading

    import turboocr_engine as te

    built = []

    class FakeOCR:
        def __init__(self, model, backend, *, layout=False, tables=False,
                     formulas=False, autorotate=False):
            built.append((model, backend, layout, tables, formulas, autorotate))

        def close(self):
            pass

    monkeypatch.setattr(te, "OCR", FakeOCR)
    monkeypatch.setattr(te, "_DEFAULT_CACHE", type(te._DEFAULT_CACHE)())

    a1 = te._default_engine("tiny", "cpu")
    b1 = te._default_engine("tiny", "cpu", layout=True)
    a2 = te._default_engine("tiny", "cpu")
    b2 = te._default_engine("tiny", "cpu", layout=True)
    assert a1 is a2 and b1 is b2 and a1 is not b1
    assert len(built) == 2  # the old slot would have built 4

    # Same-key race: N threads, one construction.
    built.clear()
    barrier = threading.Barrier(6)
    got = []

    def hit():
        barrier.wait()
        got.append(te._default_engine("small", "cpu"))

    threads = [threading.Thread(target=hit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
    assert all(g is got[0] for g in got)

    # LRU cap: a 5th key evicts the least-recently-used, which rebuilds on
    # next use; the cache never grows past the cap.
    built.clear()
    for m in ("k1", "k2", "k3", "k4", "k5"):
        monkeypatch.setattr(te, "resolve_model",
                            lambda name, _m=m: type("E", (), {"name": _m})())
        te._default_engine(m, "cpu")
    assert len(te._DEFAULT_CACHE) <= te._DEFAULT_CACHE_CAP


def test_concurrent_document_streams_share_one_worker_pool(monkeypatch):
    """All streams of one engine share `replicas` page-worker threads: any
    number of concurrent documents make progress (no gate, no permits, no
    deadlock class), and OS threads do not scale with the stream count."""
    import threading
    import time

    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=2)
    ocr.replicas = 2

    # Same-thread interleave and nesting: both previously deadlock-prone.
    got = [(a.page, b.page) for a, b in zip(
        P.OCR.read_pdf_stream(ocr, "a.pdf"), P.OCR.read_pdf_stream(ocr, "b.pdf"))]
    assert got == [(1, 1), (2, 2)]
    inner = []
    for _pr in P.OCR.read_pdf_stream(ocr, "a.pdf"):
        inner.append([q.page for q in P.OCR.read_pdf(ocr, "b.pdf").pages])
    assert inner == [[1, 2], [1, 2]]

    # CROSS-THREAD nesting at saturation — the shape every permit design
    # deadlocked on: replicas+N streams, each nesting a read on a WORKER
    # thread (different tid than the stream's).
    results = {}

    def stream_with_worker_nested(name):
        out = []
        for _pr in P.OCR.read_pdf_stream(ocr, name):
            holder = []
            t = threading.Thread(
                target=lambda h=holder: h.append(
                    len(P.OCR.read_pdf(ocr, "n.pdf").pages)),
                daemon=True)
            t.start()
            t.join(timeout=10)
            out.append(holder[0] if holder else "STARVED")
        results[name] = out

    threads = [threading.Thread(target=stream_with_worker_nested,
                                args=(f"s{i}.pdf",), daemon=True)
               for i in range(4)]  # 4 streams > replicas+1
    for t in threads:
        t.start()
    deadline = time.time() + 20
    for t in threads:
        t.join(timeout=max(0.1, deadline - time.time()))
    assert not any(t.is_alive() for t in threads), "streams starved"
    assert all(v == [2, 2] for v in results.values()), results

    # Thread bound: 20 open pumped streams share ONE pool, not 20 pools.
    before = threading.active_count()
    gens = [P.OCR.read_pdf_stream(ocr, f"m{i}.pdf") for i in range(20)]
    for g in gens:
        next(g)
    during = threading.active_count()
    assert during - before <= ocr.replicas + 2, (before, during)
    for g in gens:
        g.close()

    # text mode uses no workers at all.
    fake_pdf = __import__("sys").modules["turboocr_engine.pdf"]
    fake_pdf.iter_pdf_pages = lambda p, dpi, pages, max_pages, mode, password=None, \
            text_with_raster=False, on_error="raise": iter(
        [("text", 1, 10, 10, [("t", ((0, 0), (1, 0), (1, 1), (0, 1)))], None, [])]
    )
    assert [p.page for p in P.OCR.read_pdf_stream(ocr, "d.pdf", mode="text")] == [1]


def test_async_nested_read_inside_stream_loop(monkeypatch):
    """`await aread_pdf(b)` inside `async for ... aread_pdf_stream(a)` at
    replicas=1, MULTIPLE concurrent streams — the exact deterministic
    deadlock the permit gate produced (0% CPU, every thread parked on
    acquire). Must complete."""
    import asyncio

    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=2)
    ocr.replicas = 1

    async def one_stream(name):
        pages = []
        async for _pr in P.OCR.aread_pdf_stream(ocr, name):
            nested = await P.OCR.aread_pdf(ocr, "n.pdf")
            pages.append(len(nested.pages))
        return pages

    async def run():
        return await asyncio.wait_for(
            asyncio.gather(*[one_stream(f"s{i}.pdf") for i in range(3)]),
            timeout=20,
        )

    assert asyncio.run(run()) == [[2, 2]] * 3

def test_read_pdf_stream_validates_eagerly(monkeypatch):
    """Bad arguments raise at the CALL, not at the first next() on some other
    thread — read_pdf_stream is a validating function returning a generator."""
    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=1)
    with pytest.raises(ValueError, match="mode"):
        P.OCR.read_pdf_stream(ocr, "f.pdf", mode="bogus")
    with pytest.raises(ValueError, match="on_error"):
        P.OCR.read_pdf_stream(ocr, "f.pdf", on_error="nonsense")
    with pytest.raises(ValueError, match="max_pages"):
        P.OCR.read_pdf_stream(ocr, "f.pdf", max_pages=0)
    ocr._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        P.OCR.read_pdf_stream(ocr, "f.pdf")


def test_close_is_idempotent_and_unparks_racers():
    """close() twice (or from two threads) returns instead of hanging, and a
    reader that raced past the _closed check into the empty pool gets the
    sentinel -> RuntimeError, never an eternal block."""
    import threading
    import time

    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr._closed = False
    ocr._close_mu = threading.Lock()
    ocr._pdf_executor = None
    pipe = object()
    ocr._pipes = [pipe]
    ocr._pipe = pipe
    import queue as _q

    ocr._pool = _q.Queue()
    ocr._pool.put(pipe)

    # Simulate the TOCTOU: reader passed the flag check, close() drains, the
    # reader then parks in get(). The sentinel must wake it with an error.
    errs = []

    def racer():
        # emulate _checkout's body after the flag check
        got = ocr._pool.get()
        if got is P._POOL_CLOSED:
            ocr._pool.put(got)
            errs.append("closed")
        else:
            ocr._pool.put(got)
            errs.append("pipe")

    P.OCR.close(ocr)
    t = threading.Thread(target=racer, daemon=True)
    t.start()
    t.join(timeout=5)
    assert errs == ["closed"]

    # Idempotent + concurrent: both return promptly.
    threads = [threading.Thread(target=lambda: P.OCR.close(ocr), daemon=True)
               for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.time() + 5
    for t in threads:
        t.join(timeout=max(0.1, deadline - time.time()))
    assert not any(t.is_alive() for t in threads)

    with pytest.raises(RuntimeError, match="closed"), P.OCR._checkout(ocr):
        pass


def test_aread_pdf_stream_cancel_raises_cancelled(monkeypatch):
    """Cancelling a task mid-stream must surface CancelledError — the shared
    to_thread stepping used to close the generator from a second thread while
    next() was still running ('generator already executing')."""
    import asyncio
    import time

    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=50)
    slow_orig = ocr._read_array

    def slow_read_array(arr, **kw):
        time.sleep(0.05)
        return slow_orig(arr, **kw)

    ocr._read_array = slow_read_array

    async def run():
        async def consume():
            async for _pr in P.OCR.aread_pdf_stream(ocr, "f.pdf"):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.12)  # cancel lands mid-next()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        except BaseException as e:  # pragma: no cover
            return f"wrong: {type(e).__name__}: {e}"
        return "no error"

    assert asyncio.run(run()) == "cancelled"
    # And the gate permit was reclaimed: a fresh stream still works.
    assert [p.page for p in P.OCR.read_pdf_stream(ocr, "g.pdf", max_pages=None)][:1] == [1]

def test_pdf_password_reaches_pdfium(monkeypatch):
    """password= threads from every PDF entry point down to
    pdfium.PdfDocument(..., password=...) — captured via a fake pdfium."""
    import types

    from turboocr_engine import pdf as pdfmod

    captured = []

    class FakeDoc:
        def __init__(self, src, password=None):
            captured.append(password)

        def __len__(self):
            return 0

        def close(self):
            pass

    fake = types.SimpleNamespace(PdfDocument=FakeDoc)
    monkeypatch.setattr(pdfmod, "_import_pdfium", lambda: fake)
    data = b"%PDF-1.4 tiny"

    # The zero-page guard fires AFTER the password reaches PdfDocument, so
    # every entry point still proves the plumbing while refusing the doc.
    with pytest.raises(ValueError, match="no pages"):
        pdfmod.pdf_page_count(data, password="s3cret")
    with pytest.raises(ValueError, match="no pages"):
        list(pdfmod.extract_pdf_text(data, password="p2"))
    with pytest.raises(ValueError, match="no pages"):
        list(pdfmod.iter_pdf_pages(data, password="p3"))
    with pytest.raises(ValueError, match="no pages"):
        list(pdfmod.render_pdf(data, password="p4"))
    assert captured == ["s3cret", "p2", "p3", "p4"]
    # password type is validated before pdfium ever sees it
    with pytest.raises(ValueError, match="password must be a str"):
        pdfmod.pdf_page_count(data, password=b"bytes")

    # And the pipeline threads it through read_pdf -> iter_pdf_pages.
    import sys

    from turboocr_engine import pipeline as P

    seen = {}

    def fake_iter(p, dpi, pages, max_pages, mode, password=None,
                  text_with_raster=False, on_error="raise"):
        seen["password"] = password
        return iter(())

    fake_mod = types.SimpleNamespace(
        pdf_page_count=lambda p, password=None: 0,
        iter_pdf_pages=fake_iter,
        extract_pdf_text=lambda p, dpi, pages, max_pages, password=None: iter(()),
    )
    monkeypatch.setitem(sys.modules, "turboocr_engine.pdf", fake_mod)
    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False
    P.OCR.read_pdf(ocr, "locked.pdf", password="hunter2")
    assert seen["password"] == "hunter2"


@needs_native
@needs_fixture
@needs_models
def test_stale_ort_ep_does_not_poison_cpu_load(monkeypatch):
    """A stale ORT_EP in the caller's environment (a Linux dotfile exporting
    ORT_EP=cuda, say) must not break backend='cpu': configure_backend now
    states the EP explicitly on EVERY platform — the apple branch used to
    leave the stale value in place and the load died with 'Unknown ORT_EP'."""
    monkeypatch.setenv("ORT_EP", "cuda-from-someones-dotfile")
    ocr = turboocr.OCR("tiny", backend="cpu", models_dir=MODELS)
    page = ocr.read(FIXTURE)
    assert len(page.lines) > 5
    # And the guard restored the stale value (restoring is the env owner's
    # problem to fix, not ours to erase).
    assert os.environ["ORT_EP"] == "cuda-from-someones-dotfile"


def test_read_pdf_defaults_to_auto_mode(monkeypatch):
    """The PDF default is mode='auto' (text layer where present, OCR
    otherwise); pdf_to_searchable PINS mode='ocr' — its output embeds page
    rasters, which text-layer payloads don't carry."""
    P, ocr = _fake_pdf_ocr(monkeypatch, n_pages=1)
    import sys

    fake = sys.modules["turboocr_engine.pdf"]
    seen = []
    orig = fake.iter_pdf_pages

    def spy(p, dpi, pages, max_pages, mode, password=None,
            text_with_raster=False, on_error="raise"):
        seen.append(mode)
        return orig(p, dpi, pages, max_pages, mode)

    fake.iter_pdf_pages = spy
    P.OCR.read_pdf(ocr, "f.pdf")
    assert seen == ["auto"]

    seen.clear()
    list(P.OCR.read_pdf_stream(ocr, "f.pdf"))
    assert seen == ["auto"]

    seen.clear()

    def spy_stream(self, pdf, **kw):
        seen.append(kw.get("mode"))
        return iter(())

    monkeypatch.setattr(P.OCR, "read_pdf_stream", spy_stream)
    monkeypatch.setattr("turboocr_engine.searchable_pdf.build_searchable_pdf",
                        lambda pages, out_path=None: list(pages))
    P.OCR.pdf_to_searchable(ocr, "f.pdf", "unused.pdf")
    assert seen == ["ocr"]


def test_cli_searchable_rejects_text_mode(monkeypatch, capsys):
    """--searchable + --mode text is a contradiction (the output embeds page
    rasters) and must fail with a usage error BEFORE any engine builds."""
    from turboocr_engine import cli

    def boom(args):
        raise AssertionError("engine must not be constructed for a usage error")

    monkeypatch.setattr(cli, "_build_ocr", boom)
    rc = cli.main(["pdf", "x.pdf", "--searchable", "-o", "out.pdf",
                   "--mode", "text"])
    assert rc == 2
    assert "--mode text" in capsys.readouterr().err
    assert cli.main(["pdf", "x.pdf", "--searchable"]) == 2  # missing -o


def test_vendor_backend_clears_stale_ort_ep(monkeypatch):
    """The vendor-seam branch of configure_backend must clear a stale ORT_EP:
    aux stages (cls/layout/doc-ori/formula) load OrtEngine sessions that
    honour it — measured: OCR(backend='apple', autorotate=True) with a
    leftover ORT_EP=cuda failed to load the doc-ori model."""
    import os as _os

    from turboocr_engine import native as N

    monkeypatch.setenv("ORT_EP", "stale-garbage")
    monkeypatch.setattr(N, "resolve_engine", lambda b: "apple")
    backend, _summary = N.configure_backend("apple")
    assert backend
    assert "ORT_EP" not in _os.environ


def _rotated_pdf(tmp_path, rot):
    """Portrait page with a marker line, /Rotate applied by pdfium itself
    (reportlab's setPageRotation writes a /Rotate pdfium's RENDERER ignores,
    which would make the ink-vs-box comparison vacuous)."""
    import pypdfium2 as pdfium
    from reportlab.pdfgen import canvas as rl

    base = str(tmp_path / "rot_base.pdf")
    c = rl.Canvas(base, pagesize=(612, 792))
    c.setFont("Helvetica", 24)
    c.drawString(72, 792 - 96, "TOPLEFT MARKER TEXT WITH ENOUGH CHARS TO TRUST")
    c.showPage()
    c.save()
    p = str(tmp_path / f"rot{rot}.pdf")
    d = pdfium.PdfDocument(base)
    pg = d[0]
    pg.set_rotation(rot)
    d.save(p)
    pg.close()
    d.close()
    return p


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_pdf_text_layer_boxes_match_rendered_ink(tmp_path, rot):
    """/Rotate pages: extracted text-layer boxes must land where the RENDER
    puts the ink (the naive h-y flip was wrong for 90/180/270 — 180 was the
    dangerous one, staying in-range while pointing at the opposite corner)."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine.pdf import extract_pdf_text, render_pdf

    p = _rotated_pdf(tmp_path, rot)
    (page_no, w, h, lines), = list(extract_pdf_text(p, dpi=100))
    assert page_no == 1 and len(lines) == 1
    (_, quad), = lines
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]

    (_, arr), = list(render_pdf(p, dpi=100))
    ink = np.argwhere(arr[:, :, 0] < 128)  # dark glyph pixels
    assert len(ink) > 50
    iy0, ix0 = ink.min(axis=0)
    iy1, ix1 = ink.max(axis=0)
    # The text box must tightly cover the ink (tolerance: font metrics pad).
    assert min(xs) - 12 <= ix0 and ix1 <= max(xs) + 12, (rot, quad, (ix0, ix1))
    assert min(ys) - 12 <= iy0 and iy1 <= max(ys) + 12, (rot, quad, (iy0, iy1))
    # And page dims are consistent with the render's shape.
    assert (h, w) >= (arr.shape[0], arr.shape[1])
    assert h - arr.shape[0] <= 1 and w - arr.shape[1] <= 1


def test_auto_mode_quality_gate_rejects_thin_and_garbled_layers(tmp_path):
    """mode='auto' must NOT let a stamp hijack a scanned page: a page whose
    only text layer is a short Bates-style stamp renders for OCR; a
    substantial clean layer is served as text."""
    pytest.importorskip("pypdfium2")
    from reportlab.pdfgen import canvas as rl

    from turboocr_engine.pdf import _layer_quality, iter_pdf_pages

    # Unit: the gate itself (stats = chars, fffd, nonprint, rotation).
    assert _layer_quality((0, 0, 0, 0), 0) == "absent"
    assert _layer_quality((12, 0, 0, 0), 1) == "absent"       # stamp-thin
    # rotation no longer rejected (both transforms are ink-verified now —
    # a born-digital landscape report serves its layer like any page):
    assert _layer_quality((200, 0, 0, 90), 4) == "trusted"
    assert _layer_quality((200, 30, 0, 0), 4) == "rejected"   # mostly U+FFFD
    assert _layer_quality((200, 0, 30, 0), 4) == "rejected"   # control soup
    assert _layer_quality((200, 2, 2, 0), 4) == "trusted"

    # End to end: a stamped page (thin layer) renders; a real page serves text.
    p = str(tmp_path / "stamped.pdf")
    c = rl.Canvas(p, pagesize=(300, 200))
    c.setFont("Helvetica", 7)
    c.drawString(230, 8, "BATES 000123")  # the whole text layer
    c.showPage()
    c.setFont("Helvetica", 12)
    for k in range(4):
        c.drawString(20, 160 - 20 * k, f"real digital body text line {k} of this page")
    c.showPage()
    c.save()
    kinds = [k for k, *_ in iter_pdf_pages(p, mode="auto")]
    assert kinds == ["img", "text"]


def test_auto_mode_runs_structure_on_text_pages(monkeypatch):
    """Engines built with layout/tables/formulas must still produce them on
    text-layer pages: the raster is rendered for the structure pass while the
    text stays byte-exact from the layer (used to silently return zero
    regions on every born-digital page)."""
    import sys
    import types

    from turboocr_engine import pipeline as P
    from turboocr_engine.result import LayoutBox

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()
    ocr.has_layout = True
    ocr.has_tables = ocr.has_formulas = False

    arr = np.zeros((20, 30, 3), np.uint8)
    lines = [("layer text", ((1, 1), (9, 1), (9, 5), (1, 5)))]
    seen = {}

    def fake_iter(p, dpi, pages, max_pages, mode, password=None,
                  text_with_raster=False, on_error="raise"):
        seen["text_with_raster"] = text_with_raster
        return iter([("text", 1, 30, 20, lines,
                      arr if text_with_raster else None, [])])

    def fake_read_array(a, *, drop_score, page, keep_image, want_text=True, rotate=0):
        assert want_text is False  # structure-only run through the gate
        pr = P.PageResult(width=30, height=20, page=page)
        pr.layout.append(LayoutBox(label="text", confidence=0.9,
                                   box=((0, 0), (9, 0), (9, 9), (0, 9)), id=0))
        return pr

    ocr._read_array = fake_read_array
    monkeypatch.setitem(sys.modules, "turboocr_engine.pdf", types.SimpleNamespace(
        pdf_page_count=lambda p, password=None: 1,
        iter_pdf_pages=fake_iter,
        extract_pdf_text=lambda *a, **k: iter(()),
    ))

    doc = P.OCR.read_pdf(ocr, "f.pdf")
    assert seen["text_with_raster"] is True
    pg = doc.pages[0]
    assert [ln.text for ln in pg.lines] == ["layer text"]      # layer text kept
    assert [ln.source for ln in pg.lines] == ["pdf"]
    assert [lb.label for lb in pg.layout] == ["text"]          # structure ran
    assert pg.image is None                                     # keep_image default

    # keep_image=True on a text page stores the rendered raster.
    doc = P.OCR.read_pdf(ocr, "f.pdf", keep_image=True)
    assert doc.pages[0].image is not None


def test_on_error_skip_contains_render_failures(tmp_path, monkeypatch):
    """A page that fails to RENDER (producer side) is contained by
    on_error='skip' like an OCR failure — it used to end the whole stream."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine import pdf as pdfmod

    p = _digital_pdf(tmp_path, [None, None])  # two image-only pages
    real_render = pdfmod._render_page
    calls = {"n": 0}

    def sometimes_broken(page, scale):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("corrupt page stream")
        return real_render(page, scale)

    monkeypatch.setattr(pdfmod, "_render_page", sometimes_broken)

    out = list(pdfmod.iter_pdf_pages(p, mode="ocr", on_error="skip"))
    assert [k for k, *_ in out] == ["error", "img"]
    assert out[0][1] == 1 and "corrupt page stream" in out[0][2]

    # Default: propagate.
    calls["n"] = 0
    with pytest.raises(RuntimeError, match="corrupt page stream"):
        list(pdfmod.iter_pdf_pages(p, mode="ocr"))

    # And through the pipeline: the failed page carries the warning.
    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False

    def fake_read_array(a, *, drop_score, page, keep_image, rotate=0):
        return P.PageResult(width=a.shape[1], height=a.shape[0], page=page)

    ocr._read_array = fake_read_array
    calls["n"] = 0
    doc = P.OCR.read_pdf(ocr, p, on_error="skip")
    assert [pr.page for pr in doc.pages] == [1, 2]
    assert doc.pages[0].warnings and doc.pages[0].warnings[0].startswith(
        "page_failed: RuntimeError: corrupt")
    assert not doc.pages[1].warnings


def test_searchable_pdf_from_auto_mode_mixed_document(tmp_path):
    """A mode='auto' mixed document must produce a searchable PDF where the
    OCR'd page embeds its raster and the text-layer page is written as a
    VISIBLE text-only page — the old writer skipped source='pdf' lines and
    emitted a blank page (silent content destruction)."""
    pytest.importorskip("reportlab")
    pdfium = pytest.importorskip("pypdfium2")
    import io as _io

    from turboocr_engine.result import DocumentResult, PageResult, TextLine

    scan = PageResult(width=100, height=60, page=1, dpi=72,
                      image=np.full((60, 100, 3), 255, np.uint8))
    scan.lines.append(TextLine(text="SCANNED WORDS", confidence=0.9,
                               box=((5, 5), (90, 5), (90, 20), (5, 20))))
    digital = PageResult(width=100, height=60, page=2, dpi=72)
    digital.lines.append(TextLine(text="LAYER WORDS", confidence=1.0, source="pdf",
                                  box=((5, 5), (90, 5), (90, 20), (5, 20))))
    data = DocumentResult(pages=[scan, digital]).to_pdf_bytes()
    doc = pdfium.PdfDocument(_io.BytesIO(data))
    assert len(doc) == 2
    t1 = doc[0].get_textpage().get_text_range()
    t2 = doc[1].get_textpage().get_text_range()
    assert "SCANNED" in t1
    assert "LAYER" in t2  # the page is text-only but NOT blank

    # A failed (0x0) page mid-document takes the running page size, not 0.48pt.
    failed = PageResult(width=0, height=0, page=3, dpi=72)
    failed.warnings.append("page_failed: boom")
    data = DocumentResult(pages=[scan, failed]).to_pdf_bytes()
    doc = pdfium.PdfDocument(_io.BytesIO(data))
    w_pt, h_pt = doc[1].get_size()
    assert w_pt > 50 and h_pt > 30  # inherited the scan page's size

    # All-text-layer documents write cleanly (no rasters ever existed)...
    data = DocumentResult(pages=[digital]).to_pdf_bytes()
    assert data[:4] == b"%PDF"
    # ...while dropped-raster OCR documents still refuse, naming BOTH fixes.
    bare = PageResult(width=100, height=60, page=1)
    bare.lines.append(TextLine(text="ocr text", confidence=0.9,
                               box=((1, 1), (9, 1), (9, 9), (1, 9))))
    import warnings as W

    with W.catch_warnings():
        W.simplefilter("ignore")  # the per-page warn precedes the final raise
        with pytest.raises(ValueError, match='keep_image=True AND mode'):
            DocumentResult(pages=[bare]).to_pdf_bytes()


def test_markdown_table_cells_are_escaped():
    """Untrusted cell text must not become live inline HTML in Markdown —
    the converter unescapes to strip markup, so it must re-escape."""
    from turboocr_engine.result import PageResult, TableRegion

    from turboocr_engine.result import LayoutBox

    evil = "&lt;img src=x onerror=alert(1)&gt;"
    html = (f"<table><tr><td>{evil}</td><td>b|c</td></tr>"
            "<tr><td>x</td><td>y</td></tr></table>")
    pg = PageResult(width=10, height=10)
    box = ((0, 0), (9, 0), (9, 9), (0, 9))
    pg.layout.append(LayoutBox(label="table", confidence=0.9, box=box, id=0))
    pg.tables.append(TableRegion(html=html, score=0.9, box=box, layout_id=0))
    md = pg.to_markdown()  # layout present -> structured, tables render
    assert "|" in md  # the table actually rendered
    assert "<img" not in md and "&lt;img" in md
    assert "b\\|c" in md  # pipes still escaped


def test_from_dict_accepts_server_key_spellings():
    """The HTTP server emits bounding_box/class/confidence where this library
    writes box/label/score — from_dict must parse BOTH (it used to KeyError
    on every real server response)."""
    from turboocr_engine.result import (FormulaRegion, LayoutBox, PageResult,
                                        TableRegion, TextLine)

    bb = [[1, 2], [30, 2], [30, 12], [1, 12]]
    ln = TextLine.from_dict({"text": "hi", "confidence": 0.9, "bounding_box": bb})
    assert ln.box[0] == (1, 2) and ln.confidence == 0.9
    lb = LayoutBox.from_dict({"class": "table", "confidence": 0.8,
                              "bounding_box": bb, "id": 3})
    assert lb.label == "table" and lb.confidence == 0.8
    tr = TableRegion.from_dict({"html": "<table></table>", "confidence": 0.7,
                                "bounding_box": bb})
    assert tr.score == 0.7
    fr = FormulaRegion.from_dict({"latex": "x", "confidence": 0.6,
                                  "bounding_box": bb})
    assert fr.score == 0.6
    page = PageResult.from_dict({
        "width": 100, "height": 50,
        "results": [{"text": "t", "confidence": 0.9, "bounding_box": bb}],
        "layout": [{"class": "text", "confidence": 0.5, "bounding_box": bb}],
    })
    assert page.lines[0].text == "t" and page.layout[0].label == "text"


def test_table_to_pandas_error_contract():
    """Unparseable region HTML raises ValueError — one documented type — not
    lxml.XMLSyntaxError or pandas' internal html5lib ImportError."""
    pytest.importorskip("pandas")
    pytest.importorskip("lxml")
    from turboocr_engine.result import TableRegion

    box = ((0, 0), (1, 0), (1, 1), (0, 1))
    for bad in ("", "<table></table>", "<p>not a table</p>"):
        with pytest.raises(ValueError, match="no table"):
            TableRegion(html=bad, score=0.5, box=box).to_pandas()

    # And cell text is DATA: NA/None markers and thousands separators survive.
    ok = ("<table><tr><th>a</th><th>b</th></tr>"
          "<tr><td>NA</td><td>1,234</td></tr></table>")
    df = TableRegion(html=ok, score=0.5, box=box).to_pandas()
    assert df.iloc[0, 0] == "NA"
    assert str(df.iloc[0, 1]) == "1,234"


def test_hocr_strips_control_chars():
    """A control glyph from a PDF text layer must not make the hOCR document
    unparseable XML."""
    import xml.etree.ElementTree as ET

    from turboocr_engine.result import PageResult, TextLine

    pg = PageResult(width=100, height=50)
    pg.lines.append(TextLine(text="Total\x0c 42", confidence=0.9,
                             box=((1, 1), (50, 1), (50, 10), (1, 10))))
    doc = pg.to_hocr(full=True)
    assert "\x0c" not in doc
    ET.fromstring(doc)  # parses


def test_cli_document_formats_are_wellformed(monkeypatch, capsys, tmp_path):
    """-f hocr emits ONE parseable hOCR document; -f tsv emits ONE header
    with a page column; multi-image -f json emits ONE JSON array."""
    import json
    import xml.etree.ElementTree as ET

    from turboocr_engine import cli
    from turboocr_engine.result import DocumentResult, PageResult, TextLine

    def page(n, text):
        p = PageResult(width=100, height=50, page=n)
        p.lines.append(TextLine(text=text, confidence=0.9,
                                box=((1, 1), (50, 1), (50, 10), (1, 10))))
        return p

    doc = DocumentResult(pages=[page(1, "one"), page(2, "two")])

    class FakeOCR:
        def read_pdf(self, *a, **kw):
            return doc

        def read(self, path, **kw):
            return page(None, f"text-of-{path}")

    monkeypatch.setattr(cli, "_build_ocr", lambda args: FakeOCR())

    assert cli.main(["pdf", "d.pdf", "-f", "hocr"]) == 0
    hocr = capsys.readouterr().out
    ET.fromstring(hocr)  # one well-formed document
    assert 'id="page_1"' in hocr and 'id="page_2"' in hocr

    assert cli.main(["pdf", "d.pdf", "-f", "tsv"]) == 0
    tsv = capsys.readouterr().out.strip().splitlines()
    assert tsv[0].startswith("page\t")
    assert sum(1 for l in tsv if l.startswith("page\t")) == 1

    # SINGLE-input ocr emits the SAME document shapes as multi — the shape
    # must not depend on how many files a glob happened to match.
    assert cli.main(["ocr", "solo.png", "-f", "json"]) == 0
    d1 = json.loads(capsys.readouterr().out)
    assert list(d1) == ["pages"] and d1["pages"][0]["source"] == "solo.png"
    assert cli.main(["ocr", "solo.png", "-f", "tsv"]) == 0
    t1 = capsys.readouterr().out.strip().splitlines()
    assert t1[0].startswith("page\t")
    assert cli.main(["ocr", "solo.png", "-f", "hocr"]) == 0
    ET.fromstring(capsys.readouterr().out)

    out = tmp_path / "multi.json"
    assert cli.main(["ocr", "a.png", "b.png", "-f", "json",
                     "-o", str(out)]) == 0
    data = json.loads(out.read_text())
    assert [d["source"] for d in data["pages"]] == ["a.png", "b.png"]

    # multi-image hOCR is ONE complete document (used to stack N <html> docs)
    assert cli.main(["ocr", "a.png", "b.png", "-f", "hocr"]) == 0
    hocr2 = capsys.readouterr().out
    assert hocr2.count("<html") == 1
    ET.fromstring(hocr2)
    assert 'id="page_1"' in hocr2 and 'id="page_2"' in hocr2

    # --on-error skip: a corrupt file is noted on stderr, the good ones are
    # still emitted, and the exit code reports partial failure.
    class FlakyOCR(FakeOCR):
        def read(self, path, **kw):
            if path == "bad.png":
                raise ValueError("could not decode image file")
            return page(None, f"text-of-{path}")

    monkeypatch.setattr(cli, "_build_ocr", lambda args: FlakyOCR())
    rc = cli.main(["ocr", "a.png", "bad.png", "c.png", "-f", "json",
                   "-o", str(out), "--on-error", "skip"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "skipped bad.png" in captured.err
    data = json.loads(out.read_text())
    assert [d["source"] for d in data["pages"]] == ["a.png", "c.png"]

    monkeypatch.setattr(cli, "_build_ocr", lambda args: FakeOCR())

    # --password is plumbed through to read_pdf.
    seen = {}

    class PwOCR(FakeOCR):
        def read_pdf(self, *a, **kw):
            seen.update(kw)
            return doc

    monkeypatch.setattr(cli, "_build_ocr", lambda args: PwOCR())
    assert cli.main(["pdf", "d.pdf", "--password", "pw1"]) == 0
    capsys.readouterr()
    assert seen["password"] == "pw1"


def test_module_read_tables_key(monkeypatch):
    """turboocr_engine.read(tables=True) must build (and cache) a
    table-capable engine — the capability used to be missing from both the
    cache key and the constructor call, making it unreachable."""
    import turboocr_engine as te

    built = []

    class FakeOCR:
        def __init__(self, model, backend, *, layout=False, tables=False,
                     formulas=False, autorotate=False):
            built.append((layout, tables, formulas, autorotate))

        def read(self, image, **kw):
            return ("read", kw)

    monkeypatch.setattr(te, "OCR", FakeOCR)
    monkeypatch.setattr(te, "_DEFAULT_CACHE", type(te._DEFAULT_CACHE)())
    monkeypatch.setattr(te, "_DEFAULT_KEY_LOCKS", {})
    _, kw = te.read("x.png", tables=True)
    assert built == [(False, True, False, False)]
    assert kw["tables"] is True


def test_load_image_dtype_scaling_and_nan():
    """Non-uint8 arrays must SCALE, not clip: 16-bit scans, bool masks, and
    NaN-poisoned floats all used to become blank pages silently."""
    from turboocr_engine.imaging import load_image

    # uint16 gradient survives with spread (clipping saturated 99.6% to 255).
    g16 = (np.linspace(0, 65535, 64, dtype=np.uint16)
           .reshape(8, 8).repeat(3).reshape(8, 8, 3))
    out = load_image(g16)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert len(np.unique(out)) > 10

    # THE symmetric case: 8-bit VALUES in wide dtypes must pass through, not
    # be crushed to black by dtype-max scaling (PIL convert("I") -> int32,
    # tifffile 8-in-16, arr.astype(int) -> int64 — the common inputs).
    for dt in (np.uint16, np.int16, np.int32, np.int64, np.uint32):
        page = np.full((8, 8, 3), 255, dtype=dt)
        page[2:6, 2:6] = 0
        out = load_image(page)
        assert out.max() == 255 and out.min() == 0, dt

    # one +inf must not defeat the [0,1] rescale (posinf used to be mapped
    # to 255 BEFORE the range test, blackening everything else).
    f = np.full((4, 4, 3), 0.5, np.float32)
    f[0, 0, 0] = np.inf
    out = load_image(f)
    assert out[1, 1, 1] >= 127

    # bool mask -> 0/255, not 0/1.
    mask = np.zeros((4, 4), bool)
    mask[1, 1] = True
    out = load_image(mask)
    assert out.max() == 255

    # one NaN must not blacken the whole [0,1] float image.
    f = np.full((4, 4, 3), 0.5, np.float32)
    f[0, 0, 0] = np.nan
    out = load_image(f)
    assert out.max() >= 127  # scaled, not zeroed

    # empty inputs raise the documented ValueError, not cv2.error.
    with pytest.raises(ValueError, match="empty"):
        load_image(b"")


def test_load_image_pil_multiframe_refused():
    """A multi-frame PIL TIFF at frame 0 is refused (silent page drop), but a
    deliberately seek()ed frame and animated GIF/WebP frame 0 are honoured —
    and the error's own advice (seek + copy) must actually work."""
    from turboocr_engine.imaging import load_image

    class FakeTiff:  # duck-typed PIL image
        mode = "RGB"
        size = (4, 4)
        n_frames = 3
        format = "TIFF"
        _pos = 0

        def tell(self):
            return self._pos

        def convert(self, mode):
            return np.zeros((4, 4, 3), np.uint8)

    with pytest.raises(ValueError, match="3 frames"):
        load_image(FakeTiff())

    seeked = FakeTiff()
    seeked._pos = 2  # user picked their frame — that IS the image
    assert load_image(seeked).shape == (4, 4, 3)

    class FakeGif(FakeTiff):
        format = "GIF"  # frame 0 of an animation is the image, not data loss

    assert load_image(FakeGif()).shape == (4, 4, 3)

    class BrokenFrames(FakeTiff):
        @property
        def n_frames(self):
            raise OSError("broken file: cannot seek")

    # a raising n_frames property must not leak its own exception type
    assert load_image(BrokenFrames()).shape == (4, 4, 3)

    class OneFrame:
        mode = "RGB"
        size = (2, 2)
        n_frames = 1
        format = "TIFF"

        def convert(self, mode):
            return np.zeros((2, 2, 3), np.uint8)

    assert load_image(OneFrame()).shape == (2, 2, 3)


def test_sniff_dims_bmp_and_stuffed_jpeg():
    """BMP headers are sniffed pre-decode, and a stuffed FF 00 byte cannot
    desynchronize the JPEG walk past the ceiling."""
    import struct

    from turboocr_engine.imaging import _sniff_dims, load_image

    # INFOHEADER (dib=40): 32-bit signed dims at 18/22.
    bmp = (b"BM" + b"\x00" * 12 + struct.pack("<I", 40)
           + struct.pack("<ii", 30000, -30000) + b"\x00" * 20)
    assert _sniff_dims(bmp) == (30000, 30000)
    with pytest.raises(ValueError, match="megapixels"):
        load_image(bmp + b"\x00" * 40)

    # OS/2 BITMAPCOREHEADER (dib=12): UNSIGNED 16-bit dims at 18/20 — the
    # 32-bit misread rejected valid decodable BMPs with gigapixel nonsense.
    core = (b"BM" + b"\x00" * 12 + struct.pack("<I", 12)
            + struct.pack("<HH", 100, 50) + b"\x00" * 8)
    assert _sniff_dims(core) == (100, 50)
    # Unknown DIB size (a text file starting with "BM"): no dims claimed.
    assert _sniff_dims(b"BMW service manual, chapter 3" + b"\x00" * 20) is None

    jpg = (b"\xff\xd8" + b"\xff\x00"  # stuffed byte right after SOI
           + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08"
           + struct.pack(">HH", 50000, 50000) + b"\x03" + b"\x00" * 9)
    assert _sniff_dims(jpg) == (50000, 50000)

    # invalid env values must not disable the ceiling
    import os as _os

    from turboocr_engine.imaging import _max_image_mp

    for bad in ("-5", "nan", "abc"):
        _os.environ["TURBO_MAX_IMAGE_MP"] = bad
        try:
            assert _max_image_mp() == 96.0, bad
        finally:
            del _os.environ["TURBO_MAX_IMAGE_MP"]


def test_apple_bundle_det_probe_moves_strictly_last(tmp_path, monkeypatch):
    """With MANY graph.json files in the bundle (rec ladder + cls), the DET
    export's — the provisioned-probe — must move last of all, or an
    interrupted extraction leaves a cache the probe calls complete forever."""
    import io
    import tarfile as tf

    from turboocr_engine import models as m
    from turboocr_engine.catalog import find_model

    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.platform, "machine", lambda: "arm64")
    entry = find_model("small")
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "det_small.onnx").write_bytes(b"x")
    store = m.ModelStore(None, allow_download=True)
    store.local_dir = None
    store.cache_dir = str(cache)
    resolved = m.ResolvedModel(det=str(cache / "det_small.onnx"), rec="r",
                               dict="d", cls=None, name="small")

    names = [
        "rec_small/rec_b320/graph.json",
        "rec_small/rec_b480/graph.json",
        "cls/graph.json",
        "det_small/det_c992x768/graph.json",   # the probe file
        "rec_small/rec_b320/weights.bin",
    ]

    def fake_download(rel, dest):
        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as t:
            for name in names:
                info = tf.TarInfo(name)
                info.size = 2
                t.addfile(info, io.BytesIO(b"{}"))
        (cache / rel).write_bytes(buf.getvalue())
        return str(cache / rel)

    monkeypatch.setattr(store, "_download_apple_bundle", fake_download)

    order = []
    real_replace = os.replace

    def spy_replace(src, dst):
        order.append(os.path.relpath(dst, cache))
        return real_replace(src, dst)

    monkeypatch.setattr(m.os, "replace", spy_replace)
    assert store.ensure_apple_native(entry, resolved) is True
    assert order[-1] == os.path.join("det_small", "det_c992x768", "graph.json")
    # every non-det graph.json moved before it
    graphs = [o for o in order if o.endswith("graph.json")]
    assert graphs[-1].startswith("det_small")
    assert all(not g.startswith("det_small") for g in graphs[:-1])


def test_device_env_precedence(monkeypatch):
    """Documented precedence for device knobs: an explicit API argument
    wins; the API DEFAULT defers to the environment — CUDA_DEVICE_ID /
    OV_DEVICE / OPENVINO_DEVICE are operator interfaces per
    configuration.md, same as the DET_* overrides. (Only ORT_EP is derived
    state owned by backend= and always written.)"""
    from turboocr_engine import native as N

    monkeypatch.setattr(N, "resolve_engine", lambda b: "cpu")
    monkeypatch.setattr(N, "is_apple_silicon", lambda: False)
    monkeypatch.setenv("CUDA_DEVICE_ID", "3")
    N.configure_backend("cuda")           # default device_id=0: env wins
    assert os.environ["CUDA_DEVICE_ID"] == "3"
    N.configure_backend("cuda", device_id=2)  # explicit arg wins
    assert os.environ["CUDA_DEVICE_ID"] == "2"

    monkeypatch.setenv("OPENVINO_DEVICE", "GPU")
    monkeypatch.setattr(N, "resolve_engine", lambda b: "cpu")
    N.configure_backend("openvino")       # no device= : env survives
    assert os.environ["OPENVINO_DEVICE"] == "GPU"
    N.configure_backend("openvino", device="NPU")
    assert os.environ["OPENVINO_DEVICE"] == "NPU"

    monkeypatch.setenv("OV_DEVICE", "NPU")
    monkeypatch.setattr(N, "resolve_engine", lambda b: "intel")
    N.configure_backend("openvino")       # vendor path, no device=
    assert os.environ["OV_DEVICE"] == "NPU"
    # ORT_EP stays derived state: always cleared on the vendor path.
    assert "ORT_EP" not in os.environ

def test_log_level_authorship_by_value(monkeypatch):
    """LOG_LEVEL authorship is judged by VALUE: values we wrote may be
    updated by a later engine's verbosity; a user's value — set at ANY time,
    including after the first engine — always wins."""
    from turboocr_engine import native as N

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(N, "_log_level_written", None)
    N.set_log_level_default(False)
    assert os.environ["LOG_LEVEL"] == "warn"
    N.set_log_level_default(True)
    assert os.environ["LOG_LEVEL"] == "info"   # ours -> updated

    os.environ["LOG_LEVEL"] = "debug"          # user overrides LATER
    N.set_log_level_default(False)
    assert os.environ["LOG_LEVEL"] == "debug"  # never clobbered

    monkeypatch.setattr(N, "_log_level_written", None)
    monkeypatch.setenv("LOG_LEVEL", "error")   # pre-existing user value
    N.set_log_level_default(True)
    assert os.environ["LOG_LEVEL"] == "error"

def test_read_reading_order_reaches_the_engine():
    """read(reading_order=True) must route through run_with_layout (the gate
    auto-enables layout for it) and keep the returned order — it used to take
    the plain run() branch and silently return []."""
    import contextlib
    import types

    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._closed = False
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False

    calls = []

    class Pipe:
        def run(self, img):
            calls.append("run")
            return []

        def run_with_layout(self, img, *, layout, reading_order, tables,
                            formulas, text):
            calls.append(("run_with_layout", layout, reading_order))
            return types.SimpleNamespace(
                items=[], layout=[], tables=[], formulas=[],
                reading_order=[2, 0, 1], table_degraded=False,
                table_warning="", formula_degraded=False, formula_warning="",
                text_degraded=False, text_warning="",
            )

    pipe = Pipe()
    ocr._pipe = pipe
    ocr._checkout = lambda: contextlib.nullcontext(pipe)

    img = np.full((20, 30, 3), 255, np.uint8)
    res = P.OCR.read(ocr, img, reading_order=True)
    assert calls == [("run_with_layout", True, True)]
    assert res.reading_order == [2, 0, 1]


def _pdf_with_tounicode(tmp_path, glyph_cp):
    """A minimal raw PDF whose ToUnicode CMap maps 'A' to the given code
    point — the shape that makes PDFium report surrogate pairs per glyph."""
    hex_units = "".join(
        f"{u:04X}" for u in (
            [glyph_cp] if glyph_cp <= 0xFFFF else [
                0xD800 + ((glyph_cp - 0x10000) >> 10),
                0xDC00 + ((glyph_cp - 0x10000) & 0x3FF),
            ]))
    cmap = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
            b"/CMapName /X def 1 begincodespacerange <00> <FF> endcodespacerange\n"
            b"1 beginbfchar <41> <" + hex_units.encode() + b"> endbfchar\n"
            b"endcmap CMapName currentdict /CMap defineresource pop end end")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length 44 >>\nstream\nBT /F1 24 Tf 20 40 Td (AAA) Tj ET\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/ToUnicode 6 0 R >>",
        b"<< /Length " + str(len(cmap)).encode() + b" >>\nstream\n" + cmap
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (b"trailer\n<< /Size " + str(len(objs) + 1).encode()
            + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode()
            + b"\n%%EOF\n")
    p = tmp_path / f"cp{glyph_cp:x}.pdf"
    p.write_bytes(bytes(out))
    return str(p)


def test_pdf_astral_text_layer_is_valid_unicode(tmp_path):
    """A ToUnicode map yielding astral code points (emoji, SMP CJK) must
    produce REAL characters — lone surrogates made every to_json()/file
    write raise UnicodeEncodeError and silently blanked searchable PDFs."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine.pdf import extract_pdf_text

    p = _pdf_with_tounicode(tmp_path, 0x1F600)  # 😀
    (_page_no, _w, _h, lines), = list(extract_pdf_text(p))
    (text, _quad), = lines
    assert text == "\U0001F600" * 3
    text.encode("utf-8")  # must not raise
    assert len(text) == 3  # code points, not surrogate halves


def test_pdf_line_box_ignores_positioned_whitespace(tmp_path):
    """A positioned trailing-space run (tab leaders, empty right table cell)
    must not stretch the line box past the ink."""
    pytest.importorskip("pypdfium2")
    from reportlab.pdfgen import canvas as rl

    from turboocr_engine.pdf import extract_pdf_text

    p = str(tmp_path / "ws.pdf")
    c = rl.Canvas(p, pagesize=(612, 200))
    c.setFont("Helvetica", 14)
    c.drawString(50, 100, "Hello there this line is long enough to trust yes")
    c.drawString(500, 100, "   ")  # positioned whitespace on the same line
    c.showPage()
    c.save()
    (_, _w, _h, lines), = list(extract_pdf_text(p, dpi=72))
    text, quad = lines[0]
    assert text.startswith("Hello")
    assert max(pt[0] for pt in quad) < 450  # box ends at the glyphs, not 500+


def test_auto_gate_counts_visible_chars_not_utf16_units(tmp_path):
    """A multi-line stamp block (38 visible chars over 8 lines) must NOT be
    trusted — CountChars counted the \\r\\n separators and inflated it past
    the 50-char bar."""
    pytest.importorskip("pypdfium2")
    from reportlab.pdfgen import canvas as rl

    from turboocr_engine.pdf import iter_pdf_pages

    p = str(tmp_path / "stampblock.pdf")
    c = rl.Canvas(p, pagesize=(300, 200))
    c.setFont("Helvetica", 6)
    for k, t in enumerate(["BATES", "000123", "CONF", "EXH-A",
                           "P 1/9", "RECD", "FILED", "COPY"]):
        c.drawString(260, 180 - 12 * k, t)
    c.showPage()
    c.save()
    (kind, *_), = list(iter_pdf_pages(p, mode="auto"))
    assert kind == "img"  # rendered for OCR, stamp not trusted


def test_text_mode_contains_failures_and_flags_empty_pages(tmp_path, monkeypatch):
    """mode='text': a failing page is contained by on_error='skip' (used to
    end the whole extraction), and a layer-less page carries a no_text_layer
    warning instead of being indistinguishable from a blank page."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine import pdf as pdfmod
    from turboocr_engine import pipeline as P

    p = _digital_pdf(tmp_path, [["real digital text on page one here"], None])

    ocr = object.__new__(P.OCR)
    ocr.replicas = 1
    ocr.autorotate = False
    ocr.keep_image = None
    ocr._pdf_executor = None
    ocr._close_mu = P.threading.Lock()
    ocr._closed = False
    ocr._pipe = object()
    ocr.has_layout = ocr.has_tables = ocr.has_formulas = False

    doc = P.OCR.read_pdf(ocr, p, mode="text")
    assert doc.pages[0].lines and not doc.pages[0].warnings
    assert not doc.pages[1].lines
    assert doc.pages[1].warnings[0].startswith("no_text_layer:")

    real_extract = pdfmod._extract_page_text
    calls = {"n": 0}

    def broken_first(page, scale):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("textpage exploded")
        return real_extract(page, scale)

    monkeypatch.setattr(pdfmod, "_extract_page_text", broken_first)
    doc = P.OCR.read_pdf(ocr, p, mode="text", on_error="skip")
    assert doc.pages[0].warnings[0].startswith("page_failed: RuntimeError")
    assert doc.pages[1].warnings[0].startswith("no_text_layer:")

    calls["n"] = 0
    with pytest.raises(RuntimeError, match="textpage exploded"):
        P.OCR.read_pdf(ocr, p, mode="text")  # default still raises


def test_auto_mode_render_failure_keeps_layer_text(tmp_path, monkeypatch):
    """text_with_raster: when the STRUCTURE render fails on a page whose
    text layer extracted cleanly, the text survives with a warning — it used
    to be discarded wholesale."""
    pytest.importorskip("pypdfium2")
    from turboocr_engine import pdf as pdfmod

    p = _digital_pdf(tmp_path, [["good digital body text that the gate trusts",
                                 "second line to be extra trustworthy here"]])

    def always_broken(page, scale):
        raise RuntimeError("content stream corrupt")

    monkeypatch.setattr(pdfmod, "_render_page", always_broken)
    (payload,) = list(pdfmod.iter_pdf_pages(p, mode="auto", text_with_raster=True))
    kind, _page_no, _w, _h, lines, arr, warns = payload
    assert kind == "text" and lines and arr is None
    assert warns and warns[0].startswith("page_render_failed:")

    # And an extraction failure in auto mode falls through to render+OCR.
    monkeypatch.undo()
    monkeypatch.setattr(pdfmod, "_extract_page_text",
                        lambda page, scale: (_ for _ in ()).throw(RuntimeError("bad layer")))
    (payload,) = list(pdfmod.iter_pdf_pages(p, mode="auto"))
    assert payload[0] == "img"  # the page still rasterizes and OCRs


def test_apple_metal_axis_guard():
    """An image axis over Metal's 16384 texture cap must raise a catchable
    ValueError on the apple backend — it used to SIGABRT the whole process
    (uncatchable), and the AREA ceiling can't see it (17000x2000 = 34 MP)."""
    from turboocr_engine import pipeline as P

    ocr = object.__new__(P.OCR)
    ocr.engine = "apple"
    ocr.keep_image = None
    big = np.zeros((2000, 17000, 3), np.uint8)
    with pytest.raises(ValueError, match="16384"):
        P.OCR._read_array(ocr, big, drop_score=0.5)
    # cpu handles the same image (guard is backend-scoped) — prove the guard
    # itself doesn't fire; the fake pipe substitutes for real inference.
    import contextlib

    ocr2 = object.__new__(P.OCR)
    ocr2.engine = "cpu"
    ocr2.keep_image = False
    ocr2.has_layout = ocr2.has_tables = ocr2.has_formulas = False

    class Pipe:
        def run(self, img):
            return []

    ocr2._checkout = lambda: contextlib.nullcontext(Pipe())
    res = P.OCR._read_array(ocr2, big, drop_score=0.5)
    assert res.width == 17000


def test_from_dict_full_server_page_shape():
    """The remaining server keys parse too: orientation_deg, the degraded
    flag+message pairs (as warnings), table cells, the batch envelope, and
    the 'lines' container alias; malformed boxes fail with a named error."""
    from turboocr_engine.result import DocumentResult, PageResult, TableRegion

    bb = [[1, 2], [30, 2], [30, 12], [1, 12]]
    page = PageResult.from_dict({
        "width": 100, "height": 50, "orientation_deg": 90,
        "results": [{"text": "t", "confidence": 0.9, "bounding_box": bb}],
        "text_degraded": True, "text_warning": "det found boxes, rec empty",
        "table_degraded": True,
        "tables": [{"html": "<table></table>", "confidence": 0.8,
                    "bounding_box": bb,
                    "cells": [{"text": "a", "row": 0, "col": 0}]}],
    })
    assert page.orientation == 90
    assert "text_degraded: det found boxes, rec empty" in page.warnings
    assert "table_degraded: no detail" in page.warnings
    assert page.tables[0].cells == [{"text": "a", "row": 0, "col": 0}]
    # cells round-trip; absent stays absent
    assert "cells" in page.tables[0].to_dict()
    assert "cells" not in TableRegion(html="", score=0, box=tuple(map(tuple, bb))).to_dict()

    # 'lines' container alias no longer parses silently empty
    p2 = PageResult.from_dict({"lines": [{"text": "hi", "confidence": 1.0,
                                          "box": bb}]})
    assert p2.lines[0].text == "hi"

    # batch envelope + unknown-shape refusal
    doc = DocumentResult.from_dict({"batch_results": [{"width": 1, "height": 1,
                                                       "results": []}]})
    assert len(doc.pages) == 1
    with pytest.raises(ValueError, match="batch_results"):
        DocumentResult.from_dict({"totally": "wrong"})

    # malformed boxes fail with named errors, not min() confusion
    with pytest.raises(ValueError, match="bounding_box"):
        PageResult.from_dict({"results": [{"text": "x", "bounding_box": []}]})
    from turboocr_engine.result import LayoutBox

    lb = LayoutBox.from_dict({"label": None, "class": "table",
                              "confidence": 0.5, "bounding_box": bb})
    assert lb.label == "table"  # explicit null falls through to "class"


def test_markdown_cell_newline_and_backslash():
    from turboocr_engine.result import _html_table_to_markdown

    html = ("<table><tr><th>h1</th><th>h2</th></tr>"
            "<tr><td>a\nb</td><td>c\\|d</td></tr></table>")
    md = _html_table_to_markdown(html)
    for row in md.splitlines():
        assert row.startswith("|") and row.endswith("|")  # rows never split
    assert "a b" in md            # newline collapsed
    assert "c\\\\\\|d" in md      # backslash escaped BEFORE the pipe


def test_seam_only_backends_refuse_without_the_seam(monkeypatch):
    """backend='intel'/'amd'/'turbo' on a build without that seam must raise
    BackendUnavailable naming the wheel — 'intel' used to die with a
    ModelLoadError blaming backend 'cpu', and 'turbo' silently OCR'd on CPU
    where the docs promise a refusal."""
    from turboocr_engine import native as N
    from turboocr_engine.errors import BackendUnavailable

    monkeypatch.setattr(N, "resolve_engine", lambda b: "cpu")
    monkeypatch.setattr(N, "native_backends", lambda: ["cpu"])
    for backend, wheel_word in (("intel", "openvino"), ("amd", "rocm"),
                                ("turbo", "cuda12"), ("nvidia", "cuda12")):
        with pytest.raises(BackendUnavailable, match=wheel_word):
            N.ensure_backend_supported(backend)
    # openvino/apple carry an EP fallback in their rows: no seam refusal.
    N.ensure_backend_supported("cpu")


def test_ocr_mode_validated_and_ja_warns(monkeypatch):
    from turboocr_engine import pipeline as P

    with pytest.raises(ValueError, match="mode must be one of"):
        P.OCR("tiny", mode="nope")

    import warnings as W

    with W.catch_warnings(record=True) as rec:
        W.simplefilter("always")
        entry = P._resolve_entry(None, "ja", None)
    assert entry.name == "tiny"
    assert any("kana" in str(w.message) for w in rec)
    with W.catch_warnings(record=True) as rec:
        W.simplefilter("always")
        P._resolve_entry(None, "ja", "small")
    assert not rec  # explicit non-tiny tier: no warning


def test_wide_int_dtypes_scale_by_data_bit_depth():
    """A 16-bit scan held in int32/int64 (integer arithmetic widens uint16)
    must scale by the DATA's apparent bit depth, not the dtype max — the
    dtype divisor rounded every pixel to zero."""
    from turboocr_engine.imaging import load_image

    g = np.linspace(0, 65535, 64).astype(np.int64).reshape(8, 8)
    for dt in (np.int32, np.int64, np.uint32):
        out = load_image(g.astype(dt))
        assert out.min() == 0 and out.max() == 255, dt
        assert len(np.unique(out)) > 10, dt
    # float on a 16-bit scale: no longer saturated to two values
    out = load_image(g.astype(np.float32))
    assert out.max() == 255 and len(np.unique(out)) > 10


def test_from_dict_score_alias_on_lines_and_layout():
    from turboocr_engine.result import LayoutBox, TextLine

    bb = [[0, 0], [9, 0], [9, 9], [0, 9]]
    assert TextLine.from_dict({"text": "x", "score": 0.9, "box": bb}).confidence == 0.9
    assert LayoutBox.from_dict({"label": "t", "score": 0.7, "box": bb}).confidence == 0.7


def test_unknown_backend_passthrough_warns():
    import warnings as W

    from turboocr_engine import native as N

    with W.catch_warnings(record=True) as rec:
        W.simplefilter("always")
        try:
            N.configure_backend("bogus_ep_name")
        finally:
            os.environ.pop("ORT_EP", None)
    assert any("not a known backend name" in str(w.message) for w in rec)
