"""The library is SILENT by default — enforced, not promised.

docs/reference/python.md claims zero unrequested lines on stdout and stderr.
Before this test the claim was re-measured by hand whenever someone remembered;
these tests make a regression a red build instead of a user's bug report.

Mechanics: the engine runs in a SUBPROCESS with captured pipes, because the
noise this guards against is written by C++ (ORT session chatter, NSLog,
std::cerr banners, OpenCV's logger) straight to the process's file
descriptors — an in-process capture of sys.stdout/sys.stderr would miss all
of it. Skipped when the models are not already cached: a silence test must
not print a download progress line, and CI without the cache should skip,
not fetch.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]

#: Env that changes verbosity or writes to stdout by design — stripped so the
#: subprocess measures the DEFAULT, which is what the docs promise.
_VERBOSITY_ENV = ("LOG_LEVEL", "PROFILE_STAGES", "TURBO_OPENCV_LOG",
                  "ORT_SHARED_POOL", "VERBOSE")


def _cached(*rels: str) -> bool:
    """True when every asset is already resolvable WITHOUT a download."""
    from turboocr_engine.models import ModelStore

    store = ModelStore(None, allow_download=False)
    try:
        for rel in rels:
            store.ensure_asset(rel)
    except Exception:
        return False
    return True


def _tiny_cached() -> bool:
    from turboocr_engine.catalog import resolve_model
    from turboocr_engine.models import ModelStore

    try:
        ModelStore(None, allow_download=False).resolve(resolve_model("tiny"))
    except Exception:
        return False
    return True


def _run_silent(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _VERBOSITY_ENV}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=_PKG_DIR, env=env, timeout=300,
    )


@pytest.mark.skipif(not _tiny_cached(), reason="tiny models not in cache")
def test_default_engine_is_silent():
    r = _run_silent(
        "import numpy as np\n"
        "import turboocr_engine as t\n"
        "o = t.OCR()\n"
        "o.read(np.zeros((64, 256, 3), np.uint8))\n"
        "o.close()\n"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == "", f"stdout not silent:\n{r.stdout}"
    assert r.stderr == "", f"stderr not silent:\n{r.stderr}"


@pytest.mark.skipif(
    not (_tiny_cached() and _cached(
        "layout/layout.onnx",
        "table/slanext_encoder/SLANeXt_wired_encoder.onnx",
        "table/slanext_encoder/SLANeXt_wired_decoder.bin",
        "table/slanext_encoder/SLANeXt_dict_infer.txt",
        "formula/ppformulanet_s/inference_trt.onnx",
        "formula/ppformulanet_s/tokenizer.json",
        "doc_ori.onnx",
    )),
    reason="stage models not in cache",
)
def test_fully_loaded_engine_is_silent():
    # Every optional stage constructs AND runs — historically each stage had
    # its own banner ([CpuFormula], [cpu_layout], NSLog, OpenCV's MatExpr
    # warning), so construction-plus-first-read is exactly where noise lives.
    r = _run_silent(
        "import numpy as np\n"
        "import turboocr_engine as t\n"
        "o = t.OCR(layout=True, tables=True, formulas=True, autorotate=True)\n"
        "o.read(np.zeros((64, 256, 3), np.uint8),\n"
        "       layout=True, tables=True, formulas=True)\n"
        "o.close()\n"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == "", f"stdout not silent:\n{r.stdout}"
    assert r.stderr == "", f"stderr not silent:\n{r.stderr}"
