"""End-to-end tests for FORMULA_BACKEND=auto (per-crop CJK escalation).

The routing logic itself is unit-tested in tests/cpp/common/test_cjk_stats.cpp; this
module proves the composite backend behaves end-to-end:
  1. `auto` loads BOTH children (-S and plus-M) or refuses to serve formulas.
  2. On pure-EN formula content, `auto` output is byte-identical to a pure
     `ppformulanet_s` server — the no-EN-regression guarantee.
  3. (optional) On a CJK page, `auto` diverges from -S — escalation observable.

Opt-in: spawns GPU servers and needs both formula bundles, so it is skipped
unless TURBO_FORMULA_TESTS=1. The CJK-divergence leg additionally needs
OMNIDOC_CJK_IMAGE=<path to a Chinese formula page image> (dataset not shipped).
"""
import importlib.util
import os
from pathlib import Path

import pytest
import requests

# Load the server-spawning helpers by path (these dirs are packages, so a bare
# `from test_boot_config import ...` fails when pytest imports us as
# integration.structure.test_formula_auto_cjk). The helpers live with the
# boot-config suite under integration/service/.
_spec = importlib.util.spec_from_file_location(
    "_boot_config_helpers",
    Path(__file__).parents[1] / "service" / "test_boot_config.py")
_boot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_boot)
_Server = _boot._Server
REPO_ROOT = _boot.REPO_ROOT

pytestmark = pytest.mark.skipif(
    os.environ.get("TURBO_FORMULA_TESTS") != "1",
    reason="formula backend tests spawn GPU servers; set TURBO_FORMULA_TESTS=1",
)

FORMULA_S = REPO_ROOT / "models" / "formula" / "ppformulanet_s"
FORMULA_PM = REPO_ROOT / "models" / "formula" / "ppformulanet_plus_m"
EN_PDF = REPO_ROOT / "tests" / "fixtures" / "pdf" / "formulas.pdf"


def _need_models():
    if not (FORMULA_S.is_dir() and FORMULA_PM.is_dir()):
        pytest.skip("both formula bundles (ppformulanet_s + ppformulanet_plus_m) required")


def _formulas_from_pdf(url, pdf_bytes):
    r = requests.post(f"{url}/ocr/pdf?layout=1&formulas=1", data=pdf_bytes,
                      headers={"Content-Type": "application/pdf"}, timeout=300)
    assert r.status_code == 200, r.text[:300]
    out = []
    for page in r.json()["pages"]:
        for f in page.get("formulas", []):
            out.append(f["latex"])
    return out


def _formulas_from_image(url, img_bytes):
    r = requests.post(f"{url}/ocr/raw?layout=1&formulas=1", data=img_bytes,
                      headers={"Content-Type": "image/png"}, timeout=300)
    assert r.status_code == 200, r.text[:300]
    return sorted((f["layout_id"], f["latex"]) for f in r.json()["formulas"])


def test_auto_loads_both_children_and_matches_s_on_english():
    """auto must announce both children at boot, produce formulas on the EN
    fixture, and be byte-identical to a pure -S server on the same input
    (pure-EN crops never escalate -> zero EN regression)."""
    _need_models()
    if not EN_PDF.exists():
        pytest.skip("formulas.pdf fixture missing")
    pdf = EN_PDF.read_bytes()

    srv = _Server({"FORMULA_BACKEND": "auto"})
    try:
        srv.wait_ready()
        # stdout is a pipe; the load line was already written during boot.
        auto_latex = _formulas_from_pdf(srv.url, pdf)
        assert auto_latex, "FORMULA_BACKEND=auto produced no formulas on the EN fixture"
    finally:
        srv.close()

    srv_s = _Server({"FORMULA_BACKEND": "ppformulanet_s"})
    try:
        srv_s.wait_ready()
        s_latex = _formulas_from_pdf(srv_s.url, pdf)
    finally:
        srv_s.close()

    assert auto_latex == s_latex, (
        "auto diverged from -S on pure-EN content — escalation fired where it "
        "must not (EN regression)")


def test_auto_escalates_on_cjk_page():
    """On a Chinese formula page, auto must diverge from -S (plus-M rewrote
    the crops). Needs OMNIDOC_CJK_IMAGE since the dataset isn't shipped."""
    _need_models()
    img_path = os.environ.get("OMNIDOC_CJK_IMAGE")
    if not img_path or not Path(img_path).exists():
        pytest.skip("set OMNIDOC_CJK_IMAGE=<chinese formula page png> to run")
    img = Path(img_path).read_bytes()

    srv = _Server({"FORMULA_BACKEND": "auto"})
    try:
        srv.wait_ready()
        auto_out = _formulas_from_image(srv.url, img)
        assert auto_out, "no formulas detected on the CJK page"
    finally:
        srv.close()

    srv_s = _Server({"FORMULA_BACKEND": "ppformulanet_s"})
    try:
        srv_s.wait_ready()
        s_out = _formulas_from_image(srv_s.url, img)
    finally:
        srv_s.close()

    differing = sum(1 for a, s in zip(auto_out, s_out) if a != s)
    assert differing > 0, (
        "auto output identical to -S on a Chinese formula page — CJK "
        "escalation did not fire")


def test_auto_refuses_boot_without_plus_m(tmp_path):
    """FORMULA_BACKEND=auto with the plus-M bundle missing must fail loud at
    boot (both children are required), not serve -S silently."""
    _need_models()
    # Point FORMULA_ONNX at a copy of the -S dir whose sibling plus_m is absent.
    import shutil
    fake_root = tmp_path / "formula"
    shutil.copytree(FORMULA_S, fake_root / "ppformulanet_s", symlinks=True)
    srv = _Server({
        "FORMULA_BACKEND": "auto",
        "FORMULA_ONNX": str(fake_root / "ppformulanet_s"),
        "FORMULA_TOKENIZER": str(fake_root / "ppformulanet_s" / "tokenizer.json"),
    })
    try:
        with pytest.raises((RuntimeError, TimeoutError)):
            srv.wait_ready(timeout=60)
    finally:
        srv.close()
