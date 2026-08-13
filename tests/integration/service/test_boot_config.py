"""End-to-end boot-configuration regressions.

The unit tests in tests/cpp/detection/test_det_postprocess.cpp cover the DET_LIMIT_TYPE
resolution math deterministically. This module closes the loop the unit tests
can't: it boots a real server under the reporter's exact env and asserts that
text actually comes back — the class of bug (#23) where a legal-looking env
combination silently produces zero detections.

Opt-in: these tests spawn a fresh GPU server, so they are skipped unless
TURBO_BOOT_TESTS=1 (they would otherwise contend for the GPU with the
already-running server the rest of the suite targets). The binary is located at
build/turboocr-server relative to the repo root, or via TURBO_SERVER_BIN.
"""
import io
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(
    os.environ.get("TURBO_BOOT_TESTS") != "1",
    reason="boot-config tests spawn a GPU server; set TURBO_BOOT_TESTS=1 to run",
)


def _server_bin():
    p = os.environ.get("TURBO_SERVER_BIN")
    cand = Path(p) if p else REPO_ROOT / "build" / "turboocr-server"
    if not cand.exists():
        pytest.skip(f"server binary not found at {cand}")
    return cand


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _big_text_image():
    """Large image whose text is destroyed if the server thumbnails to 64px
    (the #23 failure mode) but readable at native resolution."""
    img = Image.new("RGB", (1600, 400), "white")
    d = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 90)
    except Exception:
        font = None
    d.text((60, 140), "BOOT CONFIG OK 2026", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _Server:
    def __init__(self, env_extra):
        self.port = _free_port()
        env = dict(os.environ)
        env.update({
            "PORT": str(self.port),
            "GRPC_PORT": str(_free_port()),
            "PIPELINE_POOL_SIZE": "1",
            "OCR_MODEL": env.get("OCR_MODEL", "tiny"),
        })
        env.update(env_extra)
        self.env = env
        self.proc = subprocess.Popen(
            [str(_server_bin())], cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def wait_ready(self, timeout=180):
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"server exited early:\n{out[-2000:]}")
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("server did not become healthy")

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


def _run_ocr(url, data):
    r = requests.post(f"{url}/ocr/raw", data=data,
                      headers={"Content-Type": "image/png"}, timeout=60)
    assert r.status_code == 200, r.text[:300]
    return r.json()["results"]


def test_det_limit_type_max_still_recognises_text():
    """Regression (GitHub #23): DET_LIMIT_TYPE=max with layout/table/formula off
    used to thumbnail every image to 64px -> zero detections. It must now keep
    native resolution and recover the text."""
    srv = _Server({
        "DET_LIMIT_TYPE": "max",
        "DISABLE_LAYOUT": "1",
        "FORMULA_BACKEND": "",
        "TABLE_BACKEND": "",
    })
    try:
        srv.wait_ready()
        results = _run_ocr(srv.url, _big_text_image())
        text = " ".join(r["text"] for r in results).lower()
        assert results, "DET_LIMIT_TYPE=max produced zero detections (issue #23)"
        assert any(t in text for t in ("boot", "config", "2026")), \
            f"text not recovered under DET_LIMIT_TYPE=max: {text!r}"
    finally:
        srv.close()


def test_det_limit_type_max_with_explicit_cap():
    """The reporter's corrected config (DET_MAX_SIDE_LIMIT=2560) must also work."""
    srv = _Server({
        "DET_LIMIT_TYPE": "max",
        "DET_MAX_SIDE_LIMIT": "2560",
        "DISABLE_LAYOUT": "1",
    })
    try:
        srv.wait_ready()
        results = _run_ocr(srv.url, _big_text_image())
        assert results, "max policy + explicit cap produced zero detections"
    finally:
        srv.close()


def test_garbage_det_env_fails_fast():
    """No-silent-failure contract: a garbage/out-of-range DET env must make the
    server REFUSE TO BOOT with a config error, not silently clamp-and-serve (and
    definitely not thumbnail every image to nothing, the #23 failure mode). This
    is the boot-validation half of the #23 hardening — the unit tests cover the
    clamp math; this proves the server actually exits on bad config."""
    srv = _Server({
        "DET_LIMIT_TYPE": "max",
        "DET_MAX_SIDE_LIMIT": "0",       # out of allowed range [32, 4096]
        "DET_LIMIT_SIDE_LEN": "junk",    # not an integer
        "DISABLE_LAYOUT": "1",
    })
    try:
        with pytest.raises((RuntimeError, TimeoutError)) as exc:
            srv.wait_ready(timeout=30)
        assert srv.proc.poll() is not None, "server kept running on invalid config"
        assert "config error" in str(exc.value).lower() \
            or "invalid configuration" in str(exc.value).lower(), \
            f"server exited but not with a config error: {exc.value}"
    finally:
        srv.close()
