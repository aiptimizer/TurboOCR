"""Integration tests for the /infer per-backend routing endpoint.

/infer is the GPU-only table/formula override route. Historically it was a
DoS/SSRF surface (inline backends spinning up ORT-CUDA sessions on the request
thread, arbitrary base_url on inline openai backends). The endpoint had no
integration coverage at all — this file exercises its full validation and
security-gate ladder, which is deterministic on any server (text-only included,
since an unconfigured backend name is rejected the same way).

A successful inference needs a configured table/formula backend, so the happy
path is gated on /capabilities and skipped on a text-only server.
"""
import pytest
import requests

from conftest import pil_to_base64, hello_image  # noqa: F401 (fixture)


def _infer(server_url, payload=None, raw=None):
    kw = {"timeout": 15}
    if raw is not None:
        kw["data"] = raw
        kw["headers"] = {"Content-Type": "application/json"}
    else:
        kw["json"] = payload
    return requests.post(f"{server_url}/infer", **kw)


def _code(r):
    try:
        return r.json()["error"]["code"]
    except Exception:
        return None


@pytest.fixture(scope="module")
def has_infer(server_url):
    """Skip the whole module if this build didn't register /infer (CPU build)."""
    caps = requests.get(f"{server_url}/capabilities", timeout=5)
    if caps.status_code != 200 or "/infer" not in caps.text:
        pytest.skip("/infer not registered on this server (GPU-only route)")
    return True


@pytest.fixture(scope="module")
def img_b64(hello_image):
    return pil_to_base64(hello_image, "PNG")


class TestInferValidation:
    def test_non_json_body_rejected(self, server_url, has_infer):
        r = _infer(server_url, raw=b"this is not json")
        assert r.status_code == 400
        assert _code(r) == "INVALID_JSON"

    def test_missing_image_rejected(self, server_url, has_infer):
        r = _infer(server_url, {"modality": "table", "backend": "x"})
        assert r.status_code == 400
        assert _code(r) == "MISSING_IMAGE"

    def test_empty_image_rejected(self, server_url, has_infer):
        r = _infer(server_url, {"image": "", "modality": "table", "backend": "x"})
        assert r.status_code == 400
        assert _code(r) == "MISSING_IMAGE"

    def test_missing_modality_rejected(self, server_url, has_infer, img_b64):
        r = _infer(server_url, {"image": img_b64, "backend": "x"})
        assert r.status_code == 400
        assert _code(r) == "INVALID_PARAMETER"

    def test_bad_modality_rejected(self, server_url, has_infer, img_b64):
        r = _infer(server_url, {"image": img_b64, "modality": "banana", "backend": "x"})
        assert r.status_code == 400
        assert _code(r) == "INVALID_PARAMETER"

    def test_non_string_modality_rejected(self, server_url, has_infer, img_b64):
        r = _infer(server_url, {"image": img_b64, "modality": 7, "backend": "x"})
        assert r.status_code == 400
        assert _code(r) == "INVALID_PARAMETER"

    def test_missing_backend_rejected(self, server_url, has_infer, img_b64):
        r = _infer(server_url, {"image": img_b64, "modality": "table"})
        assert r.status_code == 400
        assert _code(r) == "INVALID_PARAMETER"

    def test_non_string_non_object_backend_rejected(self, server_url, has_infer, img_b64):
        r = _infer(server_url, {"image": img_b64, "modality": "table", "backend": 7})
        assert r.status_code == 400
        assert _code(r) == "INVALID_PARAMETER"

    def test_unknown_named_backend_rejected(self, server_url, has_infer, img_b64):
        r = _infer(server_url,
                   {"image": img_b64, "modality": "formula", "backend": "no_such_backend"})
        assert r.status_code == 400
        assert _code(r) == "ROUTING_UNKNOWN_OVERRIDE"


class TestInferSecurityGates:
    def test_inline_openai_backend_forbidden_by_default(self, server_url, has_infer, img_b64):
        """SSRF gate: an inline kind:openai backend names an arbitrary base_url;
        it must be refused unless the operator sets TURBO_ALLOW_ADHOC_BACKENDS."""
        r = _infer(server_url, {
            "image": img_b64, "modality": "formula",
            "backend": {"kind": "openai", "base_url": "http://169.254.169.254/",
                        "model": "x"},
        })
        assert r.status_code == 403
        assert _code(r) == "ADHOC_BACKENDS_DISABLED"

    def test_inline_local_backend_rejected(self, server_url, has_infer, img_b64):
        """DoS gate: an inline kind:local backend would build ORT-CUDA sessions on
        the request thread — must be rejected (name an already-loaded backend)."""
        r = _infer(server_url, {
            "image": img_b64, "modality": "table",
            "backend": {"kind": "local", "onnx": "/tmp/evil.onnx"},
        })
        # The invariant is that it is REFUSED before any local engine is built —
        # the exact code varies (ad-hoc-local guard, or the spec is rejected as an
        # unknown/invalid engine first). A policy refusal (ADHOC_*) answers 403
        # since 2026-08-02; a malformed/unknown spec answers 400.
        code = _code(r) or ""
        assert (r.status_code, bool(code.startswith("ADHOC_"))) in ((403, True), (400, False)), \
            f"unexpected refusal shape: {r.status_code} {code}"
        assert code.startswith(("ROUTING_", "ADHOC_")) or code == "INVALID_PARAMETER", \
            f"inline local backend not refused with a routing error: {code}"
