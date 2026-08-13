"""THE GUARD TEST: every capability must work on every endpoint, both ways.

WHY THIS EXISTS
---------------
Capabilities (layout / tables / formulas / autorotate) were wired per-endpoint by
hand: `server_main.cpp` threaded the same three booleans into six different
registrar signatures — and the gRPC one takes them in a DIFFERENT ORDER
(layout, readiness, table, formula), so transposing two of them compiles
cleanly and silently disables a feature. Nothing caught that.

Worse, each endpoint parsed the client's REQUEST its own way. Measured against a
server with layout+tables+formulas+doc_ori all loaded (`Stages loaded layout=1
table=1 formula=1 doc_ori=1`, /capabilities reporting all true):

    POST /ocr  {"image": ..., "layout": true, "tables": true}   -> 200, layout 0, tables 0
    POST /ocr?layout=1                                          -> 200, layout 22   (works)
    POST /ocr?tables=1                                          -> 200, tables 0

i.e. a client can ask for a capability the server fully supports, receive 200,
and get nothing back — no error, no warning field. That is the bug class this
file exists to make impossible to reintroduce.

CONTRACT ASSERTED HERE
----------------------
1. Every capability the server ADVERTISES in /capabilities must be REQUESTABLE
   on every endpoint that returns page content.
2. Query-parameter and JSON-body forms must behave IDENTICALLY.
3. Asking for a capability the server did NOT load must fail LOUDLY (4xx, or a
   warning in the response) — never a silent 200 with the field missing.

Adding a capability therefore means adding one row to CAPABILITIES below; if any
endpoint or request form does not honour it, this test fails.

RUNNING
-------
    TURBO_BASE_URL=http://127.0.0.1:8080 pytest tests/integration/service/test_capability_matrix.py
Skips cleanly when no server is reachable, so it is safe to collect anywhere.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
import urllib.error

import pytest

BASE = os.environ.get("TURBO_BASE_URL", "http://127.0.0.1:8080")
FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures", "images", "png", "receipt.png",
)

# capability -> (request key, response field in the page JSON)
#
# This list MUST match include/turbo_ocr/core/capability_table.def. It is
# duplicated here on purpose: the whole point of the guard is to check the C++
# registry from OUTSIDE the process, so deriving it from the same source would
# make the test agree with a broken server by construction.
#
# `autorotate` has no response field of its own — it rotates the page before
# OCR — so it is exercised only by the availability/consistency checks below,
# not by the response-field assertion.
CAPABILITIES = [
    ("layout", "layout", "layout"),
    ("tables", "tables", "tables"),
    ("formulas", "formulas", "formulas"),
    ("autorotate", "autorotate", None),
]

# Endpoints that return page content and must honour every capability.
CONTENT_ENDPOINTS = ["/ocr"]


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.load(r)


def _post(path, body):
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, None


def _server_caps():
    try:
        return _get("/capabilities").get("features", {})
    except Exception:
        return None


CAPS = _server_caps()
needs_server = pytest.mark.skipif(CAPS is None, reason=f"no server at {BASE}")
needs_fixture = pytest.mark.skipif(not os.path.exists(FIXTURE), reason="no fixture image")


@pytest.fixture(scope="module")
def image_b64():
    return base64.b64encode(open(FIXTURE, "rb").read()).decode()


@needs_server
@needs_fixture
@pytest.mark.parametrize("cap,req_key,resp_key", CAPABILITIES)
@pytest.mark.parametrize("endpoint", CONTENT_ENDPOINTS)
def test_capability_is_requestable_via_query(cap, req_key, resp_key, endpoint, image_b64):
    """An ADVERTISED capability must be ACCEPTED via query parameter.

    Deliberately does NOT assert the response field is present: the emitter
    omits empty arrays by design (a documented byte-compatibility contract), so
    a page with no tables legitimately has no `tables` key. Presence is asserted
    where it is meaningful — test_formula_content_is_returned below, on a
    fixture that actually contains formulas.

    What this DOES catch is the request being rejected or erroring, i.e. a
    capability advertised as available that no endpoint can actually serve.
    """
    if not CAPS.get(cap):
        pytest.skip(f"server did not load {cap}")
    status, body = _post(f"{endpoint}?{req_key}=1", {"image": image_b64})
    assert status == 200, (
        f"{endpoint} advertises {cap}=true in /capabilities but "
        f"{endpoint}?{req_key}=1 returned HTTP {status}"
    )


@needs_server
@needs_fixture
@pytest.mark.parametrize("cap,req_key,resp_key", CAPABILITIES)
@pytest.mark.parametrize("endpoint", CONTENT_ENDPOINTS)
def test_query_and_body_forms_agree(cap, req_key, resp_key, endpoint, image_b64):
    """The JSON-body form must behave identically to the query form.

    This is the assertion that fails today: the body form is ignored outright.
    """
    if not CAPS.get(cap):
        pytest.skip(f"server did not load {cap}")
    if resp_key is None:
        pytest.skip(f"{cap} has no response field to compare")
    _, via_query = _post(f"{endpoint}?{req_key}=1", {"image": image_b64})
    _, via_body = _post(endpoint, {"image": image_b64, req_key: True})
    assert (resp_key in via_query) == (resp_key in via_body), (
        f"{endpoint}: '{req_key}' honoured as a query param but NOT in the JSON "
        f"body (or vice versa) — request parsing is not shared across forms"
    )


@needs_server
@needs_fixture
def test_unsupported_capability_fails_loudly(image_b64):
    """Asking for something the server did NOT load must never be a silent 200.

    Either reject it (4xx) or return it with a warning — but do not accept the
    request and quietly omit the result, which is indistinguishable from
    "this page genuinely had no tables".
    """
    missing = [c for c, _, _ in CAPABILITIES if not CAPS.get(c)]
    if not missing:
        pytest.skip("this server loaded every capability")
    cap = missing[0]
    status, body = _post(f"/ocr?{cap}=1", {"image": image_b64})
    loud = status >= 400 or (body is not None and body.get("warnings"))
    assert loud, (
        f"requested {cap}, which this server did not load, and got a clean "
        f"HTTP {status} with no warning — the client cannot tell that its "
        f"request was dropped"
    )


@needs_server
def test_capabilities_advertises_every_capability():
    """/capabilities must carry a key for every capability, true OR false.

    Reporting only the loaded ones is indistinguishable, to a client, from a
    build that never had the capability at all — and it is the failure mode that
    lets a capability be added to the request parser but forgotten in the one
    place clients use to discover it.
    """
    for cap, _, _ in CAPABILITIES:
        assert cap in CAPS, (
            f"/capabilities does not mention '{cap}' at all; a client cannot "
            f"tell 'not loaded here' from 'this build never supported it'"
        )
        assert isinstance(CAPS[cap], bool), (
            f"/capabilities features.{cap} is {CAPS[cap]!r}, expected a bool"
        )


@needs_server
@needs_fixture
@pytest.mark.parametrize("cap,req_key,resp_key", CAPABILITIES)
def test_rejection_uses_the_documented_error_code(cap, req_key, resp_key, image_b64):
    """A capability the server did NOT load must be refused with ITS code.

    One stable code per condition (capability_table.def owns them), so two
    endpoints cannot answer the same rejection differently.
    """
    if CAPS.get(cap):
        pytest.skip(f"server loaded {cap}")
    expected = {
        "layout": "LAYOUT_DISABLED",
        "tables": "TABLE_BACKEND_DISABLED",
        "formulas": "FORMULA_BACKEND_DISABLED",
        "autorotate": "AUTOROTATE_DISABLED",
    }[cap]
    req = urllib.request.Request(
        f"{BASE}/ocr?{req_key}=1",
        json.dumps({"image": image_b64}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180):
            pytest.fail(f"{cap}=1 was accepted but the server did not load it")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        assert expected in detail, (
            f"{cap}=1 rejected with {detail!r}; expected the documented "
            f"{expected}"
        )


@needs_server
def test_formula_content_is_returned():
    """A capability with a fixture that genuinely exercises it must produce output.

    The receipt fixture above has no formulas, so its missing `formulas` key is
    honest. This case uses a PDF that does contain them, which is what makes
    "requested and actually ran" distinguishable from "requested and silently
    dropped" — the defect this whole file guards.
    """
    if not CAPS.get("formulas"):
        pytest.skip("server did not load formulas")
    pdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fixtures", "pdf", "formulas.pdf",
    )
    if not os.path.exists(pdf):
        pytest.skip("no formulas.pdf fixture")
    req = urllib.request.Request(
        f"{BASE}/ocr/pdf?formulas=1", open(pdf, "rb").read(),
        {"Content-Type": "application/pdf"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        doc = json.load(r)
    found = sum(len(p.get("formulas", [])) for p in doc.get("pages", []))
    assert found > 0, (
        "formulas=1 on a PDF full of formulas returned none — the capability "
        "was accepted but did not run"
    )


@needs_server
@needs_fixture
def test_pdf_only_capability_is_tolerated_on_image_endpoints(image_b64):
    """?autorotate=1 on /ocr must behave like any other unsupported parameter.

    autorotate is a page-level PDF capability; /ocr cannot run it. The v3.5.0
    contract (default, lenient mode) is 200 + an x-ignored-params header — NOT
    a 400 on the capability's availability, and NOT a silent no-op with no
    header. This regressed once when the shared parser briefly parsed every
    capability on every endpoint; EndpointSpec.acts_on is what prevents that.

    Skipped under strict mode (TURBO_OCR_STRICT_QUERY_PARAMS=1 rejects every
    unsupported param with 400 INVALID_PARAMETER, which is also correct).
    """
    req = urllib.request.Request(
        f"{BASE}/ocr?autorotate=1",
        json.dumps({"image": image_b64}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            ignored = r.headers.get("x-ignored-params", "")
            assert "autorotate" in ignored, (
                f"lenient /ocr?autorotate=1 returned {r.status} without "
                f"x-ignored-params naming it (got {ignored!r}) — the client "
                f"cannot tell the flag was not honoured"
            )
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        assert e.code == 400 and "INVALID_PARAMETER" in detail, (
            f"/ocr?autorotate=1 -> HTTP {e.code} {detail!r}; only strict-mode "
            f"INVALID_PARAMETER is an acceptable rejection (AUTOROTATE_DISABLED "
            f"here means the endpoint gated a capability it cannot run)"
        )
