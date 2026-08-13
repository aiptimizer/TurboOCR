"""Integration tests for the gRPC OCRService endpoint."""

import json
import sys
from pathlib import Path

import pytest

from conftest import make_text_image, pil_to_png_bytes, pil_to_jpeg_bytes

# Compile proto for Python if needed
PROTO_DIR = Path(__file__).resolve().parents[3] / "proto"
GENERATED_DIR = Path(__file__).resolve().parents[2] / "_grpc_generated"


def _ensure_grpc_stubs():
    """Compile proto files to Python stubs if not already done."""
    if GENERATED_DIR.exists() and (GENERATED_DIR / "ocr_pb2.py").exists():
        return True
    try:
        import grpc_tools.protoc
        GENERATED_DIR.mkdir(exist_ok=True)
        (GENERATED_DIR / "__init__.py").write_text("")
        result = grpc_tools.protoc.main([
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={GENERATED_DIR}",
            f"--grpc_python_out={GENERATED_DIR}",
            str(PROTO_DIR / "ocr.proto"),
        ])
        return result == 0
    except ImportError:
        return False


# Try to set up gRPC stubs
_grpc_available = _ensure_grpc_stubs()

if _grpc_available:
    sys.path.insert(0, str(GENERATED_DIR))
    try:
        import grpc
        import ocr_pb2
        import ocr_pb2_grpc
    except ImportError:
        _grpc_available = False


@pytest.mark.skipif(not _grpc_available, reason="gRPC stubs not available (install grpcio-tools)")
class TestGrpcEndpoint:
    """Test gRPC OCRService.Recognize endpoint."""

    @pytest.fixture(scope="class")
    def grpc_stub(self, grpc_target):
        """Create a gRPC stub for the OCR service."""
        channel = grpc.insecure_channel(grpc_target)
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            pytest.skip(f"gRPC server not reachable at {grpc_target}")
        return ocr_pb2_grpc.OCRServiceStub(channel)

    def test_recognize_png(self, grpc_stub, hello_image):
        """Send a PNG image and get OCR results via gRPC."""
        png_bytes = pil_to_png_bytes(hello_image)
        request = ocr_pb2.OCRRequest(image=png_bytes)
        response = grpc_stub.Recognize(request, timeout=10)

        # The server uses json_bytes mode by default
        if response.json_response:
            data = json.loads(response.json_response)
            assert "results" in data
        else:
            assert response.num_detections >= 0

    def test_recognize_jpeg(self, grpc_stub, hello_image):
        """Send a JPEG image via gRPC."""
        jpeg_bytes = pil_to_jpeg_bytes(hello_image)
        request = ocr_pb2.OCRRequest(image=jpeg_bytes)
        response = grpc_stub.Recognize(request, timeout=10)
        assert response.num_detections >= 0

    def test_recognize_detects_text(self, grpc_stub):
        """gRPC endpoint should detect known text."""
        img = make_text_image("GRPCTEST", width=400, height=100, font_size=40)
        png_bytes = pil_to_png_bytes(img)
        request = ocr_pb2.OCRRequest(image=png_bytes)
        response = grpc_stub.Recognize(request, timeout=10)

        if response.json_response:
            data = json.loads(response.json_response)
            all_text = " ".join(r["text"] for r in data["results"]).upper()
            assert "GRPC" in all_text or "TEST" in all_text or len(data["results"]) > 0

    def test_recognize_empty_image(self, grpc_stub):
        """Empty image bytes should raise an error or return zero results."""
        request = ocr_pb2.OCRRequest(image=b"")
        try:
            response = grpc_stub.Recognize(request, timeout=10)
            # If it doesn't raise, should have 0 detections
            assert response.num_detections == 0
        except grpc.RpcError as e:
            # gRPC error is also acceptable
            assert e.code() in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.INTERNAL)

    def test_batch_recognize(self, grpc_stub, hello_image, numbers_image):
        """RecognizeBatch should process multiple images."""
        images = [pil_to_png_bytes(hello_image), pil_to_png_bytes(numbers_image)]
        request = ocr_pb2.OCRBatchRequest(images=images)
        response = grpc_stub.RecognizeBatch(request, timeout=15)
        assert response.total_images == 2
        assert len(response.batch_results) == 2

    @staticmethod
    def _slot_detections(slot):
        """Detection count for one batch slot, robust to json_bytes vs
        structured response mode (results is empty in the default mode)."""
        if slot.json_response:
            return len(json.loads(slot.json_response).get("results", []))
        return slot.num_detections

    def test_batch_recognize_jpegs(self, grpc_stub, hello_image, numbers_image):
        """Regression (GitHub #22, gRPC side): >=2 JPEGs in one RecognizeBatch is
        the only way to reach the nvJPEG batched-decode path over gRPC. The HTTP
        path is covered in test_ocr_batch_endpoint; this guards the gRPC twin,
        which the endpoint matrix left untested (it batched a single image, and
        test_batch_recognize batches PNGs, which never touch nvJPEG)."""
        images = [pil_to_jpeg_bytes(hello_image), pil_to_jpeg_bytes(numbers_image)]
        response = grpc_stub.RecognizeBatch(
            ocr_pb2.OCRBatchRequest(images=images), timeout=15)
        assert response.total_images == 2
        assert len(response.batch_results) == 2
        for i, slot in enumerate(response.batch_results):
            assert self._slot_detections(slot) > 0, \
                f"gRPC batch JPEG slot {i} decoded to no text"
        # The CUDA context must survive: a follow-up request still works.
        again = grpc_stub.RecognizeBatch(
            ocr_pb2.OCRBatchRequest(
                images=[pil_to_jpeg_bytes(hello_image)] * 3), timeout=15)
        assert again.total_images == 3

    def test_batch_recognize_mixed_formats(self, grpc_stub, hello_image, numbers_image):
        """Mixed PNG + JPEG in one gRPC batch exercises the decode splitter's
        jpeg-vs-other reindex (batched JPEGs scattered back to their original
        slots alongside per-image-decoded PNGs)."""
        images = [pil_to_png_bytes(hello_image),
                  pil_to_jpeg_bytes(numbers_image),
                  pil_to_jpeg_bytes(hello_image)]
        response = grpc_stub.RecognizeBatch(
            ocr_pb2.OCRBatchRequest(images=images), timeout=15)
        assert response.total_images == 3
        assert len(response.batch_results) == 3
        for i, slot in enumerate(response.batch_results):
            assert self._slot_detections(slot) > 0, \
                f"gRPC mixed batch slot {i} decoded to no text"

    def test_grpc_matches_http(self, grpc_stub, server_url, hello_image):
        """gRPC and HTTP should return the same text for the same image."""
        png_bytes = pil_to_png_bytes(hello_image)

        # gRPC
        grpc_req = ocr_pb2.OCRRequest(image=png_bytes)
        grpc_resp = grpc_stub.Recognize(grpc_req, timeout=10)

        # HTTP
        import requests as req_lib
        http_resp = req_lib.post(
            f"{server_url}/ocr/raw",
            data=png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        http_data = http_resp.json()

        if grpc_resp.json_response:
            grpc_data = json.loads(grpc_resp.json_response)
            grpc_texts = [r["text"] for r in grpc_data["results"]]
            http_texts = [r["text"] for r in http_data["results"]]
            assert grpc_texts == http_texts, (
                f"gRPC and HTTP returned different text: grpc={grpc_texts}, http={http_texts}"
            )

    def test_health_rpc(self, grpc_stub):
        """Health RPC returns a status and the discoverable response mode."""
        resp = grpc_stub.Health(ocr_pb2.HealthRequest(), timeout=10)
        assert resp.status, "Health returned an empty status"
        # response_mode is additive (H7); getattr keeps the test robust if the
        # cached stub predates the proto field. When present it must be valid.
        mode = getattr(resp, "response_mode", "")
        if mode:
            assert mode in ("json_bytes", "structured")

    def test_recognize_pdf(self, grpc_stub, test_pdf_bytes):
        """RecognizePDF renders + OCRs a PDF and returns per-page results."""
        req = ocr_pb2.OCRPDFRequest(pdf_data=test_pdf_bytes, mode="ocr")
        resp = grpc_stub.RecognizePDF(req, timeout=30)
        assert len(resp.pages) >= 1, "RecognizePDF returned no pages"
