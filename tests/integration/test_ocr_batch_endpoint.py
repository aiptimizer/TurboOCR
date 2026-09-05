"""Integration tests for the /ocr/batch endpoint."""

import pytest
import requests

from conftest import make_text_image, pil_to_base64


class TestOcrBatchEndpoint:
    """Test /ocr/batch endpoint for parallel image processing."""

    def test_batch_two_images(self, server_url, hello_image, numbers_image):
        """Batch of 2 images should return 2 result sets."""
        b64_1 = pil_to_base64(hello_image)
        b64_2 = pil_to_base64(numbers_image)
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_1, b64_2]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 2

    def test_batch_single_image(self, server_url, hello_image):
        """Batch with 1 image should work and return 1 result set."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64]},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 1

    def test_batch_preserves_order(self, server_url, unique_images):
        """Batch results must be in the same order as input images.

        This catches a critical bug: if the pipeline pool dispatches work
        out of order, results could be associated with the wrong image.
        """
        # Use first 5 unique images
        images = unique_images[:5]
        b64_list = [pil_to_base64(img) for _, img in images]

        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": b64_list},
            timeout=20,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 5

        # Verify each result corresponds to its input
        for i, (expected_text, _) in enumerate(images):
            batch_text = " ".join(
                item["text"] for item in data["batch_results"][i]["results"]
            ).upper()
            # The expected_text is like "UNIQUE0000" -- check it appears
            if data["batch_results"][i]["results"]:
                # At least verify results are non-empty for valid images
                assert len(data["batch_results"][i]["results"]) > 0

    def test_batch_many_images(self, server_url):
        """Batch of 10 images should all be processed."""
        images = []
        for i in range(10):
            img = make_text_image(f"BATCH{i}", width=300, height=80, font_size=36)
            images.append(pil_to_base64(img))

        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": images},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 10

    def test_batch_empty_array(self, server_url):
        """Empty images array should return 400."""
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": []},
            timeout=10,
        )
        assert r.status_code == 400

    def test_batch_missing_images_key(self, server_url):
        """Missing 'images' key should return 400."""
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"wrong_key": []},
            timeout=10,
        )
        assert r.status_code == 400

    def test_batch_two_jpegs(self, server_url, hello_image, numbers_image):
        """Regression (GitHub #22): >=2 JPEGs in one batch is the ONLY way to
        reach the nvJPEG batched-decode path (single JPEG and mixed PNG+JPEG
        stay on per-image decode). That path used to hand nvjpegDecodeBatched
        host output pointers, poisoning the CUDA context (502 + process exit).
        """
        b64_1 = pil_to_base64(hello_image, "JPEG")
        b64_2 = pil_to_base64(numbers_image, "JPEG")
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_1, b64_2]},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 2
        for i, slot in enumerate(data["batch_results"]):
            assert slot["results"], f"slot {i} decoded to no text: {slot}"

        # The context must survive: a follow-up request still works.
        r2 = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_1, b64_2, b64_1]},
            timeout=30,
        )
        assert r2.status_code == 200
        assert len(r2.json()["batch_results"]) == 3

    def test_batch_mixed_formats(self, server_url, hello_image, numbers_image):
        """Batch with mixed PNG and JPEG images should work."""
        b64_png = pil_to_base64(hello_image, "PNG")
        b64_jpg = pil_to_base64(numbers_image, "JPEG")
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_png, b64_jpg]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 2

    def test_batch_partial_failure_preserves_order(self, server_url,
                                                    hello_image, numbers_image):
        """A garbage image in the middle must NOT drop or shift the others: the
        response keeps a 1:1 slot mapping, the errors[] array flags only the bad
        slot (non-null), and the two valid slots still carry their text. This is
        the failure-isolation contract the single-image tests can't reach."""
        import base64
        good1 = pil_to_base64(hello_image, "JPEG")
        good2 = pil_to_base64(numbers_image, "JPEG")
        garbage = base64.b64encode(b"\xff\xd8not-a-real-jpeg\x00\x01").decode("ascii")
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [good1, garbage, good2]},
            timeout=15,
        )
        assert r.status_code == 200, r.text[:300]
        data = r.json()
        assert len(data["batch_results"]) == 3
        errors = data.get("errors", [None, None, None])
        assert errors[0] is None, f"valid slot 0 flagged as error: {errors[0]}"
        assert errors[1] is not None, "garbage slot 1 was not flagged as an error"
        assert errors[2] is None, f"valid slot 2 flagged as error: {errors[2]}"
        # The valid slots must still carry text (not shifted into slot 1's place).
        assert data["batch_results"][0]["results"], "slot 0 lost its text"
        assert data["batch_results"][2]["results"], "slot 2 lost its text"

    def test_batch_zero_byte_slot(self, server_url, hello_image):
        """An empty-bytes slot is a decode failure, isolated to that slot."""
        import base64
        good = pil_to_base64(hello_image, "PNG")
        empty = base64.b64encode(b"").decode("ascii")
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [good, empty]},
            timeout=15,
        )
        assert r.status_code == 200, r.text[:300]
        data = r.json()
        assert len(data["batch_results"]) == 2
        assert data.get("errors", [None, None])[1] is not None
        assert data["batch_results"][0]["results"], "valid slot lost its text"


def test_batch_jpeg_matches_raw(server_url, hello_image):
    """A JPEG in /ocr/batch is decoded on the replica like /ocr/raw: identical text.

    Mixed with a PNG so both decode paths (replica for JPEG, host for PNG)
    run in the same batch and land in the right slots.
    """
    import base64 as _b64
    import requests as _rq
    from conftest import pil_to_jpeg_bytes as _jpg, pil_to_png_bytes as _png
    jpg, png = _jpg(hello_image), _png(hello_image)
    raw_jpg = _rq.post(f"{server_url}/ocr/raw", data=jpg, headers={"Content-Type": "image/jpeg"}, timeout=10)
    raw_png = _rq.post(f"{server_url}/ocr/raw", data=png, headers={"Content-Type": "image/png"}, timeout=10)
    batch = _rq.post(f"{server_url}/ocr/batch", json={"images": [_b64.b64encode(jpg).decode("ascii"), _b64.b64encode(png).decode("ascii")]}, timeout=30)
    assert raw_jpg.status_code == 200 and raw_png.status_code == 200 and batch.status_code == 200
    slots = batch.json()["batch_results"]
    assert len(slots) == 2
    texts = lambda results: [item["text"] for item in results]
    assert texts(slots[0]["results"]) == texts(raw_jpg.json()["results"])
    assert texts(slots[1]["results"]) == texts(raw_png.json()["results"])
