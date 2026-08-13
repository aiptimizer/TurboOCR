"""Fast smoke regression tests for OCR accuracy on synthetic images.

Per-fixture scoring (PNG / JPEG / PDF against expected JSON) lives in
tests/accuracy/. This file only owns the tiny, fast smoke checks that
guard against a totally broken model.
"""

import pytest
import requests

from conftest import make_text_image, pil_to_base64, IMAGES_DIR


def _ocr_text(server_url, text, width=500, height=100, font_size=40):
    img = make_text_image(text, width=width, height=height, font_size=font_size)
    b64 = pil_to_base64(img)
    last = None
    for _ in range(3):
        last = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        if last.status_code == 200:
            return " ".join(item["text"] for item in last.json()["results"])
    assert last.status_code == 200, f"3 retries all failed ({last.status_code})"
    return ""


def _char_accuracy(expected, detected):
    """Ordered character recall: LCS(expected, detected) / len(expected).

    This was a per-character MEMBERSHIP test against the whole detected string
    (``c in d``), which is order-blind and duplicate-blind: "HELLO" vs "OLLEH"
    scored 1.0, and a model that returned the alphabet for every image passed
    every case — the exact totally-broken-model failure this suite exists to
    catch. The longest-common-subsequence ratio punishes both scrambling and
    padding while still tolerating OCR dropping or mangling a character.
    """
    if not expected:
        return 1.0
    e = expected.lower().replace(" ", "")
    d = detected.lower().replace(" ", "")
    if not e:
        return 1.0
    # O(len(e)*len(d)) DP over two short strings.
    prev = [0] * (len(d) + 1)
    for ch_e in e:
        cur = [0]
        for j, ch_d in enumerate(d):
            cur.append(max(prev[j] + 1 if ch_e == ch_d else 0, prev[j + 1], cur[-1]))
        prev = cur
    return prev[-1] / len(e)


class TestAccuracyRegression:
    MIN_RECALL = 0.70

    @pytest.mark.parametrize("text", ["HELLO", "WORLD", "12345", "ABCDEF", "Testing"])
    def test_single_word_recall(self, server_url, text):
        detected = _ocr_text(server_url, text, width=400, height=100, font_size=48)
        recall = _char_accuracy(text, detected)
        assert recall >= self.MIN_RECALL, (
            f"Recall {recall:.0%} < {self.MIN_RECALL:.0%} for '{text}'. Detected: '{detected}'"
        )

    def test_numbers_recall(self, server_url):
        detected = _ocr_text(server_url, "0123456789", width=500, height=100, font_size=48)
        assert _char_accuracy("0123456789", detected) >= self.MIN_RECALL

    def test_mixed_case(self, server_url):
        detected = _ocr_text(server_url, "Hello World", width=500, height=100, font_size=40)
        assert _char_accuracy("Hello World", detected) >= self.MIN_RECALL

    def test_multiline_recall(self, server_url):
        img = make_text_image(
            "First line here\nSecond line here", width=500, height=150, font_size=28,
        )
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        all_text = " ".join(item["text"] for item in data["results"]).lower()
        assert "first" in all_text or "line" in all_text or "second" in all_text

    def test_confidence_not_garbage(self, server_url):
        img = make_text_image("CONFIDENCE", width=500, height=100, font_size=48)
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        if not data["results"]:
            pytest.skip("No detections")
        avg_conf = sum(item["confidence"] for item in data["results"]) / len(data["results"])
        assert avg_conf > 0.5

    def test_inverted_text_detected(self, server_url):
        import sys
        sys.path.insert(0, str(IMAGES_DIR.parent))
        from generate_test_images import generate_inverted
        img = generate_inverted("INVERTED")
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert isinstance(r.json()["results"], list)
