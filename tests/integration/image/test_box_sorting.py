"""Unit tests for box sorting logic.

The sorted_boxes() function in common/geometry/box.h sorts detected text boxes in
reading order: top-to-bottom, left-to-right, using Y-quantization into
line bands (kSameLineThreshold=10). These tests verify the ordering
through the server by sending images with text at controlled positions.
"""

import pytest
import requests

from conftest import make_text_image, pil_to_base64, pil_to_png_bytes
from PIL import Image, ImageDraw, ImageFont
import os


# Scalable BOLD TTFs, by platform. These tests draw short synthetic words and
# assert their ORDER, so the glyphs must be large enough for the detector to
# find every one of them reliably — see _get_font.
_FONT_PATHS = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _get_font(size=36):
    """A real scalable TTF at `size` px — never the bitmap fallback.

    WHY THIS MATTERS: the list above used to be Linux-only, so on macOS this
    fell through to ImageFont.load_default() — a bitmap face that renders ~8px
    glyphs no matter what `size` says. At that size the detector legitimately
    misses words, and these tests then read a MISSING box as a sort-order
    violation. The failure looked like a backend bug and was a fixture bug.

    Raise rather than degrade: a missing font makes these assertions
    meaningless, and that should look like a broken environment.
    """
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise RuntimeError(
        "no scalable bold TTF found for box-sorting tests; tried:\n  "
        + "\n  ".join(_FONT_PATHS)
    )


class TestBoxSorting:
    """Test that OCR results come back in correct reading order."""

    def test_left_to_right_single_line(self, server_url):
        """Two words on the same line should appear left then right."""
        img = Image.new("RGB", (600, 100), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(36)
        draw.text((20, 20), "AAA", fill="black", font=font)
        draw.text((400, 20), "BBB", fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 2:
            pytest.skip("Need at least 2 detections for ordering test")

        # First result should be to the left of second
        first_x = results[0]["bounding_box"][0][0]
        second_x = results[1]["bounding_box"][0][0]
        assert first_x < second_x, (
            f"Expected left-to-right order: first_x={first_x} should be < second_x={second_x}"
        )

    def test_top_to_bottom(self, server_url):
        """Text on separate lines should be ordered top to bottom."""
        img = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(36)
        draw.text((50, 20), "TOP", fill="black", font=font)
        draw.text((50, 150), "BOTTOM", fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 2:
            pytest.skip("Need at least 2 detections")

        first_y = results[0]["bounding_box"][0][1]
        second_y = results[1]["bounding_box"][0][1]
        assert first_y < second_y, (
            f"Expected top-to-bottom order: first_y={first_y} < second_y={second_y}"
        )

    def test_reading_order_grid(self, server_url):
        """Numbered grid should produce results in approximate reading order.

        Uses well-separated rows with large text to ensure reliable detection.
        """
        img = Image.new("RGB", (800, 600), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(48)
        # Three rows, widely separated, two items per row
        positions = [
            (50, 20),   (450, 20),    # row 1
            (50, 220),  (450, 220),   # row 2
            (50, 420),  (450, 420),   # row 3
        ]
        labels = ["10", "20", "30", "40", "50", "60"]
        for (x, y), label in zip(positions, labels):
            draw.text((x, y), label, fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 4:
            pytest.skip(f"Only {len(results)} detections, need at least 4")

        # Verify bounding box Y coordinates are non-decreasing (reading order)
        centers_y = [sum(p[1] for p in item["bounding_box"]) / 4 for item in results]
        # Allow boxes on the same line to be in any Y order (within 50px)
        for i in range(1, len(centers_y)):
            if abs(centers_y[i] - centers_y[i - 1]) > 50:
                assert centers_y[i] > centers_y[i - 1], (
                    f"Row ordering violation at index {i}: "
                    f"y[{i-1}]={centers_y[i-1]:.0f} > y[{i}]={centers_y[i]:.0f}"
                )

    def test_same_line_y_quantization(self, server_url):
        """Boxes within kSameLineThreshold=10px of each other vertically
        should be treated as the same line and sorted by x."""
        img = Image.new("RGB", (700, 100), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(30)
        # Place B slightly lower (within 10px threshold) but to the left
        draw.text((400, 25), "AAA", fill="black", font=font)
        draw.text((50, 30), "BBB", fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 2:
            pytest.skip("Need at least 2 detections")

        # BBB is at x=50, AAA is at x=400 -- BBB should come first (left-to-right)
        first_x = results[0]["bounding_box"][0][0]
        second_x = results[1]["bounding_box"][0][0]
        assert first_x < second_x, (
            f"Y-quantization failed: items on near-same line not sorted by X. "
            f"first_x={first_x}, second_x={second_x}"
        )
