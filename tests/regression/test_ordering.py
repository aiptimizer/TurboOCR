"""Regression tests for text ordering.

Verifies that OCR results are returned in reading order (top-to-bottom,
left-to-right). Tests both generated images (original) and real test data
from tests/fixtures/images/ (business_letter.png, dense_text.png).
"""

import os

import pytest
import requests
from PIL import Image, ImageDraw, ImageFont

from conftest import pil_to_base64, load_expected, IMAGES_DIR


def _get_font(size=36):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _extract_bounding_box_centers(results):
    """Extract (center_x, center_y) for each result's bounding box."""
    centers = []
    for item in results:
        bbox = item["bounding_box"]
        cx = sum(p[0] for p in bbox) / 4
        cy = sum(p[1] for p in bbox) / 4
        centers.append((cx, cy))
    return centers


def _ocr_raw_file(server_url, file_path, timeout=30):
    """Send a real file to /ocr/raw and return the parsed JSON response."""
    suffix = file_path.suffix.lower()
    content_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "image/png")
    data = file_path.read_bytes()
    r = requests.post(
        f"{server_url}/ocr/raw",
        data=data,
        headers={"Content-Type": content_type},
        timeout=timeout,
    )
    assert r.status_code == 200, f"Failed for {file_path.name}: HTTP {r.status_code}"
    return r.json()


def _verify_reading_order(results, tolerance_y=30):
    """Verify results are in approximate reading order (top-to-bottom, left-to-right).

    Returns list of violations (empty if ordering is correct).
    Rows are grouped by Y-center within tolerance_y pixels.
    """
    if len(results) < 2:
        return []

    centers = _extract_bounding_box_centers(results)

    # Group results into rows by Y-center
    rows = []
    for i, (cx, cy) in enumerate(centers):
        placed = False
        for row in rows:
            # Check if this item belongs to an existing row
            row_y = sum(c[1] for _, c in row) / len(row)
            if abs(cy - row_y) <= tolerance_y:
                row.append((i, (cx, cy)))
                placed = True
                break
        if not placed:
            rows.append([(i, (cx, cy))])

    # Sort rows by Y position, items within rows by X position
    rows.sort(key=lambda row: sum(c[1] for _, c in row) / len(row))
    for row in rows:
        row.sort(key=lambda item: item[1][0])

    # Build expected order and check against actual
    expected_order = [idx for row in rows for idx, _ in row]
    violations = []
    for pos, (expected_idx, actual_idx) in enumerate(zip(expected_order, range(len(results)))):
        if expected_idx != actual_idx:
            violations.append((pos, expected_idx, actual_idx))

    return violations


class TestOrdering:
    """Verify results are ordered top-to-bottom, left-to-right."""

    def test_vertical_ordering(self, server_url):
        """Three words stacked vertically should come back top-to-bottom."""
        img = Image.new("RGB", (300, 400), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(36)
        draw.text((50, 20), "ALPHA", fill="black", font=font)
        draw.text((50, 160), "BRAVO", fill="black", font=font)
        draw.text((50, 300), "CHARLIE", fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 3:
            pytest.skip(f"Only {len(results)} detections, need 3")

        centers = _extract_bounding_box_centers(results)
        for i in range(1, len(centers)):
            assert centers[i][1] >= centers[i - 1][1] - 15, (
                f"Vertical ordering broken at index {i}: "
                f"y[{i-1}]={centers[i-1][1]:.0f} > y[{i}]={centers[i][1]:.0f}"
            )

    def test_horizontal_ordering_same_line(self, server_url):
        """Three words on the same line should be ordered left-to-right."""
        img = Image.new("RGB", (900, 100), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(36)
        draw.text((20, 20), "ONE", fill="black", font=font)
        draw.text((320, 20), "TWO", fill="black", font=font)
        draw.text((620, 20), "THREE", fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 3:
            pytest.skip(f"Only {len(results)} detections, need 3")

        centers = _extract_bounding_box_centers(results)
        for i in range(1, len(centers)):
            assert centers[i][0] > centers[i - 1][0], (
                f"Horizontal ordering broken: x[{i-1}]={centers[i-1][0]:.0f} "
                f">= x[{i}]={centers[i][0]:.0f}"
            )

    def test_two_column_layout(self, server_url):
        """Two-column layout should be read row-by-row, not column-by-column.

        Layout:
            LEFT1    RIGHT1
            LEFT2    RIGHT2

        Expected order: LEFT1, RIGHT1, LEFT2, RIGHT2
        (not LEFT1, LEFT2, RIGHT1, RIGHT2)
        """
        img = Image.new("RGB", (800, 300), "white")
        draw = ImageDraw.Draw(img)
        font = _get_font(30)
        # Row 1
        draw.text((50, 30), "LEFT1", fill="black", font=font)
        draw.text((500, 30), "RIGHT1", fill="black", font=font)
        # Row 2 -- well separated vertically
        draw.text((50, 180), "LEFT2", fill="black", font=font)
        draw.text((500, 180), "RIGHT2", fill="black", font=font)

        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]
        if len(results) < 4:
            pytest.skip(f"Only {len(results)} detections, need 4")

        texts = [item["text"].upper() for item in results]

        # Find positions of our markers in the results
        def _find(marker):
            for i, t in enumerate(texts):
                if marker in t:
                    return i
            return -1

        l1 = _find("LEFT1")
        r1 = _find("RIGHT1")
        l2 = _find("LEFT2")
        r2 = _find("RIGHT2")

        if any(x == -1 for x in [l1, r1, l2, r2]):
            pytest.skip(f"Could not find all markers in output: {texts}")

        # Row 1 should come before row 2
        assert l1 < l2 and r1 < r2, (
            f"Row ordering wrong: L1={l1}, R1={r1}, L2={l2}, R2={r2}. Texts: {texts}"
        )
        # Within each row, left should come before right
        assert l1 < r1, f"Left-right order wrong in row 1: L1={l1}, R1={r1}"
        assert l2 < r2, f"Left-right order wrong in row 2: L2={l2}, R2={r2}"

    def test_numbered_sequence(self, server_url, ordering_image):
        """3x3 numbered grid (1-9) should produce monotonically increasing sequence."""
        b64 = pil_to_base64(ordering_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        results = data["results"]

        nums = []
        for item in results:
            text = item["text"].strip()
            if text.isdigit():
                nums.append(int(text))

        if len(nums) < 5:
            pytest.skip(f"Only found {len(nums)} numbers: {nums}")

        # Check monotonically increasing
        violations = []
        for i in range(1, len(nums)):
            if nums[i] <= nums[i - 1]:
                violations.append((i, nums[i - 1], nums[i]))

        assert not violations, (
            f"Ordering violations: {violations}. Full sequence: {nums}"
        )


class TestRealDocumentOrdering:
    """Verify reading order on real multi-line documents."""

    def test_business_letter_ordering(self, server_url):
        """business_letter.png should return results in top-to-bottom reading order."""
        letter_path = IMAGES_DIR / "png" /"business_letter.png"
        if not letter_path.exists():
            pytest.skip("business_letter.png not found in fixtures/images/png/")

        actual = _ocr_raw_file(server_url, letter_path)
        results = actual["results"]
        if len(results) < 3:
            pytest.skip(f"Only {len(results)} detections, need at least 3")

        centers = _extract_bounding_box_centers(results)
        # Overall, Y-centers should be non-decreasing (with tolerance for same-row items)
        violations = 0
        for i in range(1, len(centers)):
            # Allow items on the same row (within 20px) to be in any Y-order
            if centers[i][1] < centers[i - 1][1] - 20:
                violations += 1

        max_allowed = max(1, len(centers) // 5)  # Allow up to 20% ordering anomalies
        assert violations <= max_allowed, (
            f"business_letter.png: {violations} vertical ordering violations "
            f"(max allowed {max_allowed})"
        )

    def test_dense_text_ordering(self, server_url):
        """dense_text.png should return results in top-to-bottom reading order."""
        dense_path = IMAGES_DIR / "png" /"dense_text.png"
        if not dense_path.exists():
            pytest.skip("dense_text.png not found in fixtures/images/png/")

        actual = _ocr_raw_file(server_url, dense_path)
        results = actual["results"]
        if len(results) < 5:
            pytest.skip(f"Only {len(results)} detections, need at least 5")

        centers = _extract_bounding_box_centers(results)
        violations = 0
        for i in range(1, len(centers)):
            if centers[i][1] < centers[i - 1][1] - 20:
                violations += 1

        max_allowed = max(1, len(centers) // 5)
        assert violations <= max_allowed, (
            f"dense_text.png: {violations} vertical ordering violations "
            f"(max allowed {max_allowed})"
        )

    def test_real_document_row_consistency(self, server_url):
        """For business_letter.png, verify within-row items are left-to-right ordered."""
        letter_path = IMAGES_DIR / "png" /"business_letter.png"
        if not letter_path.exists():
            pytest.skip("business_letter.png not found in fixtures/images/png/")

        actual = _ocr_raw_file(server_url, letter_path)
        results = actual["results"]
        if len(results) < 3:
            pytest.skip(f"Only {len(results)} detections")

        centers = _extract_bounding_box_centers(results)

        # Group into rows by Y-center (within 20px tolerance)
        rows = []
        for i, (cx, cy) in enumerate(centers):
            placed = False
            for row in rows:
                row_y = sum(c[1] for _, c in row) / len(row)
                if abs(cy - row_y) <= 20:
                    row.append((i, (cx, cy)))
                    placed = True
                    break
            if not placed:
                rows.append([(i, (cx, cy))])

        # Within each row, items should be ordered left-to-right
        violations = 0
        for row in rows:
            row_sorted = sorted(row, key=lambda item: item[1][0])
            for j in range(1, len(row)):
                if row[j][1][0] < row[j - 1][1][0]:
                    violations += 1

        max_violations = max(2, len(results) // 5)
        assert violations <= max_violations, (
            f"business_letter.png: {violations} within-row left-to-right violations "
            f"(max allowed {max_violations})"
        )
