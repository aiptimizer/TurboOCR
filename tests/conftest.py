"""Shared pytest fixtures for the Turbo OCR test suite."""

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--server-url",
        default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"),
        help="Base URL of the running OCR server (default: http://localhost:8000)",
    )
    parser.addoption(
        "--grpc-target",
        default=os.environ.get("OCR_GRPC_TARGET", "localhost:50051"),
        help="gRPC target address (default: localhost:50051)",
    )


# ---------------------------------------------------------------------------
# Server connection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def server_url(request):
    """Base URL of the OCR HTTP server."""
    url = request.config.getoption("--server-url")
    # Verify server is reachable
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Server at {url} returned {r.status_code} on /health")
    except requests.ConnectionError:
        pytest.skip(f"Server not reachable at {url}")
    return url


@pytest.fixture(scope="session")
def grpc_target(request):
    """gRPC target address."""
    return request.config.getoption("--grpc-target")


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------

# Scalable TTFs, by platform. macOS ships Helvetica/Arial; Linux distros vary.
_FONT_PATHS = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def _get_font(size=28):
    """A real scalable TTF at `size` px — never the bitmap fallback.

    WHY THIS MATTERS: the list used to be Linux-only and silently fell back to
    ImageFont.load_default(), which on macOS renders ~8px glyphs REGARDLESS of
    `size`. Every test drawing text then measured the detector's small-glyph
    limit instead of whatever it meant to test — e.g. test_box_sorting saw one
    of its two words dropped and read that as a sort-order bug.

    Failing loudly beats degrading silently: a missing font makes the whole
    suite meaningless, so it should look like a broken environment, not like a
    broken detector.
    """
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise RuntimeError(
        "no scalable TTF found for text-rendering tests; tried:\n  "
        + "\n  ".join(_FONT_PATHS)
        + "\nInstall DejaVu or Liberation fonts (Linux), or run on a system "
          "with the stock macOS fonts."
    )


@pytest.fixture(scope="session")
def font():
    """A usable font for drawing text on images."""
    return _get_font(28)


@pytest.fixture(scope="session")
def small_font():
    """A smaller font for dense text."""
    return _get_font(18)


def make_text_image(text, width=400, height=100, font_size=28, bg="white", fg="black"):
    """Create a PIL Image with the given text rendered on it."""
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    draw.text((10, 10), text, fill=fg, font=font)
    return img


def pil_to_png_bytes(img):
    """Convert PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_jpeg_bytes(img, quality=90):
    """Convert PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pil_to_base64(img, fmt="PNG"):
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Pre-generated test image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hello_image():
    """A simple image containing the word 'HELLO'."""
    return make_text_image("HELLO", width=300, height=80, font_size=40)


@pytest.fixture(scope="session")
def numbers_image():
    """Image with numbers '12345'."""
    return make_text_image("12345", width=300, height=80, font_size=40)


@pytest.fixture(scope="session")
def paragraph_image():
    """Image with multiple lines of text."""
    text = "Line one text here\nLine two more text\nLine three final"
    return make_text_image(text, width=500, height=200, font_size=24)


@pytest.fixture(scope="session")
def blank_image():
    """A blank white image with no text."""
    return Image.new("RGB", (200, 200), "white")


@pytest.fixture(scope="session")
def ordering_image():
    """Image with numbered blocks at known positions for ordering tests.

    Layout (3x3 grid):
        1  2  3
        4  5  6
        7  8  9
    """
    img = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(36)
    positions = [
        (50, 30),   (250, 30),  (450, 30),   # row 1
        (50, 150),  (250, 150), (450, 150),   # row 2
        (50, 280),  (250, 280), (450, 280),   # row 3
    ]
    for i, (x, y) in enumerate(positions, 1):
        draw.text((x, y), str(i), fill="black", font=font)
    return img


@pytest.fixture(scope="session")
def multiline_ordered_image():
    """Image with text at controlled positions for verifying top-to-bottom,
    left-to-right ordering. Uses word-like labels."""
    img = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(30)
    # Row 1
    draw.text((50, 30), "ALPHA", fill="black", font=font)
    draw.text((350, 30), "BRAVO", fill="black", font=font)
    # Row 2
    draw.text((50, 150), "CHARLIE", fill="black", font=font)
    draw.text((350, 150), "DELTA", fill="black", font=font)
    # Row 3
    draw.text((50, 280), "ECHO", fill="black", font=font)
    draw.text((350, 280), "FOXTROT", fill="black", font=font)
    return img


@pytest.fixture(scope="session")
def unique_images():
    """Generate 10 images each with unique text for concurrency tests."""
    images = []
    for i in range(10):
        text = f"UNIQUE{i:04d}"
        img = make_text_image(text, width=400, height=100, font_size=36)
        images.append((text, img))
    return images


# ---------------------------------------------------------------------------
# Real test data fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
PDF_DIR = FIXTURES_DIR / "pdf"
# Back-compat alias: TEST_DATA_DIR used to be tests/test_data; callers now use fixtures/
TEST_DATA_DIR = FIXTURES_DIR
EDGE_CASES_DIR = FIXTURES_DIR / "edge_cases"
IMAGE_EXPECTED_DIR = IMAGES_DIR / "expected"
PDF_EXPECTED_DIR = PDF_DIR / "expected"


def _load_image_expected(stem):
    path = IMAGE_EXPECTED_DIR / f"{stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _load_pdf_expected(stem):
    path = PDF_EXPECTED_DIR / f"{stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def edge_cases_dir():
    return EDGE_CASES_DIR


@pytest.fixture(scope="session")
def png_test_files():
    files = sorted((IMAGES_DIR / "png").glob("*.png"))
    return [(f, _load_image_expected(f.stem)) for f in files]


@pytest.fixture(scope="session")
def jpeg_test_files():
    files = sorted((IMAGES_DIR / "jpeg").glob("*.jpg"))
    return [(f, _load_image_expected(f.stem)) for f in files]


@pytest.fixture(scope="session")
def pdf_test_files():
    files = sorted(PDF_DIR.glob("*.pdf"))
    return [(f, _load_pdf_expected(f.stem)) for f in files]


@pytest.fixture(scope="session")
def all_png_paths():
    return sorted((IMAGES_DIR / "png").glob("*.png"))


@pytest.fixture(scope="session")
def all_jpeg_paths():
    return sorted((IMAGES_DIR / "jpeg").glob("*.jpg"))


@pytest.fixture(scope="session")
def all_pdf_paths():
    return sorted(PDF_DIR.glob("*.pdf"))


def load_expected(path):
    """Load expected JSON for a fixture image/PDF by path."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return _load_pdf_expected(p.stem)
    return _load_image_expected(p.stem)


# ---------------------------------------------------------------------------
# Async HTTP session fixture (aiohttp)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_session():
    """Session-scoped aiohttp ClientSession for async benchmarks."""
    import aiohttp
    session = aiohttp.ClientSession()
    yield session
    await session.close()
