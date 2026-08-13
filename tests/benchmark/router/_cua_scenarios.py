"""CUA-router benchmark scenarios: fixture lists, thresholds, self-validator.

Three scenarios per plan 06 §1:
  - text_only       (POST /ocr/raw)
  - formula_heavy   (POST /ocr/pdf?mode=ocr)
  - table_heavy     (POST /ocr/raw + POST /ocr/pdf?mode=ocr)

Each scenario describes the fixtures to hit, per-fixture sample count, and
the verdict thresholds the regression engine should apply.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
PDF_DIR = FIXTURES_DIR / "pdf"


# Layout classes that disqualify a fixture from the text-only scenario.
# (see include/turbo_ocr/core/layout_types.h)
#   5  = display_formula
#   11 = formula_number
#   15 = inline_formula
#   21 = table
FORBIDDEN_TEXT_CLASS_IDS: set[int] = {5, 11, 15, 21}


@dataclass
class FixtureSpec:
    path: Path
    content_type: str
    kind: str  # 'image' | 'pdf'


@dataclass
class Scenario:
    name: str
    endpoint: str
    fixtures: list[FixtureSpec]
    n_per_fixture: int = 60
    warmup: int = 10
    # absolute pass/alert/hard thresholds (None = no absolute ceiling)
    target_p50_ms: float | None = None
    hard_p50_ms: float | None = None
    halt_p50_ms: float | None = None
    p99_ms: float | None = None
    # delta vs text scenario thresholds (relative). 1.20 = +20%.
    delta_alert_ratio: float | None = None


# -- Fixture lists per plan 06 §1 ------------------------------------------------

def _png(name: str) -> FixtureSpec:
    return FixtureSpec(IMAGES_DIR / "png" / name, "image/png", "image")


def _jpg(name: str) -> FixtureSpec:
    return FixtureSpec(IMAGES_DIR / "jpeg" / name, "image/jpeg", "image")


def _pdf(name: str) -> FixtureSpec:
    return FixtureSpec(PDF_DIR / name, "application/pdf", "pdf")


TEXT_ONLY_FIXTURES: list[FixtureSpec] = [
    _png("business_letter.png"),
    _png("dense_text.png"),
    _png("multi_language.png"),
    _png("small_text.png"),
    _jpg("03_book_page.jpg"),
    _jpg("08_document_scan.jpg"),
]

FORMULA_FIXTURES: list[FixtureSpec] = [
    _pdf("formulas.pdf"),
]

TABLE_FIXTURES: list[FixtureSpec] = [
    _png("table.png"),
    _pdf("tables_document.pdf"),
]


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="text_only",
            endpoint="/ocr/raw",
            fixtures=TEXT_ONLY_FIXTURES,
            target_p50_ms=270.0,
            hard_p50_ms=280.0,
            halt_p50_ms=300.0,
            p99_ms=400.0,
        ),
        Scenario(
            name="formula_heavy",
            endpoint="/ocr/pdf?mode=ocr",
            fixtures=FORMULA_FIXTURES,
            delta_alert_ratio=1.20,
        ),
        Scenario(
            name="table_heavy",
            endpoint="/ocr/pdf?mode=ocr",  # PDF fixtures use this; image fixtures override below
            fixtures=TABLE_FIXTURES,
            delta_alert_ratio=1.20,
        ),
    ]


# -- Fixture hashing -------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def fixture_hashes(fixtures: list[FixtureSpec]) -> dict[str, str]:
    return {f.path.name: sha256_file(f.path) for f in fixtures if f.path.exists()}


# -- Self-validator (text-only) --------------------------------------------------

async def validate_text_fixtures(
    session: aiohttp.ClientSession,
    base_url: str,
    fixtures: list[FixtureSpec],
    timeout_s: float = 5.0,
) -> tuple[list[FixtureSpec], list[str]]:
    """Best-effort: POST each fixture to /layout (if exposed) and drop any
    whose returned class_ids intersect FORBIDDEN_TEXT_CLASS_IDS.

    If /layout is not available (404/connection error), all fixtures pass.
    Returns (kept_fixtures, warnings).
    """
    kept: list[FixtureSpec] = []
    warns: list[str] = []
    url = base_url.rstrip("/") + "/layout"
    for fx in fixtures:
        if not fx.path.exists():
            warns.append(f"missing fixture: {fx.path}")
            continue
        body = fx.path.read_bytes()
        try:
            async with session.post(
                url,
                data=body,
                headers={"Content-Type": fx.content_type},
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as r:
                if r.status == 404:
                    # endpoint not available; accept all and return early
                    return fixtures, ["layout endpoint unavailable; skipping self-validation"]
                if r.status != 200:
                    warns.append(f"{fx.path.name}: layout HTTP {r.status}; keeping")
                    kept.append(fx)
                    continue
                payload = await r.json()
        except Exception as e:
            warns.append(f"layout probe failed ({type(e).__name__}); skipping self-validation")
            return fixtures, warns

        class_ids = _extract_class_ids(payload)
        bad = class_ids & FORBIDDEN_TEXT_CLASS_IDS
        if bad:
            warns.append(f"{fx.path.name}: layout returned forbidden class_ids {sorted(bad)}; rejected")
            continue
        kept.append(fx)
    return kept, warns


def _extract_class_ids(payload: Any) -> set[int]:
    """Walk a layout response and collect any integer 'class_id' field."""
    out: set[int] = set()

    def walk(node: Any):
        if isinstance(node, dict):
            cid = node.get("class_id")
            if isinstance(cid, int):
                out.add(cid)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    return out


# -- Single-request senders ------------------------------------------------------

async def send_image(session: aiohttp.ClientSession, url: str, ct: str, body: bytes,
                     timing: bool = False) -> tuple[int, int, dict | None]:
    headers = {"Content-Type": ct}
    if timing:
        headers["X-Turbo-Timing"] = "1"
    async with session.post(url, data=body, headers=headers) as r:
        text = await r.read()
        timings = None
        if timing and r.status == 200:
            try:
                j = json.loads(text)
                timings = j.get("timings") if isinstance(j, dict) else None
            except Exception:
                timings = None
        return r.status, 1, timings


async def send_pdf(session: aiohttp.ClientSession, url: str, ct: str, body: bytes,
                   timing: bool = False) -> tuple[int, int, dict | None]:
    headers = {"Content-Type": ct}
    if timing:
        headers["X-Turbo-Timing"] = "1"
    async with session.post(url, data=body, headers=headers) as r:
        if r.status != 200:
            await r.read()
            return r.status, 0, None
        try:
            j = await r.json()
            pages = len(j.get("pages", [])) if isinstance(j, dict) else 0
            timings = j.get("timings") if (timing and isinstance(j, dict)) else None
            return r.status, max(pages, 1), timings
        except Exception:
            return r.status, 1, None
