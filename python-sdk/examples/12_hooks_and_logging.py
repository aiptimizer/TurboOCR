"""Observe HTTP traffic with httpx event hooks and the turboocr logger."""

import logging
from pathlib import Path

import httpx

from turboocr import Client

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
logging.getLogger("turboocr").setLevel(logging.DEBUG)


def on_request(request: httpx.Request) -> None:
    print(f"-> {request.method} {request.url.path}")


def on_response(response: httpx.Response) -> None:
    print(f"<- {response.status_code} {response.request.url.path}")


client = Client(on_request=on_request, on_response=on_response)
response = client.recognize_image(IMAGE)
print(f"recognized {len(response.results)} text items")

# Output (elapsed-ms and req-id vary every run):
# turboocr DEBUG turbo-ocr POST /ocr/raw -> 200 (<ms>ms) [req=<id>]
# -> POST /ocr/raw
# <- 200 /ocr/raw
# recognized 71 text items
