"""Pass your own httpx.Client for custom timeouts, limits, and headers."""

from pathlib import Path

import httpx

from turboocr import Client

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

http = httpx.Client(
    base_url="http://localhost:8000",
    timeout=httpx.Timeout(connect=2.0, read=60.0, write=60.0, pool=30.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    headers={"X-Trace-Source": "docs-example"},
    # verify="/path/to/ca-bundle.pem",                       # custom CA
    # cert=("/path/to/client.crt", "/path/to/client.key"),   # mTLS
)

client = Client(http_client=http)
response = client.recognize_image(IMAGE, include_blocks=True)
print(f"recognized {len(response.results)} items via shared httpx.Client")

# Output:
# recognized 71 items via shared httpx.Client
