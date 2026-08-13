"""GrpcClient: same surface as Client, gRPC transport."""

from pathlib import Path

from turboocr import GrpcClient

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

with GrpcClient(target="localhost:50051") as client:
    health = client.health()
    print(f"gRPC health: ok={health.ok}")
    response = client.recognize_image(IMAGE, include_blocks=True)
    print(f"recognized {len(response.results)} items via gRPC")

# Output:
# gRPC health: ok=True
# recognized 71 items via gRPC
